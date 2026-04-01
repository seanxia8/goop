"""Optical TPC simulation pipeline."""

from __future__ import annotations

from typing import Any, List, Optional, Union

import torch

# any array type supporting __dlpack__ (torch.Tensor, jax.Array, etc.)
ArrayLike = Union[torch.Tensor, Any]

from dataclasses import dataclass, field

from .base import ConvolutionKernelBase, DelaySamplerBase, PhotonSourceBase, TOFSamplerBase
from .digitize import DigitizationConfig
from .sampler import create_default_tof_sampler
from .delays import Delays, create_default_delays
from .kernels import create_default_kernel
from .waveform import SlicedWaveform, Waveform

@dataclass
class OpticalSimConfig:
    """Full configuration for the optical simulation pipeline.
    """

    tof_sampler: TOFSamplerBase = field(default_factory=create_default_tof_sampler)
    delays: Union[Delays, List[DelaySamplerBase]] = field(default_factory=create_default_delays)
    kernel: ConvolutionKernelBase = field(default_factory=create_default_kernel)

    aux_photon_sources: List[PhotonSourceBase] = field(default_factory=list)
    digitization: Optional[DigitizationConfig] = None

    device: str = "cuda"
    tick_ns: float = 1.0
    oversample: int = 1
    gain: float = -45.0    # per PMT gain in ADC units
    ser_jitter_std: float = 0.0      # std of multiplicative Gaussian on PE weights
    baseline_noise_std: float = 0.0  # std of per-sample Gaussian ADC noise

    def __post_init__(self):
        if not isinstance(self.oversample, int) or self.oversample < 1:
            raise ValueError(f"oversample must be an int >= 1, got {self.oversample}")
        self.n_channels = self.tof_sampler.n_channels
        if isinstance(self.delays, list):
            self.delays = Delays(self.delays)


class OpticalSimulator:
    """Full optical TPC simulation pipeline."""

    def __init__(self, config: OpticalSimConfig):
        self.config = config
        self._device = torch.device(config.device)
        self._fine_tick = config.tick_ns / config.oversample
        if config.oversample > 1:
            self._fine_kernel = config.kernel.with_tick_ns(self._fine_tick)
        else:
            self._fine_kernel = config.kernel

    @torch.no_grad()
    def simulate(
        self,
        pos: ArrayLike,
        n_photons: ArrayLike,
        t_step: ArrayLike,
        stitched: bool = True,
        subtract_t0: bool = False,
    ) -> Union[SlicedWaveform, Waveform]:
        """
        Run the full optical simulation pipeline.

        Returns SlicedWaveform if stitched=True, Waveform if stitched=False.
        """
        cfg = self.config
        device = self._device

        # auto-convert any array type supporting __dlpack__ (incl JAX) to torch.Tensor
        pos, n_photons, t_step = (
            torch.from_dlpack(a) if not isinstance(a, torch.Tensor) else a
            for a in (pos, n_photons, t_step)
        )

        if subtract_t0:
            t_step = t_step - t_step.min()

        # 1. Sample photon arrival times
        times, channels = cfg.tof_sampler.sample(pos, n_photons, t_step=t_step)

        # 2. Add stochastic delay offsets
        if times.numel() > 0:
            times = times + cfg.delays.sample(times.shape[0], device)

        # 3. Inject auxiliary photon sources (dark noise, etc.)
        if cfg.aux_photon_sources and times.numel() > 0:
            t_start = times.min().item()
            t_end = times.max().item()
            for source in cfg.aux_photon_sources:
                aux_times, aux_channels = source.sample(
                    cfg.n_channels, t_start, t_end, device
                )
                if aux_times.numel() > 0:
                    times = torch.cat([times, aux_times])
                    channels = torch.cat([channels, aux_channels])

        # 4. Build histograms (at fine resolution when oversampling)
        fine_tick = self._fine_tick
        fine_kernel_tensor = self._fine_kernel()
        extra_args = {}
        wvfm_cls = SlicedWaveform if stitched else Waveform
        if stitched:
            extra_args["kernel_extent_ns"] = fine_kernel_tensor.shape[0] * fine_tick
        if cfg.oversample > 1:
            extra_args["t0_snap_ns"] = cfg.tick_ns

        # SER amplitude jitter: per-photon multiplicative weights
        if cfg.ser_jitter_std > 0 and times.numel() > 0:
            extra_args["weights"] = torch.normal(
                1.0, cfg.ser_jitter_std, (times.shape[0],), device=device
            )

        # PE counts per channel (before convolution destroys them)
        if times.numel() > 0:
            pe_counts = torch.bincount(channels.long(), minlength=cfg.n_channels)
        else:
            pe_counts = torch.zeros(cfg.n_channels, device=device, dtype=torch.long)

        wf = wvfm_cls.from_photons(times, channels, fine_tick, cfg.n_channels, **extra_args)
        wf.attrs["pe_counts"] = pe_counts

        # 5. Convolve
        wf = wf.convolve(fine_kernel_tensor, cfg.gain)

        # 6. Downsample to output resolution
        if cfg.oversample > 1:
            wf = wf.downsample(cfg.oversample)

        # 7. Baseline noise (per-sample Gaussian, pre-digitization)
        if cfg.baseline_noise_std > 0:
            wf.adc += torch.randn_like(wf.adc) * cfg.baseline_noise_std

        # 8. Digitize (optional)
        if cfg.digitization is not None:
            wf = wf.digitize(cfg.digitization.pedestal, cfg.digitization.n_bits)

        return wf
