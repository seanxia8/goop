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
    gain: float = -45.0    # per PMT gain in ADC units

    def __post_init__(self):
        self.n_channels = self.tof_sampler.n_channels
        if isinstance(self.delays, list):
            self.delays = Delays(self.delays)
    

class OpticalSimulator:
    """Full optical TPC simulation pipeline."""


    def __init__(self, config: OpticalSimConfig):
        self.config = config
        self._device = torch.device(config.device)

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

        # 4. Build histograms
        extra_args = {}
        wvfm_cls = SlicedWaveform if stitched else Waveform
        if stitched:
            extra_args["kernel_extent_ns"] = cfg.kernel().shape[0] * cfg.tick_ns

        wf = wvfm_cls.from_photons(times, channels, cfg.tick_ns, cfg.n_channels, **extra_args)

        # 5. Convolve
        wf = wf.convolve(cfg.kernel(), cfg.gain)

        # 6. Digitize (optional)
        if cfg.digitization is not None:
            wf = wf.digitize(cfg.digitization.pedestal, cfg.digitization.n_bits)

        return wf
