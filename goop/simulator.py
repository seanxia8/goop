"""Optical TPC simulation pipeline."""

from __future__ import annotations

from typing import Any, List, Union

import torch

# any array type supporting __dlpack__ (torch.Tensor, jax.Array, etc.)
ArrayLike = Union[torch.Tensor, Any]

from dataclasses import dataclass, field

from .base import ConvolutionKernelBase, DelaySamplerBase, TOFSamplerBase
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

        # 3. Build histograms
        extra_args = {}
        wvfm_cls = SlicedWaveform if stitched else Waveform
        if stitched:
            extra_args["kernel_extent_ns"] = cfg.kernel().shape[0] * cfg.tick_ns
        
        wf = wvfm_cls.from_photons(times, channels, cfg.tick_ns, cfg.n_channels, **extra_args)

        # 4. Convolve
        return wf.convolve(cfg.kernel(), cfg.gain)
