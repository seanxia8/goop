"""Convolution kernels and waveform chunk map."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import torch

from .base import ConvolutionKernelBase

__all__ = [
    "create_default_kernel",
    "RLCKernel",
    "SERKernel",
]

def create_default_kernel() -> ConvolutionKernelBase:
    return RLCKernel


@dataclass
class RLCKernel(ConvolutionKernelBase):
    """Damped-oscillator (RLC) impulse-response kernel.

    h(t) = exp(-t / tau_relax) * sin(t / tau_osc) * normalization
    """

    tau_relax_ns: float = 55.0
    tau_osc_ns: float = 110.0
    duration_ns: float = 9000.0
    tick_ns: float = 1.0
    device: torch.device = torch.device("cpu")

    _kernel_cache: torch.Tensor = field(default=None, init=False, repr=False)

    @torch.no_grad()
    def __call__(self) -> torch.Tensor:
        if self._kernel_cache is None:
            n_ticks = int(self.duration_ns / self.tick_ns)
            t_ns = torch.arange(n_ticks, device=self.device, dtype=torch.float32) * self.tick_ns
            self._kernel_cache = (
                torch.exp(-t_ns / self.tau_relax_ns)
                * torch.sin(t_ns / self.tau_osc_ns)
                * (self.tau_osc_ns**2 + self.tau_relax_ns**2)
                / (self.tau_osc_ns * self.tau_relax_ns**2)
            )
        return self._kernel_cache


@dataclass
class SERKernel(ConvolutionKernelBase):
    """PMT single-electron response kernel with AC-coupled overshoot.

    h(t) = -A * (exp(-t/tau_f) - exp(-t/tau_r))
           + B * (exp(-t/tau_o) - exp(-t/tau_r))
    with B = A * (tau_f - tau_r) / (tau_o - tau_r) so ∫h dt = 0 and h(0) = 0.
    """

    tau_r_ns: float = 0.796
    tau_f_ns: float = 10.951
    tau_o_ns: float = 651.44
    duration_ns: float = 6000.0
    tick_ns: float = 1.0
    kernel_adc_peak: float = -25.0  # just based on figure 4 of https://arxiv.org/pdf/2406.07514
    device: torch.device = torch.device("cpu")

    _kernel_cache: torch.Tensor = field(default=None, init=False, repr=False)

    @torch.no_grad()
    def __call__(self) -> torch.Tensor:
        if self._kernel_cache is None:
            n_ticks = max(1, int(self.duration_ns / self.tick_ns))
            t_ns = torch.arange(n_ticks, device=self.device, dtype=torch.float32) * self.tick_ns

            tr, tf, to = self.tau_r_ns, self.tau_f_ns, self.tau_o_ns
            B = -(tf - tr) / (to - tr)
            main_pulse = (torch.exp(-t_ns / tf) - torch.exp(-t_ns / tr))
            overshoot = B * (torch.exp(-t_ns / to) - torch.exp(-t_ns / tr))
            self._kernel_cache = main_pulse + overshoot
            self._kernel_cache *= self.kernel_adc_peak / self._kernel_cache.abs().amax()
        return self._kernel_cache