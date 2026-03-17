"""Convolution kernels and waveform chunk map."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

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

    h(t) = -A * (exp(-t/tau_f) - exp(-t/tau_r)) + B * exp(-t/tau_o)
    with B = A * (tau_f - tau_r) / tau_o.
    """

    tau_r_ns: float = 2.5
    tau_f_ns: float = 6.0
    tau_o_ns: float = 620.0
    amplitude: float = 1.0
    duration_ns: float = 6000.0
    tick_ns: float = 1.0
    device: torch.device = torch.device("cpu")

    _kernel_cache: torch.Tensor = field(default=None, init=False, repr=False)

    @torch.no_grad()
    def __call__(self) -> torch.Tensor:
        if self._kernel_cache is None:
            n_ticks = max(1, int(self.duration_ns / self.tick_ns))
            t_ns = torch.arange(n_ticks, device=self.device, dtype=torch.float32) * self.tick_ns

            B = self.amplitude * (self.tau_f_ns - self.tau_r_ns) / self.tau_o_ns
            main_pulse = -self.amplitude * (
                torch.exp(-t_ns / self.tau_f_ns) - torch.exp(-t_ns / self.tau_r_ns)
            )
            overshoot = B * torch.exp(-t_ns / self.tau_o_ns)
            self._kernel_cache = main_pulse + overshoot
        return self._kernel_cache


def pmt_response(t, tau_r=2.5, tau_f=6.0, tau_o=620.0, A=1.0):
    """
    PMT Single Electron Response (AC-coupled, bipolar).

    Bi-exponential pulse with exponential overshoot:
    f(t) = -A[exp(-t/tau_f) - exp(-t/tau_r)] + B*exp(-t/tau_o)

    Parameters:
        t: time array (ns)
        tau_r: rise time constant (ns)
        tau_f: fall time constant (ns)
        tau_o: overshoot decay time constant (ns)
        A: amplitude scaling

    Returns:
        response: bipolar response (negative pulse, positive overshoot)
    """
    response = np.zeros_like(t, dtype=float)
    mask = t >= 0
    t_pos = t[mask]

    # Negative main pulse (bi-exponential)
    main_pulse = -A * (np.exp(-t_pos / tau_f) - np.exp(-t_pos / tau_r))

    # Positive overshoot (from zero-integral constraint: B*tau_o = A*(tau_f - tau_r))
    B = A * (tau_f - tau_r) / tau_o
    overshoot = B * np.exp(-t_pos / tau_o)

    response[mask] = main_pulse + overshoot
    return response
