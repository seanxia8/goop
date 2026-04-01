"""Convolution kernels and waveform chunk map."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .base import ConvolutionKernelBase

__all__ = [
    "create_default_kernel",
    "RLCKernel",
    "SERKernel",
]


def create_default_kernel() -> ConvolutionKernelBase:
    return RLCKernel


def _exp_integral(tau: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """∫_a^b exp(-t/tau) dt = tau * (exp(-a/tau) - exp(-b/tau))."""
    return tau * (torch.exp(-a / tau) - torch.exp(-b / tau))


def _rlc_antiderivative(
    t: torch.Tensor, tau_relax: float, tau_osc: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Antiderivative of exp(-t/tau_relax) * sin(t/tau_osc)."""
    a = torch.tensor(-1.0 / tau_relax, device=device, dtype=dtype)
    b = torch.tensor(1.0 / tau_osc, device=device, dtype=dtype)
    den = a * a + b * b
    ex = torch.exp(a * t)
    return ex * (a * torch.sin(b * t) - b * torch.cos(b * t)) / den


@dataclass
class RLCKernel(ConvolutionKernelBase):
    """Damped-oscillator (RLC) impulse-response kernel.

    h(t) = exp(-t / tau_relax) * sin(t / tau_osc) * normalization

    Discrete samples are the **average** of h(t) over each tick bin
    [n·Δ, (n+1)·Δ), not point values at n·Δ.
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
            dt = self.tick_ns
            n_ticks = int(self.duration_ns / dt)
            dtype = torch.float32
            edges = torch.arange(n_ticks + 1, device=self.device, dtype=dtype) * dt
            F = _rlc_antiderivative(
                edges, self.tau_relax_ns, self.tau_osc_ns, self.device, dtype
            )
            integral = F[1:] - F[:-1]
            c = (self.tau_osc_ns**2 + self.tau_relax_ns**2) / (
                self.tau_osc_ns * self.tau_relax_ns**2
            )
            self._kernel_cache = c * integral / dt
        return self._kernel_cache


@dataclass
class SERKernel(ConvolutionKernelBase):
    """PMT single-electron response kernel with AC-coupled overshoot.

    h(t) = (exp(-t/tau_f) - exp(-t/tau_r))
           + B * (exp(-t/tau_o) - exp(-t/tau_r))
    with B = -(tau_f - tau_r) / (tau_o - tau_r) so ∫h dt = 0 on (0, ∞).

    Discrete samples are the **average** of h(t) over each tick bin
    [n·Δ, (n+1)·Δ), then scaled so max |sample| matches |kernel_adc_peak|.
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
            dt = self.tick_ns
            n_ticks = max(1, int(self.duration_ns / dt))
            dtype = torch.float32
            tr = torch.tensor(self.tau_r_ns, device=self.device, dtype=dtype)
            tf = torch.tensor(self.tau_f_ns, device=self.device, dtype=dtype)
            to = torch.tensor(self.tau_o_ns, device=self.device, dtype=dtype)

            n = torch.arange(n_ticks, device=self.device, dtype=dtype)
            a = n * dt
            b = (n + 1) * dt

            main = _exp_integral(tf, a, b) - _exp_integral(tr, a, b)
            osh_basis = _exp_integral(to, a, b) - _exp_integral(tr, a, b)
            b_mix = -(self.tau_f_ns - self.tau_r_ns) / (self.tau_o_ns - self.tau_r_ns)
            integral = main + b_mix * osh_basis
            self._kernel_cache = integral / dt
            self._kernel_cache *= self.kernel_adc_peak / self._kernel_cache.abs().amax()
        return self._kernel_cache
