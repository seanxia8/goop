"""Convolution kernels and waveform chunk map."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch

from .base import ConvolutionKernelBase
from .waveform_utils import _next_fft_size

__all__ = [
    "create_default_kernel",
    "create_default_response",
    "RLCKernel",
    "Response",
    "ScintillationKernel",
    "SERKernel",
    "TPBExponentialKernel",
    "TPBTriexponentialKernel",
    "TTSKernel",
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
    kernel_adc_peak: float = -25.0  # based on figure 4 of https://arxiv.org/pdf/2406.07514 (SBND simulation)
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


# PDF kernels for the differentiable pipeline
#
# Each kernel is a discretized probability density: kernel[n] is the analytic
# integral of the underlying PDF over the bin [n*dt, (n+1)*dt). The kernel
# therefore sums to 1 (up to truncation at duration_ns), so convolving a photon
# count histogram with it preserves total photon count while smearing each
# photon's arrival time according to the PDF, equivalent in expectation to
# adding an independent random delay per photon as in goop/delays.py.


def _exp_pdf_bin(tau: float, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """∫_a^b (1/τ) exp(-t/τ) dt = exp(-a/τ) − exp(-b/τ).  Valid for a, b ≥ 0."""
    return torch.exp(-a / tau) - torch.exp(-b / tau)


def _gauss_pdf_bin(
    mu: float, sigma: float, a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """∫_a^b N(t; μ, σ) dt = ½(erf((b−μ)/(σ√2)) − erf((a−μ)/(σ√2)))."""
    s = sigma * math.sqrt(2.0)
    return 0.5 * (torch.erf((b - mu) / s) - torch.erf((a - mu) / s))


@dataclass
class ScintillationKernel(ConvolutionKernelBase):
    """Discretized biexponential scintillation PDF (singlet + triplet).

    Equivalent to ScintillationBiexponentialDelay in goop/delays.py; convolving
    the photon histogram with this kernel produces the same expected per-bin
    counts as adding a per-photon scintillation delay drawn from the same
    mixture distribution.
    """

    singlet_fraction: float = 0.30
    tau_singlet_ns: float = 6.0
    tau_triplet_ns: float = 1300.0
    duration_ns: float = 13000.0  # ≈10 × tau_triplet → mass loss < 1e-4
    tick_ns: float = 1.0
    device: torch.device = torch.device("cpu")

    _kernel_cache: torch.Tensor = field(default=None, init=False, repr=False)

    @torch.no_grad()
    def __call__(self) -> torch.Tensor:
        if self._kernel_cache is None:
            dt = self.tick_ns
            n_ticks = max(1, int(self.duration_ns / dt))
            n = torch.arange(n_ticks, device=self.device, dtype=torch.float32)
            a = n * dt
            b = (n + 1) * dt
            singlet = _exp_pdf_bin(self.tau_singlet_ns, a, b)
            triplet = _exp_pdf_bin(self.tau_triplet_ns, a, b)
            self._kernel_cache = (
                self.singlet_fraction * singlet
                + (1.0 - self.singlet_fraction) * triplet
            )
        return self._kernel_cache


@dataclass
class TPBExponentialKernel(ConvolutionKernelBase):
    """Discretized single-exponential TPB re-emission PDF."""

    tau_ns: float = 20.0
    duration_ns: float = 200.0  # ≈10 × tau → mass loss < 1e-4
    tick_ns: float = 1.0
    device: torch.device = torch.device("cpu")

    _kernel_cache: torch.Tensor = field(default=None, init=False, repr=False)

    @torch.no_grad()
    def __call__(self) -> torch.Tensor:
        if self._kernel_cache is None:
            dt = self.tick_ns
            n_ticks = max(1, int(self.duration_ns / dt))
            n = torch.arange(n_ticks, device=self.device, dtype=torch.float32)
            a = n * dt
            b = (n + 1) * dt
            self._kernel_cache = _exp_pdf_bin(self.tau_ns, a, b)
        return self._kernel_cache


@dataclass
class TPBTriexponentialKernel(ConvolutionKernelBase):
    r"""Discretized 4-component exponential mixture for TPB re-emission.

    PDF: I_{TPB}(t) = Σ_i a_i (1/τ_i) e^{-t/τ_i},  Σ_i a_i = 1.
    """

    tau_1_ns: float = 5.0
    tau_2_ns: float = 49.0
    tau_3_ns: float = 3550.0
    tau_4_ns: float = 309.0
    a_1: float = 0.6
    a_2: float = 0.3
    a_3: float = 0.08
    a_4: float = 0.02
    duration_ns: float = 35500.0  # ~10 * tau_3 -> mass loss < 1e-4
    tick_ns: float = 1.0
    device: torch.device = torch.device("cpu")

    _kernel_cache: torch.Tensor = field(default=None, init=False, repr=False)

    @torch.no_grad()
    def __call__(self) -> torch.Tensor:
        if self._kernel_cache is None:
            dt = self.tick_ns
            n_ticks = max(1, int(self.duration_ns / dt))
            n = torch.arange(n_ticks, device=self.device, dtype=torch.float32)
            a = n * dt
            b = (n + 1) * dt
            self._kernel_cache = (
                self.a_1 * _exp_pdf_bin(self.tau_1_ns, a, b)
                + self.a_2 * _exp_pdf_bin(self.tau_2_ns, a, b)
                + self.a_3 * _exp_pdf_bin(self.tau_3_ns, a, b)
                + self.a_4 * _exp_pdf_bin(self.tau_4_ns, a, b)
            )
        return self._kernel_cache


@dataclass
class TTSKernel(ConvolutionKernelBase):
    """Discretized Gaussian PMT transit-time-spread PDF.

    The kernel is a Gaussian centered at ``transit_time_ns`` with width
    σ = fwhm_ns / 2.35482.  Default ``transit_time_ns=55.0`` matches the
    default ``TTSDelay`` (which adds a 55 ns transit-time shift plus jitter).

    The kernel time axis is non-negative, so set ``transit_time_ns`` >= 4σ to
    avoid clipping the Gaussian's left tail (default 55 ns >> 4 × 1 ns is fine).
    Setting ``transit_time_ns=0`` clips half the mass (matches the historical
    ``TTSDelay(apply_transit_time=True)`` behavior, but means kernel sum ~= 0.5).
    """

    fwhm_ns: float = 2.4
    transit_time_ns: float = 55.0
    duration_ns: Optional[float] = None
    tick_ns: float = 1.0
    device: torch.device = torch.device("cpu")

    _kernel_cache: torch.Tensor = field(default=None, init=False, repr=False)

    @property
    def sigma_ns(self) -> float:
        return self.fwhm_ns / 2.35482

    @torch.no_grad()
    def __call__(self) -> torch.Tensor:
        if self._kernel_cache is None:
            dt = self.tick_ns
            sigma = self.sigma_ns
            duration = self.duration_ns
            if duration is None:
                duration = self.transit_time_ns + 8.0 * sigma
            n_ticks = max(1, int(duration / dt))
            n = torch.arange(n_ticks, device=self.device, dtype=torch.float32)
            a = n * dt
            b = (n + 1) * dt
            self._kernel_cache = _gauss_pdf_bin(self.transit_time_ns, sigma, a, b)
        return self._kernel_cache


@dataclass
class Response(ConvolutionKernelBase):
    """Composite kernel: list of ConvolutionKernelBase combined via FFT-multiply.

    The combined kernel is mathematically equivalent to the time-domain
    convolution of all child kernels, but computed once via the product of
    rFFTs.  This lets the whole detector response (delay PDFs ⊛ SER) be
    pre-composed into a single per-channel FFT convolution.

    ``tick_ns`` is propagated to each child via ``with_tick_ns()`` on every
    ``__call__``, so the composite kernel always evaluates its children at the
    same tick.  Each child is also moved to ``self.device`` before FFT.
    """

    kernels: List[ConvolutionKernelBase] = field(default_factory=list)
    tick_ns: float = 1.0
    device: torch.device = torch.device("cpu")

    _kernel_cache: torch.Tensor = field(default=None, init=False, repr=False)

    @torch.no_grad()
    def __call__(self) -> torch.Tensor:
        if self._kernel_cache is None:
            if not self.kernels:
                raise ValueError("Response requires at least one child kernel")
            children = [k.with_tick_ns(self.tick_ns) for k in self.kernels]
            tensors = [k().to(self.device) for k in children]
            n_out = sum(t.shape[0] for t in tensors) - (len(tensors) - 1)
            n_fft = _next_fft_size(n_out)
            prod = None
            for t in tensors:
                ft = torch.fft.rfft(t, n=n_fft)
                prod = ft if prod is None else prod * ft
            self._kernel_cache = torch.fft.irfft(prod, n=n_fft)[:n_out]
        return self._kernel_cache


def create_default_response(
    tick_ns: float = 1.0, device: torch.device = torch.device("cpu")
) -> Response:
    """Default differentiable response: Scintillation ⊛ TPB ⊛ TTS ⊛ SER."""
    return Response(
        kernels=[
            ScintillationKernel(),
            TPBExponentialKernel(),
            TTSKernel(),
            SERKernel(),
        ],
        tick_ns=tick_ns,
        device=device,
    )
