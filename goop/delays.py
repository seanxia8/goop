"""Concrete delay samplers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from .base import DelaySamplerBase

__all__ = [
    "ScintillationBiexponentialDelay",
    "TPBExponentialDelay",
    "TPBTriexponentialDelay",
    "TTSDelay",
    "Delays",
    "create_default_delays",
]

@dataclass
class ScintillationBiexponentialDelay(DelaySamplerBase):
    """Bi-exponential scintillation delay (singlet + triplet).

    Source: https://link.springer.com/article/10.1140/epjc/s10052-024-13306-3

    The slow (triplet) delay is set to 1300 ns. Other sources say ~1,530 ns, but this 1300 ns
    is taken from dedicated measurements of light signals without WLS.
    """

    singlet_fraction: float = 0.30
    tau_singlet_ns: float = 6.0
    tau_triplet_ns: float = 1300.0  # other say 1530 ns but this may be due to WLS times

    @torch.no_grad()
    def __call__(self, n_photons: int, device: torch.device) -> torch.Tensor:
        is_singlet = torch.rand(n_photons, device=device) < self.singlet_fraction
        delays = torch.empty(n_photons, device=device)
        n_s = is_singlet.sum().item()
        if n_s > 0:
            delays[is_singlet] = torch.empty(n_s, device=device).exponential_(
                1.0 / self.tau_singlet_ns
            )
        n_t = n_photons - n_s
        if n_t > 0:
            delays[~is_singlet] = torch.empty(n_t, device=device).exponential_(
                1.0 / self.tau_triplet_ns
            )
        return delays


@dataclass
class TPBExponentialDelay(DelaySamplerBase):
    """Exponential TPB wavelength-shifter re-emission delay."""

    tau_ns: float = 20.0

    @torch.no_grad()
    def __call__(self, n_photons: int, device: torch.device) -> torch.Tensor:
        return torch.empty(n_photons, device=device).exponential_(1.0 / self.tau_ns)

@dataclass
class TPBTriexponentialDelay(DelaySamplerBase):
    r"""Multi-exponential TPB wavelength-shifter re-emission delay.

    Source: https://link.springer.com/article/10.1140/epjc/s10052-024-13306-3

    I_{TPB}(t) = \sum_i a_i \frac{e^{-t/\tau_i}}{\tau_i}, \sum_i a_i = 1.
    Sampled as a discrete mixture: choose i with probability a_i, then sample an
    exponential with mean tau_i (rate 1/tau_i).
    """

    tau_1_ns: float = 5.0    # 1 - 10 ns in literature
    tau_2_ns: float = 49.0   # +/- 1 ns
    tau_3_ns: float = 3550.0 # +/- 500 ns
    tau_4_ns: float = 309.0  # +/- 10 ns
    a_1: float = 0.6         # +/- 0.01
    a_2: float = 0.3         # +/- 0.01
    a_3: float = 0.08        # +/- 0.01
    a_4: float = 0.02        # +/- 0.01

    @torch.no_grad()
    def __call__(self, n_photons: int, device: torch.device) -> torch.Tensor:
        weights = torch.tensor([self.a_1, self.a_2, self.a_3, self.a_4], device=device)
        taus = torch.tensor(
            [self.tau_1_ns, self.tau_2_ns, self.tau_3_ns, self.tau_4_ns], device=device
        )
        idx = torch.multinomial(weights, n_photons, replacement=True)
        delays = torch.empty(n_photons, device=device)
        for k in range(4):
            mask = idx == k
            n_k = int(mask.sum().item())
            if n_k > 0:
                delays[mask] = torch.empty(n_k, device=device).exponential_(
                    float((1.0 / taus[k]).item())
                )
        return delays


@dataclass
class TTSDelay(DelaySamplerBase):
    """Gaussian PMT transit-time spread.
    
    Source: https://link.springer.com/article/10.1140/epjc/s10052-024-13306-3
    """

    fwhm_ns: float = 2.4 # 
    _transit_time_ns: float = 55.0
    apply_transit_time: bool = False

    @property
    def sigma_ns(self) -> float:
        """Standard deviation of the Gaussian distribution in ns.
        
        FWHM = 2 * sqrt(2 * ln(2)) * sigma => sigma = FWHM / (2 * sqrt(2 * ln(2)))
        """
        return self.fwhm_ns / 2.35482

    @torch.no_grad()
    def __call__(self, n_photons: int, device: torch.device) -> torch.Tensor:
        return torch.normal(
            mean=(0.0 if self.apply_transit_time else self._transit_time_ns),
            std=self.sigma_ns,
            size=(n_photons,),
            device=device,
        )


class Delays:
    """Applies a sequence of delay samplers, summing their offsets.

    Parameters
    ----------
    delays : list of DelaySamplerBase instances, applied in order.
    """

    def __init__(self, delays: List[DelaySamplerBase]):
        self.delays = list(delays)

    @torch.no_grad()
    def sample(self, n_photons: int, device: torch.device) -> torch.Tensor:
        """Return (n_photons,) tensor of combined time offsets in ns."""
        total = torch.zeros(n_photons, device=device)
        for delay in self.delays:
            total.add_(delay(n_photons, device))
        return total

    def __len__(self) -> int:
        return len(self.delays)

    def __iter__(self):
        return iter(self.delays)


def create_default_delays() -> Delays:
    """Standard delay chain: scintillation + TPB + PMT TTS."""
    return Delays([ScintillationBiexponentialDelay(), TPBExponentialDelay(), TTSDelay()])