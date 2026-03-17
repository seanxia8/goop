"""Concrete delay samplers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from .base import DelaySamplerBase

__all__ = [
    "ScintillationBiexponentialDelay",
    "TPBExponentialDelay",
    "TTSDelay",
    "Delays",
    "create_default_delays",
]

@dataclass
class ScintillationBiexponentialDelay(DelaySamplerBase):
    """Bi-exponential scintillation delay (singlet + triplet)."""

    singlet_fraction: float = 0.30
    tau_singlet_ns: float = 1.0
    tau_triplet_ns: float = 1530.0

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
class TTSDelay(DelaySamplerBase):
    """Gaussian PMT transit-time spread."""

    fwhm_ns: float = 1.0

    @property
    def sigma_ns(self) -> float:
        return self.fwhm_ns / 2.35482

    @torch.no_grad()
    def __call__(self, n_photons: int, device: torch.device) -> torch.Tensor:
        return torch.normal(
            mean=0.0, std=self.sigma_ns, size=(n_photons,), device=device
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