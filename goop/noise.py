"""Auxiliary photon sources (dark noise, etc.)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from .base import PhotonSourceBase


@dataclass
class DarkNoise(PhotonSourceBase):
    """Poisson dark-count noise source for PMTs/SiPMs.

    For each channel, draws N ~ Poisson(rate_hz * window_ns * 1e-9) hits
    uniformly distributed across the time window.
    """

    rate_hz: float = 2000.0  # dark rate per channel in Hz

    @torch.no_grad()
    def sample(
        self,
        n_channels: int,
        t_start_ns: float,
        t_end_ns: float,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        window_ns = t_end_ns - t_start_ns
        if window_ns <= 0 or self.rate_hz <= 0:
            return (
                torch.zeros(0, device=device),
                torch.zeros(0, device=device, dtype=torch.long),
            )

        expected = self.rate_hz * window_ns * 1e-9  # mean hits per channel
        counts = torch.poisson(
            torch.full((n_channels,), expected, device=device)
        ).long()
        total = counts.sum().item()

        if total == 0:
            return (
                torch.zeros(0, device=device),
                torch.zeros(0, device=device, dtype=torch.long),
            )

        times = torch.rand(total, device=device) * window_ns + t_start_ns
        channels = torch.repeat_interleave(
            torch.arange(n_channels, device=device), counts
        )
        return times, channels
