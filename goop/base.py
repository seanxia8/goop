"""Abstract base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import torch


class TOFSamplerBase(ABC):
    """Base class for time-of-flight samplers."""

    @abstractmethod
    def sample(
        self,
        pos: torch.Tensor,
        n_photons: torch.Tensor,
        t_step: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample photon arrival times and channel IDs.

        Parameters
        ----------
        pos : (N, 3) positions in mm.
        n_photons : (N,) photons emitted per step.
        t_step : (N,) emission time per step in ns.

        Returns
        -------
        times : (M,) detected photon arrival times in ns.
        channels : (M,) PMT channel IDs.
        """
        ...

    @property
    @abstractmethod
    def n_channels(self) -> int:
        """Number of output channels."""
        ...

class DelaySamplerBase(ABC):
    """Base class for stochastic time-offset samplers."""

    @abstractmethod
    def __call__(self, n_photons: int, device: torch.device) -> torch.Tensor:
        """Return (n_photons,) tensor of time offsets in ns."""
        ...


class PhotonSourceBase(ABC):
    """Base class for auxiliary photon sources (dark noise, afterpulsing, etc.).

    Unlike DelaySamplerBase (which adds time offsets to existing photons),
    PhotonSourceBase creates NEW photons with their own times and channels.
    Called between delay sampling and histogramming in the pipeline.
    """

    @abstractmethod
    def sample(
        self,
        n_channels: int,
        t_start_ns: float,
        t_end_ns: float,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate auxiliary photon hits.

        Parameters
        ----------
        n_channels : Total number of PMT channels.
        t_start_ns : Start of the time window in ns.
        t_end_ns : End of the time window in ns.
        device : Torch device.

        Returns
        -------
        times : (M,) photon arrival times in ns.
        channels : (M,) PMT channel IDs.
        """
        ...


class ConvolutionKernelBase(ABC):
    """Base class for impulse-response kernels.

    Subclasses must store tick_ns and device as attributes,
    and implement __call__ (returns kernel tensor).
    """

    @abstractmethod
    def __call__(self) -> torch.Tensor:
        """Return 1-D kernel tensor (time domain)."""
        ...
