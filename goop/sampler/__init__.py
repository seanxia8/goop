"""Photon time-of-flight samplers.

Class hierarchy:

    TOFSamplerBase (ABC)                          — goop/base.py
     └── PCATOFSampler (ABC)                      — shared PCA reconstruction +
          │                                         Poisson/inverse-CDF sampling +
          │                                         differentiable ``sample_pdf``
          │                                         Abstract: ``_lookup(pos)``
          ├── TOFSampler                          — ``_lookup`` via voxel LUT
          │                                         trilinear (or nearest-neighbor)
          │                                         interpolation of a compressed
          │                                         photon library (h5)
          └── SirenTOFSampler                     — ``_lookup`` via pre-trained
                                                    SIREN network

All public names are re-exported here so ``from goop.sampler import X`` and
``from goop import X`` both keep working regardless of which submodule
defines ``X``.
"""

from .base import (
    DEFAULT_N_SIMULATED,
    DEFAULT_PLIB_PATH,
    PCATOFSampler,
)
from .lut import (
    DifferentiableTOFSampler,
    TOFSampler,
    create_default_tof_sampler,
)
from .siren import (
    DEFAULT_CFG_PATH,
    DEFAULT_CKPT_PATH,
    DEFAULT_SIRENTV_SRC,
    SirenTOFSampler,
    create_siren_tof_sampler,
)

__all__ = [
    "PCATOFSampler",
    "TOFSampler",
    "DifferentiableTOFSampler",
    "SirenTOFSampler",
    "create_default_tof_sampler",
    "create_siren_tof_sampler",
    "DEFAULT_PLIB_PATH",
    "DEFAULT_N_SIMULATED",
    "DEFAULT_CKPT_PATH",
    "DEFAULT_CFG_PATH",
    "DEFAULT_SIRENTV_SRC",
]
