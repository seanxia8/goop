"""
Voxel-LUT TOF sampler.

Reads a compressed photon library h5 and looks up ``(vis, t0, coeffs)`` per
voxel via trilinear (or nearest-neighbor) interpolation.
"""

from __future__ import annotations

import h5py
import numpy as np
import torch

from ..base import TOFSamplerBase
from .base import DEFAULT_N_SIMULATED, DEFAULT_PLIB_PATH, PCATOFSampler

__all__ = [
    "TOFSampler",
    "DifferentiableTOFSampler",
    "create_default_tof_sampler",
]


def create_default_tof_sampler(**kwargs) -> TOFSamplerBase:
    """Create a TOFSampler with the standard photon library.

    Note: ``differentiable`` is accepted for backward compatibility but is a no-op —
    every ``PCATOFSampler`` exposes both ``sample`` and ``sample_pdf``.
    """
    default_kwargs = {
        "n_simulated": DEFAULT_N_SIMULATED,
        "lazy": False,
        "device": "cuda:0",
        "interpolate": True,
        "pmt_qe": 0.12,  # incl. TPB reemission, see https://link.springer.com/article/10.1140/epjc/s10052-024-13306-3
    }
    default_kwargs.update(kwargs)
    default_kwargs.pop("differentiable", None)  # back-compat no-op
    return TOFSampler(DEFAULT_PLIB_PATH, **default_kwargs)


class TOFSampler(PCATOFSampler):
    """
    Monte Carlo photon time-of-flight sampler from a compressed photon library.

    Reads a half-detector (P PMT) compressed plib and produces full 2P-PMT
    results via x-reflection symmetry. Entirely GPU-native when lazy=False
    and device="cuda".
    """

    def __init__(self, filepath, n_simulated=1.5e7, lazy=True, device="cpu", interpolate=True, pmt_qe=None):
        dev = torch.device(device)
        self._lazy = lazy
        self._interpolate = interpolate
        self._file = None  # set below or kept None for eager mode

        with h5py.File(filepath, "r") as f:
            # Populate shared PCA fields via base helper
            basis = PCATOFSampler._read_h5_basis(filepath)
            # h5py re-open below is avoided by reading LUT tensors in this same block
            self._init_common(
                device=dev,
                n_simulated=n_simulated,
                pmt_qe=float(pmt_qe) if pmt_qe is not None else 1.0,
                n_pmts=basis["n_pmts"],
                n_components=basis["n_components"],
                log_quantile_C=basis["log_quantile_C"],
                t_max_ns=basis["t_max_ns"],
                mode=basis["mode"],
                pca_mean=basis["pca_mean"],
                pca_components=basis["pca_components"],
                u_grid=basis["u_grid"],
                numvox=basis["numvox"],
                min_xyz=basis["min_xyz"],
                max_xyz=basis["max_xyz"],
            )
            self._n_voxels = basis["n_voxels"]

            if not lazy:
                self.vis = torch.from_numpy(f["vis"][:]).float().to(self._device)
                self.t0 = torch.from_numpy(f["t0"][:]).float().to(self._device)
                self.coeffs = torch.from_numpy(f["coeffs"][:]).float().to(self._device)
            else:
                self.vis = self.t0 = self.coeffs = None

        self._file = h5py.File(filepath, "r", swmr=True) if lazy else None

    @classmethod
    def from_arrays(
        cls,
        *,
        vis: torch.Tensor,
        t0: torch.Tensor,
        coeffs: torch.Tensor,
        pca_mean: torch.Tensor,
        pca_components: torch.Tensor,
        u_grid: torch.Tensor,
        numvox: torch.Tensor,
        min_xyz: torch.Tensor,
        max_xyz: torch.Tensor,
        log_quantile_C: float = 1e-2,
        t_max_ns: float = 600.0,
        mode: str = "log_quantile",
        n_simulated: float = DEFAULT_N_SIMULATED,
        device: str = "cpu",
        interpolate: bool = True,
        pmt_qe: float = 1.0,
    ) -> "TOFSampler":
        """Construct a TOFSampler from in-memory arrays, bypassing h5 file loading.

        Useful for tests and synthetic libraries.  Shapes:
          vis: (n_voxels, n_pmts), t0: (n_voxels, n_pmts),
          coeffs: (n_voxels, n_pmts, n_components),
          pca_mean: (Q,), pca_components: (K, Q), u_grid: (Q,),
          numvox: (3,), min_xyz: (3,), max_xyz: (3,)
        """
        inst = cls.__new__(cls)
        dev = torch.device(device)
        inst._lazy = False
        inst._interpolate = interpolate
        inst._file = None
        inst._init_common(
            device=dev,
            n_simulated=n_simulated,
            pmt_qe=pmt_qe,
            n_pmts=vis.shape[1],
            n_components=coeffs.shape[2],
            log_quantile_C=log_quantile_C,
            t_max_ns=t_max_ns,
            mode=mode,
            pca_mean=pca_mean,
            pca_components=pca_components,
            u_grid=u_grid,
            numvox=numvox,
            min_xyz=min_xyz,
            max_xyz=max_xyz,
        )
        inst._n_voxels = int(vis.shape[0])
        inst.vis = vis.to(dtype=torch.float32, device=dev)
        inst.t0 = t0.to(dtype=torch.float32, device=dev)
        inst.coeffs = coeffs.to(dtype=torch.float32, device=dev)
        return inst

    # LUT fetch (trilinear or nearest-neighbor)

    def _coord_to_voxel(self, pos):
        """pos: (N, 3) -> (N,) voxel indices via flat raveled index."""
        pos = pos.to(dtype=torch.float64, device=self._device)  # (N, 3)
        frac = (pos - self._min_xyz) / (self._max_xyz - self._min_xyz + 1e-12)  # (N, 3) normalized [0,1]
        idx = (frac * self._numvox.double()).long().clamp(min=0)  # (N, 3) integer grid coords
        for d in range(3):
            idx[:, d] = idx[:, d].clamp(max=self._numvox[d] - 1)
        nx, ny, nz = self._numvox
        return idx[:, 0] + nx * (idx[:, 1] + ny * idx[:, 2])  # (N,) flat voxel id

    def _fetch(self, voxel_ids):
        """Fetch vis, t0, coeffs for voxel_ids.
        voxel_ids: (N,) -> vis: (N, P), t0: (N, P), coeffs: (N, P, K)
        """
        if self.vis is not None:
            return self.vis[voxel_ids], self.t0[voxel_ids], self.coeffs[voxel_ids]
        ids_np = voxel_ids.cpu().numpy()
        uniq, inv = np.unique(ids_np, return_inverse=True)
        v = torch.from_numpy(self._file["vis"][uniq]).float()      # (U, P)
        t = torch.from_numpy(self._file["t0"][uniq]).float()       # (U, P)
        c = torch.from_numpy(self._file["coeffs"][uniq]).float()   # (U, P, K)
        inv_t = torch.from_numpy(inv).long()                       # (N,) maps back to uniq
        return v[inv_t].to(self._device), t[inv_t].to(self._device), c[inv_t].to(self._device)

    def _trilinear_fetch(self, pos):
        """pos: (N, 3) -> interpolated (vis, t0, coeffs) via trilinear blending of 8 corner voxels."""
        pos = pos.to(dtype=torch.float64, device=self._device)
        frac = (pos - self._min_xyz) / (self._max_xyz - self._min_xyz + 1e-12)
        cont = frac * self._numvox.double() - 0.5  # continuous index, voxel-center-aligned

        idx0 = cont.long().clamp(min=0)  # floor corner (N, 3)
        idx1 = idx0 + 1                  # ceil corner (N, 3)
        for d in range(3):
            idx0[:, d].clamp_(max=self._numvox[d] - 1)
            idx1[:, d].clamp_(max=self._numvox[d] - 1)

        w = (cont - idx0.double()).clamp(0, 1).float()  # (N, 3) fractional weights

        # 8 corners: compute voxel IDs and trilinear weights
        corners = []   # list of 8 (N,) voxel ID tensors
        weights = []   # list of 8 (N,) scalar weight tensors
        nx, ny, nz = self._numvox
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    ix = idx1[:, 0] if dx else idx0[:, 0]
                    iy = idx1[:, 1] if dy else idx0[:, 1]
                    iz = idx1[:, 2] if dz else idx0[:, 2]
                    corners.append(ix + nx * (iy + ny * iz))
                    wx = w[:, 0] if dx else (1 - w[:, 0])
                    wy = w[:, 1] if dy else (1 - w[:, 1])
                    wz = w[:, 2] if dz else (1 - w[:, 2])
                    weights.append(wx * wy * wz)

        # Batch fetch all 8*N voxel IDs (dedup inside _fetch)
        N = pos.shape[0]
        all_vox = torch.cat(corners)  # (8N,)
        all_vis, all_t0, all_coeffs = self._fetch(all_vox)

        # Reshape to (8, N, ...) and blend with weights (8, N, 1)
        w8 = torch.stack(weights).unsqueeze(-1)                  # (8, N, 1)
        vis = (w8 * all_vis.view(8, N, -1)).sum(0)               # (N, P)
        t0 = (w8 * all_t0.view(8, N, -1)).sum(0)                 # (N, P)
        coeffs = (w8.unsqueeze(-1) * all_coeffs.view(8, N, self._n_pmts, -1)).sum(0)  # (N, P, K)

        return vis, t0, coeffs

    def _lookup(self, pos):
        """Dispatch to trilinear or nearest-neighbor fetch based on self._interpolate."""
        if self._interpolate:
            return self._trilinear_fetch(pos)
        vox = self._coord_to_voxel(pos)
        return self._fetch(vox)

    # ---- lifecycle override: close h5 handle -----------------------------

    def close(self):
        if getattr(self, "_file", None) is not None:
            self._file.close()
            self._file = None


# Back-compat alias: ``sample_pdf`` is now available on every ``PCATOFSampler``
# (including the regular ``TOFSampler``). External callers that reference
# ``DifferentiableTOFSampler`` continue to work unchanged.
DifferentiableTOFSampler = TOFSampler
