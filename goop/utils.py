"""Utility functions for input preprocessing."""

from __future__ import annotations

from typing import Tuple, Union, Optional

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]


def voxelize(
    pos: ArrayLike,
    n_photons: ArrayLike,
    t_step: ArrayLike,
    dx: float,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Bin segments into cubic voxels of side length ``dx`` mm.

    Within each voxel, photon counts are summed, and positions / emission
    times are averaged weighted by photon count. Total photon yield is
    exactly preserved.

    Type/device dispatch:
      * If all three inputs are ``torch.Tensor``, the work runs on
        ``pos.device`` (no host↔device copies) using ``torch.unique`` +
        ``index_add_``. On a GPU this is ~300× faster than the numpy path.
        Returns torch tensors on the same device.
      * Otherwise the inputs are treated as numpy arrays and the original
        ``np.unique`` + ``np.add.at`` path is used (faster than torch on CPU
        for this access pattern). Returns numpy arrays.

    Parameters
    ----------
    pos : (N, 3) — segment positions in mm.
    n_photons : (N,) — photon count per segment.
    t_step : (N,) — emission time per segment in ns.
    dx : float — voxel side length in mm. Must be > 0.

    Returns
    -------
    pos_vox : (M, 3) float32 — photon-weighted centroid per voxel.
    nph_vox : (M,) int64 — summed photon count per voxel.
    tns_vox : (M,) float32 — photon-weighted mean emission time per voxel.
    """
    if dx <= 0:
        raise ValueError(f"voxel size dx must be > 0, got {dx}")

    if (isinstance(pos, torch.Tensor)
            and isinstance(n_photons, torch.Tensor)
            and isinstance(t_step, torch.Tensor)):
        return _voxelize_torch(pos, n_photons, t_step, dx)

    # numpy / mixed → CPU numpy path (cast detached torch tensors as needed).
    if isinstance(pos, torch.Tensor):
        pos = pos.detach().cpu().numpy()
    if isinstance(n_photons, torch.Tensor):
        n_photons = n_photons.detach().cpu().numpy()
    if isinstance(t_step, torch.Tensor):
        t_step = t_step.detach().cpu().numpy()
    return _voxelize_numpy(pos, n_photons, t_step, dx)


def _voxelize_numpy(
    pos: np.ndarray, n_photons: np.ndarray, t_step: np.ndarray, dx: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos = np.asarray(pos, dtype=np.float32)
    n_photons = np.asarray(n_photons)
    t_step = np.asarray(t_step, dtype=np.float32)

    vox_idx = np.floor(pos / dx).astype(np.int32)
    keys = (vox_idx[:, 0].astype(np.int64) * 1_000_003
            + vox_idx[:, 1].astype(np.int64) * 1_009
            + vox_idx[:, 2].astype(np.int64))

    unique_keys, inverse = np.unique(keys, return_inverse=True)
    n_vox = len(unique_keys)

    w = n_photons.astype(np.float64)
    pos_sum = np.zeros((n_vox, 3), dtype=np.float64)
    tns_sum = np.zeros(n_vox, dtype=np.float64)
    w_sum = np.zeros(n_vox, dtype=np.float64)
    nph_sum = np.zeros(n_vox, dtype=np.int64)

    np.add.at(pos_sum, inverse, pos * w[:, None])
    np.add.at(tns_sum, inverse, t_step * w)
    np.add.at(w_sum, inverse, w)
    np.add.at(nph_sum, inverse, n_photons.astype(np.int64))

    mask = w_sum > 0
    pos_vox = (pos_sum[mask] / w_sum[mask, None]).astype(np.float32)
    tns_vox = (tns_sum[mask] / w_sum[mask]).astype(np.float32)
    nph_vox = nph_sum[mask]
    return pos_vox, nph_vox, tns_vox


def _voxelize_torch(
    pos: torch.Tensor, n_photons: torch.Tensor, t_step: torch.Tensor,
    dx: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = pos.device
    pos_t = pos.to(dtype=torch.float32)
    nph_t = n_photons.to(dtype=torch.long)
    tns_t = t_step.to(dtype=torch.float32)

    vox_idx = torch.floor(pos_t / dx).long()
    keys = (vox_idx[:, 0] * 1_000_003
            + vox_idx[:, 1] * 1_009
            + vox_idx[:, 2])

    _, inverse = torch.unique(keys, return_inverse=True)
    n_vox = int(inverse.max().item()) + 1 if inverse.numel() > 0 else 0

    nph_f64 = nph_t.to(torch.float64)
    pos_f64 = pos_t.to(torch.float64)
    tns_f64 = tns_t.to(torch.float64)

    pos_sum = torch.zeros((n_vox, 3), device=device, dtype=torch.float64)
    pos_sum.index_add_(0, inverse, pos_f64 * nph_f64.unsqueeze(1))

    tns_sum = torch.zeros(n_vox, device=device, dtype=torch.float64)
    tns_sum.index_add_(0, inverse, tns_f64 * nph_f64)

    w_sum = torch.zeros(n_vox, device=device, dtype=torch.float64)
    w_sum.index_add_(0, inverse, nph_f64)

    nph_sum = torch.zeros(n_vox, device=device, dtype=torch.long)
    nph_sum.index_add_(0, inverse, nph_t)

    mask = w_sum > 0
    pos_vox = (pos_sum[mask] / w_sum[mask].unsqueeze(1)).to(torch.float32)
    tns_vox = (tns_sum[mask] / w_sum[mask]).to(torch.float32)
    nph_vox = nph_sum[mask]
    return pos_vox, nph_vox, tns_vox

def throw_in_time_window(
    pos: ArrayLike,
    n_photons: ArrayLike,
    t_step: ArrayLike,
    labels: ArrayLike,
    time_window_ns: float,
    device: str = 'cuda',
    pdgs: Optional[ArrayLike] = None,
    de: Optional[ArrayLike] = None,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Throw in a time window."""
    if time_window_ns <= 0:
        raise ValueError(f"time window must be > 0, got {time_window_ns}")
    def _to_tensor(x, device):
        if isinstance(x, torch.Tensor):
            return x
        return torch.from_numpy(np.array(x)).to(device)

    pos = _to_tensor(pos, device)
    n_photons = _to_tensor(n_photons, device)
    t_step = _to_tensor(t_step, device)
    labels = _to_tensor(labels, device)
    if pdgs is not None:
        pdgs = _to_tensor(pdgs, device)
    if de is not None:
        de = _to_tensor(de, device)

    # necessary to bring all point's t to positive
    t_step = t_step - t_step.min()
    unique_labels = torch.unique(labels)
    mask = labels >= 0
    label_indices = torch.searchsorted(unique_labels, labels[mask])
    per_label_min = torch.full((len(unique_labels),), float("inf"), dtype=t_step.dtype, device=device)
    per_label_min.scatter_reduce_(0, label_indices, t_step[mask], reduce="amin", include_self=True)
    t_step = t_step.clone()
    t_step[mask] = t_step[mask] - per_label_min[label_indices]

    rand_t0_per_label = torch.rand(len(unique_labels), device=device) * time_window_ns
    t_step[mask] = t_step[mask] + rand_t0_per_label[label_indices]

    keep = mask & (t_step <= time_window_ns)
    pos = pos[keep]
    n_photons = n_photons[keep]
    t_step = t_step[keep]
    labels = labels[keep]
    pdgs = pdgs[keep] if pdgs is not None else None
    de = de[keep] if de is not None else None

    result = {
        "pos_mm": pos,
        "n_photons": n_photons,
        "t_step": t_step,
        "labels": labels,
        "pdgs": pdgs,
        "de": de,
    }
    return result