"""
Fast GPU-native Monte Carlo photon sampler from compressed photon library.
"""

import numpy as np
import torch
import h5py

from .base import TOFSamplerBase

__all__ = [
    "create_default_tof_sampler",
    "TOFSampler",
]

DEFAULT_PLIB_PATH = "/sdf/data/neutrino/youngsam/compressed_plib_b04_quantile_log_n50.h5"
DEFAULT_N_SIMULATED = 15_000_000


def create_default_tof_sampler(**kwargs) -> TOFSamplerBase:
    """Create a TOFSampler with the standard photon library."""
    from .sampler import TOFSampler

    default_kwargs = {
        "n_simulated": DEFAULT_N_SIMULATED,
        "lazy": False,
        "device": "cuda:0",
        "interpolate": True,
        "pmt_qe": 0.12,  #incl. TPB reemission, see https://link.springer.com/article/10.1140/epjc/s10052-024-13306-3
    }
    default_kwargs.update(kwargs)

    return TOFSampler(DEFAULT_PLIB_PATH, **default_kwargs)


class TOFSampler(TOFSamplerBase):
    """
    Monte Carlo photon time-of-flight sampler from a compressed photon library.

    Reads a half-detector (81 PMT) compressed plib and produces full 162-PMT
    results via x-reflection symmetry. Entirely GPU-native when lazy=False
    and device="cuda".
    """

    def __init__(self, filepath, n_simulated=1.5e7, lazy=True, device="cpu", interpolate=True, pmt_qe=None):
        self.n_simulated = n_simulated
        self._file = None
        self._device = torch.device(device)
        self._lazy = lazy
        self._interpolate = interpolate

        with h5py.File(filepath, "r") as f:
            self._n_voxels = f["vis"].shape[0]
            self._n_pmts = f["vis"].shape[1]
            self._n_components = f["coeffs"].shape[2]
            self._log_quantile_C = float(f.attrs.get("log_quantile_C", 1e-2))
            self._t_max_ns = float(f.attrs.get("t_max_ns", 600.0))
            self._mode = str(f.attrs.get("mode", "log_quantile"))

            self.pca_mean = torch.from_numpy(f["pca_mean"][:]).float().to(self._device)
            self.pca_components = torch.from_numpy(f["pca_components"][:]).float().to(self._device)
            self.u_grid = torch.from_numpy(f["u_grid"][:]).float().to(self._device)
            # Precompute du (probability weight per quantile point)
            self._du = torch.diff(self.u_grid, prepend=torch.zeros(1, device=self._device))

            numvox = torch.from_numpy(np.asarray(f["numvox"][:], dtype=np.int64))
            min_xyz = torch.from_numpy(np.asarray(f["min"][:], dtype=np.float64))
            max_xyz = torch.from_numpy(np.asarray(f["max"][:], dtype=np.float64))
            self._numvox = numvox.to(self._device)
            self._min_xyz = min_xyz.to(self._device)
            self._max_xyz = max_xyz.to(self._device)

            if not lazy:
                self.vis = torch.from_numpy(f["vis"][:]).float().to(self._device)
                self.t0 = torch.from_numpy(f["t0"][:]).float().to(self._device)
                self.coeffs = torch.from_numpy(f["coeffs"][:]).float().to(self._device)
            else:
                self.vis = self.t0 = self.coeffs = None

        self._file = h5py.File(filepath, "r", swmr=True) if lazy else None
        self._pmt_qe = float(pmt_qe) if pmt_qe is not None else 1.0

    @property
    def n_channels(self) -> int:
        return self._n_pmts * 2  # full detector (both sides of cathode)

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

    def _quantile_times(self, coeffs, t0):
        """Reconstruct absolute quantile times from PCA coefficients.
        coeffs: (M, K), t0: (M,) -> q_abs: (M, Q)
        where M = active pairs, K = PCA components, Q = quantile grid points.
        """
        # pca_components: (K, Q), pca_mean: (Q,)
        raw = coeffs @ self.pca_components + self.pca_mean  # (M, Q) reconstructed quantile curve
        if self._mode == "log_quantile":
            q = torch.pow(10.0, raw).sub_(self._log_quantile_C).clamp_(min=0)  # (M, Q) undo log transform
        else:
            q = raw.clamp_(min=0)  # (M, Q)
        q.add_(t0.unsqueeze(-1))  # (M, Q) shift relative -> absolute times
        return q

    @torch.no_grad()
    def sample(self, pos, n_photons, t_step, return_histogram=False,
               t_max_ns=100.0, tick_ns=0.1, chunk_size=20000):
        """
        Monte Carlo photon sampling.

        Args:
            pos: (N, 3) positions in mm. Tensor or array.
            n_photons: int or (N,) array/tensor — photons emitted per position.
            t_step: (N,) array/tensor — per-step emission time in ns.
            return_histogram: if True, return (162, n_bins) Poisson-sampled histogram.
                              if False (default), return (times_ns, channel_ids) tensors.
            t_max_ns: time range in ns for histogram.
            tick_ns: bin width in ns for histogram.
            chunk_size: positions per chunk (controls GPU memory).

        Returns:
            Histogram mode: (162, n_bins) int32 tensor of photon counts.
            Raw mode: tuple (times, channels) — 1D tensors of all detected photons.
        """
        pos = torch.as_tensor(pos, dtype=torch.float32, device=self._device)
        if pos.dim() == 1:
            pos = pos.unsqueeze(0)
        N = pos.shape[0]
        P = self._n_pmts  # 81
        n_pmts_full = P * 2  # 162
        n_bins = int(round(t_max_ns / tick_ns))

        if isinstance(n_photons, (int, float)):
            scale = torch.full((N,), n_photons / self.n_simulated, device=self._device)
        else:
            scale = torch.as_tensor(n_photons, dtype=torch.float32, device=self._device) / self.n_simulated

        if t_step is not None:
            t_step = torch.as_tensor(t_step, dtype=torch.float32, device=self._device)

        if return_histogram:
            return self._sample_histogram(pos, scale, N, P, n_pmts_full, n_bins, tick_ns, chunk_size, t_step)
        else:
            return self._sample_raw(pos, scale, N, P, n_pmts_full, chunk_size, t_step)

    def _sample_histogram(self, pos, scale, N, P, n_pmts_full, n_bins, tick_ns, chunk_size, t_step):
        """Scatter-add approach: only reconstruct quantile functions for active pairs.

        Opaque cathode at x=0: x<=0 sources illuminate PMTs 0..(P-1) only,
        x>0 sources illuminate PMTs P..(2P-1) only. x>0 positions are mirrored
        into the plib range before lookup.
        """
        out_flat = torch.zeros(n_pmts_full * n_bins, device=self._device)  # (162*n_bins,)
        du = self._du        # (Q,) probability weight per quantile bin
        inv_tick = 1.0 / tick_ns

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk_pos = pos[start:end]       # (C, 3)
            chunk_scale = scale[start:end]   # (C,)
            C = end - start
            chunk_t = t_step[start:end] if t_step is not None else None  # (C,) or None

            on_pos_side = chunk_pos[:, 0] > 0   # (C,) bool — which side of cathode
            pos_lookup = chunk_pos.clone()       # (C, 3)
            pos_lookup[on_pos_side, 0] = -pos_lookup[on_pos_side, 0]  # mirror x>0 to x<0

            v, t0, coeffs = self._lookup(pos_lookup)  # (C,P), (C,P), (C,P,K)

            expected = (v * chunk_scale.unsqueeze(1) * self._pmt_qe).clamp_(min=0)  # (C, P)
            counts = torch.poisson(expected)       # (C, P) Poisson-sampled integer counts
            active_mask = counts > 0.5             # (C, P) bool — pairs with ≥1 photon

            n_active = active_mask.sum().item()    # M = number of active pairs
            if n_active == 0:
                continue

            active_counts = counts[active_mask]    # (M,)
            active_coeffs = coeffs[active_mask]    # (M, K)
            active_t0 = t0[active_mask]            # (M,)

            # assign full-detector PMT channel ids (0..161)
            pmt_offset = torch.where(on_pos_side, P, 0).unsqueeze(1).expand(C, P)  # (C, P)
            pmt_base = torch.arange(P, device=self._device).unsqueeze(0).expand(C, -1)  # (C, P)
            active_pmt = (pmt_base + pmt_offset)[active_mask]  # (M,) channel ids in [0,161]

            q_abs = self._quantile_times(active_coeffs, active_t0)  # (M, Q) absolute quantile times

            if chunk_t is not None:
                t_expanded = chunk_t.unsqueeze(1).expand(C, P)       # (C, P)
                active_t_emit = t_expanded[active_mask]              # (M,)
                q_abs = q_abs + active_t_emit.unsqueeze(-1)          # (M, Q) shift by emission time

            # convert quantile times to histogram bins and scatter-add
            bin_idx = (q_abs * inv_tick).long()                      # (M, Q) time bin indices
            in_window = (bin_idx >= 0) & (bin_idx < n_bins)          # (M, Q) bool — in time range
            bin_idx.clamp_(0, n_bins - 1)                            # (M, Q) clamped
            flat_idx = active_pmt.unsqueeze(-1) * n_bins + bin_idx   # (M, Q) flat index into output
            weights = active_counts.unsqueeze(-1) * du.unsqueeze(0) * in_window  # (M, Q) weighted counts
            out_flat.scatter_add_(0, flat_idx.reshape(-1), weights.reshape(-1))  # accumulate

        return out_flat.reshape(n_pmts_full, n_bins).to(torch.int32)  # (162, n_bins)

    def _sample_raw(self, pos, scale, N, P, n_pmts_full, chunk_size, t_step):
        """Inverse-CDF sampling for raw (time, channel) pairs.

        Same cathode convention as _sample_histogram: x<=0 -> PMTs 0..(P-1),
        x>0 -> PMTs P..(2P-1).
        """
        all_times = []
        all_ch = []
        u_grid = self.u_grid  # (Q,) uniform quantile grid

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk_pos = pos[start:end]       # (C, 3)
            chunk_scale = scale[start:end]   # (C,)
            C = end - start
            chunk_t = t_step[start:end] if t_step is not None else None  # (C,) or None

            on_pos_side = chunk_pos[:, 0] > 0   # (C,) bool
            pos_lookup = chunk_pos.clone()       # (C, 3)
            pos_lookup[on_pos_side, 0] = -pos_lookup[on_pos_side, 0]  # mirror x>0 to x<0

            v, t0, coeffs = self._lookup(pos_lookup)  # (C,P), (C,P), (C,P,K)

            expected = (v * chunk_scale.unsqueeze(1) * self._pmt_qe).clamp_(min=0)  # (C, P)
            counts = torch.poisson(expected).long()  # (C, P) integer photon counts
            active_mask = counts > 0                 # (C, P) bool

            n_active = active_mask.sum().item()      # M = number of active pairs
            if n_active == 0:
                continue

            active_counts = counts[active_mask]      # (M,)
            active_coeffs = coeffs[active_mask]      # (M, K)
            active_t0 = t0[active_mask]              # (M,)

            # assign full-detector PMT channel ids (0..161)
            pmt_offset = torch.where(on_pos_side, P, 0).unsqueeze(1).expand(C, P)  # (C, P)
            pmt_base = torch.arange(P, device=self._device).unsqueeze(0).expand(C, -1)  # (C, P)
            active_pmt = (pmt_base + pmt_offset)[active_mask]  # (M,) channel ids in [0,161]

            q_abs = self._quantile_times(active_coeffs, active_t0)  # (M, Q) absolute quantile times

            # expand each active pair by its photon count
            total_photons = active_counts.sum().item()  # T = total photons in this chunk
            pair_idx = torch.repeat_interleave(
                torch.arange(n_active, device=self._device), active_counts
            )  # (T,) which active pair each photon belongs to

            # inverse-CDF: draw uniform, interpolate into quantile function
            u = torch.rand(total_photons, device=self._device)  # (T,) ~ U(0,1)

            idx = torch.searchsorted(u_grid, u).clamp(1, u_grid.shape[0] - 1)  # (T,) right bin edge
            u_lo, u_hi = u_grid[idx - 1], u_grid[idx]      # (T,) bracketing u-grid values
            t_lo = q_abs[pair_idx, idx - 1]                 # (T,) quantile time at left edge
            t_hi = q_abs[pair_idx, idx]                     # (T,) quantile time at right edge
            frac = ((u - u_lo) / (u_hi - u_lo + 1e-12)).clamp_(0, 1)  # (T,) interpolation weight
            t_samp = t_lo + frac * (t_hi - t_lo)           # (T,) sampled arrival times

            if chunk_t is not None:
                t_expanded = chunk_t.unsqueeze(1).expand(C, P)  # (C, P)
                active_t_emit = t_expanded[active_mask]         # (M,)
                t_samp = t_samp + active_t_emit[pair_idx]       # (T,) add emission time offset

            all_times.append(t_samp)              # (T,)
            all_ch.append(active_pmt[pair_idx])   # (T,) PMT channel per photon

        if all_times:
            return torch.cat(all_times), torch.cat(all_ch)
        return torch.zeros(0, device=self._device), torch.zeros(
            0, device=self._device, dtype=torch.long
        )

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def __del__(self):
        self.close()