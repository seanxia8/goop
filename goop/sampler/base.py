"""
Shared base for PCA-compressed photon-library TOF samplers.

``PCATOFSampler`` holds the quantile-time reconstruction + sampling machinery
(Poisson + inverse-CDF, differentiable PDF deposition) that is independent of
*how* ``(vis, t0, coeffs)`` are produced. Subclasses implement only
``_lookup(pos)``.
"""

from abc import abstractmethod

import h5py
import numpy as np
import torch

from ..base import TOFSamplerBase

__all__ = [
    "PCATOFSampler",
    "DEFAULT_PLIB_PATH",
    "DEFAULT_N_SIMULATED",
]

DEFAULT_PLIB_PATH = "/sdf/data/neutrino/youngsam/compressed_plib_b04_quantile_log_n50.h5"
DEFAULT_N_SIMULATED = 15_000_000


class PCATOFSampler(TOFSamplerBase):
    """Abstract base for PCA-compressed TOF samplers.

    Subclasses implement only ``_lookup(pos) -> (vis, t0, coeffs)`` — how the
    per-PMT visibility, t0, and PCA coefficients are obtained (voxel LUT,
    neural network, ...). Everything else — quantile reconstruction, Poisson +
    inverse-CDF sampling (``sample``), and the differentiable PDF deposition
    path (``sample_pdf``) — is shared here.

    All subclasses must populate the following fields in their ``__init__``
    (directly or via ``_init_common`` / ``_read_h5_basis``):

        ``_device``, ``_n_pmts``, ``_n_components``, ``_pmt_qe``,
        ``n_simulated``, ``_log_quantile_C``, ``_t_max_ns``, ``_mode``,
        ``pca_mean`` (Q,), ``pca_components`` (K, Q), ``u_grid`` (Q,),
        ``_du`` (Q,), ``_numvox`` (3,), ``_min_xyz`` (3,), ``_max_xyz`` (3,).

    Full-detector output: the compressed plib is half-detector (x ≤ 0, P
    PMTs); x > 0 positions are x-mirrored before ``_lookup`` and assigned
    PMT ids P..(2P-1). ``n_channels == 2 * _n_pmts``.
    """

    def _init_common(
        self,
        *,
        device,
        n_simulated,
        pmt_qe,
        n_pmts,
        n_components,
        log_quantile_C,
        t_max_ns,
        mode,
        pca_mean,
        pca_components,
        u_grid,
        numvox,
        min_xyz,
        max_xyz,
    ):
        """Populate every shared PCA-sampler field. Call from subclass __init__."""
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        self.n_simulated = float(n_simulated)
        self._pmt_qe = float(pmt_qe)
        self._n_pmts = int(n_pmts)
        self._n_components = int(n_components)
        self._log_quantile_C = float(log_quantile_C)
        self._t_max_ns = float(t_max_ns)
        self._mode = str(mode)
        self.pca_mean = pca_mean.to(dtype=torch.float32, device=self._device)
        self.pca_components = pca_components.to(dtype=torch.float32, device=self._device)
        self.u_grid = u_grid.to(dtype=torch.float32, device=self._device)
        self._du = torch.diff(self.u_grid, prepend=torch.zeros(1, device=self._device))
        self._numvox = numvox.to(dtype=torch.long, device=self._device)
        self._min_xyz = min_xyz.to(dtype=torch.float64, device=self._device)
        self._max_xyz = max_xyz.to(dtype=torch.float64, device=self._device)

    @staticmethod
    def _read_h5_basis(filepath):
        """Read shared PCA-basis + voxel-grid + PMT positions from a compressed plib.

        Returns a dict of ready-to-pass fields plus ``pmt_pos`` (np.ndarray) and
        ``n_voxels`` (int). Does not load the per-voxel LUT tensors (vis/t0/coeffs).
        """
        with h5py.File(filepath, "r") as f:
            return dict(
                n_pmts=int(f["vis"].shape[1]),
                n_components=int(f["coeffs"].shape[2]),
                log_quantile_C=float(f.attrs.get("log_quantile_C", 1e-2)),
                t_max_ns=float(f.attrs.get("t_max_ns", 600.0)),
                mode=str(f.attrs.get("mode", "log_quantile")),
                pca_mean=torch.from_numpy(f["pca_mean"][:]).float(),
                pca_components=torch.from_numpy(f["pca_components"][:]).float(),
                u_grid=torch.from_numpy(f["u_grid"][:]).float(),
                numvox=torch.from_numpy(np.asarray(f["numvox"][:], dtype=np.int64)),
                min_xyz=torch.from_numpy(np.asarray(f["min"][:], dtype=np.float64)),
                max_xyz=torch.from_numpy(np.asarray(f["max"][:], dtype=np.float64)),
                pmt_pos=np.asarray(f["pmt_pos"][:]) if "pmt_pos" in f else None,
                n_voxels=int(f["vis"].shape[0]),
            )

    # abstract lookup

    @abstractmethod
    def _lookup(self, pos: torch.Tensor):
        """Return (vis, t0, coeffs) for each position in ``pos`` (N, 3).

        Shapes: vis (N, P), t0 (N, P), coeffs (N, P, K). Must be differentiable
        with respect to ``pos`` if gradients are desired. Input positions are
        assumed to be on the plib's half-detector side (x <= 0); callers handle
        x-mirroring before invoking this method.
        """

    @property
    def n_channels(self) -> int:
        return self._n_pmts * 2  # full detector (both sides of cathode)

    # PCA reconstruction

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
            return_histogram: if True, return (2P, n_bins) Poisson-sampled histogram.
                              if False (default), return (times, channels, source_idx).
            t_max_ns: time range in ns for histogram.
            tick_ns: bin width in ns for histogram.
            chunk_size: positions per chunk (controls GPU memory).

        Returns:
            Histogram mode: (2P, n_bins) int32 tensor of photon counts.
            Raw mode: tuple (times, channels, source_idx) — 1D tensors of all detected photons.
        """
        pos = torch.as_tensor(pos, dtype=torch.float32, device=self._device)
        if pos.dim() == 1:
            pos = pos.unsqueeze(0)
        N = pos.shape[0]
        P = self._n_pmts
        n_pmts_full = P * 2
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
        out_flat = torch.zeros(n_pmts_full * n_bins, device=self._device)  # (2P*n_bins,)
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

            # assign full-detector PMT channel ids (0..2P-1)
            pmt_offset = torch.where(on_pos_side, P, 0).unsqueeze(1).expand(C, P)  # (C, P)
            pmt_base = torch.arange(P, device=self._device).unsqueeze(0).expand(C, -1)  # (C, P)
            active_pmt = (pmt_base + pmt_offset)[active_mask]  # (M,) channel ids in [0, 2P)

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

        return out_flat.reshape(n_pmts_full, n_bins).to(torch.int32)  # (2P, n_bins)

    def _sample_raw(self, pos, scale, N, P, n_pmts_full, chunk_size, t_step):
        """Inverse-CDF sampling for raw (time, channel, source_idx) triples.

        Same cathode convention as _sample_histogram: x<=0 -> PMTs 0..(P-1),
        x>0 -> PMTs P..(2P-1).
        """
        all_times = []
        all_ch = []
        all_source = []
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

            # Track which input position each active pair belongs to
            pos_local, _ = active_mask.nonzero(as_tuple=True)  # (M,) position index within chunk

            # assign full-detector PMT channel ids (0..2P-1)
            pmt_offset = torch.where(on_pos_side, P, 0).unsqueeze(1).expand(C, P)  # (C, P)
            pmt_base = torch.arange(P, device=self._device).unsqueeze(0).expand(C, -1)  # (C, P)
            active_pmt = (pmt_base + pmt_offset)[active_mask]  # (M,) channel ids in [0, 2P)

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

            all_times.append(t_samp)                         # (T,)
            all_ch.append(active_pmt[pair_idx])              # (T,) PMT channel per photon
            all_source.append(pos_local[pair_idx] + start)   # (T,) global position index per photon

        if all_times:
            return torch.cat(all_times), torch.cat(all_ch), torch.cat(all_source)
        empty = torch.zeros(0, device=self._device)
        return empty, empty.long(), empty.long()

    def sample_pdf(
        self,
        pos,
        n_photons,
        t_step,
        expected_eps: float = 1e-9,
        chunk_size: int = 20000,
    ):
        """Deposit per-PMT expected PDF as weighted synthetic photons.

        Replaces ``torch.poisson(expected)`` with ``expected`` directly (no
        Poisson sampling — gradients flow to ``n_photons`` and to ``v``,
        which itself depends on ``pos`` via ``_lookup``), and returns
        synthetic ``(times, channels, weights)`` triples so that histogramming
        with those weights reproduces the per-PMT expected PDF.

        Returns
        -------
        times : (M*Q,) float32 — quantile times in ns (absolute, with t_step shift)
        channels : (M*Q,) int64 — full-detector PMT id (0..2P-1)
        weights : (M*Q,) float32 — probability mass per (pair, quantile bin),
            equal to ``expected[pair] * du[bin]`` so ``Σ weights[pair_q] = expected[pair]``.

        ``M`` is the number of (position, PMT) pairs with ``expected > expected_eps``;
        ``Q`` is the number of quantile grid points in the PCA basis.
        """
        pos = pos if isinstance(pos, torch.Tensor) else torch.as_tensor(pos)
        pos = pos.to(dtype=torch.float32, device=self._device)
        if pos.dim() == 1:
            pos = pos.unsqueeze(0)
        N = pos.shape[0]
        P = self._n_pmts

        if isinstance(n_photons, (int, float)):
            scale = torch.full(
                (N,), float(n_photons) / self.n_simulated, device=self._device
            )
        else:
            n_ph_t = n_photons if isinstance(n_photons, torch.Tensor) else torch.as_tensor(n_photons)
            scale = n_ph_t.to(dtype=torch.float32, device=self._device) / self.n_simulated

        if t_step is not None:
            t_step_t = t_step if isinstance(t_step, torch.Tensor) else torch.as_tensor(t_step)
            t_step = t_step_t.to(dtype=torch.float32, device=self._device)

        all_times, all_ch, all_w = [], [], []

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk_pos = pos[start:end]
            chunk_scale = scale[start:end]
            C = end - start
            chunk_t = t_step[start:end] if t_step is not None else None

            on_pos_side = chunk_pos[:, 0] > 0
            pos_lookup = chunk_pos.clone()
            pos_lookup[on_pos_side, 0] = -pos_lookup[on_pos_side, 0]

            v, t0, coeffs = self._lookup(pos_lookup)        # (C,P), (C,P), (C,P,K)
            expected = v * chunk_scale.unsqueeze(1) * self._pmt_qe  # (C, P) — differentiable

            # Threshold to skip negligible pairs (gradient is zero below; fine
            # for noise-floor entries, threshold is safely small).
            active_mask = expected.detach() > expected_eps
            if not bool(active_mask.any()):
                continue

            active_expected = expected[active_mask]      # (M,)
            active_coeffs = coeffs[active_mask]          # (M, K)
            active_t0 = t0[active_mask]                  # (M,)

            pmt_offset = torch.where(on_pos_side, P, 0).unsqueeze(1).expand(C, P)
            pmt_base = torch.arange(P, device=self._device).unsqueeze(0).expand(C, -1)
            active_pmt = (pmt_base + pmt_offset)[active_mask]  # (M,)

            q_abs = self._quantile_times(active_coeffs, active_t0)  # (M, Q)

            if chunk_t is not None:
                t_expanded = chunk_t.unsqueeze(1).expand(C, P)
                active_t_emit = t_expanded[active_mask]
                q_abs = q_abs + active_t_emit.unsqueeze(-1)

            Q = q_abs.shape[1]
            times = q_abs.reshape(-1)                                                # (M*Q,)
            channels = active_pmt.unsqueeze(-1).expand(-1, Q).reshape(-1)            # (M*Q,)
            weights = (active_expected.unsqueeze(-1) * self._du.unsqueeze(0)).reshape(-1)  # (M*Q,)

            all_times.append(times)
            all_ch.append(channels)
            all_w.append(weights)

        if all_times:
            return torch.cat(all_times), torch.cat(all_ch), torch.cat(all_w)
        empty = torch.zeros(0, device=self._device)
        return empty, empty.long(), empty

    def close(self):
        """No-op by default; LUT subclass overrides to close its h5 handle."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
