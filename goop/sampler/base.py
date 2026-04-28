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
        self._cumdu = torch.cat([
            torch.zeros(1, device=self._device),
            self._du.cumsum(0),
        ])  # (Q+1,) — CDF values at each u-grid point, with 0 prepended.
        # q_stride=1 keeps all Q u-grid points in sample_pdf. Users can set
        # ``sampler.q_stride = K`` (or pass ``q_stride=K`` to sample_pdf) to
        # subsample every K-th quantile point — cuts the (M, Q) tensors
        # proportionally. ``du`` is recomputed from the subsampled grid so
        # total probability mass is preserved.
        self.q_stride = 1
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

    @property
    def t_max_ns(self) -> float:
        """Width of the per-segment quantile-time window the basis covers."""
        return self._t_max_ns

    # shared helpers used by sample_pdf and _histogram_chunk

    def _resolve_q_stride(self, q_stride):
        """Return ``(q_idx, du_eff)`` for the requested quantile-grid stride.

        Precedence: explicit kwarg > sampler attribute > 1 (no subsample).
        Stride>1 preserves total probability mass by recomputing ``du`` from
        the subsampled grid.
        """
        stride = int(q_stride if q_stride is not None else getattr(self, "q_stride", 1))
        if stride <= 1:
            return None, self._du
        Q_full = self.u_grid.shape[0]
        q_idx = torch.arange(0, Q_full, stride, device=self._device)
        u_sub = self.u_grid[q_idx]
        du_eff = torch.diff(u_sub, prepend=torch.zeros(1, device=self._device))
        return q_idx, du_eff

    def _mirror_x(self, pos_chunk):
        """Mirror x>0 sources into the half-detector basis (cathode symmetry).

        Returns ``(pos_lookup, on_pos_side)``: ``pos_lookup`` is a clone of
        ``pos_chunk`` with x signs flipped on x>0 rows so a single LUT/SIREN
        call handles both halves; ``on_pos_side`` records which rows were
        flipped, for routing the output back to the correct PMT id.
        """
        on_pos_side = pos_chunk[:, 0] > 0
        pos_lookup = pos_chunk.clone()
        pos_lookup[on_pos_side, 0] = -pos_lookup[on_pos_side, 0]
        return pos_lookup, on_pos_side

    def _active_pmt_ids(self, on_pos_side, active_mask):
        """Assemble full-detector PMT ids for ``active_mask`` rows.

        Channels 0..P-1 cover x<=0 sources; channels P..2P-1 cover x>0 sources
        via the cathode-symmetry trick. ``on_pos_side`` (from ``_mirror_x``)
        selects the offset.
        """
        P = self._n_pmts
        C = on_pos_side.shape[0]
        pmt_offset = torch.where(on_pos_side, P, 0).unsqueeze(1).expand(C, P)
        pmt_base = torch.arange(P, device=self._device).unsqueeze(0).expand(C, -1)
        return (pmt_base + pmt_offset)[active_mask]

    # PCA reconstruction

    def _quantile_times(self, coeffs, t0, q_idx=None):
        """Reconstruct absolute quantile times from PCA coefficients.
        coeffs: (M, K), t0: (M,) -> q_abs: (M, Q)
        where M = active pairs, K = PCA components, Q = quantile grid points.

        If ``q_idx`` is given, the PCA basis is subsampled to only the columns
        named by ``q_idx``, so the output is ``(M, len(q_idx))`` and the
        ``(M, Q)`` full matrix is never materialised. This is the memory-saving
        path used by ``sample_pdf`` when ``q_stride > 1``.
        """
        if q_idx is None:
            comp = self.pca_components  # (K, Q)
            mean = self.pca_mean         # (Q,)
        else:
            comp = self.pca_components.index_select(1, q_idx)
            mean = self.pca_mean.index_select(0, q_idx)
        raw = coeffs @ comp + mean  # (M, Q_eff)
        if self._mode == "log_quantile":
            q = (torch.pow(10.0, raw) - self._log_quantile_C).clamp(min=0)
        else:
            q = raw.clamp(min=0)
        return q + t0.unsqueeze(-1)

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

        #return out_flat.reshape(n_pmts_full, n_bins).to(torch.int32)  # (2P, n_bins)
        return out_flat.reshape(n_pmts_full, n_bins).to(torch.float32)  # (2P, n_bins)
        
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

        if len(all_times) > 0:
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
        q_stride: int | None = None,
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

        q_idx, du_eff = self._resolve_q_stride(q_stride)

        all_times, all_ch, all_w = [], [], []

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk_pos = pos[start:end]
            chunk_scale = scale[start:end]
            C = end - start
            chunk_t = t_step[start:end] if t_step is not None else None

            pos_lookup, on_pos_side = self._mirror_x(chunk_pos)
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
            active_pmt = self._active_pmt_ids(on_pos_side, active_mask)  # (M,)

            # Build q_abs directly at the subsampled quantile indices (if any)
            # — avoids ever materialising the full (M, Q_full) matrix.
            q_abs = self._quantile_times(active_coeffs, active_t0, q_idx=q_idx)

            if chunk_t is not None:
                t_expanded = chunk_t.unsqueeze(1).expand(C, P)
                active_t_emit = t_expanded[active_mask]
                q_abs = q_abs + active_t_emit.unsqueeze(-1)

            Q = q_abs.shape[1]
            times = q_abs.reshape(-1)                                                # (M*Q,)
            channels = active_pmt.unsqueeze(-1).expand(-1, Q).reshape(-1)            # (M*Q,)
            weights = (active_expected.unsqueeze(-1) * du_eff.unsqueeze(0)).reshape(-1)  # (M*Q,)

            all_times.append(times)
            all_ch.append(channels)
            all_w.append(weights)

        if all_times:
            return torch.cat(all_times), torch.cat(all_ch), torch.cat(all_w)
        empty = torch.zeros(0, device=self._device)
        return empty, empty.long(), empty

    # streaming (chunked) histogram

    def _histogram_chunk(
        self,
        pos_chunk: torch.Tensor,
        scale_chunk: torch.Tensor,
        tns_chunk: torch.Tensor,
        tick_ns: float,
        n_bins: int,
        t0_ref: float,
        du_eff: torch.Tensor,
        q_idx,
        expected_eps: float,
    ) -> torch.Tensor:
        """Scatter one chunk's weighted ghost photons into a fresh (2P, n_bins)
        dense histogram. Returns the chunk-local hist; same PE content as
        ``sample_pdf`` would emit for this chunk, but without materialising the
        flat ``(M·Q,)`` photon lists (they stay inside this function's scope).

        Designed to be wrapped in ``torch.utils.checkpoint.checkpoint(...,
        use_reentrant=False)`` so the transient ``(M, Q)`` tensors are not
        saved for backward — they're recomputed on demand.
        """
        P = self._n_pmts
        n_ch = 2 * P
        C = pos_chunk.shape[0]

        pos_lookup, on_pos_side = self._mirror_x(pos_chunk)
        v, t0, coeffs = self._lookup(pos_lookup)                    # (C,P), (C,P), (C,P,K)
        expected = v * scale_chunk.unsqueeze(1) * self._pmt_qe       # (C, P)
        active_mask = expected.detach() > expected_eps

        chunk_hist = torch.zeros(n_ch, n_bins, device=self._device, dtype=torch.float32)
        if not bool(active_mask.any()):
            return chunk_hist

        active_expected = expected[active_mask]                      # (M,)
        active_coeffs   = coeffs[active_mask]                        # (M, K)
        active_t0       = t0[active_mask]                            # (M,)
        active_pmt      = self._active_pmt_ids(on_pos_side, active_mask)  # (M,)

        # (M, Q) — subsampled directly via q_idx so the full (M, Q_full) is never built.
        q_abs = self._quantile_times(active_coeffs, active_t0, q_idx=q_idx)

        t_expanded = tns_chunk.unsqueeze(1).expand(C, P)
        active_t_emit = t_expanded[active_mask]                      # (M,)
        q_abs = q_abs + active_t_emit.unsqueeze(-1)                  # (M, Q)

        weights = active_expected.unsqueeze(-1) * du_eff.unsqueeze(0)  # (M, Q)

        bin_idx = ((q_abs - t0_ref) / tick_ns).long()                # (M, Q)
        in_window = (bin_idx >= 0) & (bin_idx < n_bins)
        bin_idx = bin_idx.clamp(0, n_bins - 1)
        weights = weights * in_window                                 # zero out-of-window

        # Fused scatter: flatten to (2P·n_bins,) with composite (channel, bin) index.
        flat_idx = active_pmt.unsqueeze(-1) * n_bins + bin_idx        # (M, Q)
        flat_hist = chunk_hist.view(-1).scatter_add(
            0, flat_idx.reshape(-1), weights.reshape(-1)
        )
        return flat_hist.view(n_ch, n_bins)

    def histogram_pdf(
        self,
        pos,
        n_photons,
        t_step,
        tick_ns: float,
        n_bins: int,
        t0_ref: float,
        expected_eps: float = 1e-9,
        chunk_size: int = 5000,
        q_stride: int | None = None,
        use_checkpoint: bool = True,
    ) -> torch.Tensor:
        """Differentiable per-PMT PDF deposition as a dense histogram.

        Same mathematical output as ``sample_pdf(...)`` + ``from_photons(...)`` followed
        by ``SlicedWaveform.deslice()`` aligned to ``t0_ref`` / ``n_bins``, but
        memory-per-call is ``O(output + one_chunk_work)`` instead of ``O(N·P·Q)``
        — because each chunk's ``(M, Q)`` ghost-photon tensors are shed from the
        autograd graph via ``torch.utils.checkpoint`` and recomputed during
        backward.

        Parameters
        ----------
        pos, n_photons, t_step : position / yield / emission-time arrays
            (same shapes and semantics as ``sample_pdf``).
        tick_ns : float
            Time-bin width.
        n_bins : int
            Number of time bins in the output histogram. Bins with indices
            outside ``[0, n_bins)`` are silently dropped.
        t0_ref : float
            Absolute time (ns) corresponding to bin 0. Bin centres lie at
            ``t0_ref + (b + 0.5) * tick_ns``.
        expected_eps : float, optional
            Per-PMT yield threshold; pairs below are skipped (zero-gradient
            region — safe for noise-floor entries). Default ``1e-9``.
        chunk_size : int, optional
            Position chunking for gradient checkpointing. Default ``5000``.
            Lower = smaller peak memory, higher = less recompute overhead.
        q_stride : int, optional
            Quantile-grid stride. Explicit kwarg > ``self.q_stride`` > ``1``.

        Returns
        -------
        hist : (2*P, n_bins) float32 tensor, differentiable w.r.t. ``pos`` and
            ``n_photons``.
        """
        pos = pos if isinstance(pos, torch.Tensor) else torch.as_tensor(pos)
        pos = pos.to(dtype=torch.float32, device=self._device)
        if pos.dim() == 1:
            pos = pos.unsqueeze(0)
        N = pos.shape[0]
        P = self._n_pmts
        n_ch = 2 * P

        if isinstance(n_photons, (int, float)):
            scale = torch.full(
                (N,), float(n_photons) / self.n_simulated, device=self._device
            )
        else:
            n_ph_t = n_photons if isinstance(n_photons, torch.Tensor) else torch.as_tensor(n_photons)
            scale = n_ph_t.to(dtype=torch.float32, device=self._device) / self.n_simulated

        if t_step is None:
            t_step_t = torch.zeros(N, device=self._device, dtype=torch.float32)
        else:
            t_step_t = t_step if isinstance(t_step, torch.Tensor) else torch.as_tensor(t_step)
            t_step_t = t_step_t.to(dtype=torch.float32, device=self._device)

        q_idx, du_eff = self._resolve_q_stride(q_stride)

        hist = torch.zeros(n_ch, n_bins, device=self._device, dtype=torch.float32)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            args = (
                pos[start:end], scale[start:end], t_step_t[start:end],
                float(tick_ns), int(n_bins), float(t0_ref),
                du_eff, q_idx, float(expected_eps),
            )
            if use_checkpoint:
                from torch.utils.checkpoint import checkpoint
                chunk_hist = checkpoint(
                    self._histogram_chunk, *args, use_reentrant=False,
                )
            else:
                chunk_hist = self._histogram_chunk(*args)
            hist = hist + chunk_hist

        return hist

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
