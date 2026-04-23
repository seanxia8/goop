"""Waveform types for the optical TPC simulation pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from .digitize import digitize as _digitize
from .waveform_utils import _next_fft_size, _slice_channel


@dataclass
class Waveform:
    """Per-channel 1D data on a shared global time axis.

    All channels have the same length and share a single time origin (t0).
    Used for both pre-convolution histograms and post-convolution waveforms.
    """

    adc: torch.Tensor   # (n_channels, n_bins)
    t0: float           # global time origin in ns
    tick_ns: float      # time step in ns
    n_channels: int     # number of PMT channels
    attrs: dict = field(default_factory=dict)  # arbitrary metadata (e.g. pe_counts)

    @staticmethod
    def from_photons(
        times: torch.Tensor,
        channels: torch.Tensor,
        tick_ns: float,
        n_channels: int,
        t0: float = None,
        t0_snap_ns: float = None,
        weights: torch.Tensor = None,
    ) -> Waveform:
        """Build per-channel histograms on a shared global time axis.

        NOTE: this materializes the full global histogram in memory.
        For events with huge time ranges, this may cause an OOM.

        Parameters
        ----------
        t0 : Global time origin in ns. Defaults to times.min(),
             snapped to the tick boundary.
        t0_snap_ns : Grid spacing for t0 snapping (default: tick_ns).
             Use a coarser value when oversampling to keep t0 on the
             output grid.
        weights : Per-photon weights (default: 1.0 per photon).
        """
        device = times.device

        if times.numel() == 0:
            return Waveform(
                adc=torch.zeros(n_channels, 1, device=device),
                t0=t0 or 0.0, tick_ns=tick_ns, n_channels=n_channels,
            )
        if t0 is None:
            snap = t0_snap_ns or tick_ns
            t0 = (times.min() / snap).floor().item() * snap
        shifted = times - t0
        if (shifted < 0).any():
            raise ValueError(
                f"Some photon times are before t0={t0:.2f} ns "
            )
        n_bins = int(shifted.max().item() / tick_ns) + 1

        data = torch.zeros(n_channels, n_bins, device=device, dtype=torch.float32)
        bin_idx = (shifted / tick_ns).long().clamp(max=n_bins - 1)

        flat_idx = channels.long() * n_bins + bin_idx
        w = weights if weights is not None else torch.ones_like(flat_idx, dtype=torch.float32)
        data.view(-1).scatter_add_(0, flat_idx, w)

        return Waveform(adc=data, t0=t0, tick_ns=tick_ns, n_channels=n_channels)

    def slice(self, kernel_extent_ns: float) -> SlicedWaveform:
        """Compress by removing dead gaps (zero-stretches > kernel_extent_ns)."""
        kernel_extent_bins = int(kernel_extent_ns / self.tick_ns)
        device = self.adc.device
        all_adc: List[torch.Tensor] = []
        offsets_list = [0]
        all_t0: List[float] = []
        all_pmt: List[int] = []

        for ch in range(self.n_channels):
            compressed, bin_starts, t0_starts = _slice_channel(
                self.adc[ch], self.t0, self.tick_ns, kernel_extent_bins
            )
            n_ch_chunks = len(bin_starts)
            for k in range(n_ch_chunks):
                c0 = bin_starts[k]
                c1 = bin_starts[k + 1] if k + 1 < n_ch_chunks else compressed.numel()
                all_adc.append(compressed[c0:c1])
                offsets_list.append(offsets_list[-1] + (c1 - c0))
                all_t0.append(t0_starts[k])
                all_pmt.append(ch)

        return SlicedWaveform(
            adc=torch.cat(all_adc) if all_adc else torch.zeros(0, device=device),
            offsets=torch.tensor(offsets_list, device=device, dtype=torch.long),
            t0_ns=torch.tensor(all_t0, device=device, dtype=torch.float32),
            pmt_id=torch.tensor(all_pmt, device=device, dtype=torch.long),
            tick_ns=self.tick_ns,
            n_channels=self.n_channels,
            n_bins=self.adc.shape[1],
            attrs=dict(self.attrs),
        )

    def digitize(self, pedestal: float, n_bits: int) -> Waveform:
        """Apply pedestal, quantize, and saturate to ADC range."""
        return Waveform(
            adc=_digitize(self.adc, pedestal, n_bits),
            t0=self.t0, tick_ns=self.tick_ns, n_channels=self.n_channels,
            attrs=dict(self.attrs),
        )

    def convolve(self, kernel: torch.Tensor, gain: float) -> Waveform:
        """FFT-convolve all channels with kernel and apply gain."""
        conv_ticks = kernel.shape[0]
        n_ch, n_tick = self.adc.shape
        n_fft = _next_fft_size(n_tick + conv_ticks - 1)

        padded = F.pad(self.adc, (0, n_fft - n_tick))
        k_fft = torch.fft.rfft(kernel, n=n_fft)
        result = gain * torch.fft.irfft(
            torch.fft.rfft(padded) * k_fft.unsqueeze(0), n=n_fft
        )
        out_len = n_tick + conv_ticks - 1
        return Waveform(
            adc=result[:, :out_len], t0=self.t0,
            tick_ns=self.tick_ns, n_channels=self.n_channels,
            attrs=dict(self.attrs),
        )

    def downsample(self, factor: int) -> Waveform:
        """Average groups of `factor` consecutive bins into coarser bins."""
        if factor == 1:
            return self
        n_ch, n_bins = self.adc.shape
        remainder = n_bins % factor
        data = self.adc
        if remainder != 0:
            data = F.pad(data, (0, factor - remainder))
        n_coarse = data.shape[1] // factor
        coarse = data.reshape(n_ch, n_coarse, factor).mean(dim=2)
        return Waveform(
            adc=coarse, t0=self.t0,
            tick_ns=self.tick_ns * factor, n_channels=self.n_channels,
            attrs=dict(self.attrs),
        )


@dataclass
class SlicedWaveform:
    """Compressed waveform stored as flat CSR-style arrays.

    Each chunk is a contiguous segment of ADC data tagged with a PMT channel
    and a real-time origin.  Chunks are ordered by (pmt_id, t0_ns).

    Fields
    ------
    adc      : (total_bins,)  all chunks concatenated
    offsets  : (n_chunks+1,)  CSR boundaries — chunk k = adc[offsets[k]:offsets[k+1]]
    t0_ns    : (n_chunks,)    real-time start of each chunk in ns
    pmt_id   : (n_chunks,)    PMT channel of each chunk
    tick_ns  : time bin width in ns
    n_channels : total number of PMT channels
    """

    adc: torch.Tensor       # (total_bins,)
    offsets: torch.Tensor   # (n_chunks+1,)
    t0_ns: torch.Tensor    # (n_chunks,)
    pmt_id: torch.Tensor    # (n_chunks,)
    tick_ns: float
    n_channels: int
    n_bins: int = None       # expected desliced length (optional, set by slice/convolve)
    attrs: dict = field(default_factory=dict)  # arbitrary metadata (e.g. pe_counts)

    @property
    def n_chunks(self) -> int:
        return self.pmt_id.numel()

    def chunk(self, k: int) -> torch.Tensor:
        """Return ADC data for chunk k."""
        return self.adc[self.offsets[k]:self.offsets[k + 1]]

    @staticmethod
    def from_photons(
        times: torch.Tensor,
        channels: torch.Tensor,
        tick_ns: float,
        n_channels: int,
        kernel_extent_ns: float,
        t0_snap_ns: float = None,
        weights: torch.Tensor = None,
    ) -> SlicedWaveform:
        """Build compressed per-channel histograms directly from photon times.

        Parameters
        ----------
        t0_snap_ns : Grid spacing for t0 snapping (default: tick_ns).
             Use a coarser value when oversampling to keep chunk t0
             values on the output grid.
        weights : Per-photon weights (default: 1.0 per photon).
        """
        snap = t0_snap_ns or tick_ns
        device = times.device
        all_adc: List[torch.Tensor] = []
        offsets_list = [0]
        all_t0: List[float] = []
        all_pmt: List[int] = []

        for ch in range(n_channels):
            ch_mask = channels == ch
            ch_times = times[ch_mask]

            if ch_times.numel() == 0:
                all_adc.append(torch.zeros(1, device=device))
                offsets_list.append(offsets_list[-1] + 1)
                all_t0.append(0.0)
                all_pmt.append(ch)
                continue

            sort_idx = ch_times.sort().indices
            sorted_t = ch_times[sort_idx]
            ch_w = weights[ch_mask][sort_idx] if weights is not None else None
            diffs = torch.diff(sorted_t)
            ch_t0 = (sorted_t[0] / snap).floor() * snap

            excess = (diffs - kernel_extent_ns).clamp(min=0)
            cum_dead = torch.cat([torch.zeros(1, device=device), excess.cumsum(0)])
            compressed_t = sorted_t - cum_dead - ch_t0

            n_bins = int(compressed_t[-1].item() / tick_ns) + 1
            bin_idx = (compressed_t / tick_ns).long().clamp(max=n_bins - 1)
            hist = torch.zeros(n_bins, device=device, dtype=torch.float32)
            w = ch_w if ch_w is not None else torch.ones_like(bin_idx, dtype=torch.float32)
            hist.scatter_add_(0, bin_idx, w)

            gap_mask = diffs > kernel_extent_ns
            gap_indices = torch.where(gap_mask)[0]

            if gap_indices.numel() > 0:
                comp_bins = (compressed_t[gap_indices + 1] / tick_ns).long()
                real_times = (sorted_t[gap_indices + 1] / snap).floor() * snap
                chunk_bin_starts = [0] + comp_bins.tolist()
                chunk_time_starts = [ch_t0.item()] + real_times.tolist()
            else:
                chunk_bin_starts = [0]
                chunk_time_starts = [ch_t0.item()]

            n_ch_chunks = len(chunk_bin_starts)
            for k in range(n_ch_chunks):
                c0 = chunk_bin_starts[k]
                c1 = chunk_bin_starts[k + 1] if k + 1 < n_ch_chunks else hist.numel()
                all_adc.append(hist[c0:c1])
                offsets_list.append(offsets_list[-1] + (c1 - c0))
                all_t0.append(chunk_time_starts[k])
                all_pmt.append(ch)

        return SlicedWaveform(
            adc=torch.cat(all_adc) if all_adc else torch.zeros(0, device=device),
            offsets=torch.tensor(offsets_list, device=device, dtype=torch.long),
            t0_ns=torch.tensor(all_t0, device=device, dtype=torch.float32),
            pmt_id=torch.tensor(all_pmt, device=device, dtype=torch.long),
            tick_ns=tick_ns,
            n_channels=n_channels,
        )

    def deslice(self, fill: Optional[float] = None) -> Waveform:
        """Decompress all channels back to a shared global time axis.

        Parameters
        ----------
        fill : float, optional
            Value for dead (non-active) regions.  Defaults to
            ``attrs["pedestal"]`` if present, otherwise ``0.0``.
        """
        if fill is None:
            fill = float(self.attrs.get("pedestal", 0.0))

        if self.n_chunks == 0:
            return Waveform(
                adc=torch.full((self.n_channels, 1), fill, device=self.adc.device),
                t0=0.0, tick_ns=self.tick_ns, n_channels=self.n_channels,
                attrs=dict(self.attrs),
            )

        global_t0 = self.t0_ns.min().item()

        if self.n_bins is not None:
            n_bins = self.n_bins
        else:
            global_t_end = max(
                self.t0_ns[k].item()
                + int(self.offsets[k + 1] - self.offsets[k]) * self.tick_ns
                for k in range(self.n_chunks)
            )
            n_bins = max(1, int((global_t_end - global_t0) / self.tick_ns))

        device = self.adc.device
        data = torch.full((self.n_channels, n_bins), fill, device=device, dtype=torch.float32)

        for k in range(self.n_chunks):
            ch = int(self.pmt_id[k].item())
            chunk_data = self.adc[self.offsets[k]:self.offsets[k + 1]]
            r0 = int((self.t0_ns[k].item() - global_t0) / self.tick_ns)
            chunk_len = chunk_data.numel()
            end = min(r0 + chunk_len, n_bins)
            data[ch, r0:end] = chunk_data[:end - r0]

        return Waveform(
            adc=data, t0=global_t0,
            tick_ns=self.tick_ns, n_channels=self.n_channels,
            attrs=dict(self.attrs),
        )

    def deslice_channel(self, channel: int, fill: Optional[float] = None) -> Tuple[float, torch.Tensor]:
        """Decompress one channel. Returns (t0_ns, 1D waveform tensor).

        Parameters
        ----------
        fill : float, optional
            Value for dead regions.  Defaults to ``attrs["pedestal"]``
            if present, otherwise ``0.0``.
        """
        if fill is None:
            fill = float(self.attrs.get("pedestal", 0.0))

        if channel >= self.n_channels:
            raise IndexError(
                f"channel {channel} out of range for {self.n_channels} channels"
            )

        ch_indices = torch.where(self.pmt_id == channel)[0]

        if ch_indices.numel() == 0:
            return 0.0, torch.full((1,), fill, device=self.adc.device, dtype=self.adc.dtype)

        ch_t0 = float('inf')
        ch_t_end = float('-inf')
        for idx in ch_indices:
            k = idx.item()
            chunk_len = int(self.offsets[k + 1] - self.offsets[k])
            t_start = self.t0_ns[k].item()
            t_end = t_start + chunk_len * self.tick_ns
            ch_t0 = min(ch_t0, t_start)
            ch_t_end = max(ch_t_end, t_end)

        n_real = max(1, int((ch_t_end - ch_t0) / self.tick_ns))
        real_wf = torch.full((n_real,), fill, device=self.adc.device, dtype=self.adc.dtype)

        for idx in ch_indices:
            k = idx.item()
            chunk_data = self.adc[self.offsets[k]:self.offsets[k + 1]]
            r0 = int((self.t0_ns[k].item() - ch_t0) / self.tick_ns)
            chunk_len = chunk_data.numel()
            end = min(r0 + chunk_len, n_real)
            real_wf[r0:end] = chunk_data[:end - r0]

        return ch_t0, real_wf

    def digitize(self, pedestal: float, n_bits: int) -> SlicedWaveform:
        """Apply pedestal, quantize, and saturate to ADC range."""
        return SlicedWaveform(
            adc=_digitize(self.adc, pedestal, n_bits),
            offsets=self.offsets.clone(),
            t0_ns=self.t0_ns.clone(),
            pmt_id=self.pmt_id.clone(),
            tick_ns=self.tick_ns,
            n_channels=self.n_channels,
            n_bins=self.n_bins,
            attrs=dict(self.attrs),
        )

    def convolve(self, kernel: torch.Tensor, gain: float) -> SlicedWaveform:
        """FFT-convolve each chunk independently with kernel and apply gain."""
        conv_ticks = kernel.shape[0]
        device = self.adc.device
        new_pieces: List[torch.Tensor] = []
        new_offsets = [0]
        k_fft_cache: dict = {}

        for k in range(self.n_chunks):
            chunk_data = self.adc[self.offsets[k]:self.offsets[k + 1]]
            out_len = chunk_data.numel() + conv_ticks - 1
            n_fft = _next_fft_size(out_len)

            if n_fft not in k_fft_cache:
                k_fft_cache[n_fft] = torch.fft.rfft(kernel, n=n_fft)

            padded = F.pad(chunk_data, (0, n_fft - chunk_data.numel()))
            result = gain * torch.fft.irfft(
                torch.fft.rfft(padded) * k_fft_cache[n_fft], n=n_fft
            )
            new_pieces.append(result[:out_len])
            new_offsets.append(new_offsets[-1] + out_len)

        new_n_bins = self.n_bins + conv_ticks - 1 if self.n_bins is not None else None
        return SlicedWaveform(
            adc=torch.cat(new_pieces) if new_pieces else torch.zeros(0, device=device),
            offsets=torch.tensor(new_offsets, device=device, dtype=torch.long),
            t0_ns=self.t0_ns.clone(),
            pmt_id=self.pmt_id.clone(),
            tick_ns=self.tick_ns,
            n_channels=self.n_channels,
            n_bins=new_n_bins,
            attrs=dict(self.attrs),
        )

    def downsample(self, factor: int) -> SlicedWaveform:
        """Average groups of `factor` consecutive fine bins into coarser bins."""
        if factor == 1:
            return self
        coarse_tick = self.tick_ns * factor
        device = self.adc.device
        new_pieces: List[torch.Tensor] = []
        new_offsets = [0]

        for k in range(self.n_chunks):
            chunk = self.adc[self.offsets[k]:self.offsets[k + 1]]
            remainder = chunk.numel() % factor
            if remainder != 0:
                chunk = F.pad(chunk, (0, factor - remainder))
            coarse_chunk = chunk.reshape(-1, factor).mean(dim=1)
            new_pieces.append(coarse_chunk)
            new_offsets.append(new_offsets[-1] + coarse_chunk.numel())

        new_n_bins = math.ceil(self.n_bins / factor) if self.n_bins is not None else None
        return SlicedWaveform(
            adc=torch.cat(new_pieces) if new_pieces else torch.zeros(0, device=device),
            offsets=torch.tensor(new_offsets, device=device, dtype=torch.long),
            t0_ns=self.t0_ns.clone(),
            pmt_id=self.pmt_id.clone(),
            tick_ns=coarse_tick,
            n_channels=self.n_channels,
            n_bins=new_n_bins,
            attrs=dict(self.attrs),
        )
    
    def align(sw: SlicedWaveform, fill: float = 0.0) -> SlicedWaveform:
        """Rewrites each active channel in `sw` as a single chunk spanning the global
        [min_t0, max_t_end] window, padding gaps with `fill`. Placeholder chunks
        (t0=0.0 with a zero adc first bin) are excluded from the window calculation.
        """
        
        device = sw.adc.device
        chunk_lengths = sw.offsets[1:] - sw.offsets[:-1]

        # Exclude chunks with t0=0 when clear real activity exists elsewhere.
        # This handles both explicit placeholders and channels with no photons
        # that were assigned t0=0 by any construction path.
        # Filtering by ADC as well
        nonzero_mask = sw.t0_ns > 0.0
        if nonzero_mask.any():
            min_nonzero_t0 = sw.t0_ns[nonzero_mask].min()

            # Check only the first bin of each chunk
            first_bin_vals = sw.adc[sw.offsets[:-1]]

            # Placeholder: t0 below real activity threshold AND first bin is zero
            is_placeholder = (sw.t0_ns < min_nonzero_t0) & (first_bin_vals == 0.0)
        else:
            is_placeholder = torch.zeros(sw.n_chunks, device=device, dtype=torch.bool)

        active = torch.where(~is_placeholder)[0]

        if active.numel() == 0:
            return SlicedWaveform(
                adc=torch.zeros(0, device=device, dtype=torch.float32),
                offsets=torch.zeros(1, device=device, dtype=torch.long),
                t0_ns=torch.zeros(0, device=device, dtype=torch.float32),
                pmt_id=torch.zeros(0, device=device, dtype=torch.long),
                tick_ns=sw.tick_ns,
                n_channels=sw.n_channels,
                n_bins=sw.n_bins,
                attrs=dict(sw.attrs),
            )

        active_channels = torch.unique(sw.pmt_id[active])
        active_channels_list = active_channels.tolist()
        n_active_ch = active_channels.numel()

        # Per-channel earliest start and latest end, then take global extremes
        ch_t0_list = []
        ch_t_end_list = []
        for ch in active_channels_list:
            ch_active = active[sw.pmt_id[active] == ch]
            ch_t0 = sw.t0_ns[ch_active].min().item()
            ch_t_end = (sw.t0_ns[ch_active] + chunk_lengths[ch_active].float() * sw.tick_ns).max().item()
            ch_t0_list.append(ch_t0)
            ch_t_end_list.append(ch_t_end)

        t0_global = min(ch_t0_list)
        t_end_global = max(ch_t_end_list)
        n_bins_global = max(1, int(round((t_end_global - t0_global) / sw.tick_ns)))

        # Pre-filled with `fill`. left-pad comes from r0 offset, right-pad is implicit
        new_adc = torch.full(
            (n_active_ch * n_bins_global,), fill, device=device, dtype=torch.float32
        )

        for i, ch in enumerate(active_channels_list):
            ch_active = active[sw.pmt_id[active] == ch]
            row = new_adc[i * n_bins_global : (i + 1) * n_bins_global]
            for k in ch_active.tolist():
                chunk_data = sw.adc[sw.offsets[k] : sw.offsets[k + 1]]
                # r0 is offset from the global t0, giving left-padding for late-starting channels
                r0 = int(round((sw.t0_ns[k].item() - t0_global) / sw.tick_ns))
                chunk_len = chunk_data.numel()
                end = min(r0 + chunk_len, n_bins_global)
                if end > r0:
                    row[r0:end] = chunk_data[: end - r0]

        new_offsets = torch.arange(
            0, (n_active_ch + 1) * n_bins_global, n_bins_global,
            device=device, dtype=torch.long,
        )
        new_t0_ns = torch.full((n_active_ch,), t0_global, device=device, dtype=torch.float32)

        return SlicedWaveform(
            adc=new_adc,
            offsets=new_offsets,
            t0_ns=new_t0_ns,
            pmt_id=active_channels.to(dtype=torch.long),
            tick_ns=sw.tick_ns,
            n_channels=sw.n_channels,
            n_bins=n_bins_global,
            attrs=dict(sw.attrs),
        )