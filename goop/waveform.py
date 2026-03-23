"""Waveform types for the optical TPC simulation pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
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

    data: torch.Tensor  # (n_channels, n_bins)
    t0: float           # global time origin in ns
    tick_ns: float      # time step in ns
    n_channels: int     # number of PMT channels

    @staticmethod
    def from_photons(
        times: torch.Tensor,
        channels: torch.Tensor,
        tick_ns: float,
        n_channels: int,
        t0: float = None,
    ) -> Waveform:
        """Build per-channel histograms on a shared global time axis.

        NOTE: this materializes the full global histogram in memory.
        For events with huge time ranges, this may cause an OOM.

        Parameters
        ----------
        t0 : Global time origin in ns. Defaults to times.min(),
             snapped to the tick boundary.
        """
        device = times.device

        if times.numel() == 0:
            return Waveform(
                data=torch.zeros(n_channels, 1, device=device),
                t0=t0 or 0.0, tick_ns=tick_ns, n_channels=n_channels,
            )
        if t0 is None:
            t0 = (times.min() / tick_ns).floor().item() * tick_ns
        shifted = times - t0
        if (shifted < 0).any():
            raise ValueError(
                f"Some photon times are before t0={t0:.2f} ns "
            )
        n_bins = int(shifted.max().item() / tick_ns) + 1

        data = torch.zeros(n_channels, n_bins, device=device, dtype=torch.float32)
        bin_idx = (shifted / tick_ns).long().clamp(max=n_bins - 1)

        flat_idx = channels.long() * n_bins + bin_idx
        data.view(-1).scatter_add_(
            0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32)
        )

        return Waveform(data=data, t0=t0, tick_ns=tick_ns, n_channels=n_channels)

    def slice(self, kernel_extent_ns: float) -> SlicedWaveform:
        """Compress by removing dead gaps (zero-stretches > kernel_extent_ns)."""
        kernel_extent_bins = int(kernel_extent_ns / self.tick_ns)
        device = self.data.device
        all_adc: List[torch.Tensor] = []
        offsets_list = [0]
        all_t0: List[float] = []
        all_pmt: List[int] = []

        for ch in range(self.n_channels):
            compressed, bin_starts, t0_starts = _slice_channel(
                self.data[ch], self.t0, self.tick_ns, kernel_extent_bins
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
            n_bins=self.data.shape[1],
        )

    def digitize(self, pedestal: float, n_bits: int) -> Waveform:
        """Apply pedestal, quantize, and saturate to ADC range."""
        return Waveform(
            data=_digitize(self.data, pedestal, n_bits),
            t0=self.t0, tick_ns=self.tick_ns, n_channels=self.n_channels,
        )

    def convolve(self, kernel: torch.Tensor, gain: float) -> Waveform:
        """FFT-convolve all channels with kernel and apply gain."""
        conv_ticks = kernel.shape[0]
        n_ch, n_tick = self.data.shape
        n_fft = _next_fft_size(n_tick + conv_ticks - 1)

        padded = F.pad(self.data, (0, n_fft - n_tick))
        k_fft = torch.fft.rfft(kernel, n=n_fft)
        result = gain * torch.fft.irfft(
            torch.fft.rfft(padded) * k_fft.unsqueeze(0), n=n_fft
        )
        out_len = n_tick + conv_ticks - 1
        return Waveform(
            data=result[:, :out_len], t0=self.t0,
            tick_ns=self.tick_ns, n_channels=self.n_channels,
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
    offsets: torch.Tensor    # (n_chunks+1,)
    t0_ns: torch.Tensor     # (n_chunks,)
    pmt_id: torch.Tensor    # (n_chunks,)
    tick_ns: float
    n_channels: int
    n_bins: int = None       # expected desliced length (optional, set by slice/convolve)

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
    ) -> SlicedWaveform:
        """Build compressed per-channel histograms directly from photon times."""
        device = times.device
        all_adc: List[torch.Tensor] = []
        offsets_list = [0]
        all_t0: List[float] = []
        all_pmt: List[int] = []

        for ch in range(n_channels):
            ch_times = times[channels == ch]

            if ch_times.numel() == 0:
                all_adc.append(torch.zeros(1, device=device))
                offsets_list.append(offsets_list[-1] + 1)
                all_t0.append(0.0)
                all_pmt.append(ch)
                continue

            sorted_t = ch_times.sort().values
            diffs = torch.diff(sorted_t)
            ch_t0 = (sorted_t[0] / tick_ns).floor() * tick_ns

            excess = (diffs - kernel_extent_ns).clamp(min=0)
            cum_dead = torch.cat([torch.zeros(1, device=device), excess.cumsum(0)])
            compressed_t = sorted_t - cum_dead - ch_t0

            n_bins = int(compressed_t[-1].item() / tick_ns) + 1
            bin_idx = (compressed_t / tick_ns).long().clamp(max=n_bins - 1)
            hist = torch.zeros(n_bins, device=device, dtype=torch.float32)
            hist.scatter_add_(0, bin_idx, torch.ones_like(bin_idx, dtype=torch.float32))

            gap_mask = diffs > kernel_extent_ns
            gap_indices = torch.where(gap_mask)[0]

            if gap_indices.numel() > 0:
                comp_bins = (compressed_t[gap_indices + 1] / tick_ns).long()
                real_times = (sorted_t[gap_indices + 1] / tick_ns).floor() * tick_ns
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

    def deslice(self) -> Waveform:
        """Decompress all channels back to a shared global time axis."""
        if self.n_chunks == 0:
            return Waveform(
                data=torch.zeros(self.n_channels, 1, device=self.adc.device),
                t0=0.0, tick_ns=self.tick_ns, n_channels=self.n_channels,
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
        data = torch.zeros(self.n_channels, n_bins, device=device, dtype=torch.float32)

        for k in range(self.n_chunks):
            ch = int(self.pmt_id[k].item())
            chunk_data = self.adc[self.offsets[k]:self.offsets[k + 1]]
            r0 = int((self.t0_ns[k].item() - global_t0) / self.tick_ns)
            chunk_len = chunk_data.numel()
            end = min(r0 + chunk_len, n_bins)
            data[ch, r0:end] = chunk_data[:end - r0]

        return Waveform(
            data=data, t0=global_t0,
            tick_ns=self.tick_ns, n_channels=self.n_channels,
        )

    def deslice_channel(self, channel: int) -> Tuple[float, torch.Tensor]:
        """Decompress one channel. Returns (t0_ns, 1D waveform tensor)."""
        if channel >= self.n_channels:
            raise IndexError(
                f"channel {channel} out of range for {self.n_channels} channels"
            )

        ch_indices = torch.where(self.pmt_id == channel)[0]

        if ch_indices.numel() == 0:
            return 0.0, torch.zeros(1, device=self.adc.device, dtype=self.adc.dtype)

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
        real_wf = torch.zeros(n_real, device=self.adc.device, dtype=self.adc.dtype)

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
        )