from __future__ import annotations

import math
import torch
from typing import List, Tuple


def _next_fft_size(n: int) -> int:
    """Round up to the next size that is efficient for FFT (power of 2)."""
    return 1 << math.ceil(math.log2(max(n, 1)))


def _slice_channel(
    data: torch.Tensor,
    t0: float,
    tick_ns: float,
    kernel_extent_bins: int,
) -> Tuple[torch.Tensor, List[int], List[float]]:
    """Compress a single channel by removing zero-stretches > kernel_extent_bins.

    Returns (compressed_data, chunk_bin_starts, chunk_t0_ns).
    """
    if data.numel() <= 1:
        return data, [0], [t0]

    n = data.numel()
    is_zero = data == 0

    if not is_zero.any():
        return data, [0], [t0]

    nonzero_idx = torch.where(~is_zero)[0]

    if nonzero_idx.numel() == 0:
        return data, [0], [t0]

    gaps = torch.diff(nonzero_idx) - 1
    keep_mask = torch.ones(n, dtype=torch.bool, device=data.device)

    large_gaps = torch.where(gaps > kernel_extent_bins)[0]
    chunk_bin_starts = [0]
    chunk_t0_ns = [t0]

    for g_idx in large_gaps:
        gap_start = nonzero_idx[g_idx].item() + 1
        gap_end = nonzero_idx[g_idx + 1].item()
        remove_start = gap_start + kernel_extent_bins
        keep_mask[remove_start:gap_end] = False

    compressed = data[keep_mask]

    for g_idx in large_gaps:
        resume_bin = nonzero_idx[g_idx + 1].item()
        kept_so_far = int(keep_mask[:resume_bin].sum().item())
        chunk_bin_starts.append(kept_so_far)
        chunk_t0_ns.append(t0 + resume_bin * tick_ns)

    return compressed, chunk_bin_starts, chunk_t0_ns