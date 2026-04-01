"""HDF5 I/O for GOOP SlicedWaveform results."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    import h5py

from .waveform import SlicedWaveform

if TYPE_CHECKING:
    from .simulator import OpticalSimConfig


def write_config_light(
    f: h5py.File,
    config: OpticalSimConfig,
    dataset_name: str = "",
    file_index: int = 0,
    source_file: str = "",
    n_events: int = 0,
    global_event_offset: int = 0,
) -> None:
    """Write file-level config to ``/config/`` group. Idempotent."""
    if "config" in f:
        return
    g = f.create_group("config")
    a = g.attrs

    # Simulator parameters
    a["tick_ns"] = config.tick_ns
    a["n_channels"] = config.n_channels
    a["n_pmts_per_side"] = config.n_channels // 2
    a["gain"] = config.gain
    a["oversample"] = config.oversample
    a["ser_jitter_std"] = config.ser_jitter_std
    a["baseline_noise_std"] = config.baseline_noise_std

    # Digitization
    digitized = config.digitization is not None
    a["digitized"] = digitized
    if digitized:
        a["n_bits"] = config.digitization.n_bits
        a["pedestal"] = config.digitization.pedestal

    # Kernel
    kernel = config.kernel
    a["kernel_type"] = type(kernel).__name__
    if hasattr(kernel, "duration_ns"):
        a["kernel_duration_ns"] = kernel.duration_ns

    # Caller-supplied
    a["dataset_name"] = dataset_name
    a["file_index"] = file_index
    a["source_file"] = source_file
    a["n_events"] = n_events
    a["global_event_offset"] = global_event_offset


def _split_and_write_side(
    group: h5py.Group,
    waveform: SlicedWaveform,
    mask: torch.Tensor,
    pmt_offset: int,
    digitized: bool,
    n_bits: int = 0,
) -> None:
    """Extract chunks matching *mask*, remap pmt_id, and write to *group*."""
    indices = torch.where(mask)[0]
    if indices.numel() == 0:
        group.create_dataset("adc", data=np.array([], dtype=np.float32), compression="gzip")
        group.create_dataset("offsets", data=np.array([0], dtype=np.int64), compression="gzip")
        group.create_dataset("t0_ns", data=np.array([], dtype=np.float32), compression="gzip")
        group.create_dataset("pmt_id", data=np.array([], dtype=np.int32), compression="gzip")
        return

    offsets = waveform.offsets
    parts = [waveform.adc[offsets[k]:offsets[k + 1]] for k in indices]
    adc = torch.cat(parts)
    lengths = torch.tensor(
        [offsets[k + 1] - offsets[k] for k in indices],
        dtype=torch.long,
    )
    new_offsets = torch.zeros(len(indices) + 1, dtype=torch.long)
    torch.cumsum(lengths, dim=0, out=new_offsets[1:])

    t0_ns = waveform.t0_ns[indices]
    pmt_id = waveform.pmt_id[indices] - pmt_offset

    adc_np = adc.cpu().numpy()
    adc_kwargs = {"compression": "gzip"}
    if digitized:
        adc_np = adc_np.clip(0, 65535).round().astype(np.uint16)
        if n_bits > 0:
            adc_kwargs["scaleoffset"] = n_bits
    group.create_dataset("adc", data=adc_np, **adc_kwargs)
    group.create_dataset("offsets", data=new_offsets.numpy().astype(np.int64), compression="gzip")
    group.create_dataset("t0_ns", data=t0_ns.cpu().numpy().astype(np.float32), compression="gzip")
    group.create_dataset("pmt_id", data=pmt_id.cpu().numpy().astype(np.int32), compression="gzip")


def save_event_light(
    f: h5py.File,
    event_key: str,
    waveform: SlicedWaveform,
    source_event_idx: int = 0,
    digitized: bool = False,
    n_bits: int = 0,
    n_pmts_per_side: int = 81,
) -> None:
    """Save one :class:`SlicedWaveform` split into east/west sub-groups.

    Parameters
    ----------
    n_bits : int
        When > 0 and *digitized* is True, passed as ``scaleoffset`` to the
        ``adc`` dataset for better compression (e.g. 14 for a 14-bit ADC).
    """
    evt = f.create_group(event_key)
    evt.attrs["source_event_idx"] = source_event_idx

    # PE counts per side
    pe = waveform.attrs.get("pe_counts")
    if pe is not None:
        pe_np = pe.cpu().numpy() if isinstance(pe, torch.Tensor) else np.asarray(pe)
        evt.create_dataset("pe_counts_east", data=pe_np[:n_pmts_per_side].astype(np.int32), compression="gzip")
        evt.create_dataset("pe_counts_west", data=pe_np[n_pmts_per_side:].astype(np.int32), compression="gzip")

    pmt_id = waveform.pmt_id
    east_mask = pmt_id < n_pmts_per_side
    west_mask = ~east_mask

    _split_and_write_side(evt.create_group("east"), waveform, east_mask, 0, digitized, n_bits)
    _split_and_write_side(evt.create_group("west"), waveform, west_mask, n_pmts_per_side, digitized, n_bits)


def load_event_light(
    f: h5py.File,
    event_key: str,
    device: str = "cpu",
) -> SlicedWaveform:
    """Load one event back into a merged :class:`SlicedWaveform`."""
    cfg = f["config"].attrs
    tick_ns = float(cfg["tick_ns"])
    n_channels = int(cfg["n_channels"])
    n_pmts = int(cfg["n_pmts_per_side"])

    parts_adc = []
    parts_offsets = []
    parts_t0 = []
    parts_pmt = []
    running = 0

    for side, pmt_offset in [("east", 0), ("west", n_pmts)]:
        g = f[event_key][side]
        adc = torch.tensor(g["adc"][:], dtype=torch.float32, device=device)
        offsets = torch.tensor(g["offsets"][:], dtype=torch.long, device=device)
        t0_ns = torch.tensor(g["t0_ns"][:], dtype=torch.float32, device=device)
        pmt_id = torch.tensor(g["pmt_id"][:], dtype=torch.long, device=device) + pmt_offset

        parts_adc.append(adc)
        parts_offsets.append(offsets[:-1] + running)
        running += adc.numel()
        parts_t0.append(t0_ns)
        parts_pmt.append(pmt_id)

    all_offsets = torch.cat(parts_offsets + [torch.tensor([running], device=device)])

    attrs: dict = {}
    evt = f[event_key]
    if "pe_counts_east" in evt and "pe_counts_west" in evt:
        pe_e = torch.tensor(evt["pe_counts_east"][:], dtype=torch.long, device=device)
        pe_w = torch.tensor(evt["pe_counts_west"][:], dtype=torch.long, device=device)
        attrs["pe_counts"] = torch.cat([pe_e, pe_w])

    return SlicedWaveform(
        adc=torch.cat(parts_adc),
        offsets=all_offsets,
        t0_ns=torch.cat(parts_t0),
        pmt_id=torch.cat(parts_pmt),
        tick_ns=tick_ns,
        n_channels=n_channels,
        attrs=attrs,
    )
