"""HDF5 I/O for GOOP SlicedWaveform results."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

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
    label_key: str = "volume",
    n_labels: int = 2,
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
    a["label_key"] = label_key
    a["n_labels"] = n_labels
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

def _write_sliced_waveform(
    group: h5py.Group,
    waveform: SlicedWaveform,
    digitized: bool,
    n_bits: int = 0,
) -> None:
    """Write a single SlicedWaveform to *group*."""
    adc_np = waveform.adc.cpu().numpy()
    adc_kwargs = {"compression": "gzip", "shuffle": True}
    if digitized:
        adc_np = adc_np.clip(0, 65535).round().astype(np.uint16)
        if n_bits > 0:
            adc_kwargs["scaleoffset"] = n_bits
    group.create_dataset("adc", data=adc_np, **adc_kwargs)
    group.create_dataset("offsets", data=waveform.offsets.cpu().numpy().astype(np.int64), compression="gzip")
    group.create_dataset("t0_ns", data=waveform.t0_ns.cpu().numpy().astype(np.float32), compression="gzip")
    group.create_dataset("pmt_id", data=waveform.pmt_id.cpu().numpy().astype(np.int32), compression="gzip")

def save_event_tpc(
    f: h5py.File,
    event_key: str,
    positions: np.ndarray,
    n_photons: np.ndarray,
    t_step: np.ndarray,
    labels: np.ndarray,
    source_event_idx: int = 0,
) -> np.ndarray:
    """Save per-volume tpc interaction objects for one event.
    Parameters
    ----------
    positions : np.ndarray
        (N, 3) array of interaction positions
    n_photons : np.ndarray
        (N,) array of interaction photon counts
    t_step : np.ndarray
        (N,) array of interaction time steps
    labels : np.ndarray
        (N,) array of interaction labels
    source_event_idx : int
        Index of the source event
    """
    evt = f.create_group(event_key)
    evt.attrs["source_event_idx"] = source_event_idx 
    interaction_ids = np.unique(labels)    
    evt.attrs["n_interactions"] = len(interaction_ids)

    for i, int_id in enumerate(interaction_ids):
        s_indices = np.where(labels == int_id)[0]
        int_grp = evt.create_group(f"interaction_{i}")
        int_grp.create_dataset("n_photons", data=n_photons[s_indices].astype(np.int32), compression="gzip")
        int_grp.create_dataset("t_step", data=t_step[s_indices].astype(np.float32), compression="gzip")
        int_grp.create_dataset("labels", data=np.array([int_id], dtype=np.int32), compression="gzip")
        int_grp.create_dataset("positions", data=positions[s_indices].astype(np.float32), compression="gzip")

    return interaction_ids

def save_event_light_w_tpc(
    f: h5py.File,
    event_key: str,
    waveforms: List[SlicedWaveform],
    source_event_idx: int = 0,
    tpc_interactions: np.ndarray = None,
    digitized: bool = False,
    n_bits: int = 0,
) -> None:
    if tpc_interactions is None:
        save_event_light(f, event_key, waveforms, source_event_idx, digitized, n_bits)
    else:
        evt = f.require_group(event_key)
        evt.attrs["n_labels"] = len(tpc_interactions)
        for i, int_id in enumerate(tpc_interactions):
            int_grp = evt[f"interaction_{i}"]
            assert int_grp is not None, "Interaction group must be created"
            ### new waveform format here ###

def save_event_light(
    f: h5py.File,
    event_key: str,
    waveforms: List[SlicedWaveform],
    source_event_idx: int = 0,
    digitized: bool = False,
    n_bits: int = 0,
) -> None:
    """Save per-volume :class:`SlicedWaveform` objects for one event.

    Parameters
    ----------
    waveforms : list[SlicedWaveform]
        One waveform per detector volume (from separate GOOP runs).
    n_bits : int
        When > 0 and *digitized* is True, passed as ``scaleoffset`` to the
        ``adc`` dataset for better compression (e.g. 14 for a 14-bit ADC).
    """
    evt = f.create_group(event_key)
    evt.attrs["source_event_idx"] = source_event_idx
    evt.attrs["n_labels"] = len(waveforms)

    for v, wvfm in enumerate(waveforms):
        label_val = wvfm.attrs.get("label", v)
        vol_grp = evt.create_group(f"label_{label_val}")

        # PE counts
        pe = wvfm.attrs.get("pe_counts")
        if pe is not None:
            pe_np = pe.cpu().numpy() if isinstance(pe, torch.Tensor) else np.asarray(pe)
            vol_grp.create_dataset("pe_counts", data=pe_np.astype(np.int32), compression="gzip")

        # Waveform data
        _write_sliced_waveform(vol_grp, wvfm, digitized, n_bits)


def load_event_tpc(
    f: h5py.File,
    event_key: str,
    device: str = "cpu",
) -> List:
    cfg = f["config"].attrs
    evt = f[event_key]
    n_interactions = evt.attrs["n_interactions"]
    results = []
    for i in range(n_interactions):
        int_grp = evt[f"interaction_{i}"]
        assert int_grp is not None, "Interaction group must be created"
        positions = torch.tensor(int_grp["positions"][:], dtype=torch.float32, device=device)
        n_photons = torch.tensor(int_grp["n_photons"][:], dtype=torch.int32, device=device)
        t_step = torch.tensor(int_grp["t_step"][:], dtype=torch.float32, device=device)
        labels = torch.tensor(int_grp["labels"][:], dtype=torch.int32, device=device)
        results.append(
            {
                "positions": positions,
                "n_photons": n_photons,
                "t_step": t_step,
                "labels": labels,
            }
        )
    return results

def load_event_light(
    f: h5py.File,
    event_key: str,
    device: str = "cpu",
) -> List[SlicedWaveform]:
    """Load per-label :class:`SlicedWaveform` objects for one event.

    Returns a list of SlicedWaveforms (one per label). To get the combined
    signal, deslice each and sum the resulting dense waveforms.
    """
    cfg = f["config"].attrs
    tick_ns = float(cfg["tick_ns"])
    n_channels = int(cfg["n_channels"])
    pedestal = float(cfg["pedestal"]) if cfg.get("digitized", False) else 0.0

    evt = f[event_key]

    # Discover label groups (label_0, label_1, ...) sorted by label value
    label_keys = sorted(
        [k for k in evt.keys() if k.startswith("label_")],
        key=lambda k: int(k.split("_", 1)[1]),
    )

    result = []
    for lk in label_keys:
        g = evt[lk]
        adc = torch.tensor(g["adc"][:], dtype=torch.float32, device=device)
        offsets = torch.tensor(g["offsets"][:], dtype=torch.long, device=device)
        t0_ns = torch.tensor(g["t0_ns"][:], dtype=torch.float32, device=device)
        pmt_id = torch.tensor(g["pmt_id"][:], dtype=torch.long, device=device)

        label_val = int(lk.split("_", 1)[1])
        attrs: dict = {"label": label_val, "pedestal": pedestal}
        if "pe_counts" in g:
            attrs["pe_counts"] = torch.tensor(g["pe_counts"][:], dtype=torch.long, device=device)

        result.append(SlicedWaveform(
            adc=adc,
            offsets=offsets,
            t0_ns=t0_ns,
            pmt_id=pmt_id,
            tick_ns=tick_ns,
            n_channels=n_channels,
            attrs=attrs,
        ))

    return result
