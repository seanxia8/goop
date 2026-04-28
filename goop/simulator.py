"""Optical TPC simulation pipeline."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import torch

# any array type supporting __dlpack__ (torch.Tensor, jax.Array, etc.)
ArrayLike = Union[torch.Tensor, Any]

from dataclasses import dataclass, field

from .base import ConvolutionKernelBase, DelaySamplerBase, PhotonSourceBase, TOFSamplerBase
from .digitize import DigitizationConfig
from .sampler import create_default_tof_sampler
from .delays import Delays, create_default_delays
from .kernels import create_default_kernel
from .waveform import SlicedWaveform, Waveform

@dataclass
class OpticalSimConfig:
    """Full configuration for the optical simulation pipeline.
    """

    tof_sampler: TOFSamplerBase = field(default_factory=create_default_tof_sampler)
    delays: Union[Delays, List[DelaySamplerBase]] = field(default_factory=list)
    kernel: ConvolutionKernelBase = field(default_factory=create_default_kernel)

    aux_photon_sources: List[PhotonSourceBase] = field(default_factory=list)
    digitization: Optional[DigitizationConfig] = None

    device: str = "cuda"
    tick_ns: float = 1.0
    oversample: int = 1
    gain: float = -45.0    # per PMT gain in ADC units
    ser_jitter_std: float = 0.0      # std of multiplicative Gaussian on PE weights
    baseline_noise_std: float = 0.0  # std of per-sample Gaussian ADC noise
    calibration_mode: bool = False
    n_modules: int = 1
    n_labels_to_simulate: int = 3
    time_window_ns: Optional[float] = None

    # for diff-sim
    streaming: bool = True
    stream_chunk_size: int = 5000
    stream_checkpoint: bool = True    # per-chunk gradient checkpoint in histogram_pdf;
                                       # disable for small N (e.g. after voxelization)

    def __post_init__(self):
        if not isinstance(self.oversample, int) or self.oversample < 1:
            raise ValueError(f"oversample must be an int >= 1, got {self.oversample}")
        self.n_channels = self.tof_sampler.n_channels * self.n_modules
        if isinstance(self.delays, list):
            self.delays = Delays(self.delays)
        self.n_labels_to_simulate = min(self.n_labels_to_simulate, 30)

class OpticalSimulator:
    """Full optical TPC simulation pipeline."""

    def __init__(self, config: OpticalSimConfig):
        self.config = config
        self._device = torch.device(config.device)
        self._fine_tick = config.tick_ns / config.oversample
        if config.oversample > 1:
            self._fine_kernel = config.kernel.with_tick_ns(self._fine_tick)
        else:
            self._fine_kernel = config.kernel


    def _simulate(
        self,
        times: torch.Tensor,
        channels: torch.Tensor,
        stitched: bool,
        *,
        n_channels: Optional[int] = None,
        add_baseline_noise: bool = True,
    ) -> Union[SlicedWaveform, Waveform]:
        """Histogram → SER jitter → convolve → downsample → noise → digitize.

        Aux photon sources must be injected into *times*/*channels* by the
        caller before this method is called.
        """
        cfg = self.config
        device = self._device
        fine_tick = self._fine_tick
        fine_kernel_tensor = self._fine_kernel()

        if n_channels is None:
            n_channels = cfg.n_channels

        wvfm_cls = SlicedWaveform if stitched else Waveform
        extra_args: dict = {}
        if stitched:
            extra_args["kernel_extent_ns"] = fine_kernel_tensor.shape[0] * fine_tick
        if cfg.oversample > 1:
            extra_args["t0_snap_ns"] = cfg.tick_ns
        if cfg.ser_jitter_std > 0 and times.numel() > 0:
            extra_args["weights"] = torch.normal(
                1.0, cfg.ser_jitter_std, (times.shape[0],), device=device
            )

        if times.numel() > 0:
            pe_counts = torch.bincount(channels.long(), minlength=n_channels)
        else:
            pe_counts = torch.zeros(n_channels, device=device, dtype=torch.long)

        wf = wvfm_cls.from_photons(times, channels, fine_tick, n_channels, **extra_args)
        wf.attrs["pe_counts"] = pe_counts

        wf = wf.convolve(fine_kernel_tensor, cfg.gain)
        if cfg.oversample > 1:
            wf = wf.downsample(cfg.oversample)

        if add_baseline_noise and cfg.baseline_noise_std > 0:
            wf.adc += torch.randn_like(wf.adc) * cfg.baseline_noise_std
        if cfg.digitization is not None:
            wf = wf.digitize(cfg.digitization.pedestal, cfg.digitization.n_bits)
            wf.attrs["pedestal"] = cfg.digitization.pedestal

        return wf

    @staticmethod
    def _split_by_label(
        wf: SlicedWaveform, n_channels: int, unique_labels: torch.Tensor,
    ) -> List[SlicedWaveform]:
        """Split a virtual-channel SlicedWaveform into per-label waveforms."""
        device = wf.adc.device
        results: List[SlicedWaveform] = []
        for li, lbl in enumerate(unique_labels):
            ch_lo = li * n_channels
            ch_hi = ch_lo + n_channels
            idx = ((wf.pmt_id >= ch_lo) & (wf.pmt_id < ch_hi)).nonzero(as_tuple=True)[0]

            if idx.numel() == 0:
                sub = SlicedWaveform(
                    adc=torch.empty(0, device=device),
                    offsets=torch.tensor([0], device=device, dtype=torch.long),
                    t0_ns=torch.empty(0, device=device),
                    pmt_id=torch.empty(0, device=device, dtype=torch.long),
                    tick_ns=wf.tick_ns, n_channels=n_channels,
                    attrs={"pe_counts": wf.attrs["pe_counts"][ch_lo:ch_hi],
                           "label": lbl.item()},
                )
            else:
                starts = wf.offsets[idx]
                ends = wf.offsets[idx + 1]
                lengths = ends - starts
                sample_indices = torch.cat([
                    torch.arange(s, e, device=device)
                    for s, e in zip(starts.tolist(), ends.tolist())
                ])
                sub = SlicedWaveform(
                    adc=wf.adc[sample_indices],
                    offsets=torch.cat([torch.tensor([0], device=device), lengths.cumsum(0)]),
                    t0_ns=wf.t0_ns[idx],
                    pmt_id=wf.pmt_id[idx] - ch_lo,
                    tick_ns=wf.tick_ns, n_channels=n_channels,
                    attrs={"pe_counts": wf.attrs["pe_counts"][ch_lo:ch_hi],
                           "label": lbl.item()},
                )
            results.append(sub)
        return results

    def _simulate_labeled_batch(
        self,
        times: torch.Tensor,
        channels: torch.Tensor,
        photon_labels: torch.Tensor,
        batch_labels: torch.Tensor,
        stitched: bool,
        add_baseline_noise: bool,
    ) -> List[SlicedWaveform]:
        """Run the virtual-channel pipeline for a batch of labels."""
        cfg = self.config
        device = self._device
        n_ch = cfg.n_channels
        n_batch = len(batch_labels)

        # Select photons belonging to this batch
        batch_mask = torch.isin(photon_labels, batch_labels)
        b_times = times[batch_mask]
        b_channels = channels[batch_mask]
        b_photon_labels = photon_labels[batch_mask]

        # Remap to contiguous virtual channels for this batch
        label_idx = torch.searchsorted(batch_labels, b_photon_labels)
        virtual_channels = label_idx * n_ch + b_channels

        # Per-label aux sources in virtual channel space
        if cfg.aux_photon_sources and b_times.numel() > 0:
            b_times_real = b_times  # snapshot before aux sources grow b_times
            for li in range(n_batch):
                lbl_mask = b_photon_labels == batch_labels[li]
                lbl_t = b_times_real[lbl_mask]
                if lbl_t.numel() == 0:
                    continue
                t_start, t_end = lbl_t.min().item(), lbl_t.max().item()
                for source in cfg.aux_photon_sources:
                    aux_t, aux_ch = source.sample(n_ch, t_start, t_end, device)
                    if aux_t.numel() > 0:
                        b_times = torch.cat([b_times, aux_t])
                        virtual_channels = torch.cat([virtual_channels, li * n_ch + aux_ch])

        combined = self._simulate(
            b_times, virtual_channels, stitched,
            n_channels=n_ch * n_batch, add_baseline_noise=add_baseline_noise,
        )
        return self._split_by_label(combined, n_ch, batch_labels)

    @torch.no_grad()
    def simulate(
        self,
        pos: ArrayLike,
        n_photons: ArrayLike,
        t_step: ArrayLike,
        labels: Optional[ArrayLike] = None,
        pdgs: Optional[ArrayLike] = None,
        de: Optional[ArrayLike] = None,
        stitched: bool = True,
        subtract_t0: bool = False,
        add_baseline_noise: bool = True,
        label_batch_size: Optional[int] = None,
        return_tpc: bool = False,
    ) -> Union[SlicedWaveform, Waveform, List[SlicedWaveform]]:
        """
        Run the full optical simulation pipeline.

        Parameters
        ----------
        labels : optional (N,) integer array
            Per-position label (e.g. volume ID, interaction ID).  When
            provided, photon channels are remapped into disjoint virtual
            channel spaces (one per label) so the entire pipeline runs in
            a single batched call.  The combined waveform is then split
            back into per-label ``SlicedWaveform`` objects.
        label_batch_size : optional int
            Maximum number of unique labels to process in one virtual-channel
            batch.  When the number of unique labels exceeds this, they are
            processed in groups to limit GPU memory.  TOF sampling and delays
            are still computed once for all labels.
        return_tpc : bool
            When True and *labels* is provided, return a tuple
            ``(waveforms, positions, n_photons, t_step, labels)`` where the
            TPC arrays reflect all manipulations (t0 subtraction, random
            shift, window filtering) so they can be passed directly to
            ``save_event_light_w_tpc``.

        Returns
        -------
        SlicedWaveform or Waveform when *labels* is None.
        list[SlicedWaveform] when *labels* is provided and *return_tpc* is False.
        tuple(list[SlicedWaveform], np.ndarray, np.ndarray, np.ndarray, np.ndarray)
            when *labels* is provided and *return_tpc* is True.
        """
        cfg = self.config
        device = self._device

        pos, n_photons, t_step = (
            torch.from_dlpack(a) if not isinstance(a, torch.Tensor) else a
            for a in (pos, n_photons, t_step)
        )
        if pdgs is not None:
            pdgs = torch.from_dlpack(pdgs) if not isinstance(pdgs, torch.Tensor) else pdgs
        if de is not None:
            de = torch.from_dlpack(de) if not isinstance(de, torch.Tensor) else de
        pos = pos.to(device=device, dtype=torch.float32)
        n_photons = n_photons.to(device=device)
        t_step = t_step.to(device=device, dtype=torch.float32)
        if pdgs is not None:
            pdgs = pdgs.to(device=device, dtype=torch.float32)
        if de is not None:
            de = de.to(device=device, dtype=torch.float32)


        if labels is not None:
            labels = (
                torch.from_dlpack(labels) if not isinstance(labels, torch.Tensor) else labels
            ).to(dtype=torch.long, device=device)
            # unique_tpc_labels: sorted tensor of ALL unique valid labels.
            # Used for t_step preprocessing so every label is normalised correctly.
            unique_tpc_labels = torch.unique(labels[labels >= 0])

        if subtract_t0:
            if labels is not None:
                # Step 1: normalise each label group so its minimum t_step is 0.
                # Uses unique_tpc_labels (all labels, sorted) so searchsorted is valid
                # for every label present in the input.
                valid_mask = labels >= 0
                label_indices = torch.searchsorted(unique_tpc_labels, labels.clamp(min=0))
                per_label_min = torch.full(
                    (len(unique_tpc_labels),), float("inf"), dtype=t_step.dtype, device=device
                )
                per_label_min.scatter_reduce_(
                    0, label_indices[valid_mask], t_step[valid_mask], reduce="amin", include_self=True
                )
                t_step = t_step - per_label_min[label_indices]
            else:
                t_step = t_step - t_step.min()

        if labels is not None and cfg.time_window_ns is not None:
            valid_mask = labels >= 0
            # Again use unique_tpc_labels (sorted, all labels) for searchsorted.
            label_indices = torch.searchsorted(unique_tpc_labels, labels[valid_mask])
            # Step 2: random per-label start time within [0, time_window_ns)
            rand_t0_per_label = torch.rand(len(unique_tpc_labels), device=device) * cfg.time_window_ns
            # Only shift valid-label points; label=-1 points keep their original t_step.
            t_step = t_step.clone()
            t_step[valid_mask] = t_step[valid_mask] + rand_t0_per_label[label_indices]
            # Step 3: drop valid-label points that exceed the window; always keep label=-1.
            keep = valid_mask & (t_step <= cfg.time_window_ns)
            pos_masked, n_photons_masked, t_step_masked, labels_masked = pos[keep], n_photons[keep], t_step[keep], labels[keep]
            pdgs_masked, de_masked = pdgs[keep], de[keep]
            times, channels, source_idx = cfg.tof_sampler.sample(pos_masked, n_photons_masked, t_step=t_step_masked)

        else:
            # 1. TOF sampling (batched across all positions)
            times, channels, source_idx = cfg.tof_sampler.sample(pos, n_photons, t_step=t_step)
            # No time-window filter: all valid positions are "kept".
            pos_masked, n_photons_masked, t_step_masked = pos, n_photons, t_step
            pdgs_masked, de_masked = pdgs, de
            labels_masked = labels if labels is not None else None

        # 2. Stochastic delays
        if times.numel() > 0:
            times = times + cfg.delays.sample(times.shape[0], device)

        # Labeled mode — choose which labels to generate waveforms for.
        # This is done *after* t_step preprocessing so the subset selection
        # does not corrupt the normalisation of non-simulated labels.
        if labels is not None:
            if cfg.n_labels_to_simulate is not None and cfg.n_labels_to_simulate < len(unique_tpc_labels):
                perm = torch.randperm(len(unique_tpc_labels), device=device)[:cfg.n_labels_to_simulate]
                # Sort so that downstream searchsorted calls remain valid.
                unique_labels_to_simulate = unique_tpc_labels[perm].sort().values
            else:
                unique_labels_to_simulate = unique_tpc_labels

            photon_labels = labels_masked[source_idx] if times.numel() > 0 else torch.empty(0, dtype=torch.long, device=device)
            n_labels = len(unique_labels_to_simulate)
            bs = label_batch_size or n_labels
            results: List[SlicedWaveform] = []
            for start in range(0, n_labels, bs):
                batch_labels = unique_labels_to_simulate[start:start + bs]
                results.extend(self._simulate_labeled_batch(
                    times, channels, photon_labels, batch_labels,
                    stitched, add_baseline_noise,
                ))

            if return_tpc:
                # Keep only TPC points whose label was actually simulated,
                # so the returned arrays align 1-to-1 with `results`.
                sim_mask = torch.isin(labels_masked, unique_labels_to_simulate)
                pdgs_masked, de_masked = pdgs_masked[sim_mask], de_masked[sim_mask]
                return (
                    results,
                    pos_masked[sim_mask].cpu().numpy(),
                    n_photons_masked[sim_mask].cpu().numpy(),
                    t_step_masked[sim_mask].cpu().numpy(),
                    labels_masked[sim_mask].cpu().numpy(),
                    pdgs_masked.cpu().numpy(),
                    de_masked.cpu().numpy(),
                )
            return results

        # Unlabeled mode
        if cfg.aux_photon_sources and times.numel() > 0:
            t_start, t_end = times.min().item(), times.max().item()
            for source in cfg.aux_photon_sources:
                aux_t, aux_ch = source.sample(cfg.n_channels, t_start, t_end, device)
                if aux_t.numel() > 0:
                    times = torch.cat([times, aux_t])
                    channels = torch.cat([channels, aux_ch])

        return self._simulate(times, channels, stitched, add_baseline_noise=add_baseline_noise)
