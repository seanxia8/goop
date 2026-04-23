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
    delays: Union[Delays, List[DelaySamplerBase]] = field(default_factory=create_default_delays)
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
            for li in range(n_batch):
                lbl_mask = b_photon_labels == batch_labels[li]
                lbl_t = b_times[lbl_mask]
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
        stitched: bool = True,
        subtract_t0: bool = False,
        add_baseline_noise: bool = True,
        label_batch_size: Optional[int] = None,
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

        Returns
        -------
        SlicedWaveform or Waveform when *labels* is None.
        list[SlicedWaveform] when *labels* is provided.
        """
        cfg = self.config
        device = self._device

        # auto-convert any array type supporting __dlpack__ (incl JAX) to torch.Tensor
        pos, n_photons, t_step = (
            torch.from_dlpack(a) if not isinstance(a, torch.Tensor) else a
            for a in (pos, n_photons, t_step)
        )

        if labels is not None:
            labels = (
                torch.from_dlpack(labels) if not isinstance(labels, torch.Tensor) else labels
            ).to(dtype=torch.long, device=device)

        if subtract_t0:
            t_step = t_step - t_step.min()
        
        if labels is not None:
            unique_tpc_labels = torch.unique(labels)

            if cfg.time_window_ns is not None:
                rand_t0_per_label = torch.rand(len(unique_tpc_labels), device=labels.device) * cfg.time_window_ns
                label_indices = torch.searchsorted(unique_tpc_labels, labels)
                rand_t0 = rand_t0_per_label[label_indices]
                t_step += rand_t0            

        # 1. TOF sampling (batched across all positions)
        times, channels, source_idx = cfg.tof_sampler.sample(pos, n_photons, t_step=t_step)

        # 2. Stochastic delays
        if times.numel() > 0:
            times = times + cfg.delays.sample(times.shape[0], device)

        # Labeled mode
        if labels is not None:
            photon_labels = labels[source_idx] if times.numel() > 0 else torch.empty(0, dtype=torch.long, device=device)
            unique_labels = torch.unique(labels)[:cfg.n_labels_to_simulate]
            n_labels = len(unique_labels)            
            bs = label_batch_size or n_labels

            results: List[SlicedWaveform] = []
            for start in range(0, n_labels, bs):
                batch_labels = unique_labels[start:start + bs]
                results.extend(self._simulate_labeled_batch(
                    times, channels, photon_labels, batch_labels,
                    stitched, add_baseline_noise,
                ))
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
