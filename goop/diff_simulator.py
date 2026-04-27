"""Differentiable optical-simulation pipeline.

Replaces stochastic per-photon delay sampling with deterministic convolution
against a composite ``Response`` kernel that includes the delay PDFs, and
replaces stochastic per-photon TOF sampling with a deterministic PDF
deposition via ``DifferentiableTOFSampler.sample_pdf``.

Required config
---------------
- ``cfg.kernel`` is a ``Response`` (or any ``ConvolutionKernelBase``) that
  already encodes the delay PDFs the user wants — typically
  ``create_default_response()``.
- ``cfg.tof_sampler`` exposes a ``sample_pdf(...)`` method
  (e.g. ``DifferentiableTOFSampler``).
"""

from __future__ import annotations

from typing import Any, Optional, Union

import torch

from .digitize import digitize_ste
from .simulator import OpticalSimConfig, OpticalSimulator
from .waveform import SlicedWaveform, Waveform

ArrayLike = Union[torch.Tensor, Any]

# Padding constants for V2 sparse streaming time-grouping.
_GAP_THRESHOLD_PAD_NS = 100.0  # safety margin so a gap-split can't bisect a kernel-extent tail
_N_BINS_GROUP_PAD = 10         # extra bins so late photons aren't truncated by integer rounding


def as_dlpack(tensor: ArrayLike) -> torch.Tensor:
    """Convert a __dlpack__-supporting array to a torch.Tensor, no-op if already one.

    Note: ``torch.from_dlpack`` fails on tensors that require_grad, so we
    must short-circuit on torch.Tensor inputs to preserve the autograd graph.
    """
    if isinstance(tensor, torch.Tensor):
        return tensor
    return torch.from_dlpack(tensor)

class DifferentiableOpticalSimulator(OpticalSimulator):
    """OpticalSimulator with stochastic delays replaced by kernel convolution.

    The delay PDFs (Scintillation, TPB, TTS) must already be folded into
    ``config.kernel`` — typically a ``Response`` composite.  Instead of
    drawing per-photon delays, the photon histogram is convolved with the
    full delay-and-detector response in a single FFT pass.
    """

    def __init__(self, config: OpticalSimConfig):
        if not hasattr(config.tof_sampler, "sample_pdf"):
            raise ValueError(
                "DifferentiableOpticalSimulator requires a TOF sampler that exposes "
                "`sample_pdf(...)` (e.g. PCATOFSampler subclasses). Got "
                f"{type(config.tof_sampler).__name__}."
            )
        super().__init__(config)

    def simulate(
        self,
        pos: ArrayLike,
        n_photons: ArrayLike,
        t_step: ArrayLike,
        stitched: bool = True,
        subtract_t0: bool = False,
        add_baseline_noise: bool = True,
    ) -> SlicedWaveform:
        """Run the deterministic PDF-deposition pipeline through the diff
        TOF sampler and the composite ``Response`` kernel.

        ``stitched`` must be ``True`` (SlicedWaveform output).  Labeled mode
        is not supported here — the PDF-deposition path doesn't track
        per-photon source positions.
        """
        if not stitched:
            raise ValueError(
                "DifferentiableOpticalSimulator.simulate(..., stitched=False) is "
                "not supported."
            )
        cfg = self.config
        device = self._device

        # auto-convert any array supporting __dlpack__ to torch.Tensor
        pos, n_photons, t_step = map(as_dlpack, (pos, n_photons, t_step))

        if subtract_t0:
            t_step = t_step - t_step.min()

        if cfg.streaming:
            return self._simulate_streaming(
                pos, n_photons, t_step, add_baseline_noise=add_baseline_noise
            )

        times, channels, weights = cfg.tof_sampler.sample_pdf(
            pos, n_photons, t_step=t_step
        )

        # Inject auxiliary photon sources (e.g. DarkNoise) with unit weights.
        # Dark hits are independent of (pos, n_photons), so they don't
        # interfere with the gradient path back to those inputs.
        if cfg.aux_photon_sources and times.numel() > 0:
            t_start, t_end = times.min().item(), times.max().item()
            for source in cfg.aux_photon_sources:
                aux_t, aux_ch = source.sample(cfg.n_channels, t_start, t_end, device)
                if aux_t.numel() > 0:
                    times = torch.cat([times, aux_t])
                    channels = torch.cat([channels, aux_ch])
                    weights = torch.cat([weights, torch.ones_like(aux_t)])

        return self._simulate(
            times, channels, stitched, add_baseline_noise=add_baseline_noise,
            weights=weights,
        )

    def _simulate(
        self,
        times: torch.Tensor,
        channels: torch.Tensor,
        stitched: bool,
        *,
        n_channels: Optional[int] = None,
        add_baseline_noise: bool = True,
        weights: Optional[torch.Tensor] = None,
    ) -> SlicedWaveform:
        """Histogram (with PDF weights, optional SER jitter) → convolve →
        downsample → baseline noise.

        ``weights`` carries the per-synthetic-photon probability mass from
        ``sample_pdf`` (plus unit weights from any aux photon sources);
        gradients flow through it to ``n_photons`` and, via the trilinear
        voxel interpolation, to ``pos``.

        Optional noise:
        - SER jitter (``ser_jitter_std > 0``): multiplies weights by
          ``N(1, σ)`` per photon — the random factor is a constant in the
          autograd graph, so gradient through ``weights`` is preserved.
        - Baseline noise (``baseline_noise_std > 0`` and
          ``add_baseline_noise=True``): adds ``N(0, σ)`` per ADC bin —
          additive constant, preserves all gradients.

        Digitization is rejected at construction time.
        """
        cfg = self.config
        device = self._device
        fine_tick = self._fine_tick
        fine_kernel_tensor = self._fine_kernel()
        if n_channels is None:
            n_channels = cfg.n_channels

        if cfg.ser_jitter_std > 0 and weights.numel() > 0:
            jitter = torch.normal(
                1.0, cfg.ser_jitter_std, weights.shape, device=device
            )
            weights = weights * jitter

        wvfm_cls = SlicedWaveform if stitched else Waveform
        extra_args: dict = {"weights": weights}
        if stitched:
            extra_args["kernel_extent_ns"] = fine_kernel_tensor.shape[0] * fine_tick
        if cfg.oversample > 1:
            extra_args["t0_snap_ns"] = cfg.tick_ns

        if times.numel() > 0:
            pe_counts = torch.zeros(n_channels, device=device, dtype=torch.float32)
            pe_counts = pe_counts.scatter_add(0, channels.long(), weights)
        else:
            pe_counts = torch.zeros(n_channels, device=device, dtype=torch.float32)

        wf = wvfm_cls.from_photons(times, channels, fine_tick, n_channels, **extra_args)
        wf.attrs["pe_counts"] = pe_counts

        wf = wf.convolve(fine_kernel_tensor, cfg.gain)
        if cfg.oversample > 1:
            wf = wf.downsample(cfg.oversample)

        if add_baseline_noise and cfg.baseline_noise_std > 0:
            wf.adc = wf.adc + torch.randn_like(wf.adc) * cfg.baseline_noise_std

        if cfg.digitization is not None:
            # STE: forward applies round+clamp; backward passes gradient
            # through as identity.  Biased gradient (true gradient is zero
            # almost everywhere), but useful for training.
            wf.adc = digitize_ste(
                wf.adc, cfg.digitization.pedestal, cfg.digitization.n_bits,
            )
            wf.attrs["pedestal"] = cfg.digitization.pedestal
        return wf


    def _simulate_streaming(
        self,
        pos: torch.Tensor,
        n_photons: torch.Tensor,
        t_step: torch.Tensor,
        add_baseline_noise: bool = True,
    ) -> SlicedWaveform:
        """Sparse streaming variant: group segments by time proximity, build a
        small dense histogram per group via ``histogram_pdf`` (checkpoint-ed),
        convolve each with a small FFT, and assemble into a ``SlicedWaveform``.

        Peak memory is independent of both ``N`` (segment count, via
        per-chunk checkpointing inside ``histogram_pdf``) **and** total event
        time span (via time-grouping + per-group FFTs instead of one
        monolithic FFT). For a full LArTPC event (N ≈ 100 k, 3 ms span):
        ~5 GB vs ~23 GB for the dense V1 streaming path.
        """
        cfg = self.config
        device = self._device
        fine_tick = self._fine_tick
        fine_kernel_tensor = self._fine_kernel()
        sampler = cfg.tof_sampler
        n_ch = cfg.n_channels

        if cfg.ser_jitter_std > 0:
            import warnings
            warnings.warn(
                "ser_jitter_std is ignored in streaming mode (no per-photon weights); "
                "use streaming=False if you need SER jitter."
            )

        # Most PCA samplers expose ``t_max_ns`` as a property; a generic mock
        # sampler may not. Default to 600 ns (the shipped basis's window).
        t_window = getattr(sampler, "t_max_ns", 600.0)
        kernel_extent_ns = float(fine_kernel_tensor.shape[0]) * fine_tick
        gap_threshold = kernel_extent_ns + t_window + _GAP_THRESHOLD_PAD_NS

        # ---- 1. Sort segments by t_step and split at natural time gaps ----
        if t_step.numel() == 0:
            return SlicedWaveform(
                adc=torch.zeros(0, device=device),
                offsets=torch.tensor([0], device=device, dtype=torch.long),
                t0_ns=torch.zeros(0, device=device),
                pmt_id=torch.zeros(0, device=device, dtype=torch.long),
                tick_ns=fine_tick, n_channels=n_ch,
                attrs={"pe_counts": torch.zeros(n_ch, device=device)},
            )

        order = t_step.detach().argsort()
        t_sorted = t_step[order].detach()
        gaps = torch.diff(t_sorted)
        split_points = (gaps > gap_threshold).nonzero(as_tuple=True)[0] + 1
        groups = torch.tensor_split(order, split_points.cpu())

        # Pre-compute every group's t0_g and n_bins_g in two batched ops, then
        # one combined .tolist() — collapses 2 host-syncs per group (.min().item()
        # and .max().item()) into 2 syncs total.
        group_min = torch.stack([t_step[g].detach().min() for g in groups])
        group_max = torch.stack([t_step[g].detach().max() for g in groups])
        t0_starts = (group_min / fine_tick).floor() * fine_tick
        n_bins_t = (
            ((group_max - t0_starts + t_window) / fine_tick).floor().long()
            + _N_BINS_GROUP_PAD
        )
        t0_starts_cpu = t0_starts.tolist()
        n_bins_cpu = n_bins_t.tolist()

        # ---- 2. Per-group: histogram_pdf → small Waveform → convolve -----
        all_adc = []
        all_offsets = [0]
        all_t0 = []
        all_pmt = []
        total_pe = torch.zeros(n_ch, device=device, dtype=torch.float32)

        for gi, g_idx in enumerate(groups):
            t_g = t_step[g_idx]
            t0_g = float(t0_starts_cpu[gi])
            n_bins_g = int(n_bins_cpu[gi])

            hist_g = sampler.histogram_pdf(
                pos[g_idx], n_photons[g_idx], t_g,
                tick_ns=fine_tick, n_bins=n_bins_g, t0_ref=t0_g,
                chunk_size=cfg.stream_chunk_size,
                use_checkpoint=cfg.stream_checkpoint,
            )
            total_pe = total_pe + hist_g.sum(dim=1)

            # Inject aux photon sources for this group's time window.
            if cfg.aux_photon_sources:
                t_start_g = t0_g
                t_end_g = t0_g + n_bins_g * fine_tick
                for source in cfg.aux_photon_sources:
                    aux_t, aux_ch = source.sample(n_ch, t_start_g, t_end_g, device)
                    if aux_t.numel() == 0:
                        continue
                    aux_bin = ((aux_t - t0_g) / fine_tick).long().clamp(0, n_bins_g - 1)
                    aux_flat = aux_ch.long() * n_bins_g + aux_bin
                    hist_g = hist_g.reshape(-1).scatter_add(
                        0, aux_flat, torch.ones_like(aux_t, dtype=hist_g.dtype)
                    ).reshape(n_ch, n_bins_g)

            wf_g = Waveform(
                adc=hist_g, t0=t0_g, tick_ns=fine_tick, n_channels=n_ch,
            )
            wf_g = wf_g.convolve(fine_kernel_tensor, cfg.gain)
            if cfg.oversample > 1:
                wf_g = wf_g.downsample(cfg.oversample)

            # Extract per-PMT chunks, skipping zero channels — vectorized.
            active_mask = wf_g.adc.detach().abs().amax(dim=1) > 1e-12
            active_chs = active_mask.nonzero(as_tuple=True)[0]
            n_active = active_chs.numel()
            if n_active > 0:
                active_adc = wf_g.adc[active_chs]
                chunk_len = active_adc.shape[1]
                all_adc.append(active_adc.reshape(-1))
                base = all_offsets[-1]
                all_offsets.extend(base + (i + 1) * chunk_len for i in range(n_active))
                all_t0.extend([wf_g.t0] * n_active)
                all_pmt.extend(active_chs.tolist())

        # ---- 3. Assemble SlicedWaveform -----------------------------------
        tick_out = fine_tick * cfg.oversample if cfg.oversample > 1 else fine_tick
        sw = SlicedWaveform(
            adc=torch.cat(all_adc) if all_adc else torch.zeros(0, device=device),
            offsets=torch.tensor(all_offsets, device=device, dtype=torch.long),
            t0_ns=torch.tensor(all_t0, device=device, dtype=torch.float32),
            pmt_id=torch.tensor(all_pmt, device=device, dtype=torch.long),
            tick_ns=tick_out,
            n_channels=n_ch,
            attrs={"pe_counts": total_pe},
        )

        if add_baseline_noise and cfg.baseline_noise_std > 0:
            sw.adc = sw.adc + torch.randn_like(sw.adc) * cfg.baseline_noise_std

        if cfg.digitization is not None:
            sw.adc = digitize_ste(
                sw.adc, cfg.digitization.pedestal, cfg.digitization.n_bits,
            )
            sw.attrs["pedestal"] = cfg.digitization.pedestal

        return sw

