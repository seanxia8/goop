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
        _assert_differentiable(config)
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


def _assert_differentiable(cfg: OpticalSimConfig) -> None:
    """Reject configs incompatible with gradient flow.

    The only hard requirement is a TOF sampler with ``sample_pdf``.  Noise
    knobs preserve gradient flow because their random draws are constants
    in the autograd graph; ``digitization`` is handled by an STE wrapper
    in ``_simulate``.
    """
    if not hasattr(cfg.tof_sampler, "sample_pdf"):
        raise ValueError(
            "DifferentiableOpticalSimulator requires a TOF sampler that exposes "
            "`sample_pdf(...)` (e.g. `DifferentiableTOFSampler`).  Got "
            f"{type(cfg.tof_sampler).__name__}."
        )
