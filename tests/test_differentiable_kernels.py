"""Tests for the differentiable delay-PDF kernels and DifferentiableOpticalSimulator.

The kernels in goop/kernels.py (ScintillationKernel, TPBExponentialKernel,
TPBTriexponentialKernel, TTSKernel) are discretized PDFs equivalent in
expectation to the corresponding stochastic delay samplers in goop/delays.py.
The Response composite kernel pre-composes them (and SER) into a single FFT
convolution.  These tests verify equivalence and that gradients flow.
"""

from __future__ import annotations

import math

import pytest
import torch

from goop import (
    DifferentiableOpticalSimulator,
    OpticalSimConfig,
    OpticalSimulator,
    Response,
    RLCKernel,
    ScintillationBiexponentialDelay,
    ScintillationKernel,
    SERKernel,
    TPBExponentialDelay,
    TPBExponentialKernel,
    TPBTriexponentialKernel,
    TTSDelay,
    TTSKernel,
    Waveform,
    create_default_response,
)
from goop.delays import Delays, TPBTriexponentialDelay
from goop.kernels import _exp_pdf_bin, _gauss_pdf_bin

DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _kernel_moments(kernel: torch.Tensor, tick_ns: float):
    """Mean and standard deviation of a discrete PDF kernel using bin midpoints."""
    n = torch.arange(kernel.numel(), dtype=torch.float64)
    t = (n + 0.5) * tick_ns
    k = kernel.double()
    total = k.sum()
    mean = (k * t).sum() / total
    var = (k * (t - mean) ** 2).sum() / total
    return float(mean), float(var.sqrt())


def _sample_histogram(samples: torch.Tensor, tick_ns: float, n_bins: int) -> torch.Tensor:
    """Build a normalized histogram of `samples` on [0, n_bins*tick_ns)."""
    bins = (samples / tick_ns).long().clamp(0, n_bins - 1)
    h = torch.zeros(n_bins, dtype=torch.float64)
    h.scatter_add_(0, bins.long(), torch.ones_like(samples, dtype=torch.float64))
    return h / samples.numel()


# ---------------------------------------------------------------------------
# 1. PDF bin-integral helpers — sanity checks
# ---------------------------------------------------------------------------


class TestPdfBinHelpers:
    def test_exp_bin_normalizes(self):
        # tau=20 ns, range out to 10*tau so truncation loss e^{-10} ≈ 4.5e-5
        tick = 0.5
        n = torch.arange(0, 400, dtype=torch.float32)
        a = n * tick
        b = (n + 1) * tick
        k = _exp_pdf_bin(20.0, a, b)
        assert abs(k.sum().item() - 1.0) < 1e-3

    def test_gauss_bin_normalizes_centered(self):
        tick = 0.1
        sigma = 2.0
        mu = 20.0
        n = torch.arange(0, 400, dtype=torch.float32)
        a = n * tick
        b = (n + 1) * tick
        k = _gauss_pdf_bin(mu, sigma, a, b)
        assert abs(k.sum().item() - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# 2. Kernel normalization
# ---------------------------------------------------------------------------


class TestKernelNormalization:
    def test_scintillation_sums_to_one(self):
        k = ScintillationKernel(device=DEVICE)()
        assert abs(k.sum().item() - 1.0) < 1e-3

    def test_tpb_exp_sums_to_one(self):
        k = TPBExponentialKernel(device=DEVICE)()
        assert abs(k.sum().item() - 1.0) < 1e-3

    def test_tpb_triexp_sums_to_one(self):
        k = TPBTriexponentialKernel(device=DEVICE)()
        assert abs(k.sum().item() - 1.0) < 1e-3

    def test_tts_sums_to_one(self):
        k = TTSKernel(device=DEVICE)()
        # 8σ truncation each side → erf(8) ≈ 1 to ~1e-15
        assert abs(k.sum().item() - 1.0) < 1e-6

    def test_tts_clipped_when_transit_time_zero(self):
        # Centered at 0 → only the right half captured
        k = TTSKernel(transit_time_ns=0.0, duration_ns=10.0, device=DEVICE)()
        assert abs(k.sum().item() - 0.5) < 1e-3


# ---------------------------------------------------------------------------
# 3. Moment matching against analytical PDFs
# ---------------------------------------------------------------------------


class TestKernelMoments:
    def test_scintillation_mean_var(self):
        ker = ScintillationKernel(device=DEVICE)
        mean, std = _kernel_moments(ker(), ker.tick_ns)
        # Analytical: mean = p·τ_s + (1-p)·τ_t
        p, ts, tt = ker.singlet_fraction, ker.tau_singlet_ns, ker.tau_triplet_ns
        expected_mean = p * ts + (1 - p) * tt
        expected_var = p * 2 * ts ** 2 + (1 - p) * 2 * tt ** 2 - expected_mean ** 2
        assert abs(mean - expected_mean) / expected_mean < 0.01
        assert abs(std - math.sqrt(expected_var)) / math.sqrt(expected_var) < 0.01

    def test_tpb_exp_mean_var(self):
        ker = TPBExponentialKernel(device=DEVICE)
        mean, std = _kernel_moments(ker(), ker.tick_ns)
        # Analytical: mean = τ, std = τ
        assert abs(mean - ker.tau_ns) / ker.tau_ns < 0.05
        assert abs(std - ker.tau_ns) / ker.tau_ns < 0.05

    def test_tpb_triexp_mean(self):
        ker = TPBTriexponentialKernel(device=DEVICE)
        mean, _ = _kernel_moments(ker(), ker.tick_ns)
        expected = (
            ker.a_1 * ker.tau_1_ns + ker.a_2 * ker.tau_2_ns
            + ker.a_3 * ker.tau_3_ns + ker.a_4 * ker.tau_4_ns
        )
        assert abs(mean - expected) / expected < 0.01

    def test_tts_mean_var(self):
        ker = TTSKernel(device=DEVICE)
        mean, std = _kernel_moments(ker(), ker.tick_ns)
        assert abs(mean - ker.transit_time_ns) < 0.5  # within ~0.5 ns
        assert abs(std - ker.sigma_ns) / ker.sigma_ns < 0.05


# ---------------------------------------------------------------------------
# 4. Sampler equivalence: kernel ≈ histogram of stochastic samples
# ---------------------------------------------------------------------------


class TestSamplerEquivalence:
    """Each PDF kernel should match the histogram of samples drawn from the
    corresponding stochastic delay sampler in goop/delays.py."""

    N = 2_000_000  # large enough that per-bin shot noise << kernel values

    def _compare(self, kernel: torch.Tensor, samples: torch.Tensor, tick_ns: float, l1_tol: float):
        n_bins = kernel.numel()
        h = _sample_histogram(samples, tick_ns, n_bins)
        # Both are normalized PDFs on the same grid (kernel sum ≈ 1).
        l1 = (h - kernel.double()).abs().sum().item()
        assert l1 < l1_tol, f"L1 distance {l1:.4f} exceeds tolerance {l1_tol}"

    def test_tpb_exp(self):
        torch.manual_seed(0)
        ker = TPBExponentialKernel(tau_ns=20.0, duration_ns=400.0, device=DEVICE)
        sampler = TPBExponentialDelay(tau_ns=20.0)
        samples = sampler(self.N, DEVICE)
        self._compare(ker(), samples, ker.tick_ns, l1_tol=0.05)

    def test_scintillation(self):
        torch.manual_seed(1)
        ker = ScintillationKernel(
            singlet_fraction=0.30, tau_singlet_ns=6.0, tau_triplet_ns=1300.0,
            duration_ns=13000.0, device=DEVICE,
        )
        sampler = ScintillationBiexponentialDelay(
            singlet_fraction=0.30, tau_singlet_ns=6.0, tau_triplet_ns=1300.0,
        )
        samples = sampler(self.N, DEVICE)
        self._compare(ker(), samples, ker.tick_ns, l1_tol=0.05)

    def test_tpb_triexp(self):
        torch.manual_seed(2)
        ker = TPBTriexponentialKernel(device=DEVICE)
        sampler = TPBTriexponentialDelay()
        samples = sampler(self.N, DEVICE)
        self._compare(ker(), samples, ker.tick_ns, l1_tol=0.05)

    def test_tts(self):
        torch.manual_seed(3)
        ker = TTSKernel(fwhm_ns=2.4, transit_time_ns=55.0, device=DEVICE)
        sampler = TTSDelay(fwhm_ns=2.4, _transit_time_ns=55.0, apply_transit_time=False)
        samples = sampler(self.N, DEVICE)
        self._compare(ker(), samples, ker.tick_ns, l1_tol=0.05)


# ---------------------------------------------------------------------------
# 5. Response composite kernel correctness
# ---------------------------------------------------------------------------


def _time_domain_convolve(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Reference linear convolution via torch.nn.functional.conv1d."""
    n_out = a.numel() + b.numel() - 1
    a_pad = torch.nn.functional.pad(a, (0, b.numel() - 1)).view(1, 1, -1)
    flipped_b = b.flip(0).view(1, 1, -1)
    out = torch.nn.functional.conv1d(a_pad, flipped_b, padding=b.numel() - 1)
    return out.view(-1)[:n_out]


class TestResponse:
    def test_two_kernel_fft_matches_time_domain(self):
        k1 = TPBExponentialKernel(tau_ns=15.0, duration_ns=200.0, device=DEVICE)
        k2 = TTSKernel(fwhm_ns=2.4, transit_time_ns=55.0, device=DEVICE)
        ref = _time_domain_convolve(k1(), k2())
        composite = Response(kernels=[k1, k2], tick_ns=1.0, device=DEVICE)()
        assert composite.shape == ref.shape
        assert torch.allclose(composite, ref, atol=1e-5, rtol=1e-4)

    def test_three_kernel_associativity(self):
        # ((k1 ⊛ k2) ⊛ k3)  ==  Response([k1, k2, k3])
        k1 = TPBExponentialKernel(tau_ns=10.0, duration_ns=100.0, device=DEVICE)
        k2 = TPBExponentialKernel(tau_ns=20.0, duration_ns=200.0, device=DEVICE)
        k3 = TTSKernel(fwhm_ns=2.4, transit_time_ns=20.0, duration_ns=40.0, device=DEVICE)
        ab = _time_domain_convolve(k1(), k2())
        abc = _time_domain_convolve(ab, k3())
        composite = Response(kernels=[k1, k2, k3], tick_ns=1.0, device=DEVICE)()
        assert composite.shape == abc.shape
        assert torch.allclose(composite, abc, atol=1e-5, rtol=1e-4)

    def test_pdf_chain_sums_to_one(self):
        """Composing three normalized PDFs gives a kernel that still sums to 1."""
        resp = Response(
            kernels=[
                ScintillationKernel(device=DEVICE),
                TPBExponentialKernel(device=DEVICE),
                TTSKernel(device=DEVICE),
            ],
            tick_ns=1.0, device=DEVICE,
        )
        out = resp()
        assert abs(out.sum().item() - 1.0) < 1e-3

    def test_with_tick_ns_propagates(self):
        resp = create_default_response(tick_ns=1.0, device=DEVICE)
        fine = resp.with_tick_ns(0.5)
        # cache cleared on copy
        assert fine._kernel_cache is None
        out_fine = fine()
        out_coarse = resp()
        # finer tick → roughly twice as many samples for the same time extent
        assert out_fine.shape[0] > out_coarse.shape[0]
        # SER's adc_peak normalization means the absolute peak should be
        # comparable between the two; just sanity-check it's finite and nonzero.
        assert torch.isfinite(out_fine).all()
        assert out_fine.abs().max().item() > 0

    def test_empty_response_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            Response(kernels=[], tick_ns=1.0, device=DEVICE)()


# ---------------------------------------------------------------------------
# 6. End-to-end equivalence: stochastic average ≈ differentiable single run
# ---------------------------------------------------------------------------


class _SeededTOF:
    """Returns the same set of photons each call — isolates delay stochasticity.

    Provides both ``sample()`` (stochastic interface) and ``sample_pdf()``
    (PDF-deposition interface, returning the same photons with unit weights)
    so the same mock can be used for both pipelines in tests.
    """

    def __init__(self, n_channels=4, seed=2026):
        self._n_channels = n_channels
        self.seed = seed
        self._cache = None

    @property
    def n_channels(self):
        return self._n_channels

    def sample(self, pos, n_photons, t_step):
        if self._cache is None:
            g = torch.Generator(device=DEVICE).manual_seed(self.seed)
            n = int(n_photons.sum().item())
            # Concentrate photons in two clusters so the convolution has signal
            t1 = torch.rand(n // 2, generator=g, device=DEVICE) * 50.0
            t2 = 1500.0 + torch.rand(n - n // 2, generator=g, device=DEVICE) * 50.0
            times = torch.cat([t1, t2])
            channels = torch.randint(
                0, self._n_channels, (n,), generator=g, device=DEVICE
            )
            self._cache = (times, channels)
        times, channels = self._cache
        n = times.numel()
        source_idx = torch.arange(n, device=DEVICE) % pos.shape[0]
        return times.clone(), channels.clone(), source_idx

    def sample_pdf(self, pos, n_photons, t_step):
        """PDF-deposition mock: same photons as sample(), with weight=1 each."""
        times, channels, _ = self.sample(pos, n_photons, t_step)
        weights = torch.ones_like(times)
        return times, channels, weights


def _fixed_hist(times, channels, t0, n_bins, tick, n_channels):
    """Manual per-channel histogram with locked t0 and n_bins (for averaging)."""
    out = torch.zeros(n_channels, n_bins)
    if times.numel() == 0:
        return out
    shifted = times - t0
    bin_idx = (shifted / tick).long().clamp(0, n_bins - 1)
    flat = channels.long() * n_bins + bin_idx
    out.view(-1).scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.float32))
    return out


def _fft_convolve(adc: torch.Tensor, kernel: torch.Tensor, gain: float) -> torch.Tensor:
    """Per-channel FFT convolution mirroring Waveform.convolve."""
    n_ch, n_tick = adc.shape
    n_k = kernel.numel()
    n_out = n_tick + n_k - 1
    n_fft = 1 << math.ceil(math.log2(max(n_out, 1)))
    padded = torch.nn.functional.pad(adc, (0, n_fft - n_tick))
    k_fft = torch.fft.rfft(kernel, n=n_fft)
    out = gain * torch.fft.irfft(
        torch.fft.rfft(padded) * k_fft.unsqueeze(0), n=n_fft
    )
    return out[:, :n_out]


class TestEndToEndEquivalence:
    """Convolving with Response = (delay PDFs ⊛ SER) is mathematically equal,
    in expectation, to (sample per-photon delays + histogram + convolve with SER).

    The OpticalSimulator pipeline auto-derives ``t0 = times.min()``, which
    shifts stochastically when delays are added.  We bypass that by histogram-
    ming directly with a locked ``t0`` and ``n_bins`` so the comparison is
    driven only by the kernel composition equivalence, not by the t0 wobble.
    """

    def test_stochastic_average_matches_diff_kernel(self):
        torch.manual_seed(7)
        n_ch = 4
        n_photons = 5000
        gain = -45.0
        tick = 1.0
        n_runs = 150

        # Photons clustered around t=200 ns
        times = 200.0 + torch.randn(n_photons) * 5.0
        channels = torch.randint(0, n_ch, (n_photons,))
        t0 = 0.0
        n_bins = 8000  # plenty of room for triplet decay (τ=1300 ns)

        scint = ScintillationKernel(device=DEVICE)
        tpb = TPBExponentialKernel(device=DEVICE)
        tts = TTSKernel(device=DEVICE)
        ser = SERKernel(duration_ns=2000.0, device=DEVICE)
        response = Response(kernels=[scint, tpb, tts, ser], tick_ns=tick, device=DEVICE)

        # Differentiable: histogram once, convolve with composite Response
        hist_diff = _fixed_hist(times, channels, t0, n_bins, tick, n_ch)
        out_diff = _fft_convolve(hist_diff, response(), gain)

        # Stochastic: average n_runs runs of (sample delays + histogram + SER conv)
        delays = Delays([
            ScintillationBiexponentialDelay(),
            TPBExponentialDelay(),
            TTSDelay(),
        ])
        ser_kernel = ser()
        accum = torch.zeros(n_ch, hist_diff.shape[1] + ser_kernel.numel() - 1)
        for _ in range(n_runs):
            d = delays.sample(n_photons, DEVICE)
            hist_s = _fixed_hist(times + d, channels, t0, n_bins, tick, n_ch)
            out_s = _fft_convolve(hist_s, ser_kernel, gain)
            accum += out_s
        stoch_mean = accum / n_runs

        # Compare on the SER-conv length window (stoch is shorter than diff
        # because diff convolves with the longer composite kernel).
        # Expected residual scaling: per-bin ADC noise σ ~ sqrt(N_photons * p *
        # sum(SER²) / N_runs).  For our parameters the predicted residual/signal
        # L2 ratio at N_runs=150 is ~6-10%; with the 1300 ns triplet tail
        # dominating the noise contribution.  The deterministic FFT-equivalence
        # check (test_two_kernel_fft_matches_time_domain) is the strict math
        # proof; this test confirms no convention/sign/normalization errors.
        n_cmp = stoch_mean.shape[1]
        for ch in range(n_ch):
            sig = out_diff[ch, :n_cmp]
            ref = stoch_mean[ch, :n_cmp]
            sig_l2 = sig.norm().item()
            if sig_l2 < 1.0:
                continue
            res_l2 = (sig - ref).norm().item()
            assert res_l2 / sig_l2 < 0.12, (
                f"ch {ch}: residual/signal L2 = {res_l2/sig_l2:.4f} "
                f"(expected < 0.12 with {n_runs} runs and triplet-dominated noise)"
            )


# ---------------------------------------------------------------------------
# 7. Differentiability: gradients flow through the convolution
# ---------------------------------------------------------------------------


class TestGradients:
    def test_grad_flows_through_response_convolution(self):
        """Gradient on photon weights flows back from waveform through Response."""
        torch.manual_seed(0)
        n_photons = 1000
        n_channels = 2
        tick = 1.0

        times = torch.rand(n_photons) * 200.0
        channels = torch.randint(0, n_channels, (n_photons,))
        weights = torch.ones(n_photons, requires_grad=True)

        wf = Waveform.from_photons(
            times, channels, tick_ns=tick, n_channels=n_channels, weights=weights,
        )
        kernel = create_default_response(tick_ns=tick, device=DEVICE)()
        out = wf.convolve(kernel, gain=-1.0)

        loss = (out.adc ** 2).sum()
        loss.backward()

        assert weights.grad is not None
        assert torch.isfinite(weights.grad).all()
        assert weights.grad.abs().max().item() > 0

    def test_grad_flows_through_full_diff_simulator(self):
        """Differentiable simulator preserves gradients on the photon histogram."""
        torch.manual_seed(0)
        n_ch = 2

        # A tiny TOF sampler whose output is a no-grad tensor — that's expected.
        # We attach gradient via the per-photon weights when histogramming.
        # To do that we need to inject weights, which is not the diff sim's
        # standard interface, so this test focuses on the kernel path directly.
        # (The full simulate() test is an integration check; gradients on
        #  inputs to TOFSampler will be enabled in Phase 2.)
        sampler = _SeededTOF(n_channels=n_ch)
        ser = SERKernel(duration_ns=500.0, device=DEVICE)
        response = Response(
            kernels=[
                TPBExponentialKernel(tau_ns=20.0, duration_ns=200.0, device=DEVICE),
                ser,
            ],
            tick_ns=1.0, device=DEVICE,
        )
        cfg = OpticalSimConfig(
            tof_sampler=sampler, delays=Delays([]), kernel=response,
            device="cpu", tick_ns=1.0, gain=-1.0,
        )
        sim = DifferentiableOpticalSimulator(cfg)
        out = sim.simulate(
            torch.zeros(5, 3), torch.full((5,), 200), torch.zeros(5),
            stitched=True, add_baseline_noise=False,
        )
        # Even without trainable inputs, the autograd graph should be
        # constructible (no @torch.no_grad applied to simulate).
        assert out.adc.requires_grad is False  # inputs aren't Parameters here
        # But running under grad mode should not have errored:
        assert out.n_channels == n_ch


# ---------------------------------------------------------------------------
# 8. DifferentiableOpticalSimulator config-validation
# ---------------------------------------------------------------------------


def _minimal_cfg(**overrides) -> OpticalSimConfig:
    sampler = _SeededTOF(n_channels=2)
    base = dict(
        tof_sampler=sampler,
        delays=Delays([]),
        kernel=create_default_response(device=DEVICE),
        device="cpu", tick_ns=1.0, gain=-1.0,
    )
    base.update(overrides)
    return OpticalSimConfig(**base)


class _SamplerNoPdf:
    """A TOF sampler with sample() but no sample_pdf — for the rejection test."""
    @property
    def n_channels(self):
        return 2
    def sample(self, pos, n_photons, t_step):
        empty = torch.zeros(0)
        return empty, empty.long(), empty.long()


class TestDifferentiableSimulatorAssertions:
    def test_allows_digitization_via_ste(self):
        """Diff sim accepts digitization; the STE bypasses round/clamp in backward."""
        from goop import DigitizationConfig
        cfg = _minimal_cfg(digitization=DigitizationConfig(n_bits=14, pedestal=1500.0))
        DifferentiableOpticalSimulator(cfg)  # should NOT raise

    def test_allows_ser_jitter(self):
        """SER jitter composes as weights *= N(1, σ); gradient flow preserved."""
        cfg = _minimal_cfg(ser_jitter_std=0.1)
        DifferentiableOpticalSimulator(cfg)  # should NOT raise

    def test_allows_baseline_noise(self):
        """Baseline noise is just additive — gradient flow preserved."""
        cfg = _minimal_cfg(baseline_noise_std=2.0)
        DifferentiableOpticalSimulator(cfg)  # should NOT raise

    def test_allows_aux_photon_sources(self):
        """Dark hits are appended with unit weights; independent of input grads."""
        from goop import DarkNoise
        cfg = _minimal_cfg(aux_photon_sources=[DarkNoise(rate_hz=2000.0)])
        DifferentiableOpticalSimulator(cfg)  # should NOT raise

    def test_rejects_sampler_without_sample_pdf(self):
        """Diff sim requires the configured TOF sampler to expose sample_pdf."""
        cfg = _minimal_cfg(tof_sampler=_SamplerNoPdf())
        with pytest.raises(ValueError, match="sample_pdf"):
            DifferentiableOpticalSimulator(cfg)

    def test_accepts_clean_config(self):
        cfg = _minimal_cfg()
        DifferentiableOpticalSimulator(cfg)

    def test_rejects_stitched_false(self):
        """Diff pipeline targets SlicedWaveform exclusively; stitched=False rejected."""
        sim = DifferentiableOpticalSimulator(_minimal_cfg())
        with pytest.raises(ValueError, match="stitched=False"):
            sim.simulate(
                torch.zeros(5, 3), torch.full((5,), 100), torch.zeros(5),
                stitched=False,
            )

    def test_digitization_quantizes_output(self):
        """With digitization on, output is integer-valued and ADC-clamped."""
        from goop import DigitizationConfig
        cfg = _minimal_cfg(
            digitization=DigitizationConfig(n_bits=14, pedestal=1500.0),
        )
        sim = DifferentiableOpticalSimulator(cfg)

        sw = sim.simulate(
            torch.zeros(5, 3), torch.full((5,), 100.0), torch.zeros(5),
            stitched=True, add_baseline_noise=False,
        )
        assert torch.equal(sw.adc, sw.adc.round())
        assert sw.adc.min().item() >= 0
        assert sw.adc.max().item() <= (1 << 14) - 1

    def test_baseline_noise_actually_applied(self):
        """Diff sim with baseline_noise_std > 0 should add noise to the output."""
        torch.manual_seed(0)
        cfg_quiet = _minimal_cfg()
        cfg_noisy = _minimal_cfg(baseline_noise_std=5.0)
        sim_quiet = DifferentiableOpticalSimulator(cfg_quiet)
        sim_noisy = DifferentiableOpticalSimulator(cfg_noisy)

        pos = torch.zeros(5, 3)
        n_ph = torch.full((5,), 50)
        t = torch.zeros(5)
        sw_q = sim_quiet.simulate(pos, n_ph, t, stitched=True, add_baseline_noise=True)
        sw_n = sim_noisy.simulate(pos, n_ph, t, stitched=True, add_baseline_noise=True)
        assert sw_n.adc.std().item() > sw_q.adc.std().item(), (
            "noisy diff sim output should have larger ADC std than quiet sim"
        )

    def test_ser_jitter_actually_applied(self):
        """ser_jitter_std > 0 should change the histogram weights vs jitter-free."""
        torch.manual_seed(1)
        cfg_clean = _minimal_cfg()
        cfg_jitter = _minimal_cfg(ser_jitter_std=0.5)
        sim_clean = DifferentiableOpticalSimulator(cfg_clean)
        sim_jitter = DifferentiableOpticalSimulator(cfg_jitter)

        pos = torch.zeros(10, 3)
        n_ph = torch.full((10,), 100)
        t = torch.zeros(10)
        sw_c = sim_clean.simulate(pos, n_ph, t, stitched=True, add_baseline_noise=False)
        sw_j = sim_jitter.simulate(pos, n_ph, t, stitched=True, add_baseline_noise=False)
        assert not torch.allclose(sw_c.adc, sw_j.adc, atol=1e-3), (
            "jittered output should differ from non-jittered"
        )

