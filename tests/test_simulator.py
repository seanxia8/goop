"""Unit tests for the optical TPC simulation pipeline."""

import pytest
import torch

from goop import (
    DarkNoise,
    Delays,
    DigitizationConfig,
    OpticalSimConfig,
    OpticalSimulator,
    RLCKernel,
    ScintillationBiexponentialDelay,
    SlicedWaveform,
    TPBExponentialDelay,
    TTSDelay,
    Waveform,
    create_default_delays,
)
from goop.kernels import _rlc_antiderivative
from goop.digitize import digitize

DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Mock TOF sampler
# ---------------------------------------------------------------------------

class MockTOFSampler:
    """Deterministic sampler for unit tests."""

    def __init__(self, n_channels: int = 4, seed: int = 42):
        self._n_channels = n_channels
        self.seed = seed

    @property
    def n_channels(self):
        return self._n_channels

    def sample(self, pos, n_photons, t_step):
        g = torch.Generator(device=DEVICE).manual_seed(self.seed)
        n = int(n_photons.sum().item()) if not isinstance(n_photons, int) else n_photons
        n_per_cluster = n // 2
        t1 = torch.rand(n_per_cluster, generator=g, device=DEVICE) * 200.0
        t2 = 2000.0 + torch.rand(n - n_per_cluster, generator=g, device=DEVICE) * 200.0
        times = torch.cat([t1, t2])
        channels = torch.randint(0, self._n_channels, (times.shape[0],), generator=g, device=DEVICE)
        times = times + t_step.mean()
        source_idx = torch.arange(n, device=DEVICE) % pos.shape[0]
        return times, channels, source_idx


class _SeededMockTOF:
    """Returns same photons every call (for stitched vs full comparison)."""

    def __init__(self, n_channels=4, seed=123):
        self._n_channels = n_channels
        self.seed = seed
        self._data = None

    def _generate(self, n_photons):
        g = torch.Generator(device=DEVICE).manual_seed(self.seed)
        n = int(n_photons.sum().item())
        n_half = n // 2
        t1 = torch.rand(n_half, generator=g, device=DEVICE) * 200.0
        t2 = 2000.0 + torch.rand(n - n_half, generator=g, device=DEVICE) * 200.0
        times = torch.cat([t1, t2])
        channels = torch.randint(0, self._n_channels, (n,), generator=g, device=DEVICE)
        return times, channels

    @property
    def n_channels(self):
        return self._n_channels

    def sample(self, pos, n_photons, t_step):
        if self._data is None:
            self._data = self._generate(n_photons)
        n = self._data[0].shape[0]
        source_idx = torch.arange(n, device=DEVICE) % pos.shape[0]
        return self._data[0].clone(), self._data[1].clone(), source_idx


# ---------------------------------------------------------------------------
# Delay sampler tests
# ---------------------------------------------------------------------------

class TestScintillationDelay:
    def test_shape(self):
        assert ScintillationBiexponentialDelay()(10_000, DEVICE).shape == (10_000,)

    def test_nonnegative(self):
        assert (ScintillationBiexponentialDelay()(50_000, DEVICE) >= 0).all()

    def test_singlet_fraction(self):
        """Fraction of singlet-like (fast) delays should match singlet_fraction."""
        torch.manual_seed(0)
        d = ScintillationBiexponentialDelay(singlet_fraction=0.30, tau_singlet_ns=1.0, tau_triplet_ns=1530.0)
        samples = d(500_000, DEVICE)
        singlet_frac = (samples < 50.0).float().mean().item()
        assert abs(singlet_frac - 0.30) < 0.03, f"singlet fraction {singlet_frac:.3f}, expected ~0.30"


class TestTPBDelay:
    def test_shape_and_nonneg(self):
        out = TPBExponentialDelay(tau_ns=20.0)(10_000, DEVICE)
        assert out.shape == (10_000,)
        assert (out >= 0).all()

    def test_mean(self):
        torch.manual_seed(0)
        assert abs(TPBExponentialDelay(tau_ns=20.0)(200_000, DEVICE).mean().item() - 20.0) < 1.0


class TestTTSDelay:
    def test_shape(self):
        assert TTSDelay(fwhm_ns=1.0)(10_000, DEVICE).shape == (10_000,)

    def test_std(self):
        torch.manual_seed(0)
        assert abs(TTSDelay(fwhm_ns=1.0)(200_000, DEVICE).std().item() - 1.0 / 2.35482) < 0.02


# ---------------------------------------------------------------------------
# Kernel tests
# ---------------------------------------------------------------------------

class TestRLCKernel:
    def test_shape(self):
        assert RLCKernel(duration_ns=9000.0, device=DEVICE)().shape == (9000,)

    def test_first_tick_is_bin_average_not_point_sample(self):
        ker = RLCKernel(device=DEVICE)
        k = ker()
        edges = torch.tensor([0.0, ker.tick_ns], device=DEVICE, dtype=torch.float32)
        F = _rlc_antiderivative(
            edges, ker.tau_relax_ns, ker.tau_osc_ns, DEVICE, torch.float32
        )
        c = (ker.tau_osc_ns**2 + ker.tau_relax_ns**2) / (
            ker.tau_osc_ns * ker.tau_relax_ns**2
        )
        ref = c * (F[1] - F[0]) / ker.tick_ns
        assert torch.allclose(k[0], ref, atol=1e-5, rtol=1e-5)

    def test_decays_monotonically(self):
        """Peak envelope should decrease over time (damped oscillator)."""
        k = RLCKernel(device=DEVICE)()
        abs_k = k.abs()
        n = abs_k.numel()
        first_quarter_peak = abs_k[:n // 4].max().item()
        last_quarter_peak = abs_k[3 * n // 4:].max().item()
        assert last_quarter_peak < 1e-4 * first_quarter_peak, (
            "kernel does not decay: tail peak should be negligible vs head peak"
        )


# ---------------------------------------------------------------------------
# Waveform type tests
# ---------------------------------------------------------------------------

def _make_simulator(n_channels=4):
    return OpticalSimulator(OpticalSimConfig(
        tof_sampler=MockTOFSampler(n_channels=n_channels),
        delays=Delays([]),
        kernel=RLCKernel(duration_ns=2000.0, device=DEVICE),
        device="cpu", tick_ns=1.0, gain=-45.0,
    ))


class TestWaveformFromPhotons:
    def test_basic(self):
        times = torch.tensor([0.0, 1.0, 2.0, 100.0, 101.0])
        channels = torch.tensor([0, 0, 0, 1, 1])
        wf = Waveform.from_photons(times, channels, tick_ns=1.0, n_channels=2)
        assert wf.n_channels == 2
        assert wf.adc.shape[0] == 2
        assert wf.adc[0].sum().item() == 3
        assert wf.adc[1].sum().item() == 2

    def test_shared_time_axis(self):
        times = torch.tensor([0.0, 1.0, 500.0])
        channels = torch.tensor([0, 0, 1])
        wf = Waveform.from_photons(times, channels, tick_ns=1.0, n_channels=2)
        assert wf.adc.shape[1] == 501
        assert wf.adc[0].sum().item() == 2
        assert wf.adc[1].sum().item() == 1

    def test_t0_preserved(self):
        times = torch.tensor([100.0, 200.0])
        channels = torch.tensor([0, 0])
        wf = Waveform.from_photons(times, channels, tick_ns=1.0, n_channels=1, t0=100.0)
        assert wf.t0 == 100.0
        assert wf.adc.shape[1] == 101

    def test_empty(self):
        wf = Waveform.from_photons(
            torch.zeros(0), torch.zeros(0, dtype=torch.long),
            tick_ns=1.0, n_channels=2,
        )
        assert wf.adc.shape == (2, 1)


# ---------------------------------------------------------------------------
# Edge cases and expected failures
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_single_photon(self):
        wf = Waveform.from_photons(
            torch.tensor([50.0]), torch.tensor([0]), tick_ns=1.0, n_channels=2,
        )
        assert wf.adc[0].sum().item() == 1
        assert wf.adc[1].sum().item() == 0

    def test_all_photons_same_time(self):
        times = torch.full((1000,), 42.0)
        channels = torch.zeros(1000, dtype=torch.long)
        wf = Waveform.from_photons(times, channels, tick_ns=1.0, n_channels=1)
        assert wf.adc[0].max().item() == 1000
        assert (wf.adc[0] > 0).sum().item() == 1

    def test_all_photons_same_channel(self):
        times = torch.rand(100) * 100.0
        channels = torch.full((100,), 2, dtype=torch.long)
        wf = Waveform.from_photons(times, channels, tick_ns=1.0, n_channels=4)
        assert wf.adc[0].sum().item() == 0
        assert wf.adc[1].sum().item() == 0
        assert wf.adc[2].sum().item() == 100
        assert wf.adc[3].sum().item() == 0

    def test_slice_no_gaps(self):
        """Slicing a dense waveform (no gaps) should not change it."""
        times = torch.rand(1000) * 100.0
        channels = torch.zeros(1000, dtype=torch.long)
        wf = Waveform.from_photons(times, channels, tick_ns=1.0, n_channels=1)
        sliced = wf.slice(kernel_extent_ns=470.0)
        # single chunk for the only channel, total adc bins = original bins
        assert sliced.adc.numel() == wf.adc.shape[1]
        assert torch.allclose(sliced.adc, wf.adc[0])
        # only 1 chunk total
        assert sliced.n_chunks == 1

    def test_slice_empty_channel(self):
        """Slicing when a channel has no photons."""
        wf = Waveform.from_photons(
            torch.tensor([5.0]), torch.tensor([0]), tick_ns=1.0, n_channels=2,
        )
        sliced = wf.slice(kernel_extent_ns=100.0)
        # 2 channels → at least 2 chunks (one per channel, even if empty)
        assert sliced.n_chunks >= 2
        assert sliced.n_channels == 2

    def test_deslice_single_chunk(self):
        """Deslicing with a single chunk should be trivial identity."""
        data = torch.rand(100)
        sw = SlicedWaveform(
            adc=data,
            offsets=torch.tensor([0, 100]),
            t0_ns=torch.tensor([10.0]),
            pmt_id=torch.tensor([0]),
            tick_ns=1.0, n_channels=1,
        )
        t0, recovered = sw.deslice_channel(0)
        assert t0 == 10.0
        assert torch.allclose(recovered, data)

    def test_deslice_channel_out_of_range(self):
        """Accessing a channel beyond n_channels should raise."""
        sw = SlicedWaveform(
            adc=torch.zeros(1),
            offsets=torch.tensor([0, 1]),
            t0_ns=torch.tensor([0.0]),
            pmt_id=torch.tensor([0]),
            tick_ns=1.0, n_channels=1,
        )
        with pytest.raises(IndexError):
            sw.deslice_channel(1)

    def test_convolve_correctness(self):
        """Convolution of a single-bin impulse should reproduce the kernel."""
        wf = Waveform.from_photons(
            torch.tensor([0.0]), torch.tensor([0]),
            tick_ns=1.0, n_channels=1,
        )
        kernel = RLCKernel(duration_ns=500.0, device=DEVICE)()
        convolved = wf.convolve(kernel, gain=1.0)
        n_k = kernel.numel()
        assert convolved.adc.shape[0] == 1
        assert torch.allclose(convolved.adc[0, :n_k], kernel, atol=1e-5)

    def test_times_before_t0_raises(self):
        with pytest.raises(ValueError, match="before t0"):
            Waveform.from_photons(
                torch.tensor([-5.0, 0.0, 10.0]),
                torch.tensor([0, 0, 0]),
                tick_ns=1.0, n_channels=1, t0=0.0,
            )

    def test_times_at_t0_is_fine(self):
        wf = Waveform.from_photons(
            torch.tensor([0.0, 0.0, 10.0]),
            torch.tensor([0, 0, 0]),
            tick_ns=1.0, n_channels=1, t0=0.0,
        )
        assert wf.adc[0].sum().item() == 3

    def test_delays_zero_photons(self):
        d = Delays([ScintillationBiexponentialDelay(), TPBExponentialDelay()])
        out = d.sample(0, DEVICE)
        assert out.shape == (0,)

    def test_sliced_from_photons_empty(self):
        sw = SlicedWaveform.from_photons(
            torch.zeros(0), torch.zeros(0, dtype=torch.long),
            tick_ns=1.0, n_channels=2, kernel_extent_ns=470.0,
        )
        assert sw.n_chunks == 2  # one trivial chunk per empty channel
        assert sw.n_channels == 2
        wf = sw.deslice()
        assert isinstance(wf, Waveform)

    def test_sliced_from_photons_single_photon_per_channel(self):
        sw = SlicedWaveform.from_photons(
            torch.tensor([100.0, 200.0]),
            torch.tensor([0, 1]),
            tick_ns=1.0, n_channels=2, kernel_extent_ns=470.0,
        )
        assert sw.adc.sum().item() == 2


class TestSimulateReturnsTypes:
    def test_stitched_returns_sliced(self):
        sim = _make_simulator()
        result = sim.simulate(torch.zeros(10, 3), torch.full((10,), 100), torch.zeros(10), stitched=True)
        assert isinstance(result, SlicedWaveform)
        assert result.n_chunks >= 4  # at least one chunk per channel
        assert result.n_channels == 4

    def test_full_returns_waveform(self):
        sim = _make_simulator()
        result = sim.simulate(torch.zeros(10, 3), torch.full((10,), 100), torch.zeros(10), stitched=False)
        assert isinstance(result, Waveform)
        assert result.adc.dim() == 2
        assert result.adc.shape[0] == 4

    def test_empty_photons(self):
        mock = MockTOFSampler(n_channels=4)
        mock.sample = lambda pos, n_ph, t_step: (
            torch.zeros(0, device=DEVICE),
            torch.zeros(0, device=DEVICE, dtype=torch.long),
            torch.zeros(0, device=DEVICE, dtype=torch.long),
        )
        sim = OpticalSimulator(OpticalSimConfig(
            tof_sampler=mock, delays=Delays([]), kernel=RLCKernel(device=DEVICE),
            device="cpu",
        ))
        result = sim.simulate(torch.zeros(1, 3), torch.zeros(1, dtype=torch.int32), torch.zeros(1))
        assert isinstance(result, SlicedWaveform)


class TestDeslice:
    def test_roundtrip(self):
        sim = _make_simulator()
        sliced = sim.simulate(torch.zeros(10, 3), torch.full((10,), 100), torch.zeros(10), stitched=True)
        assert isinstance(sliced, SlicedWaveform)

        t0, ch_wf = sliced.deslice_channel(0)
        assert ch_wf.numel() > 0
        assert isinstance(t0, float)

        full = sliced.deslice()
        assert isinstance(full, Waveform)
        assert full.n_channels == 4
        assert full.adc.dim() == 2


# ---------------------------------------------------------------------------
# Closure tests: slice <-> deslice roundtrip
# ---------------------------------------------------------------------------

class TestFromPhotonsCrossConstruction:
    """Compare Waveform.from_photons vs SlicedWaveform.from_photons paths."""

    def _make_photons(self):
        torch.manual_seed(99)
        times = torch.cat([
            torch.rand(500) * 200.0,
            2000.0 + torch.rand(500) * 200.0,
        ])
        channels = torch.cat([
            torch.zeros(250, dtype=torch.long),
            torch.ones(250, dtype=torch.long),
            torch.zeros(250, dtype=torch.long),
            torch.ones(250, dtype=torch.long),
        ])
        return times, channels

    def test_sliced_from_photons_deslice_matches_waveform_from_photons(self):
        times, channels = self._make_photons()
        extent = 470.0

        full = Waveform.from_photons(times, channels, tick_ns=1.0, n_channels=2,
                                     t0=times.min().item())
        recovered = SlicedWaveform.from_photons(
            times, channels, tick_ns=1.0, n_channels=2, kernel_extent_ns=extent,
        ).deslice()

        for ch in range(2):
            assert abs(full.adc[ch].sum().item() - recovered.adc[ch].sum().item()) < 1e-3, (
                f"ch {ch}: total count mismatch"
            )

    def test_waveform_slice_vs_sliced_from_photons_photon_counts(self):
        times, channels = self._make_photons()
        extent = 470.0

        via_slice = Waveform.from_photons(
            times, channels, tick_ns=1.0, n_channels=2,
            t0=times.min().item(),
        ).slice(extent)
        via_direct = SlicedWaveform.from_photons(
            times, channels, tick_ns=1.0, n_channels=2, kernel_extent_ns=extent,
        )

        for ch in range(2):
            # total photon counts per channel must match
            ch_mask_s = via_slice.pmt_id == ch
            ch_mask_d = via_direct.pmt_id == ch
            total_s = sum(via_slice.chunk(k).sum().item()
                         for k in torch.where(ch_mask_s)[0])
            total_d = sum(via_direct.chunk(k).sum().item()
                         for k in torch.where(ch_mask_d)[0])
            assert abs(total_s - total_d) < 1e-3, f"ch {ch}: total count mismatch"
            # same number of chunks per channel
            assert ch_mask_s.sum() == ch_mask_d.sum(), f"ch {ch}: chunk count mismatch"


class TestSliceDesliceClosure:
    """Verify that slice then deslice recovers the original waveform."""

    def _make_waveform(self):
        times = torch.cat([
            torch.rand(500) * 200.0,
            2000.0 + torch.rand(500) * 200.0,
        ])
        channels = torch.cat([
            torch.zeros(250, dtype=torch.long),
            torch.ones(250, dtype=torch.long),
            torch.zeros(250, dtype=torch.long),
            torch.ones(250, dtype=torch.long),
        ])
        return Waveform.from_photons(times, channels, tick_ns=1.0, n_channels=2)

    def test_slice_then_deslice_preserves_data(self):
        torch.manual_seed(42)
        wf = self._make_waveform()
        recovered = wf.slice(kernel_extent_ns=470.0).deslice()

        assert recovered.n_channels == wf.n_channels
        assert recovered.tick_ns == wf.tick_ns

        n_cmp = min(wf.adc.shape[1], recovered.adc.shape[1])
        for ch in range(wf.n_channels):
            assert torch.allclose(wf.adc[ch, :n_cmp], recovered.adc[ch, :n_cmp], atol=1e-6), (
                f"ch {ch}: data mismatch after slice->deslice"
            )

    def test_slice_then_deslice_preserves_t0(self):
        torch.manual_seed(42)
        wf = self._make_waveform()
        recovered = wf.slice(kernel_extent_ns=470.0).deslice()
        assert abs(wf.t0 - recovered.t0) < 1.0

    def test_slice_actually_compresses(self):
        """Waveform with a large gap must compress to strictly fewer bins."""
        torch.manual_seed(42)
        wf = self._make_waveform()
        sliced = wf.slice(kernel_extent_ns=470.0)
        orig_bins = wf.adc.shape[1] * wf.n_channels
        sliced_bins = sliced.adc.numel()
        assert sliced_bins < orig_bins, (
            f"no compression occurred: {sliced_bins} >= {orig_bins} bins"
        )
        compression_ratio = sliced_bins / orig_bins
        assert compression_ratio < 0.8, f"compression ratio {compression_ratio:.2f} too weak"

    def test_convolve_then_slice_deslice(self):
        torch.manual_seed(42)
        wf = self._make_waveform()
        kernel = RLCKernel(duration_ns=2000.0, device=DEVICE)()
        convolved = wf.convolve(kernel, gain=-45.0)
        recovered = convolved.slice(kernel_extent_ns=470.0).deslice()

        n_cmp = min(convolved.adc.shape[1], recovered.adc.shape[1])
        for ch in range(wf.n_channels):
            assert torch.allclose(
                convolved.adc[ch, :n_cmp], recovered.adc[ch, :n_cmp], atol=1e-5
            ), f"ch {ch}: convolved data mismatch after slice->deslice"

    def test_slice_convolve_deslice_matches_convolve(self):
        """wf.slice().convolve().deslice() should match wf.convolve()."""
        torch.manual_seed(42)
        wf = self._make_waveform()
        kernel = RLCKernel(duration_ns=2000.0, device=DEVICE)()

        full = wf.convolve(kernel, gain=-45.0)
        via_sliced = wf.slice(kernel_extent_ns=2000.0).convolve(kernel, gain=-45.0).deslice()

        assert full.adc.shape == via_sliced.adc.shape, (
            f"shape mismatch: {full.adc.shape} vs {via_sliced.adc.shape}"
        )
        for ch in range(wf.n_channels):
            peak = full.adc[ch].abs().max().item()
            if peak < 1e-10:
                continue
            residual = (via_sliced.adc[ch] - full.adc[ch]).abs().max().item()
            assert residual / peak < 1e-5, (
                f"ch {ch}: max |residual|/peak = {residual/peak:.6f}"
            )

    def test_slice_deslice_with_pipeline(self):
        n_ch = 4
        mock = _SeededMockTOF(n_channels=n_ch)
        kernel = RLCKernel(duration_ns=2000.0, device=DEVICE)
        config = OpticalSimConfig(
            tof_sampler=mock, delays=Delays([]), kernel=kernel,
            device="cpu", tick_ns=1.0, gain=-45.0,
        )
        sim = OpticalSimulator(config)
        pos = torch.zeros(200, 3)
        n_ph = torch.full((200,), 500)
        t_step = torch.zeros(200)

        sliced = sim.simulate(pos, n_ph, t_step, stitched=True)
        full = sim.simulate(pos, n_ph, t_step, stitched=False)
        recovered = sliced.deslice()

        n_cmp = min(recovered.adc.shape[1], full.adc.shape[1])
        for ch in range(n_ch):
            residual = recovered.adc[ch, :n_cmp] - full.adc[ch, :n_cmp]
            peak = full.adc[ch, :n_cmp].abs().max().item()
            if peak < 1e-10:
                continue
            max_rel = residual.abs().max().item() / peak
            assert max_rel < 1e-5


# ---------------------------------------------------------------------------
# Core validation: stitched vs full
# ---------------------------------------------------------------------------

class TestStitchedMatchesFull:
    @pytest.mark.parametrize("tick_ns", [0.5, 1.0, 2.0, 5.0])
    def test_residual_below_threshold(self, tick_ns):
        """Stitched->deslice should match full convolution across tick sizes."""
        n_ch = 4
        mock = _SeededMockTOF(n_channels=n_ch)
        kernel = RLCKernel(duration_ns=2000.0, tick_ns=tick_ns, device=DEVICE)
        config = OpticalSimConfig(
            tof_sampler=mock, delays=Delays([]), kernel=kernel,
            device="cpu", tick_ns=tick_ns, gain=-45.0,
        )
        sim = OpticalSimulator(config)
        pos = torch.zeros(200, 3)
        n_ph = torch.full((200,), 500)
        t_step = torch.zeros(200)

        sliced = sim.simulate(pos, n_ph, t_step, stitched=True)
        full = sim.simulate(pos, n_ph, t_step, stitched=False)

        assert isinstance(sliced, SlicedWaveform)
        assert isinstance(full, Waveform)

        for ch in range(n_ch):
            t0, real_wf = sliced.deslice_channel(ch)
            full_ch = full.adc[ch]

            n_cmp = min(real_wf.numel(), full_ch.numel())
            if n_cmp == 0:
                continue

            residual = real_wf[:n_cmp] - full_ch[:n_cmp]
            peak = full_ch[:n_cmp].abs().max().item()
            if peak < 1e-10:
                continue
            max_rel = residual.abs().max().item() / peak
            assert max_rel < 1e-5, (
                f"tick_ns={tick_ns}, ch {ch}: max |residual|/peak = {max_rel:.6f} exceeds 1e-5"
            )


class TestDifferentTickNs:

    @pytest.mark.parametrize("tick_ns", [0.5, 1.0, 2.0, 10.0])
    def test_photon_count_preserved(self, tick_ns):
        torch.manual_seed(77)
        times = torch.rand(1000) * 500.0
        channels = torch.randint(0, 3, (1000,))

        wf = Waveform.from_photons(times, channels, tick_ns=tick_ns, n_channels=3)
        for ch in range(3):
            expected = (channels == ch).sum().item()
            assert abs(wf.adc[ch].sum().item() - expected) < 1e-3

    @pytest.mark.parametrize("tick_ns", [0.5, 1.0, 2.0, 10.0])
    def test_slice_deslice_closure_across_ticks(self, tick_ns):
        torch.manual_seed(77)
        times = torch.cat([torch.rand(500) * 100.0, 1000.0 + torch.rand(500) * 100.0])
        channels = torch.randint(0, 2, (1000,))

        wf = Waveform.from_photons(times, channels, tick_ns=tick_ns, n_channels=2)
        recovered = wf.slice(kernel_extent_ns=470.0).deslice()

        n_cmp = min(wf.adc.shape[1], recovered.adc.shape[1])
        for ch in range(2):
            assert torch.allclose(wf.adc[ch, :n_cmp], recovered.adc[ch, :n_cmp], atol=1e-6), (
                f"tick_ns={tick_ns}, ch {ch}: slice->deslice mismatch"
            )

    def test_convolution_peak_consistent_across_ticks(self):
        torch.manual_seed(77)
        peaks = {}
        for tick_ns in [0.5, 1.0, 2.0]:
            times = torch.full((100,), 50.0)
            channels = torch.zeros(100, dtype=torch.long)
            wf = Waveform.from_photons(times, channels, tick_ns=tick_ns, n_channels=1)
            kernel = RLCKernel(duration_ns=500.0, tick_ns=tick_ns, device=DEVICE)
            convolved = wf.convolve(kernel(), gain=-1.0)
            peaks[tick_ns] = convolved.adc[0].abs().max().item()

        min_peak = min(peaks.values())
        max_peak = max(peaks.values())
        assert min_peak > 0, "zero peak detected"
        assert max_peak / min_peak < 2.0, (
            f"peaks vary too much across tick sizes: {peaks}"
        )


# ---------------------------------------------------------------------------
# Digitization tests
# ---------------------------------------------------------------------------

class TestDigitize:
    def test_pedestal_added(self):
        data = torch.zeros(2, 10)
        result = digitize(data, pedestal=100.0, n_bits=14)
        assert (result == 100.0).all()

    def test_quantization(self):
        data = torch.tensor([0.3, 0.7, 1.5, 2.5])
        result = digitize(data, pedestal=0.0, n_bits=14)
        # torch.round uses banker's rounding: 0.5 -> 0, 1.5 -> 2, 2.5 -> 2
        assert result[0].item() == 0.0
        assert result[1].item() == 1.0

    def test_saturation_upper(self):
        data = torch.tensor([20000.0])
        result = digitize(data, pedestal=0.0, n_bits=14)
        assert result.item() == 16383.0

    def test_saturation_lower(self):
        data = torch.tensor([-5000.0])
        result = digitize(data, pedestal=100.0, n_bits=14)
        assert result.item() == 0.0

    def test_14bit_range(self):
        data = torch.linspace(-2000, 20000, 1000)
        result = digitize(data, pedestal=1500.0, n_bits=14)
        assert result.min().item() >= 0
        assert result.max().item() <= 16383

    def test_10bit_range(self):
        data = torch.linspace(-2000, 5000, 1000)
        result = digitize(data, pedestal=500.0, n_bits=10)
        assert result.min().item() >= 0
        assert result.max().item() <= 1023

    def test_all_values_integer(self):
        data = torch.randn(100) * 500 + 1000
        result = digitize(data, pedestal=1500.0, n_bits=14)
        assert torch.equal(result, result.round())


class TestDigitizeSTE:
    """digitize_ste: same forward as digitize, identity backward."""

    def test_forward_matches_digitize(self):
        from goop.digitize import digitize_ste
        data = torch.linspace(-2000.0, 20000.0, 500)
        ref = digitize(data, pedestal=1500.0, n_bits=14)
        ste = digitize_ste(data, pedestal=1500.0, n_bits=14)
        assert torch.equal(ref, ste), "STE forward must match plain digitize exactly"

    def test_backward_is_identity(self):
        from goop.digitize import digitize_ste
        x = torch.tensor([100.0, 1500.0, 16000.0, -5000.0], requires_grad=True)
        # Sum of STE outputs → expected gradient wrt x is 1 everywhere
        digitize_ste(x, pedestal=1500.0, n_bits=14).sum().backward()
        assert torch.allclose(x.grad, torch.ones_like(x))

    def test_plain_digitize_blocks_grad(self):
        """Sanity check: plain digitize *should* have zero/no gradient."""
        x = torch.tensor([100.0, 1500.0], requires_grad=True)
        out = digitize(x, pedestal=1500.0, n_bits=14)
        # round/clamp produce zero gradient → x.grad stays None or is zero
        try:
            out.sum().backward()
        except RuntimeError:
            return  # no grad_fn, as expected
        assert x.grad is None or x.grad.abs().max().item() == 0


class TestWaveformDigitize:
    def test_waveform_digitize(self):
        wf = Waveform(
            adc=torch.randn(2, 100) * 100,
            t0=0.0, tick_ns=1.0, n_channels=2,
        )
        digitized = wf.digitize(pedestal=1500.0, n_bits=14)
        assert isinstance(digitized, Waveform)
        assert digitized.adc.shape == wf.adc.shape
        assert digitized.t0 == wf.t0
        assert digitized.tick_ns == wf.tick_ns
        assert digitized.adc.min().item() >= 0
        assert digitized.adc.max().item() <= 16383

    def test_sliced_digitize(self):
        sw = SlicedWaveform(
            adc=torch.randn(200) * 100,
            offsets=torch.tensor([0, 100, 200]),
            t0_ns=torch.tensor([0.0, 500.0]),
            pmt_id=torch.tensor([0, 1]),
            tick_ns=1.0, n_channels=2,
        )
        digitized = sw.digitize(pedestal=1500.0, n_bits=14)
        assert isinstance(digitized, SlicedWaveform)
        assert digitized.adc.shape == sw.adc.shape
        assert digitized.adc.min().item() >= 0
        assert digitized.adc.max().item() <= 16383
        assert torch.equal(digitized.offsets, sw.offsets)

    def test_sliced_digitize_deslice_matches_waveform_digitize(self):
        torch.manual_seed(55)
        times = torch.rand(500) * 200.0
        channels = torch.randint(0, 2, (500,))

        wf = Waveform.from_photons(times, channels, tick_ns=1.0, n_channels=2)
        kernel = RLCKernel(duration_ns=500.0, device=DEVICE)()
        convolved = wf.convolve(kernel, gain=-20.0)
        wf_digitized = convolved.digitize(pedestal=1500.0, n_bits=14)

        sliced = convolved.slice(kernel_extent_ns=500.0)
        sw_digitized = sliced.digitize(pedestal=1500.0, n_bits=14)
        recovered = sw_digitized.deslice()

        n_cmp = min(wf_digitized.adc.shape[1], recovered.adc.shape[1])
        for ch in range(2):
            assert torch.equal(
                wf_digitized.adc[ch, :n_cmp], recovered.adc[ch, :n_cmp]
            ), f"ch {ch}: digitized slice->deslice mismatch"


# ---------------------------------------------------------------------------
# Dark noise tests
# ---------------------------------------------------------------------------

class TestDarkNoise:
    def test_shape_and_types(self):
        dn = DarkNoise(rate_hz=2000.0)
        times, channels = dn.sample(4, 0.0, 1e6, DEVICE)
        assert times.dim() == 1
        assert channels.dim() == 1
        assert times.shape == channels.shape

    def test_time_range(self):
        torch.manual_seed(0)
        dn = DarkNoise(rate_hz=100_000.0)  # high rate for statistics
        times, channels = dn.sample(4, 100.0, 1e6, DEVICE)
        if times.numel() > 0:
            assert times.min().item() >= 100.0
            assert times.max().item() <= 1e6

    def test_channel_range(self):
        torch.manual_seed(0)
        dn = DarkNoise(rate_hz=100_000.0)
        times, channels = dn.sample(4, 0.0, 1e6, DEVICE)
        if channels.numel() > 0:
            assert channels.min().item() >= 0
            assert channels.max().item() < 4

    def test_mean_count(self):
        torch.manual_seed(0)
        dn = DarkNoise(rate_hz=1000.0)
        # 1 ms window, 4 channels, 1 kHz rate → expect ~1 hit per channel, ~4 total
        counts = []
        for _ in range(1000):
            t, c = dn.sample(4, 0.0, 1e6, DEVICE)  # 1e6 ns = 1 ms
            counts.append(t.numel())
        mean_count = sum(counts) / len(counts)
        # expected: 4 channels * 1000 Hz * 1e-3 s = 4.0
        assert abs(mean_count - 4.0) < 1.0, f"mean dark hits {mean_count}, expected ~4"

    def test_zero_rate_empty(self):
        dn = DarkNoise(rate_hz=0.0)
        times, channels = dn.sample(4, 0.0, 1e6, DEVICE)
        assert times.numel() == 0
        assert channels.numel() == 0

    def test_zero_window_empty(self):
        dn = DarkNoise(rate_hz=2000.0)
        times, channels = dn.sample(4, 100.0, 100.0, DEVICE)
        assert times.numel() == 0


# ---------------------------------------------------------------------------
# Integration: digitization + dark noise in pipeline
# ---------------------------------------------------------------------------

class TestPipelineDigitization:
    def test_backward_compat(self):
        """No digitization or aux sources → identical to original behavior."""
        sim = _make_simulator()
        pos = torch.zeros(10, 3)
        n_ph = torch.full((10,), 100)
        t = torch.zeros(10)
        result = sim.simulate(pos, n_ph, t, stitched=True)
        assert isinstance(result, SlicedWaveform)
        # Output should be float, not quantized
        assert result.adc.dtype == torch.float32

    def test_digitization_produces_integers(self):
        sim = OpticalSimulator(OpticalSimConfig(
            tof_sampler=MockTOFSampler(n_channels=4),
            delays=Delays([]),
            kernel=RLCKernel(duration_ns=2000.0, device=DEVICE),
            device="cpu", tick_ns=1.0, gain=-45.0,
            digitization=DigitizationConfig(n_bits=14, pedestal=1500.0),
        ))
        result = sim.simulate(
            torch.zeros(10, 3), torch.full((10,), 100), torch.zeros(10),
            stitched=False,
        )
        assert isinstance(result, Waveform)
        assert torch.equal(result.adc, result.adc.round())
        assert result.adc.min().item() >= 0
        assert result.adc.max().item() <= 16383

    def test_dark_noise_in_pipeline(self):
        sim = OpticalSimulator(OpticalSimConfig(
            tof_sampler=MockTOFSampler(n_channels=4),
            delays=Delays([]),
            kernel=RLCKernel(duration_ns=2000.0, device=DEVICE),
            device="cpu", tick_ns=1.0, gain=-45.0,
            aux_photon_sources=[DarkNoise(rate_hz=2000.0)],
        ))
        result = sim.simulate(
            torch.zeros(10, 3), torch.full((10,), 100), torch.zeros(10),
            stitched=False,
        )
        assert isinstance(result, Waveform)

    def test_full_sbnd_like_pipeline(self):
        """Dark noise + digitization together."""
        sim = OpticalSimulator(OpticalSimConfig(
            tof_sampler=MockTOFSampler(n_channels=4),
            delays=Delays([]),
            kernel=RLCKernel(duration_ns=2000.0, tick_ns=2.0, device=DEVICE),
            device="cpu", tick_ns=2.0, gain=-20.0,
            aux_photon_sources=[DarkNoise(rate_hz=2000.0)],
            digitization=DigitizationConfig(n_bits=14, pedestal=1500.0),
        ))
        result = sim.simulate(
            torch.zeros(10, 3), torch.full((10,), 100), torch.zeros(10),
            stitched=True,
        )
        assert isinstance(result, SlicedWaveform)
        assert result.adc.min().item() >= 0
        assert result.adc.max().item() <= 16383
        assert torch.equal(result.adc, result.adc.round())


# ---------------------------------------------------------------------------
# Oversampling tests
# ---------------------------------------------------------------------------

from goop.kernels import SERKernel


class TestKernelWithTickNs:
    def test_rlc_shape_changes(self):
        k = RLCKernel(duration_ns=500.0, tick_ns=1.0, device=DEVICE)
        k_fine = k.with_tick_ns(0.25)
        assert k_fine().shape[0] == 4 * k().shape[0]

    def test_ser_shape_changes(self):
        k = SERKernel(duration_ns=1000.0, tick_ns=1.0, device=DEVICE)
        k_fine = k.with_tick_ns(0.5)
        assert k_fine().shape[0] == 2 * k().shape[0]

    def test_cache_cleared(self):
        k = RLCKernel(duration_ns=500.0, tick_ns=1.0, device=DEVICE)
        _ = k()  # populate cache
        k_fine = k.with_tick_ns(0.5)
        assert k_fine._kernel_cache is None

    def test_original_unchanged(self):
        k = RLCKernel(duration_ns=500.0, tick_ns=1.0, device=DEVICE)
        original = k()
        _ = k.with_tick_ns(0.25)
        assert torch.equal(k(), original)


class TestWaveformDownsample:
    def test_identity(self):
        wf = Waveform(adc=torch.randn(2, 20), t0=0.0, tick_ns=0.5, n_channels=2)
        ds = wf.downsample(1)
        assert ds is wf

    def test_shape_and_tick(self):
        wf = Waveform(adc=torch.randn(2, 20), t0=0.0, tick_ns=0.25, n_channels=2)
        ds = wf.downsample(4)
        assert ds.adc.shape == (2, 5)
        assert ds.tick_ns == 1.0

    def test_signal_preserved(self):
        """Mean amplitude should be preserved after downsampling."""
        wf = Waveform(adc=torch.randn(2, 40), t0=0.0, tick_ns=0.25, n_channels=2)
        ds = wf.downsample(4)
        for ch in range(2):
            assert abs(wf.adc[ch].mean().item() - ds.adc[ch].mean().item()) < 1e-5

    def test_t0_preserved(self):
        wf = Waveform(adc=torch.randn(2, 20), t0=5.0, tick_ns=0.5, n_channels=2)
        ds = wf.downsample(2)
        assert ds.t0 == 5.0

    def test_uneven_bins_padded(self):
        """Bins not divisible by factor should still work."""
        wf = Waveform(adc=torch.ones(1, 7), t0=0.0, tick_ns=0.5, n_channels=1)
        ds = wf.downsample(4)
        assert ds.adc.shape == (1, 2)
        # first group: mean of [1,1,1,1] = 1.0; second: mean of [1,1,1,0] = 0.75
        assert abs(ds.adc[0, 0].item() - 1.0) < 1e-6
        assert abs(ds.adc[0, 1].item() - 0.75) < 1e-6


class TestSlicedWaveformDownsample:
    def test_identity(self):
        sw = SlicedWaveform(
            adc=torch.randn(20), offsets=torch.tensor([0, 20]),
            t0_ns=torch.tensor([0.0]), pmt_id=torch.tensor([0]),
            tick_ns=0.5, n_channels=1,
        )
        ds = sw.downsample(1)
        assert ds is sw

    def test_tick_updated(self):
        sw = SlicedWaveform(
            adc=torch.randn(20), offsets=torch.tensor([0, 20]),
            t0_ns=torch.tensor([0.0]), pmt_id=torch.tensor([0]),
            tick_ns=0.25, n_channels=1,
        )
        ds = sw.downsample(4)
        assert ds.tick_ns == 1.0
        assert ds.adc.numel() == 5

    def test_signal_preserved(self):
        sw = SlicedWaveform(
            adc=torch.randn(40), offsets=torch.tensor([0, 20, 40]),
            t0_ns=torch.tensor([0.0, 100.0]), pmt_id=torch.tensor([0, 1]),
            tick_ns=0.25, n_channels=2,
        )
        ds = sw.downsample(4)
        assert abs(sw.adc[:20].mean().item() - ds.adc[:5].mean().item()) < 1e-5
        assert abs(sw.adc[20:].mean().item() - ds.adc[5:].mean().item()) < 1e-5


class TestOversamplePipeline:
    """Oversampled pipeline should match direct fine-resolution + manual downsample."""

    def test_oversample_matches_fine_resolution(self):
        """oversample=4 at tick_ns=1.0 should match tick_ns=0.25 then downsample."""
        n_ch = 4
        mock = _SeededMockTOF(n_channels=n_ch)
        kernel = RLCKernel(duration_ns=500.0, tick_ns=1.0, device=DEVICE)

        # oversampled pipeline
        config_os = OpticalSimConfig(
            tof_sampler=mock, delays=Delays([]), kernel=kernel,
            device="cpu", tick_ns=1.0, oversample=4, gain=-1.0,
        )
        sim_os = OpticalSimulator(config_os)
        result_os = sim_os.simulate(
            torch.zeros(50, 3), torch.full((50,), 200), torch.zeros(50),
            stitched=False,
        )

        # manual fine-resolution pipeline
        mock2 = _SeededMockTOF(n_channels=n_ch)
        kernel_fine = RLCKernel(duration_ns=500.0, tick_ns=0.25, device=DEVICE)
        config_fine = OpticalSimConfig(
            tof_sampler=mock2, delays=Delays([]), kernel=kernel_fine,
            device="cpu", tick_ns=0.25, gain=-1.0,
        )
        sim_fine = OpticalSimulator(config_fine)
        result_fine = sim_fine.simulate(
            torch.zeros(50, 3), torch.full((50,), 200), torch.zeros(50),
            stitched=False,
        )
        result_ds = result_fine.downsample(4)

        n_cmp = min(result_os.adc.shape[1], result_ds.adc.shape[1])
        for ch in range(n_ch):
            peak = result_ds.adc[ch, :n_cmp].abs().max().item()
            if peak < 1e-10:
                continue
            residual = (result_os.adc[ch, :n_cmp] - result_ds.adc[ch, :n_cmp]).abs().max().item()
            assert residual / peak < 1e-4, (
                f"ch {ch}: oversampled vs fine residual {residual/peak:.6f}"
            )


class TestOversampleStitchedMatchesFull:
    """Stitched vs full closure should hold with oversampling."""

    @pytest.mark.parametrize("oversample", [1, 2, 4])
    def test_closure(self, oversample):
        n_ch = 4
        mock = _SeededMockTOF(n_channels=n_ch)
        kernel = RLCKernel(duration_ns=2000.0, tick_ns=1.0, device=DEVICE)
        config = OpticalSimConfig(
            tof_sampler=mock, delays=Delays([]), kernel=kernel,
            device="cpu", tick_ns=1.0, oversample=oversample, gain=-45.0,
        )
        sim = OpticalSimulator(config)
        pos = torch.zeros(50, 3)
        n_ph = torch.full((50,), 200)
        t_step = torch.zeros(50)

        sliced = sim.simulate(pos, n_ph, t_step, stitched=True)
        full = sim.simulate(pos, n_ph, t_step, stitched=False)
        recovered = sliced.deslice()

        assert recovered.tick_ns == full.tick_ns
        n_cmp = min(recovered.adc.shape[1], full.adc.shape[1])
        for ch in range(n_ch):
            peak = full.adc[ch, :n_cmp].abs().max().item()
            if peak < 1e-10:
                continue
            residual = (recovered.adc[ch, :n_cmp] - full.adc[ch, :n_cmp]).abs().max().item()
            assert residual / peak < 1e-3, (
                f"oversample={oversample}, ch {ch}: residual/peak={residual/peak:.6f}"
            )


class TestOversampleBackwardCompat:
    """oversample=1 should produce identical results to no oversampling."""

    def test_identical_output(self):
        n_ch = 4
        mock1 = _SeededMockTOF(n_channels=n_ch)
        mock2 = _SeededMockTOF(n_channels=n_ch)
        kernel = RLCKernel(duration_ns=500.0, tick_ns=1.0, device=DEVICE)

        config1 = OpticalSimConfig(
            tof_sampler=mock1, delays=Delays([]), kernel=kernel,
            device="cpu", tick_ns=1.0, gain=-45.0,
        )
        config2 = OpticalSimConfig(
            tof_sampler=mock2, delays=Delays([]), kernel=kernel,
            device="cpu", tick_ns=1.0, oversample=1, gain=-45.0,
        )

        pos = torch.zeros(50, 3)
        n_ph = torch.full((50,), 200)
        t = torch.zeros(50)

        r1 = OpticalSimulator(config1).simulate(pos, n_ph, t, stitched=False)
        r2 = OpticalSimulator(config2).simulate(pos, n_ph, t, stitched=False)

        assert torch.allclose(r1.adc, r2.adc, atol=1e-6)
        assert r1.t0 == r2.t0
        assert r1.tick_ns == r2.tick_ns


# ---------------------------------------------------------------------------
# SER amplitude jitter tests
# ---------------------------------------------------------------------------

class TestSERJitter:
    def test_no_jitter_unchanged(self):
        """ser_jitter_std=0.0 produces identical results to default."""
        mock1 = _SeededMockTOF(n_channels=4)
        mock2 = _SeededMockTOF(n_channels=4)
        kernel = RLCKernel(duration_ns=500.0, tick_ns=1.0, device=DEVICE)
        pos = torch.zeros(50, 3)
        n_ph = torch.full((50,), 200)
        t = torch.zeros(50)

        r1 = OpticalSimulator(OpticalSimConfig(
            tof_sampler=mock1, delays=Delays([]), kernel=kernel,
            device="cpu", tick_ns=1.0, gain=-45.0,
        )).simulate(pos, n_ph, t, stitched=False)
        r2 = OpticalSimulator(OpticalSimConfig(
            tof_sampler=mock2, delays=Delays([]), kernel=kernel,
            device="cpu", tick_ns=1.0, gain=-45.0, ser_jitter_std=0.0,
        )).simulate(pos, n_ph, t, stitched=False)

        assert torch.allclose(r1.adc, r2.adc, atol=1e-6)

    def test_jitter_weights_not_integer(self):
        """With jitter, histogram bins are no longer exact integers."""
        torch.manual_seed(42)
        times = torch.rand(500) * 200.0
        channels = torch.randint(0, 2, (500,))
        weights = torch.normal(1.0, 0.1, (500,))
        wf = Waveform.from_photons(times, channels, tick_ns=1.0, n_channels=2, weights=weights)
        # at least some bins should have non-integer values
        fractional = (wf.adc - wf.adc.round()).abs()
        assert fractional.max().item() > 0.01

    def test_jitter_mean_unbiased(self):
        """Total photon count should be approximately preserved (mean weight ~ 1)."""
        torch.manual_seed(0)
        times = torch.rand(100_000) * 500.0
        channels = torch.randint(0, 2, (100_000,))
        weights = torch.normal(1.0, 0.1, (100_000,))
        wf = Waveform.from_photons(times, channels, tick_ns=1.0, n_channels=2, weights=weights)
        total = wf.adc.sum().item()
        assert abs(total - 100_000) / 100_000 < 0.01  # within 1%

    def test_jitter_in_pipeline(self):
        """Full pipeline with jitter runs without error."""
        sim = OpticalSimulator(OpticalSimConfig(
            tof_sampler=MockTOFSampler(n_channels=4),
            delays=Delays([]),
            kernel=RLCKernel(duration_ns=500.0, device=DEVICE),
            device="cpu", tick_ns=1.0, gain=-45.0,
            ser_jitter_std=0.1,
        ))
        result = sim.simulate(
            torch.zeros(10, 3), torch.full((10,), 100), torch.zeros(10),
            stitched=True,
        )
        assert isinstance(result, SlicedWaveform)

    def test_jitter_sliced_weights_match(self):
        """Jittered SlicedWaveform photon counts match jittered Waveform."""
        torch.manual_seed(99)
        times = torch.rand(1000) * 200.0
        channels = torch.randint(0, 2, (1000,))
        weights = torch.normal(1.0, 0.1, (1000,))

        wf = Waveform.from_photons(times, channels, tick_ns=1.0, n_channels=2, weights=weights)
        sw = SlicedWaveform.from_photons(
            times, channels, tick_ns=1.0, n_channels=2,
            kernel_extent_ns=470.0, weights=weights,
        )
        for ch in range(2):
            expected = wf.adc[ch].sum().item()
            ch_mask = sw.pmt_id == ch
            actual = sum(sw.chunk(k).sum().item() for k in torch.where(ch_mask)[0])
            assert abs(expected - actual) < 1e-3, f"ch {ch}: count mismatch"


# ---------------------------------------------------------------------------
# Baseline noise tests
# ---------------------------------------------------------------------------

class TestBaselineNoise:
    def test_no_noise_unchanged(self):
        """baseline_noise_std=0.0 produces identical results to default."""
        mock1 = _SeededMockTOF(n_channels=4)
        mock2 = _SeededMockTOF(n_channels=4)
        kernel = RLCKernel(duration_ns=500.0, tick_ns=1.0, device=DEVICE)
        pos = torch.zeros(50, 3)
        n_ph = torch.full((50,), 200)
        t = torch.zeros(50)

        r1 = OpticalSimulator(OpticalSimConfig(
            tof_sampler=mock1, delays=Delays([]), kernel=kernel,
            device="cpu", tick_ns=1.0, gain=-45.0,
        )).simulate(pos, n_ph, t, stitched=False)
        r2 = OpticalSimulator(OpticalSimConfig(
            tof_sampler=mock2, delays=Delays([]), kernel=kernel,
            device="cpu", tick_ns=1.0, gain=-45.0, baseline_noise_std=0.0,
        )).simulate(pos, n_ph, t, stitched=False)

        assert torch.allclose(r1.adc, r2.adc, atol=1e-6)

    def test_noise_std_correct(self):
        """Noise added to a zero-signal waveform should have the configured std."""
        torch.manual_seed(0)
        # use a mock that returns no photons → waveform is all zeros before noise
        mock = MockTOFSampler(n_channels=4)
        mock.sample = lambda pos, n_ph, t_step: (
            torch.zeros(0, device=DEVICE),
            torch.zeros(0, device=DEVICE, dtype=torch.long),
            torch.zeros(0, device=DEVICE, dtype=torch.long),
        )
        sim = OpticalSimulator(OpticalSimConfig(
            tof_sampler=mock, delays=Delays([]),
            kernel=RLCKernel(duration_ns=100.0, device=DEVICE),
            device="cpu", tick_ns=1.0, gain=-1.0,
            baseline_noise_std=2.6,
        ))
        result = sim.simulate(
            torch.zeros(1, 3), torch.zeros(1, dtype=torch.int32), torch.zeros(1),
            stitched=False,
        )
        # convolution of empty histogram is ~zero, noise dominates
        measured_std = result.adc.std().item()
        assert abs(measured_std - 2.6) < 0.5, f"measured std {measured_std:.2f}, expected ~2.6"

    def test_noise_in_pipeline_with_digitize(self):
        """Baseline noise + digitization pipeline runs correctly."""
        sim = OpticalSimulator(OpticalSimConfig(
            tof_sampler=MockTOFSampler(n_channels=4),
            delays=Delays([]),
            kernel=RLCKernel(duration_ns=500.0, device=DEVICE),
            device="cpu", tick_ns=1.0, gain=-20.0,
            baseline_noise_std=2.6,
            digitization=DigitizationConfig(n_bits=14, pedestal=1500.0),
        ))
        result = sim.simulate(
            torch.zeros(10, 3), torch.full((10,), 100), torch.zeros(10),
            stitched=False,
        )
        assert isinstance(result, Waveform)
        assert torch.equal(result.adc, result.adc.round())
        assert result.adc.min().item() >= 0
        assert result.adc.max().item() <= 16383

    def test_noise_sliced(self):
        """Baseline noise works with stitched output."""
        sim = OpticalSimulator(OpticalSimConfig(
            tof_sampler=MockTOFSampler(n_channels=4),
            delays=Delays([]),
            kernel=RLCKernel(duration_ns=500.0, device=DEVICE),
            device="cpu", tick_ns=1.0, gain=-20.0,
            baseline_noise_std=2.6,
        ))
        result = sim.simulate(
            torch.zeros(10, 3), torch.full((10,), 100), torch.zeros(10),
            stitched=True,
        )
        assert isinstance(result, SlicedWaveform)
