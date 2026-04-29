"""End-to-end tests on a real edepsim event with the production light-yield
and the SIREN photon-transport sampler.

The fixture is event 0 from the standard sirentv test file. Per-segment photon
counts come from `DetectorSimulator.process_event_light` (Birks recombination +
LAr scintillation yield), positions are global mm coordinates, and times are
in ns. The event is voxelized at 20 mm to ~12 k segments, ~371 M photons,
with a ~3.3 ms time span (cosmic-ray afterglow tail).

Tests run on GPU and are skipped when CUDA, the photon library, or the SIREN
checkpoint paths are unavailable. The matrix and gradient tests guard against
regressions in the full pipeline (memory, basic numeric sanity); the
stoch-vs-diff equivalence test verifies that averaging the stochastic sim
across throws converges to the differentiable expectation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from goop import (
    OpticalSimConfig,
    OpticalSimulator,
    Response,
    ScintillationKernel,
    SERKernel,
    SlicedWaveform,
    TPBExponentialKernel,
    TTSKernel,
    Waveform,
)
from goop.delays import (
    Delays,
    ScintillationBiexponentialDelay,
    TPBExponentialDelay,
    TTSDelay,
)
from goop.diff_simulator import DifferentiableOpticalSimulator
from goop.sampler.siren import (
    DEFAULT_CFG_PATH,
    DEFAULT_CKPT_PATH,
    DEFAULT_SIRENTV_SRC,
    create_siren_tof_sampler,
)
from goop.sampler.base import DEFAULT_PLIB_PATH


FIXTURE_PATH = Path(__file__).parent / "data" / "edepsim_event0.npz"


def _have_siren_assets() -> bool:
    """SIREN tests need: a CUDA device, the photon library, the ckpt, the cfg,
    the sirentv source tree, and the fixture .npz."""
    return (
        torch.cuda.is_available()
        and Path(DEFAULT_PLIB_PATH).exists()
        and Path(DEFAULT_CKPT_PATH).exists()
        and Path(DEFAULT_CFG_PATH).exists()
        and Path(DEFAULT_SIRENTV_SRC).exists()
        and FIXTURE_PATH.exists()
    )


pytestmark = pytest.mark.skipif(
    not _have_siren_assets(),
    reason="real-edepsim suite requires CUDA + SIREN ckpt + plib + fixture",
)


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda")


@pytest.fixture(scope="module")
def edepsim_event(device):
    """Load the real fixture: ~12 k segments, ~371 M photons, 3.3 ms span."""
    data = np.load(FIXTURE_PATH)
    pos = torch.from_numpy(data["pos"]).to(device, torch.float32)
    nph = torch.from_numpy(data["n_photons"]).to(device, torch.float32)
    t = torch.from_numpy(data["t_step"]).to(device, torch.float32)
    return pos, nph, t


@pytest.fixture(scope="module")
def siren_sampler(device):
    return create_siren_tof_sampler(
        device=str(device), autocast_dtype=torch.bfloat16, use_checkpoint=True,
    )


@pytest.fixture(scope="module")
def stoch_sim(siren_sampler, device):
    """Production-style stochastic sim: per-photon delay chain + SER kernel."""
    cfg = OpticalSimConfig(
        tof_sampler=siren_sampler,
        delays=Delays([
            ScintillationBiexponentialDelay(),
            TPBExponentialDelay(),
            TTSDelay(),
        ]),
        kernel=SERKernel(duration_ns=2000.0, tick_ns=1.0, device=device),
        device=str(device), tick_ns=1.0, gain=-45.0,
    )
    return OpticalSimulator(cfg)


@pytest.fixture(scope="module")
def diff_sim(siren_sampler, device):
    """Production-style diff sim: full delay chain folded into the kernel."""
    cfg = OpticalSimConfig(
        tof_sampler=siren_sampler,
        delays=Delays([]),
        kernel=Response(
            kernels=[
                ScintillationKernel(device=device),
                TPBExponentialKernel(device=device),
                TTSKernel(device=device),
                SERKernel(duration_ns=2000.0, device=device),
            ],
            tick_ns=1.0, device=device,
        ),
        device=str(device), tick_ns=1.0, gain=-45.0,
        stream_checkpoint=False,  # ~12 k segments split across ~9 time-groups
    )
    return DifferentiableOpticalSimulator(cfg)


# ---------------------------------------------------------------------------
# 1. Matrix: each (simulator, output-shape) combo must run end-to-end on a
#    full real event without OOM, and produce non-trivial output.
# ---------------------------------------------------------------------------


class TestRealEventMatrix:
    """All four (stoch/diff) × (sliced/dense) combos run end-to-end at full
    physical photon count. Each test resets peak memory and asserts the run
    finished and produced signal."""

    @staticmethod
    def _check(wf, n_channels):
        assert wf.n_channels == n_channels
        assert wf.adc.abs().sum().item() > 0
        assert torch.isfinite(wf.adc).all()

    def test_stoch_sliced(self, stoch_sim, edepsim_event, device):
        torch.cuda.reset_peak_memory_stats(device)
        pos, nph, t = edepsim_event
        torch.manual_seed(0)
        sw = stoch_sim.simulate(pos, nph, t, stitched=True, add_baseline_noise=False)
        assert isinstance(sw, SlicedWaveform)
        assert sw.n_chunks > 0
        self._check(sw, stoch_sim.config.n_channels)

    def test_stoch_dense(self, stoch_sim, edepsim_event, device):
        torch.cuda.reset_peak_memory_stats(device)
        pos, nph, t = edepsim_event
        torch.manual_seed(0)
        wf = stoch_sim.simulate(pos, nph, t, stitched=False, add_baseline_noise=False)
        assert isinstance(wf, Waveform)
        self._check(wf, stoch_sim.config.n_channels)

    def test_diff_sliced(self, diff_sim, edepsim_event, device):
        torch.cuda.reset_peak_memory_stats(device)
        pos, nph, t = edepsim_event
        sw = diff_sim.simulate(pos, nph, t, stitched=True, add_baseline_noise=False)
        assert isinstance(sw, SlicedWaveform)
        assert sw.n_chunks > 0
        self._check(sw, diff_sim.config.n_channels)

    def test_diff_dense(self, diff_sim, edepsim_event, device):
        torch.cuda.reset_peak_memory_stats(device)
        pos, nph, t = edepsim_event
        wf = diff_sim.simulate(pos, nph, t, stitched=False, add_baseline_noise=False)
        assert isinstance(wf, Waveform)
        self._check(wf, diff_sim.config.n_channels)


# ---------------------------------------------------------------------------
# 2. Backward through the full pipeline must reach `n_photons` and `pos`.
# ---------------------------------------------------------------------------


class TestRealEventBackward:
    def test_diff_grad_through_n_photons(self, diff_sim, edepsim_event, device):
        torch.cuda.reset_peak_memory_stats(device)
        pos, nph, t = edepsim_event
        nph_g = nph.detach().clone().requires_grad_(True)
        sw = diff_sim.simulate(pos, nph_g, t, stitched=True, add_baseline_noise=False)
        sw.adc.pow(2).sum().backward()
        assert nph_g.grad is not None
        assert torch.isfinite(nph_g.grad).all()
        assert nph_g.grad.abs().max().item() > 0

    def test_diff_grad_through_pos(self, diff_sim, edepsim_event, device):
        torch.cuda.reset_peak_memory_stats(device)
        pos, nph, t = edepsim_event
        pos_g = pos.detach().clone().requires_grad_(True)
        sw = diff_sim.simulate(pos_g, nph, t, stitched=True, add_baseline_noise=False)
        sw.adc.pow(2).sum().backward()
        assert pos_g.grad is not None
        assert torch.isfinite(pos_g.grad).all()
        assert pos_g.grad.abs().max().item() > 0

    def test_align_grad_propagates_real_event(self, diff_sim, edepsim_event, device):
        """`SlicedWaveform.align()` must preserve the autograd chain on the
        production diff path: loss on the *aligned* waveform should backprop
        all the way to ``n_photons``."""
        torch.cuda.reset_peak_memory_stats(device)
        pos, nph, t = edepsim_event
        nph_g = nph.detach().clone().requires_grad_(True)
        sw = diff_sim.simulate(pos, nph_g, t, stitched=True, add_baseline_noise=False)
        aligned = sw.align(fill=0.0)
        assert aligned.adc.requires_grad, "align must keep adc on the autograd graph"
        aligned.adc.pow(2).sum().backward()
        assert nph_g.grad is not None
        assert torch.isfinite(nph_g.grad).all()
        assert nph_g.grad.abs().max().item() > 0


# ---------------------------------------------------------------------------
# 3. Statistical equivalence: averaging stoch over throws ≈ diff expectation.
# ---------------------------------------------------------------------------


class TestStochDiffEquivalence:
    """Diff deposits expected PDFs; stoch samples discrete photons + delays.
    Averaging stoch over a few throws should converge to diff in expectation.

    With ~371 M photons the *summed-across-channels* total yield is constrained
    very tightly (channel-categorical noise cancels), so 3 throws is enough.
    """

    def test_total_yield_matches(self, stoch_sim, diff_sim, edepsim_event, device):
        pos, nph, t = edepsim_event
        n_throws = 3

        wf_d = diff_sim.simulate(
            pos, nph, t, stitched=False, add_baseline_noise=False,
        )
        diff_total = wf_d.adc.sum(dim=1)        # (n_channels,)

        stoch_totals = torch.zeros_like(diff_total)
        for seed in range(n_throws):
            torch.manual_seed(seed)
            wf_s = stoch_sim.simulate(
                pos, nph, t, stitched=False, add_baseline_noise=False,
            )
            stoch_totals = stoch_totals + wf_s.adc.sum(dim=1)
        stoch_total = stoch_totals / n_throws

        rel_total = (
            (stoch_total.sum() - diff_total.sum()).abs()
            / diff_total.sum().abs()
        ).item()
        assert rel_total < 0.05, (
            f"averaged stoch *total* yield diverges from diff by "
            f"rel_total={rel_total:.4f} (n_throws={n_throws})"
        )
