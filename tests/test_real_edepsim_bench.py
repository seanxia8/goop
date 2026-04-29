"""Performance benchmarks on the real-edepsim event with the SIREN sampler.

Companion to ``test_real_edepsim.py`` — same fixtures, same skip gate, but
times the matrix instead of asserting numeric correctness. Not run by default
(use ``--benchmark-only``) so the normal dev loop stays fast.

Workflow
--------
Save a baseline once on a known-good commit::

    pytest tests/test_real_edepsim_bench.py --benchmark-only --benchmark-save=main

Compare a feature branch against that baseline and fail on >25 % median
regression on any benchmark::

    pytest tests/test_real_edepsim_bench.py --benchmark-only \\
           --benchmark-compare=main --benchmark-compare-fail=median:25%

Baselines are written under ``.benchmarks/`` (gitignored — they're
machine-specific because they depend on GPU model + SIREN ckpt path).

Notes
-----
- ``benchmark.pedantic(...)`` with ``iterations=1, rounds=5, warmup_rounds=1``
  because each call is 0.1–2 s; the default many-iteration mode would balloon
  runtime without improving statistics.
- Each timed callable ends with ``torch.cuda.synchronize()`` so the wall-clock
  measurement reflects actual GPU work, not async-launch latency.
"""

from __future__ import annotations

import torch

# Reuse the asset-skip + fixtures from the functional suite. pytest discovers
# ``pytestmark`` from the imported module's namespace so the same skip logic
# applies here without duplication.
from test_real_edepsim import (  # noqa: F401 — fixtures used via pytest injection
    pytestmark,
    device,
    edepsim_event,
    siren_sampler,
    stoch_sim,
    diff_sim,
)


_PEDANTIC = dict(iterations=1, rounds=5, warmup_rounds=1)


def _sync_call(fn, device):
    """Wrap ``fn`` so the timer captures actual GPU completion."""
    def go():
        out = fn()
        torch.cuda.synchronize(device)
        return out
    return go


# ---------------------------------------------------------------------------
# Forward-only matrix
# ---------------------------------------------------------------------------


def test_bench_stoch_sliced_fwd(benchmark, stoch_sim, edepsim_event, device):
    pos, nph, t = edepsim_event
    def call():
        torch.manual_seed(0)
        return stoch_sim.simulate(
            pos, nph, t, stitched=True, add_baseline_noise=False,
        )
    benchmark.pedantic(_sync_call(call, device), **_PEDANTIC)


def test_bench_stoch_dense_fwd(benchmark, stoch_sim, edepsim_event, device):
    pos, nph, t = edepsim_event
    def call():
        torch.manual_seed(0)
        return stoch_sim.simulate(
            pos, nph, t, stitched=False, add_baseline_noise=False,
        )
    benchmark.pedantic(_sync_call(call, device), **_PEDANTIC)


def test_bench_diff_sliced_fwd(benchmark, diff_sim, edepsim_event, device):
    pos, nph, t = edepsim_event
    def call():
        with torch.no_grad():
            return diff_sim.simulate(
                pos, nph, t, stitched=True, add_baseline_noise=False,
            )
    benchmark.pedantic(_sync_call(call, device), **_PEDANTIC)


def test_bench_diff_dense_fwd(benchmark, diff_sim, edepsim_event, device):
    pos, nph, t = edepsim_event
    def call():
        with torch.no_grad():
            return diff_sim.simulate(
                pos, nph, t, stitched=False, add_baseline_noise=False,
            )
    benchmark.pedantic(_sync_call(call, device), **_PEDANTIC)


# ---------------------------------------------------------------------------
# Forward + backward (diff sim only — stoch has no gradients)
# ---------------------------------------------------------------------------


def test_bench_diff_sliced_fwd_bwd(benchmark, diff_sim, edepsim_event, device):
    pos, nph, t = edepsim_event
    def call():
        nph_g = nph.detach().clone().requires_grad_(True)
        sw = diff_sim.simulate(
            pos, nph_g, t, stitched=True, add_baseline_noise=False,
        )
        sw.adc.pow(2).sum().backward()
        return nph_g.grad
    benchmark.pedantic(_sync_call(call, device), **_PEDANTIC)


def test_bench_diff_dense_fwd_bwd(benchmark, diff_sim, edepsim_event, device):
    pos, nph, t = edepsim_event
    def call():
        nph_g = nph.detach().clone().requires_grad_(True)
        wf = diff_sim.simulate(
            pos, nph_g, t, stitched=False, add_baseline_noise=False,
        )
        wf.adc.pow(2).sum().backward()
        return nph_g.grad
    benchmark.pedantic(_sync_call(call, device), **_PEDANTIC)
