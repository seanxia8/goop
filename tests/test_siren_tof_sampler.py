"""Verification for ``SirenTOFSampler``.

Compares ``DifferentiableOpticalSimulator`` outputs between the default voxel-LUT
sampler and the SIREN-backed sampler, then confirms gradients flow through the
SIREN path all the way back to the input positions.

Run: ``python test_siren_tof_sampler.py``  (under ``mamba activate py310_torch``).
"""

from __future__ import annotations

import sys
import traceback

import torch
import torch.nn.functional as F

from goop import (
    DifferentiableOpticalSimulator,
    OpticalSimConfig,
    Response,
    SERKernel,
    create_default_tof_sampler,
    create_siren_tof_sampler,
)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[test] device = {DEVICE}")
torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _make_kernel():
    ser = SERKernel(duration_ns=2000.0, device=DEVICE)
    return Response(kernels=[ser], tick_ns=1.0, device=DEVICE)


def build_simulators():
    lut_cfg = OpticalSimConfig(
        tof_sampler=create_default_tof_sampler(device=str(DEVICE)),
        kernel=_make_kernel(),
        device=str(DEVICE),
        tick_ns=1.0,
    )
    siren_cfg = OpticalSimConfig(
        tof_sampler=create_siren_tof_sampler(device=str(DEVICE)),
        kernel=_make_kernel(),
        device=str(DEVICE),
        tick_ns=1.0,
    )
    return (
        DifferentiableOpticalSimulator(lut_cfg),
        DifferentiableOpticalSimulator(siren_cfg),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def align_and_pad(a, ta, b, tb, tick_ns):
    """Shift two (n_ch, n_bins) arrays to a common t0 and pad to common length."""
    common_t0 = min(ta, tb)
    shift_a = int(round((ta - common_t0) / tick_ns))
    shift_b = int(round((tb - common_t0) / tick_ns))
    A = F.pad(a, (shift_a, 0))
    B = F.pad(b, (shift_b, 0))
    n = max(A.shape[1], B.shape[1])
    A = F.pad(A, (0, n - A.shape[1]))
    B = F.pad(B, (0, n - B.shape[1]))
    return A, B


def per_channel_stats(adc_lut, adc_siren):
    """Return (correlations, energy_ratios) for channels where both have signal."""
    corrs = []
    energy_ratios = []
    for ch in range(adc_lut.shape[0]):
        a = adc_lut[ch].float()
        b = adc_siren[ch].float()
        ea = a.pow(2).sum().item()
        eb = b.pow(2).sum().item()
        # skip channels that are nearly empty in both
        if ea < 1e-6 and eb < 1e-6:
            continue
        a_c = a - a.mean()
        b_c = b - b.mean()
        denom = (a_c.norm() * b_c.norm()).item() + 1e-12
        r = (a_c * b_c).sum().item() / denom
        corrs.append(r)
        if ea > 0 and eb > 0:
            energy_ratios.append(eb / ea)
    return corrs, energy_ratios


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_channel_count(sim_siren):
    assert sim_siren.config.tof_sampler.n_channels == 162, (
        f"expected 162 channels, got {sim_siren.config.tof_sampler.n_channels}"
    )
    print("[1/3] n_channels == 162  ✓")


def check_waveform_similarity(sim_lut, sim_siren):
    # Positions straddling x=0 to exercise the half-detector mirror
    pos = torch.tensor(
        [
            [-1500.0, 500.0, 100.0],
            [-500.0, -800.0, -1200.0],
            [-1000.0, 100.0, 900.0],
            [600.0, 0.0, 0.0],
            [1500.0, -300.0, 800.0],
        ],
        device=DEVICE,
    )
    n_ph = torch.full((pos.shape[0],), 5e5, device=DEVICE)
    t_step = torch.zeros(pos.shape[0], device=DEVICE)

    with torch.no_grad():
        wf_lut = sim_lut.simulate(pos, n_ph, t_step).deslice()
        wf_siren = sim_siren.simulate(pos, n_ph, t_step).deslice()

    assert wf_lut.tick_ns == wf_siren.tick_ns, (
        f"tick_ns mismatch: lut={wf_lut.tick_ns} vs siren={wf_siren.tick_ns}"
    )
    A, B = align_and_pad(wf_lut.adc, wf_lut.t0, wf_siren.adc, wf_siren.t0, wf_lut.tick_ns)

    corrs, eratios = per_channel_stats(A, B)
    if not corrs:
        raise RuntimeError("no active channels found — check simulator settings")
    c = torch.tensor(corrs)
    e = torch.tensor(eratios)
    print(
        f"[2/3] {len(corrs)} active PMTs | "
        f"corr: mean={c.mean():.3f}, median={c.median():.3f}, >0.7 frac={(c > 0.7).float().mean():.2f} | "
        f"E_siren/E_lut: median={e.median():.3f} ({e.min():.2f}–{e.max():.2f})"
    )
    # Similarity thresholds: SIREN is a ~%-level approximation of the LUT.
    assert c.mean() > 0.5, f"mean per-PMT corr = {c.mean():.3f} (expected > 0.5)"
    assert (c > 0.7).float().mean() > 0.5, (
        f"only {(c > 0.7).float().mean():.2f} of active PMTs exceed corr=0.7"
    )
    assert e.median() > 0.3 and e.median() < 3.0, (
        f"median energy ratio {e.median():.3f} is far from 1"
    )
    print("[2/3] waveform similarity  ✓")


def check_gradient_flow(sim_siren):
    pos = torch.tensor(
        [[-1500.0, 500.0, 100.0], [-800.0, -200.0, 300.0]],
        device=DEVICE,
        requires_grad=True,
    )
    n_ph = torch.full((pos.shape[0],), 5e5, device=DEVICE)
    t_step = torch.zeros(pos.shape[0], device=DEVICE)

    wf = sim_siren.simulate(pos, n_ph, t_step)
    loss = wf.adc.float().pow(2).sum()
    loss.backward()
    assert pos.grad is not None, "pos.grad is None — autograd graph broken"
    assert torch.isfinite(pos.grad).all(), f"non-finite gradient: {pos.grad}"
    gnorm = pos.grad.norm().item()
    assert gnorm > 0, f"zero gradient norm: {gnorm}"
    print(f"[3/3] grad flow OK | ||pos.grad|| = {gnorm:.3e}  ✓")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("[test] building simulators (LUT + SIREN) ...")
    sim_lut, sim_siren = build_simulators()

    check_channel_count(sim_siren)
    check_waveform_similarity(sim_lut, sim_siren)
    check_gradient_flow(sim_siren)

    print("\nSirenTOFSampler verification PASSED")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
