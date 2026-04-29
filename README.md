# GOOP: GPU-Optimized Optical Propagation

GPU-accelerated optical simulation for LArTPCs. Simulates photon transport, stochastic timing delays, and electronics response to produce per-PMT waveforms from energy deposit steps.

## Pipeline

```
edepsim steps (position, dE, t)
    |
    v
1. Photon yield:  dE * light_yield -> n_photons per step
2. TOF sampling:  photon library lookup + pmt QE -> arrival times & PMT channels
3. Delays:        scintillation + TPB re-emission + PMT TTS jitter
4. Aux sources:   dark noise injection (+ afterpulsing, crosstalk, etc.)
5. Histogramming: bin photon times into per-channel waveforms (with optional SER jitter)
6. Convolution:   FFT convolve with detector impulse response (SER kernel)
7. Downsample:    average oversampled fine bins to output resolution (if oversample > 1)
8. Baseline noise: per-sample Gaussian ADC noise (optional)
9. Digitization:  pedestal + quantization + ADC saturation (optional)
    |
    v
SlicedWaveform (compressed) or Waveform (full)
```

## Quickstart

```python
import torch
from goop import (
    OpticalSimConfig, OpticalSimulator,
)
from goop.kernels import SERKernel
from goop.delays import ScintillationBiexponentialDelay, TPBExponentialDelay, TTSDelay
from goop.sampler import create_default_tof_sampler

# configure
config = OpticalSimConfig(
    tof_sampler=create_default_tof_sampler(lazy=True),
    delays=[
        ScintillationBiexponentialDelay(),
        TPBExponentialDelay(tau_ns=20.0),
        TTSDelay(fwhm_ns=1.0),
    ],
    kernel=SERKernel(device=torch.device("cuda")),
    gain=1.0,
    tick_ns=1.0,        # output bin width (ns)
    oversample=10,      # convolve at 0.1 ns internally, downsample to 1 ns output
)
sim = OpticalSimulator(config)

# simulate (inputs are (N,3) positions in mm, (N,) photon counts, (N,) times in ns)
# accepts torch tensors or JAX arrays (auto-converted via dlpack)
result = sim.simulate(pos, n_photons, t_step)
result.attrs["pe_counts"]  # (n_channels,) detected PE per PMT
```

### Differentiable mode

The same `OpticalSimulator` runs in expectation mode when called with
`stochastic=False`: per-photon Poisson sampling becomes deterministic PDF
deposition, per-photon delay sampling becomes a `Response` composite-kernel
convolution (Scint ⊛ TPB ⊛ TTS ⊛ SER), and ADC quantization is wrapped in
a straight-through estimator. Gradients flow from `pos`, `n_photons` all
the way through to `sw.adc` and `sw.attrs["pe_counts"]`.

```python
from goop import (
    OpticalSimulator, OpticalSimConfig,
    Response, ScintillationKernel, TPBExponentialKernel, TTSKernel, SERKernel,
    create_default_tof_sampler,
)
from goop.delays import Delays

device = torch.device("cuda")

config = OpticalSimConfig(
    tof_sampler=create_default_tof_sampler(device=str(device)),
    delays=Delays([]),  # delays are now in the kernel
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
)
sim = OpticalSimulator(config)

# pos / n_photons / t_step can carry requires_grad=True
n_photons = torch.full((n_pos,), 10_000.0, device=device, requires_grad=True)
sw = sim.simulate(pos, n_photons, t_step, stochastic=False)

loss = sw.adc.pow(2).sum()
loss.backward()           # n_photons.grad now populated
```

### TOF samplers: voxel LUT vs SIREN

Every PCA-compressed TOF sampler inherits from `PCATOFSampler` (quantile-time reconstruction; Poisson + inverse-CDF in stochastic mode, deterministic PDF deposition in `stochastic=False` mode — the dispatch is on the `stochastic=` kwarg of `sample_photons` / `sample_histogram`). Two concrete backends ship in `goop.sampler`:

- **`TOFSampler`** (voxel LUT) — trilinear interpolation of `(vis, t0, coeffs)` from the compressed photon library. Fast on GPU but memory-heavy (the full LUT is ≈24 GB at the default 1.6 M voxels × 81 PMTs × 50 coefficients).
- **`SirenTOFSampler`** — the LUT lookup is replaced by a pre-trained SIREN network (`sirentv.models.pca_siren.PcaSiren`) that predicts `(vis, log_t0, coeffs)` in the *same* PCA basis. Keeps the entire lookup differentiable end-to-end (gradients flow through the network back to input positions), with essentially constant memory and wall-clock cost similar to the LUT in practice.

Both factories default to the standard photon library at `/sdf/data/neutrino/youngsam/compressed_plib_b04_quantile_log_n50.h5`:

```python
from goop import create_default_tof_sampler, create_siren_tof_sampler

# LUT — trilinear-interp lookup from the compressed plib
lut = create_default_tof_sampler(device="cuda:0")

# SIREN — neural lookup; ckpt/cfg/plib defaults point at version-67 epoch-2000
siren = create_siren_tof_sampler(device="cuda:0")

# Swap either into the simulator config
config = OpticalSimConfig(tof_sampler=siren, ...)
```

### Input voxelization

The diff-sim cost scales linearly with the number of input segments N. `voxelize` bins nearby segments into cubic voxels, summing photon counts and photon-weighting positions/times. Total photon yield is exactly preserved.

```python
from goop import voxelize

# 10 mm voxels: reduces with no detectable waveform change
pos_v, nph_v, tns_v = voxelize(pos_mm, n_photons, t_step_ns, dx=10.0)

# Then simulate on the reduced arrays
sw = sim.simulate(torch.from_numpy(pos_v).to(device), ...)
```

### Memory-efficient diff-sim setup

By default the differentiable simulator builds the full event histogram in one pass, which can exceed GPU memory for large events (~100 k segments and up). Here are a few options you can try:

1. **`autocast_dtype=torch.bfloat16`** on `create_siren_tof_sampler` — runs the SIREN network in bf16. Halves activation memory at no measurable accuracy cost.

2. **`use_checkpoint=True`** on the SIREN sampler — wraps the network in `torch.utils.checkpoint`, recomputing hidden activations during backward instead of storing them. Trades ~15 % wall-time for ~6× lower SIREN-side memory.

```python
from goop import (
    OpticalSimConfig, DifferentiableOpticalSimulator, Optical 
    Response, SERKernel, create_siren_tof_sampler, voxelize,
)
from goop.delays import Delays

# 1. Voxelize the raw input cloud (host-side, one-time).
pos_v, nph_v, tns_v = voxelize(pos, n_photons, t_step, dx=20.0)

# 2. SIREN sampler: bf16 + per-layer checkpoint.
sampler = create_siren_tof_sampler(
    device="cuda",
    autocast_dtype=torch.bfloat16,
    use_checkpoint=True,
)

# 3. Per-chunk checkpointing off. Streaming is on by default.
cfg = OpticalSimConfig(
    tof_sampler=sampler,
    delays=Delays([]),
    kernel=Response(kernels=[SERKernel(duration_ns=2000.0, device="cuda")],
                    tick_ns=1.0, device="cuda"),
    device="cuda", tick_ns=1.0, gain=-45.0,
    pos_batch_size=5_000, checkpoint=False,
)
sim = OpticalSimulator(cfg)
# call with stochastic=False to engage the differentiable path:
# sw = sim.simulate(pos_v, nph_v, tns_v, stochastic=False)
```

This is the recommended setup for Adam fits on full events.

### Per-label waveforms

Split output by any per-position grouping (detector volume, interaction ID, etc.). TOF sampling and delays are batched; the rest of the pipeline runs in a single call via virtual-channel remapping, then the combined waveform is split by label.

```python
labels = torch.tensor([0, 0, 0, 1, 1])  # one label per position
waveforms = sim.simulate(pos, n_photons, t_step, labels=labels)
# returns list[SlicedWaveform], one per unique label value
```

### With digitization and dark noise

```python
from goop.noise import DarkNoise
from goop.digitize import DigitizationConfig

config = OpticalSimConfig(
    tof_sampler=create_default_tof_sampler(lazy=True),
    delays=[
        ScintillationBiexponentialDelay(),
        TPBExponentialDelay(tau_ns=20.0),
        TTSDelay(fwhm_ns=2.4),
    ],
    kernel=SERKernel(device=torch.device("cuda")),
    tick_ns=2.0,               # 500 MHz output
    gain=-20.0,                # ADC counts per PE
    oversample=4,              # convolve at 0.5 ns internally
    ser_jitter_std=0.1,        # 10% PE-to-PE gain fluctuation
    baseline_noise_std=2.6,    # ADC baseline noise (std in ADC counts)
    aux_photon_sources=[DarkNoise(rate_hz=2000.0)],
    digitization=DigitizationConfig(n_bits=14, pedestal=1500.0),
)
result = OpticalSimulator(config).simulate(pos, n_photons, t_step)
# result.adc values are now integers in [0, 16383]
```

### Output types

**`SlicedWaveform`** (`sliced=True`, default) — compressed CSR format, only stores active regions:
```python
result.adc       # (total_bins,)  all chunks concatenated
result.offsets   # (n_chunks+1,)  CSR boundaries: chunk k = adc[offsets[k]:offsets[k+1]]
result.t0_ns     # (n_chunks,)    real-time start per chunk (ns)
result.pmt_id    # (n_chunks,)    PMT channel per chunk
result.tick_ns   # time bin width (ns)
result.n_chunks  # number of chunks

result.chunk(k)            # get ADC data for chunk k
result.deslice()           # decompress to full Waveform
result.deslice_channel(ch) # decompress one PMT -> (t0, 1D tensor)
```

**`Waveform`** (`sliced=False`) — full shared time axis:
```python
result.adc       # (n_channels, n_bins) tensor
result.t0        # time origin (ns)
result.tick_ns   # time bin width (ns)
result.attrs     # dict of metadata (e.g. result.attrs["pe_counts"])
```

Both types carry an `attrs` dict for arbitrary metadata. The simulator populates `attrs["pe_counts"]` with per-channel detected PE counts.

Without digitization enabled, waveform values are **float32** (raw analog response). With `DigitizationConfig`, output values are integer-valued float32 in `[0, 2^n_bits - 1]`.

## Saving & loading waveforms

`goop.io` provides HDF5 I/O that saves per-volume waveforms (one `SlicedWaveform` per detector volume).

### Saving

```python
import h5py
from goop import OpticalSimConfig, OpticalSimulator, write_config_light, save_event_light

config = OpticalSimConfig(...)
sim = OpticalSimulator(config)

with h5py.File("light_output.h5", "w") as f:
    write_config_light(f, config, label_key="volume", n_labels=2,
                       source_file="input.h5", n_events=100)

    for i in range(100):
        waveforms = sim.simulate(pos, n_photons, t_step,
                                 labels=volume_labels,
                                 add_baseline_noise=False)  # noise-free for compression
        save_event_light(f, f"event_{i:03d}", waveforms,
                         source_event_idx=i,
                         digitized=True,   # store adc as uint16
                         n_bits=14)        # scaleoffset for compression
```

### Loading

```python
from goop import load_event_light

with h5py.File("light_output.h5", "r") as f:
    wfs = load_event_light(f, "event_000", device="cuda")
    # returns list[SlicedWaveform], one per label

    for wf in wfs:
        wf.attrs["pe_counts"]     # (n_channels,) PE counts per PMT
        wf.deslice()              # decompress to full (n_channels, n_bins) Waveform
        wf.deslice_channel(42)    # single PMT -> (t0, 1D tensor)
```

### HDF5 layout

```
/config/                     file-level metadata (tick_ns, gain, label_key, n_labels, etc.)
/event_000/
    attrs: source_event_idx, n_labels
    label_0/
        adc        (N,) uint16 or float32   # gzip + shuffle
        offsets    (K+1,) int64             # CSR chunk boundaries
        t0_ns      (K,) float32             # chunk time origins
        pmt_id     (K,) int32               # global PMT index
        pe_counts  (n_channels,) int32      # PE per PMT from label 0
    label_1/
        (same structure)
```

When `digitized=True`, adc is stored as `uint16` with gzip+shuffle for optimal compression. Otherwise stored as `float32`.

The TOF sampler also returns a per-photon `source_idx` mapping each detected photon back to its input position, enabling the label-based splitting described above.

## Extensibility

Each pipeline component is defined by an abstract base class (`base.py`), making it easy to swap in new physics models.

**Delay samplers** (subclass `DelaySamplerBase`):
| Class | Model |
|-------|-------|
| `ScintillationBiexponentialDelay` | Bi-exponential (singlet + triplet) LAr scintillation timing |
| `TPBExponentialDelay` | Exponential TPB wavelength-shifter re-emission |
| `TTSDelay` | Gaussian PMT transit-time spread |

**Convolution kernels** (subclass `ConvolutionKernelBase`):
| Class | Model |
|-------|-------|
| `SERKernel` | PMT single-electron response (bi-exponential pulse + AC-coupled overshoot) |
| `RLCKernel` | Damped-oscillator (RLC circuit) impulse response |

**Auxiliary photon sources** (subclass `PhotonSourceBase`):
| Class | Model |
|-------|-------|
| `DarkNoise` | Poisson dark-count noise, uniform in time per channel |

**Digitization** (`DigitizationConfig`):
| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_bits` | 14 | ADC bit depth (14-bit → [0, 16383]) |
| `pedestal` | 1500.0 | Baseline offset in ADC counts |

**Simulator-level options** (`OpticalSimConfig`):
| Parameter | Default | Description |
|-----------|---------|-------------|
| `oversample` | 1 | Internal oversampling factor; bins and convolves at `tick_ns / oversample`, then averages down |
| `ser_jitter_std` | 0.0 | Std of multiplicative Gaussian on per-PE weights (models dynode gain fluctuations) |
| `baseline_noise_std` | 0.0 | Std of per-sample Gaussian ADC noise added post-convolution |
| `streaming` | True | (Diff-sim only) Time-grouped streaming histogram — peak memory scales with the largest activity burst, not the whole event. Set to `False` only for debugging / `sample_pdf` parity checks. |
| `stream_chunk_size` | 5000 | Segments per gradient-checkpointed micro-batch inside `histogram_pdf` |
| `stream_checkpoint` | True | Per-chunk `torch.utils.checkpoint` in the streaming path; turn off at small N (e.g. after voxelization) |


**`simulate()` options**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `labels` | None | Per-position integer labels; splits output into `list[SlicedWaveform]` via virtual-channel batching |
| `sliced` | True | Return `SlicedWaveform` (True) or dense `Waveform` (False) |
| `stochastic` | True | Poisson + per-photon delay sampling (True) or deterministic PDF deposition + differentiable (False) |
| `subtract_t0` | False | Shift `t_step` so the minimum is zero |
| `add_baseline_noise` | True | Add Gaussian baseline noise; set False when saving to disk for better compression |

Custom components just need to implement the relevant ABC and can be passed directly to `OpticalSimConfig`.

## Worked example

See [`notebooks/minimal_example.ipynb`](notebooks/minimal_example.ipynb) for a complete walkthrough from edepsim loading through final waveform visualization.

## Package structure

```
goop/
    simulator.py       OpticalSimConfig, OpticalSimulator (stochastic + expectation)
    waveform.py        Waveform, SlicedWaveform
    waveform_utils.py  helper functions (slicing, FFT utilities)
    kernels.py         RLCKernel, SERKernel, ScintillationKernel, TPBExponentialKernel,
                       TPBTriexponentialKernel, TTSKernel, Response (composite via FFT)
    delays.py          ScintillationBiexponentialDelay, TPBExponentialDelay, TTSDelay
    noise.py           DarkNoise (auxiliary photon sources)
    digitize.py        DigitizationConfig, digitize, digitize_ste (STE for backprop)
    utils.py           voxelize (input point-cloud coarsening)
    sampler/
        base.py        PCATOFSampler (shared PCA reconstruction + sample / sample_pdf)
        lut.py         TOFSampler, DifferentiableTOFSampler (voxel LUT lookup)
        siren.py       SirenTOFSampler (pre-trained SIREN network lookup)
    io.py              HDF5 save/load for per-volume SlicedWaveforms
    base.py            abstract base classes
```

## Tests

```bash
# default dev loop (skips benchmarks)
pytest tests/ -q

# real-event tests skip automatically without GPU + SIREN assets;
# regenerate the fixture from a fresh edepsim file with
python tests/data/regen_edepsim_event0.py
```

### Performance benchmarks

Speed regressions on the realistic-event matrix are tracked with `pytest-benchmark`. Six benchmarks (`stoch/diff × sliced/dense fwd` + `diff fwd+bwd` × `sliced/dense`) live in `tests/test_real_edepsim_bench.py`. They are skipped by default (`pytest.ini: addopts = --benchmark-skip`).

```bash
# Run benchmarks (~18 s on GPU):
pytest tests/test_real_edepsim_bench.py --benchmark-only

# Save a baseline on a known-good commit:
pytest tests/test_real_edepsim_bench.py --benchmark-only --benchmark-save=main

# Compare a feature branch against the baseline; fail on >25% median regression:
pytest tests/test_real_edepsim_bench.py --benchmark-only \
       --benchmark-compare=main --benchmark-compare-fail=median:25%
```

Baselines are saved under `.benchmarks/` (gitignored — machine-specific).

Reference numbers from a single A100-class GPU on event 0 (12 k voxelized segments, 371 M photons, 3.3 ms span) — useful as a sanity check that your local install is in the right ballpark, not a contract:

| Benchmark | Median |
|---|---:|
| `diff sliced fwd` | 157 ms |
| `diff dense fwd` | 171 ms |
| `diff sliced fwd+bwd` | 194 ms |
| `diff dense fwd+bwd` | 221 ms |
| `stoch sliced fwd` | 671 ms |
| `stoch dense fwd` | 654 ms |

## TODO

