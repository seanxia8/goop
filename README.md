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
result = sim.simulate(pos, n_photons, t_step, stitched=True)
result.attrs["pe_counts"]  # (n_channels,) detected PE per PMT
```

### Differentiable simulator

`DifferentiableOpticalSimulator` runs the same pipeline with all gradient-blocking ops replaced or modified: all per-photon delay sampling (e.g., scintillation / TPB reemission delays) becomes a `Response` composite-kernel convolution (Scint * TPB * TTS * SER), per-photon TOF sampling becomes deterministic PDF deposition (which gives the expectation), and ADC quantization is wrapped in a straight-through estimator (`digitize_ste`). 

Gradients flow from `pos`, `t_step`, `n_photons` alll the way through to `sw.adc` and `sw.attrs["pe_counts"]`.

```python
from goop import (
    DifferentiableOpticalSimulator, DifferentiableTOFSampler,
    OpticalSimConfig, Response, ScintillationKernel,
    TPBExponentialKernel, TTSKernel, SERKernel,
)
from goop.delays import Delays
from goop.sampler import DEFAULT_PLIB_PATH

device = torch.device("cuda")

config = OpticalSimConfig(
    tof_sampler=DifferentiableTOFSampler(DEFAULT_PLIB_PATH, device=str(device)),
    delays=Delays([]),  # delays are now in the kernel — see below
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
sim = DifferentiableOpticalSimulator(config)

# n_photons (and pos/t_step) can carry requires_grad=True
n_photons = torch.full((n_pos,), 10_000.0, device=device, requires_grad=True)
sw = sim.simulate(pos, n_photons, t_step, stitched=True)

loss = sw.adc.pow(2).sum()
loss.backward()           # n_photons.grad now populated
```

### Per-label waveforms

Split output by any per-position grouping (detector volume, interaction ID, etc.). TOF sampling and delays are batched; the rest of the pipeline runs in a single call via virtual-channel remapping, then the combined waveform is split by label.

```python
labels = torch.tensor([0, 0, 0, 1, 1])  # one label per position
waveforms = sim.simulate(pos, n_photons, t_step, labels=labels, stitched=True)
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

**`SlicedWaveform`** (`stitched=True`, default) — compressed CSR format, only stores active regions:
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

**`Waveform`** (`stitched=False`) — full shared time axis:
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
                                 labels=volume_labels, stitched=True,
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

**`simulate()` options**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `labels` | None | Per-position integer labels; splits output into `list[SlicedWaveform]` via virtual-channel batching |
| `stitched` | True | Return `SlicedWaveform` (True) or dense `Waveform` (False) |
| `subtract_t0` | False | Shift `t_step` so the minimum is zero |
| `add_baseline_noise` | True | Add Gaussian baseline noise; set False when saving to disk for better compression |

Custom components just need to implement the relevant ABC and can be passed directly to `OpticalSimConfig`.

## Worked example

See [`notebooks/minimal_example.ipynb`](notebooks/minimal_example.ipynb) for a complete walkthrough from edepsim loading through final waveform visualization.

## Package structure

```
goop/
    simulator.py       OpticalSimConfig, OpticalSimulator
    diff_simulator.py  DifferentiableOpticalSimulator
    waveform.py        Waveform, SlicedWaveform
    waveform_utils.py  helper functions (slicing, FFT utilities)
    kernels.py         RLCKernel, SERKernel, ScintillationKernel, TPBExponentialKernel,
                       TPBTriexponentialKernel, TTSKernel, Response (composite via FFT)
    delays.py          ScintillationBiexponentialDelay, TPBExponentialDelay, TTSDelay
    noise.py           DarkNoise (auxiliary photon sources)
    digitize.py        DigitizationConfig, digitize, digitize_ste (STE for backprop)
    sampler.py         TOFSampler, DifferentiableTOFSampler (PCA-compressed photon library)
    io.py              HDF5 save/load for per-volume SlicedWaveforms
    base.py            abstract base classes
```

## Tests

```bash
python -m pytest tests/ -v
```

## TODO

1. benchmarking for speed/mem. usage