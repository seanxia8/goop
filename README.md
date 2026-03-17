# GOOP: GPU-Optimized Optical Propagation

GPU-accelerated optical simulation for LArTPCs. Simulates photon transport, stochastic timing delays, and electronics response to produce per-PMT waveforms from energy deposit steps.

## Pipeline

```
edepsim steps (position, dE, t)
    |
    v
1. Photon yield:  dE * light_yield -> n_photons per step
2. TOF sampling:  photon library lookup + pmt QE (+ conversion efficiency in future?) -> arrival times & PMT channels
3. Delays:        scintillation + TPB re-emission + PMT TTS jitter
4. Histogramming: bin photon times into per-channel waveforms
5. Convolution:   FFT convolve with detector impulse response (SER kernel)
    |
    v
SlicedWaveform (compressed) or Waveform (full)
```

## Quickstart

```python
import torch
from goop import (
    OpticalSimConfig, OpticalSimulator,
    SERKernel, ScintillationBiexponentialDelay, TPBExponentialDelay, TTSDelay,
)
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
    tick_ns=1.0,
)
sim = OpticalSimulator(config)

# simulate (inputs are (N,3) positions in mm, (N,) photon counts, (N,) times in ns)
# accepts torch tensors or JAX arrays (auto-converted via dlpack)
result = sim.simulate(pos, n_photons, t_step, stitched=True)
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
result.data      # (n_channels, n_bins) tensor
result.t0        # time origin (ns)
result.tick_ns   # time bin width (ns)
```

Note that waveform ADC values are currently stored as **float32** and are **not digitized** (no ADC quantization, no noise floor, no saturation). The output is the raw analog response of the detector model. Digitization and realistic noise injection are TODO.

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

Custom components just need to implement `__call__` and can be passed directly to `OpticalSimConfig`.

## Worked example

See [`notebooks/minimal_example.ipynb`](notebooks/minimal_example.ipynb) for a complete walkthrough from edepsim loading through final waveform visualization.

## Package structure

```
goop/
    simulator.py      OpticalSimConfig, OpticalSimulator
    waveform.py       Waveform, SlicedWaveform
    waveform_utils.py helper functions (slicing, FFT utilities)
    kernels.py        RLCKernel, SERKernel (impulse response models)
    delays.py         ScintillationBiexponentialDelay, TPBExponentialDelay, TTSDelay
    sampler.py        TOFSampler (PCA-compressed photon library)
    base.py           abstract base classes
```

## Tests

```bash
python -m pytest test_simulator.py -v
```

## TODO

1. digitization of ADC
2. saving mechanism for PMT waveforms
3. benchmarking for speed/mem. usage