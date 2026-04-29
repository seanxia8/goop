# GOOP Production Pipeline

Batch optical simulation of particle events in a liquid argon TPC: jaxtpc photon generation followed by GOOP waveform production, producing structured HDF5 output with per-PMT SlicedWaveform data.

## Contents

```
production/
├── run_batch.py         # Main batch simulation script
├── load.py              # HDF5 load/decode functions
└── README.md            # This file
```

## Usage

From the project root:

```bash
# Basic run (5 events, digitization on)
python3 production/run_batch.py --data events.h5 --events 5

# Full options
python3 production/run_batch.py \
    --data mpvmpr_20.h5 \
    --config jaxtpc/config/cubic_wireplane_config.yaml \
    --dataset myrun \
    --outdir output/ \
    --events 1000 \
    --events-per-file 100 \
    --n-bits 15 \
    --oversample 10 \
    --dark-noise \
    --baseline-noise-std 2.6
```

### CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--data` | `out.h5` | Input HDF5 file (edep-sim output) |
| `--config` | `jaxtpc/config/cubic_wireplane_config.yaml` | Detector geometry YAML |
| `--dataset` | `sim` | Dataset name prefix for output files |
| `--outdir` | `.` | Output directory (creates `sensor/` subdir) |
| `--events` | all | Number of events to process |
| `--events-per-file` | 1000 | Events per output HDF5 file |
| `--label-key` | `interaction` | Per-waveform label: `interaction`, `track`, `ancestor`, or `volume` |
| `--n-bits` | 15 | ADC bit depth |
| `--pedestal` | `0.9 * (2^n_bits - 1)` | ADC pedestal value |
| `--max-pe-per-pmt` | 90000 | PE scale for gain calculation |
| `--no-digitize` | (digitize ON) | Disable ADC digitization |
| `--dark-noise` | off | Enable dark noise |
| `--dark-noise-rate` | 2000.0 | Dark noise rate in Hz |
| `--baseline-noise-std` | 0.0 | Gaussian baseline noise std |
| `--ser-jitter-std` | 0.1 | SER weight jitter std |
| `--tick-ns` | 1.0 | Output time bin width in ns |
| `--oversample` | 10 | Internal oversampling factor |
| `--total-pad` | 250,000 | Max deposits per side (jaxtpc JIT shape) |
| `--response-chunk-size` | 50,000 | jaxtpc response chunk size |
| `--sampler` | `lut` | TOF sampler backend: `lut` (eager voxel LUT) or `siren` (SIREN neural net) |
| `--voxel-dx` | 0.0 | Voxelize input segments to a cubic grid of side length `dx` (mm) before goop simulate, performed per-label group. `0` disables. |
| `--workers` | 2 | Number of save worker threads (0 = serial) |
| `--seed` | 42 | Random seed |
| `--align` | off | Align chunks when returning `List[SlicedWaveform]` |

## Pipeline

For each event, `run_batch.py` performs:

1. **Load** particle step data from HDF5 via `load_event`
2. **Light generation** via jaxtpc `process_event_light` — yields per-volume photon counts, positions (mm), and deposit times (us)
3. **Extract** GOOP inputs: `positions_mm`, `ceil(photons).int32`, `t0_us * 1000` (-> ns), per-deposit labels (interaction ID by default; configurable via `--label-key`)
4. **Voxelize** *(optional, when `--voxel-dx > 0`)* — bin segments per-label-group into a cubic grid of side length `dx` mm via `voxelize_labeled`. Photon counts are summed within each voxel; positions and times are photon-weighted means. 4. **Voxelize** *(optional, when `--voxel-dx > 0`)* — bin segments per-label-group into a cubic grid of side length `dx` mm via `voxelize_labeled`. Photon counts are summed within each voxel; positions and times are photon-weighted means. Empirically a `dx` of 10 mm results in 20x less point simulated but indistinguishable waveforms.
5. **Optical simulation** via GOOP `OpticalSimulator.simulate` — produces one `SlicedWaveform` per unique label value:
   - TOF sampling: either eager voxel LUT (default, `--sampler lut`) or SIREN neural net (`--sampler siren`); both use the same PCA basis (50 components, quantile-log) and inverse-CDF photon-time draw
   - Stochastic delays: scintillation (bi-exponential), TPB re-emission (tri-exponential), PMT TTS (Gaussian)
   - Optional dark noise injection
   - Histogramming into per-PMT time bins
   - FFT convolution with SER kernel (10 us duration)
   - Optional baseline noise + ADC digitization (15-bit, pedestal offset, saturation clamping)
6. **Save** per-label `SlicedWaveform` to HDF5 via `save_event_light`

### Label Keys

The `--label-key` flag controls how deposits are grouped into separate waveforms:

| Key | Field | Description |
|---|---|---|
| `interaction` | `interaction_ids` | One waveform per interaction vertex (default) |
| `track` | `track_ids` | One waveform per GEANT4 particle track |
| `ancestor` | `ancestor_track_ids` | One waveform per primary shower ancestor |
| `volume` | synthetic | One waveform per detector volume (east/west) |

### TOF Sampler Choice

`--sampler lut` (default) loads the full PCA-compressed photon library into GPU memory at startup (~26 GB). Each sample lookup is a fast trilinear interpolation on the resident grid.

`--sampler siren` uses a pre-trained `PcaSiren` network that predicts the same `(visibility, t0, PCA coefficients)` tuple as the LUT, but on demand via a network forward — no resident voxel grid. Memory footprint drops from ~26 GB to ~500 MB; per-call cost is higher because every unique segment position pays a network forward. **Voxelization (`--voxel-dx`) helps SIREN much more than LUT** because it dramatically reduces the number of unique positions hitting the network (see Performance below).

### Voxelization

`--voxel-dx FLOAT` (mm) bins input segments into cubic voxels before the simulate call. For each label group separately, segments at the same voxel index are merged: photon counts summed, positions and times averaged photon-weighted. Total photon yield is exactly preserved.

Reduction is ~15–20× at 20 mm. The simulator output is unchanged in expectation; the only cost is the spatial averaging of segments within a voxel (relevant if intra-voxel timing structure matters, which it generally doesn't at this voxel scale because per-photon delays add ~1 ns of TTS jitter on top).

## Output File Format

One file type per batch, split by `events_per_file`:

```
{dataset}_sensor_{NNNN}.h5   — per-PMT optical waveforms (SlicedWaveform)
```

### Sensor File Schema

```
/config/
    attrs: dataset_name, source_file, n_events, global_event_offset,
           tick_ns, n_channels, gain, oversample, ser_jitter_std,
           baseline_noise_std, digitized, n_bits, pedestal,
           kernel_type, label_key, n_labels

/event_{NNN}/
    attrs: source_event_idx, n_labels
    label_{id}/                            one per unique label value
        adc         (N,) uint16            ADC samples (if digitized)
                    (N,) float32           raw signal (if not digitized)
        offsets     (K+1,) int64           CSR chunk boundaries
        t0_ns       (K,) float32           real-time origin per chunk (ns)
        pmt_id      (K,) int32             PMT channel index per chunk
        pe_counts   (C,) int32             PE count per PMT channel
```

**Decode:**
```python
# Chunk k spans adc[offsets[k]:offsets[k+1]]
# with time origin t0_ns[k] on PMT channel pmt_id[k]
# Time axis: t0_ns[k] + arange(chunk_len) * tick_ns
```

## Viewing Output

```python
from production.load import get_file_path, build_viz_config, load_event, list_events

path = get_file_path('output/', 'myrun', file_index=0)
config = build_viz_config(path)  # minimal config from HDF5 metadata
events = list_events(path)       # ['event_000', 'event_001', ...]

waveforms = load_event(path, event_idx=0, device='cpu')
# Returns List[SlicedWaveform], one per volume

for wf in waveforms:
    dense = wf.deslice()  # -> Waveform with shape (n_channels, n_bins)
    print(dense.data.shape, wf.attrs['pe_counts'].sum().item(), 'PEs')
```

## Model Sizes

| Component | Size | Notes |
|---|---|---|
| Photon library (LUT sampler) | ~26 GB | PCA-compressed, quantile-log, 50 components. Loaded eagerly into GPU VRAM. |
| SIREN sampler | ~500 MB | Pre-trained `PcaSiren` checkpoint; predicts the same `(vis, t0, coeffs)` as the LUT on demand. |
| SER kernel | ~10K samples | 10 us duration at 1 ns tick, oversampled 10x internally |
| 162 PMT channels | 81 per volume | x-reflection symmetry maps half-detector library to full coverage |

## Size Reference

Benchmarked on `out.h5` (MPV/MPR events, ~180K deposits/event average), 15-bit digitization, oversample=10, no baseline noise, `--label-key interaction`:

| Metric | Value |
|---|---|
| Per event | ~6 MB |
| Per 1000 events | ~6 GB |
| Typical PE count | ~1.3M PEs/event |
| Typical interactions/event | ~15 |
| Typical chunks | ~2,500/event |

## Threading Architecture

The pipeline uses three concurrent mechanisms to overlap I/O with GPU computation:

```
Prefetch thread:  read N+1 ──────── read N+2 ──────── read N+3 ────────
Main thread:      light+GOOP N ──── light+GOOP N+1 ── light+GOOP N+2 ──
Save workers:     save N-1 ──────── save N ─────────── save N+1 ────────
```

### 1. Prefetch thread

A single background thread pre-reads the next event from HDF5 via `load_event` (CPU-only: HDF5 read + numpy array construction). The result is held in a `Future` that the main thread collects at the start of the next iteration. This hides the ~0.15s HDF5 read latency behind GPU work.

Only CPU work runs in this thread — all GPU work (JAX light generation, PyTorch GOOP simulation) stays in the main thread to avoid CUDA contention.

### 2. Main thread (GPU)

Runs all GPU computation sequentially:
- **jaxtpc** `process_event_light` (JAX) — photon yield calculation (~0.02s)
- **voxelize_labeled** (torch on GPU, optional) — JAX → torch via dlpack (zero-copy), then per-label `torch.unique` + `index_add_` (~20 ms when `--voxel-dx > 0`, dominated by the per-label Python loop; 0 when off)
- **GOOP** `simulate` (PyTorch) — TOF sampling, delays, histogramming, FFT convolution
- GPU → CPU tensor transfer before queuing to save workers

### 3. Save workers (`--workers N`, default 2)

Background threads that pull completed events from a bounded queue and write them to HDF5. The gzip compression inside h5py releases the GIL, allowing workers to compress while the main thread simulates.

- A **file lock** serializes HDF5 writes (one writer at a time)
- Queue depth = `workers + 2` to absorb event size variation
- Each queued item carries its own `(event_key, waveforms, source_event_idx)`, so event ordering is guaranteed regardless of which worker processes it
- With `--workers 0`: save runs synchronously in the main thread (no threading)

### Tuning `--workers`

| Workers | Behavior | When to use |
|---|---|---|
| 0 | Serial — save blocks the main thread (~3s/event overhead) | Debugging, minimal memory |
| 1 | Single background saver — hides most save latency | Low-memory systems |
| 2 | Default — 2 savers overlap with GPU work | Recommended |
| 3+ | Diminishing returns — HDF5 writes serialize through the lock | Not recommended |

## Performance

Benchmarked on NVIDIA A100-SXM4-40GB, `--label-key interaction`, `--workers 2`, `--oversample 10`, `--n-bits 15` (all defaults), `out.h5` events 0–2 (~95k–264k segments, ~140M–407M photons / event).

### Sampler × voxelization sweep (avg per-event wall-clock)

| Stage | LUT | LUT, 10 mm voxels | SIREN | SIREN, 10 mm voxels |
|---|---|---|---|---|
| Load (HDF5 read)            | 0.05 s | 0.05 s | 0.05 s | 0.05 s |
| Light generation (jaxtpc)   | 0.01 s | 0.01 s | 0.01 s | 0.01 s |
| Voxelization (per-label)    | —      | 0.02 s | —      | 0.02 s |
| GOOP simulation             | 0.89 s | 0.74 s | 4.20 s | 1.04 s |
| Save (queue put)            | 0.01 s | 0.01 s | 0.01 s | 0.01 s |
| Loop overhead (gc + queue drain) | 0.30 s | 0.14 s | 0.28 s | 0.14 s |
| **Total / event**           | **1.26 s** | **0.97 s** | **4.56 s** | **1.27 s** |

At `--voxel-dx 10` the per-event voxel count drops from ~95k–264k raw segments to ~6k–21k voxels; with 1 ns ticks, the final waveform is preserved exactly.

The "Loop overhead" row is the wall-clock between per-event timed stages: `gc.collect()` and tensor reference releases at the bottom of the per-event loop, plus the end-of-file `save_queue.join()` that waits for trailing background saves. It scales with how big each event's tensors were (LUT/SIREN at no-vox hold ~370 M-photon intermediates, so the gc / dealloc cost is correspondingly larger).

With `--workers 0` the save stage runs synchronously and adds ~0.24 s / event to every total above (and removes the `save_queue.join()` portion of loop overhead).

## Tips

A voxel size of 10 mm is around the maximum voxel size you can use before seeing unphysical changes in the raw downstream waveforms.

If running into OOM issues when using the non-lazy LUT, switch to using Siren as the sampler (`--sampler siren`); set the voxelization to 10 mm (`--voxel-dx 10`), otherwise there will be a bottleneck due to the number of Siren evaluations scaling with the number of input segments.

