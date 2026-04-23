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
| `--lazy` | off | Use lazy (disk-backed) photon library loading |
| `--workers` | 2 | Number of save worker threads (0 = serial) |
| `--seed` | 42 | Random seed |
| `--align` | off | Align chunks when returning `List[SlicedWaveform]` |

## Pipeline

For each event, `run_batch.py` performs:

1. **Load** particle step data from HDF5 via `load_event`
2. **Light generation** via jaxtpc `process_event_light` — yields per-volume photon counts, positions (mm), and deposit times (us)
3. **Extract** GOOP inputs: `positions_mm`, `ceil(photons).int32`, `t0_us * 1000` (-> ns), per-deposit labels (interaction ID by default; configurable via `--label-key`)
4. **Optical simulation** via GOOP `OpticalSimulator.simulate` — produces one `SlicedWaveform` per unique label value:
   - Photon library lookup + TOF sampling (PCA-compressed, inverse-CDF)
   - Stochastic delays: scintillation (bi-exponential), TPB re-emission (tri-exponential), PMT TTS (Gaussian)
   - Optional dark noise injection
   - Histogramming into per-PMT time bins
   - FFT convolution with SER kernel (10 us duration)
   - Optional baseline noise + ADC digitization (15-bit, pedestal offset, saturation clamping)
5. **Save** per-label `SlicedWaveform` to HDF5 via `save_event_light`

### Label Keys

The `--label-key` flag controls how deposits are grouped into separate waveforms:

| Key | Field | Description |
|---|---|---|
| `interaction` | `interaction_ids` | One waveform per interaction vertex (default) |
| `track` | `track_ids` | One waveform per GEANT4 particle track |
| `ancestor` | `ancestor_track_ids` | One waveform per primary shower ancestor |
| `volume` | synthetic | One waveform per detector volume (east/west) |

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
| Photon library | 26 GB | PCA-compressed, quantile-log, 50 components. Loaded into GPU VRAM (eager) or disk-backed (lazy) |
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

### 1. Prefetch thread (1 thread, not configurable)

A single background thread pre-reads the next event from HDF5 via `load_event` (CPU-only: HDF5 read + numpy array construction). The result is held in a `Future` that the main thread collects at the start of the next iteration. This hides the ~0.15s HDF5 read latency behind GPU work.

Only CPU work runs in this thread — all GPU work (JAX light generation, PyTorch GOOP simulation) stays in the main thread to avoid CUDA contention.

### 2. Main thread (GPU)

Runs all GPU computation sequentially:
- **jaxtpc** `process_event_light` (JAX) — photon yield calculation (~0.02s)
- **GOOP** `simulate` (PyTorch) — TOF sampling, delays, histogramming, FFT convolution (~2.3s)
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

Benchmarked on NVIDIA A100-SXM4-40GB, `--label-key interaction`, `--workers 2`, eager photon library.
Timing averaged over events 3–7 (after 3 warmup events):

| Stage | Serial (`--workers 0`) | Threaded (`--workers 2`) |
|---|---|---|
| Load (HDF5 read) | ~0.15s | ~0.00s (prefetched) |
| Light generation (jaxtpc) | ~0.02s | ~0.02s |
| GOOP simulation | ~2.3s | ~2.3s |
| Save (HDF5 write) | ~3.0s | ~0.04s (queued) |
| **Total/event** | **~5.5s** | **~2.4s** |

- Warmup (JIT + photon library load): ~22s one-time cost
- GOOP simulator creation: ~12–17s (photon library decompression)
- Sim time scales with photon count: ~1.8s for small events (~180M photons), ~2.6s for large (~400M photons)
