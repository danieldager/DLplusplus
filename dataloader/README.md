# Dataloader++

Feature loader and data processing framework for Meta's speech training
infrastructure. See [`docs/DATALOADER_DESIGN.md`](../docs/DATALOADER_DESIGN.md)
for the full design document.

## End-to-End Workflow

Two steps, one config file.

### Step 1 — Extract features (run once, from the login node)

```bash
python -m dataloader.prepare configs/seedlings.json
```

This submits VAD, VTC, SNR, and Noise jobs in parallel (SLURM), followed by
a packaging job that writes `.wav`-encoded clips into WebDataset `.tar` shards.
It blocks until everything is done, or you can fire-and-forget with `--no-wait`.

**Config file format** (`configs/seedlings.json`):
```json
{
  "dataset_dir": "output/seedlings_1",
  "pipeline": {
    "vad_threshold": 0.5,
    "vtc_threshold": 0.5,
    "target_sr": 16000,
    "audio_fmt": "wav",
    "max_clip_s": 600
  },
  "filters": {
    "min_snr_db": 10.0,
    "required_labels": ["KCHI"]
  },
  "loader": {
    "batch_size": 16,
    "num_workers": 4
  }
}
```

CLI options:
```
--sample 0.1     Process only 10% of files (useful for testing)
--no-wait        Fire-and-forget: submit jobs and exit immediately
--force          Wipe outputs and reprocess from scratch
--verbose        Enable debug logging
```

### Step 2 — Create a DataLoader (in your training script)

```python
from dataloader import create_dataloader

loader = create_dataloader("configs/seedlings.json")

for batch in loader:
    print(batch.waveforms.shape)    # (B, 1, T) — padded waveform tensor
    print(batch.attention_mask)     # (B, T)    — True where real audio
    print(batch.snr_db)             # (B,)      — per-clip SNR
    print(batch.wav_ids)            # list[str] — clip identifiers
    break
```

The same config drives both steps. `filters.*` fields are applied at DataLoader
creation time — no reprocessing. Changing a filter and calling
`create_dataloader()` again is instant.

## Quick Start

```python
from dataloader import (
    Collator,
    Compose,
    DataBatch,
    DataProcessor,
    FeatureLoader,
    FeatureProcessor,
    ManifestJoiner,
    MetadataManifest,
    MetadataStore,
)
```

## Package Structure

| Module | Purpose |
|---|---|
| `processor/` | Feature Processor ABCs — offline metadata extraction |
| `adapters/` | Read-only wrappers over pipeline outputs (VAD, VTC, SNR, Noise) |
| `loader/` | Feature Loader ABCs — waveform + metadata I/O |
| `manifest/` | Manifest schema, Big Join, unified metadata store |
| `transform/` | Runtime data transforms (segment, resample, encode) |
| `batch/` | Collation and `DataBatch` containers |
| `dataset/` | PyTorch Dataset implementations (WebDataset streaming) |
| `config.py` | `PipelineConfig`, `FilterConfig`, `LoaderConfig`, `DatasetConfig` |
| `build.py` | `build_manifest()` — Big Join + filtering convenience function |
| `create.py` | `create_dataloader()` — end-to-end config → DataLoader |
| `types.py` | Shared type aliases and enums |
