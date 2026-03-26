# Dataloader++ Progress Tracker

> **Copilot**: Read this file at the start of every session. Update it after
> completing each implementation step, decision, or design change.

---

## Current Phase: Phase 2 — Adapters & Integration Prep

### Phase Summary

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 — Interface Design | ✅ Complete | ABCs, types, manifest joiner, transforms, collator, dataset |
| Phase 2 — Adapters & Refactors | 🔨 In Progress | Wrap pipeline stages, extract loaders, prepare for upstream |
| Phase 3 — Concrete Feature Loader | ⬜ Not Started | WebDataset loader, raw file loader, sampler |
| Phase 4 — Validation | ⬜ Not Started | End-to-end tests, benchmarks |

---

## Decisions Log

### D1: Waveform Processing — Dual-Mode Class (2026-03-26)

**Decision**: Build a single `WaveformProcessor` class that supports both:
- **Online mode**: Modifies waveforms at dataload time (inside the `DataProcessor`
  transform pipeline). Lightweight, no disk writes.
- **Offline mode**: During feature extraction, saves the modified waveform to
  disk alongside provenance metadata (what transform was applied, parameters,
  source file). Future loads skip the transform and read the pre-processed file.

**Rationale**: The user wants flexibility — apply transforms on-the-fly for
experimentation, or bake them in for production. One class, two code paths.

**Status**: ✅ Implemented (`dataloader/transform/waveform.py`).

### D2: Masking — Deferred (2026-03-26)

**Decision**: Park the question of where masking lives (transform vs. collator)
until we have more information on what masking is needed for and how it would be
implemented in the upstream codebase.

**Status**: ⏸ Parked.

### D3: metasr-internal / fs2 Interface Compatibility (2026-03-26)

**Decision**: We do not currently have access to the upstream signatures. Build a
thin **compatibility shim layer** with clearly documented interface boundaries.
When upstream signatures become available, we adapt the shim without rewriting core
logic.

**Approach**:
- Define our own `DataBatch` / `SpeechDataset` / `SpeechCollator` with clean
  interfaces.
- Add a `dataloader/compat/` module (or similar) that can translate between
  our types and upstream types once known.
- Document every public-facing return type and method signature so the mapping
  is explicit.

**Status**: 🔨 To implement (compat layer is Phase 2).

### D4: Storage Format — .pt as Primary Everywhere (2026-03-26)

**Decision**: Use `.pt` (PyTorch serialization) as the **sole default** metadata
storage format. All new metadata flows through `PtStore`. Legacy backends
(`NpzStore`, `ParquetStore`, `JsonStore`) are retained for backward compat with
existing pipeline outputs only.

**Rationale**:
- `.pt` supports dicts, tensors, scalars, lists natively — one format for everything.
- Avoids the pipeline having to deal with format differences.
- `default_store(root)` factory always returns `PtStore`.
- Parquet stays only for manifest joins (columnar data); JSON only for
  human-readable provenance/config files written by `WaveformProcessor`.

**Action items**:
- [x] Add `PtStore(MetadataStore)` implementation.
- [x] Add `default_store()` factory.
- [x] Update `MetadataFormat` enum (`.pt` already present).
- [ ] Migrate adapter `save()`/`load()` to use `PtStore` in Phase 2 adapters.

**Status**: ✅ Implemented.

### D5: Phoneme Alignments — Deferred (2026-03-26)

**Decision**: Not in scope for Phase 2. Infrastructure supports it; will add
`PhonemeProcessor` when needed.

**Status**: ⏸ Parked.

### D6: DataBatch — More Tensor-Centric (2026-03-26)

**Decision**: Refactor `DataBatch` to favor named tensor fields over
`metadata: list[MetadataDict]`. Per-sample metadata dicts should be
projected into batch-level tensors wherever possible. Keep a
`metadata: list[MetadataDict]` escape hatch for non-tensorizable data.

**Action items**:
- [x] Add explicit tensor fields: `snr_db`, `c50_db`, `durations_s`.
- [x] Collator populates these fields; model code reads tensors directly.
- [x] Keep `metadata` list for debugging / non-tensor info only.
- [x] Add `wav_ids: list[str]` for sample identification.

**Status**: ✅ Implemented.

### D7: Distributed / Streaming — WebDataset IterableDataset (2026-03-26)

**Decision**: Implement `WebDatasetSpeechDataset` as a proper `IterableDataset`
following the user's proven pattern from their token training repo:
- `wds.WebDataset(urls, resampled=True)` for infinite epoch streaming.
- `wds.shardlists.split_by_node` for multi-node partitioning.
- `wds.shardlists.split_by_worker` for DataLoader worker partitioning.
- Shuffle buffer for sample-level randomness.
- Map-style `EvalDataset` variant for deterministic evaluation.

**Reference**: User's `TokenDataset` / `EvalDataset` implementation.

**Status**: 🔨 To implement (Phase 2–3).

### D8: C50 Clarity Metric (2026-03-26)

**Decision**: Add `c50_db` as a first-class tensor field in `DataBatch`
alongside `snr_db`. Both are per-sample scalar metrics extracted by the
Brouhaha pipeline.

**Status**: ✅ Implemented.

---

## Implementation Queue (Phase 2)

Priority order:

1. ~~**`PtStore`** — Add `.pt` metadata storage backend~~ ✅
2. ~~**`WaveformProcessor`** — Dual-mode (online/offline) waveform transforms~~ ✅
3. ~~**`DataBatch` refactor** — Tensor-centric fields (`snr_db`, `c50_db`, `durations_s`, `wav_ids`)~~ ✅
4. ~~**`WebDatasetSpeechDataset`** — IterableDataset with distributed support~~ ✅
5. ~~**Compat shim** — Placeholder for upstream type mapping~~ ✅
6. **`dataloader/adapters/`** — Wrap VAD, VTC, SNR, Noise as `FeatureProcessor`
7. **`PipelineManifestBuilder`** — Extract Big Join orchestration from `package.py`
8. **Loader utilities** — Extract load functions from `package.py`

---

## Open Questions

- **Q1**: What are the exact masking requirements? (attention masks, prediction
  masks, label masks — which are needed, at what granularity?)
- **Q2**: What are the metasr-internal `SpeechDataset` / `SpeechCollatorWithMasking`
  signatures? (blocked until access is granted)
- **Q3**: Should offline waveform processing produce a new manifest entry linking
  `wav_id` → processed file path, or overwrite the original?

---

## File Inventory

Files created/modified as part of Dataloader++:

| File | Phase | Status |
|------|-------|--------|
| `docs/DATALOADER_DESIGN.md` | 1 | ✅ Complete |
| `docs/PROGRESS.md` | 2 | ✅ Active (this file) |
| `dataloader/__init__.py` | 1 | ✅ Complete |
| `dataloader/types.py` | 1 | ✅ Complete |
| `dataloader/processor/base.py` | 1 | ✅ Complete |
| `dataloader/processor/registry.py` | 1 | ✅ Complete |
| `dataloader/loader/base.py` | 1 | ✅ Complete |
| `dataloader/loader/waveform.py` | 1 | ✅ Complete |
| `dataloader/loader/metadata.py` | 1 | ✅ Complete |
| `dataloader/manifest/schema.py` | 1 | ✅ Complete |
| `dataloader/manifest/joiner.py` | 1 | ✅ Complete |
| `dataloader/manifest/store.py` | 1→2 | ✅ Complete (PtStore + default_store) |
| `dataloader/transform/base.py` | 1 | ✅ Complete |
| `dataloader/transform/audio.py` | 1 | ✅ Complete |
| `dataloader/transform/label.py` | 1 | ✅ Complete |
| `dataloader/transform/waveform.py` | 2 | ✅ WaveformProcessor + Denoiser |
| `dataloader/batch/base.py` | 1 | ✅ Complete |
| `dataloader/batch/data_batch.py` | 1→2 | ✅ Tensor-centric (snr_db, c50_db, durations_s, wav_ids) |
| `dataloader/batch/speech.py` | 1→2 | ✅ Collates snr_db, c50_db, durations_s |
| `dataloader/dataset/base.py` | 1 | ✅ Complete |
| `dataloader/dataset/webdataset.py` | 2 | ✅ WebDatasetSpeechDataset + EvalSpeechDataset |
| `dataloader/compat/__init__.py` | 2 | ✅ Created |
| `dataloader/compat/upstream.py` | 2 | ✅ Shim (to/from upstream batch/sample) |
| `.github/copilot-instructions.md` | 1 | ✅ Updated |
