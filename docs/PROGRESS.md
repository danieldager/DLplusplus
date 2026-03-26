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

**Status**: 🔨 To implement.

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

### D4: Storage Format — .pt as Primary, JSON for Human-Readable (2026-03-26)

**Decision**: Use `.pt` (PyTorch serialization) as the primary metadata storage
format for tensor-heavy data (SNR arrays, noise embeddings, frame-level labels).
Keep JSON for lightweight human-readable metadata (pipeline config, provenance,
manifest summaries).

**Rationale**:
- `.pt` is faster to load/save for tensor data than NPZ and avoids
  numpy↔torch conversion overhead.
- `.pt` supports arbitrary nested Python objects (dicts, lists, tensors)
  natively.
- Space-efficient for float32/int64 arrays vs. JSON serialization.
- JSON stays for config/provenance because it's human-readable and
  diffable.

**Action items**:
- [ ] Add `PtStore(MetadataStore)` implementation.
- [ ] Migrate `NpzStore` usage to `PtStore` where data is tensor-native.
- [ ] Keep `JsonStore` for config/provenance metadata.
- [ ] Update `MetadataFormat` enum to reflect `.pt` as default.

**Status**: 🔨 To implement.

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
- [ ] Add explicit tensor fields for common metadata (SNR, durations, etc.)
- [ ] Collator populates these fields; model code reads tensors directly.
- [ ] Keep `metadata` list for debugging / non-tensor info only.

**Status**: 🔨 To implement.

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

---

## Implementation Queue (Phase 2)

Priority order:

1. **`PtStore`** — Add `.pt` metadata storage backend
2. **`WaveformProcessor`** — Dual-mode (online/offline) waveform transforms
3. **`DataBatch` refactor** — Tensor-centric fields
4. **`dataloader/adapters/`** — Wrap VAD, VTC, SNR, Noise as `FeatureProcessor`
5. **`PipelineManifestBuilder`** — Extract Big Join orchestration from `package.py`
6. **Loader utilities** — Extract load functions from `package.py`
7. **`WebDatasetSpeechDataset`** — IterableDataset with distributed support
8. **Compat shim** — Placeholder for upstream type mapping

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
| `dataloader/manifest/store.py` | 1 | ✅ Complete (needs PtStore) |
| `dataloader/transform/base.py` | 1 | ✅ Complete |
| `dataloader/transform/audio.py` | 1 | ✅ Complete |
| `dataloader/transform/label.py` | 1 | ✅ Complete |
| `dataloader/batch/base.py` | 1 | ✅ Complete |
| `dataloader/batch/data_batch.py` | 1 | ✅ Complete (needs tensor refactor) |
| `dataloader/batch/speech.py` | 1 | ✅ Complete |
| `dataloader/dataset/base.py` | 1 | ✅ Complete |
| `dataloader/dataset/webdataset.py` | 1 | ⬜ Stub only |
| `.github/copilot-instructions.md` | 1 | ✅ Updated |
