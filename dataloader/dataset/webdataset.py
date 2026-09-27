"""WebDataset-backed speech datasets for distributed streaming.

Provides :class:`WebDatasetSpeechDataset` (streaming ``IterableDataset``
for training) and :class:`EvalSpeechDataset` (map-style ``Dataset`` for
deterministic evaluation). Both read ``.tar`` shards produced by
``src/pipeline/package.py``.

Shard partitioning follows the WebDataset convention:
- Multi-node: ``wds.shardlists.split_by_node``
- Multi-worker: ``wds.shardlists.split_by_worker``

Samples are ``(waveform, sample_rate, metadata)`` triples, compatible with
:class:`~dataloader.transform.base.DataProcessor` and
:class:`~dataloader.batch.speech.SpeechCollator`.
"""

from __future__ import annotations

import io
import logging
import random
from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf
import torch
import webdataset as wds
from torch.utils.data import Dataset, IterableDataset

from dataloader.transform.base import DataProcessor
from dataloader.types import MetadataDict, SampleRate, Waveform

log = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────
_DEFAULT_SHUFFLE_BUFFER = 1000
_DEFAULT_SEED = 42


def _decode_audio(
    data: bytes, target_sr: int | None = None
) -> tuple[Waveform, SampleRate]:
    """Decode audio bytes to a ``(channels, samples)`` tensor."""
    with io.BytesIO(data) as buf:
        audio, sr = sf.read(buf, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(audio.T)  # (samples, ch) → (ch, samples)

    if target_sr is not None and sr != target_sr:
        try:
            import torchaudio.functional as F

            waveform = F.resample(waveform, orig_freq=sr, new_freq=target_sr)
        except ImportError:
            from scipy.signal import resample as scipy_resample

            n_out = int(waveform.shape[-1] * target_sr / sr)
            arr = np.asarray(
                scipy_resample(waveform.numpy(), n_out, axis=-1),
                dtype=np.float32,
            )
            waveform = torch.from_numpy(arr)
        sr = target_sr

    return waveform, int(sr)


def _decode_sample(
    sample: dict,
    audio_key: str,
    metadata_keys: list[str],
    target_sr: int | None,
) -> tuple[Waveform, SampleRate, MetadataDict] | None:
    """Decode a raw WebDataset sample dict into our triple format."""
    audio_data = sample.get(audio_key)
    if audio_data is None:
        log.warning("No '%s' found in sample, skipping.", audio_key)
        return None

    waveform, sr = _decode_audio(audio_data, target_sr)

    metadata: MetadataDict = {}
    # Extract wav_id from the __key__ field.
    key = sample.get("__key__", "")
    metadata["wav_id"] = key.split("/")[-1] if "/" in key else key

    # Load metadata from .pt / .npy / .json keys.
    for mk in metadata_keys:
        val = sample.get(mk)
        if val is None:
            continue
        # Strip extension for the metadata dict key.
        clean_key = mk.rsplit(".", 1)[0] if "." in mk else mk
        if isinstance(val, dict):
            metadata.update(val)
        elif mk.endswith(".json") or mk == "json":
            import json as _json
            try:
                parsed = _json.loads(val)
                if isinstance(parsed, dict):
                    metadata.update(parsed)
                else:
                    metadata[clean_key] = parsed
            except Exception:
                metadata[clean_key] = val
        else:
            metadata[clean_key] = val

    return waveform, sr, metadata


# ── Streaming dataset (training) ─────────────────────────────────────────────


class WebDatasetSpeechDataset(IterableDataset):
    """Streaming speech dataset backed by WebDataset tar shards.

    Yields ``(waveform, sample_rate, metadata)`` triples. Shard splitting
    for multi-node / multi-worker is handled automatically.

    Parameters
    ----------
    shard_urls:
        List of ``.tar`` shard paths (or a glob string).
    audio_key:
        Extension key for audio within the tar (e.g. ``"flac"``).
    metadata_keys:
        Extension keys for metadata within the tar (e.g.
        ``["metadata.json", "snr.pt"]``).
    target_sr:
        If set, resample all audio to this rate on decode.
    processor:
        Optional :class:`DataProcessor` pipeline applied to each sample.
    shuffle_buffer:
        Number of samples to buffer for shuffling.
    seed:
        Random seed for shard-level shuffling.
    allowed_ids:
        If set, only yield samples whose source ``uid`` (or ``wav_id``)
        is in this set. Used for manifest-level filtering.
    """

    def __init__(
        self,
        shard_urls: list[str] | str,
        audio_key: str = "wav",
        metadata_keys: list[str] | None = None,
        target_sr: SampleRate | None = None,
        processor: DataProcessor | None = None,
        shuffle_buffer: int = _DEFAULT_SHUFFLE_BUFFER,
        seed: int = _DEFAULT_SEED,
        allowed_ids: set[str] | None = None,
    ) -> None:
        if isinstance(shard_urls, str):
            self._urls = sorted(
                str(p) for p in Path(shard_urls).parent.glob(Path(shard_urls).name)
            )
            if not self._urls:
                self._urls = [shard_urls]
        else:
            self._urls = list(shard_urls)

        self._audio_key = audio_key
        self._metadata_keys = metadata_keys or []
        self._target_sr = target_sr
        self._processor = processor
        self._shuffle_buffer = shuffle_buffer
        self._seed = seed
        self._allowed_ids = allowed_ids

        random.seed(self._seed)
        random.shuffle(self._urls)

    def __iter__(self) -> Iterator[tuple[Waveform, SampleRate, MetadataDict]]:
        dataset = wds.WebDataset(  # type: ignore
            self._urls,
            resampled=True,
            shardshuffle=False,
            nodesplitter=wds.shardlists.split_by_node,
            handler=wds.warn_and_continue,  # type: ignore
        ).shuffle(self._shuffle_buffer)
        dataset = dataset.compose(wds.shardlists.split_by_worker)

        for sample in dataset:
            result = _decode_sample(
                sample,
                self._audio_key,
                self._metadata_keys,
                self._target_sr,
            )
            if result is None:
                continue

            waveform, sr, metadata = result

            # Filter by allowed source file IDs (from manifest filtering).
            if self._allowed_ids is not None:
                uid = metadata.get("uid", metadata.get("wav_id", ""))
                if uid not in self._allowed_ids:
                    continue

            if self._processor is not None:
                waveform, sr, metadata = self._processor(waveform, sr, metadata)

            yield waveform, sr, metadata

    def __repr__(self) -> str:
        return (
            f"WebDatasetSpeechDataset(shards={len(self._urls)}, "
            f"audio_key={self._audio_key!r})"
        )


# ── Map-style dataset (evaluation) ───────────────────────────────────────────


class EvalSpeechDataset(Dataset):
    """Map-style speech dataset for deterministic evaluation.

    Pre-loads a fixed number of samples from WebDataset shards into memory,
    guaranteeing identical iteration order across runs.

    Parameters
    ----------
    shard_urls:
        List of ``.tar`` shard paths.
    num_samples:
        Maximum number of samples to load.
    audio_key:
        Extension key for audio within the tar.
    metadata_keys:
        Extension keys for metadata.
    target_sr:
        If set, resample all audio to this rate.
    processor:
        Optional :class:`DataProcessor` pipeline.
    seed:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        shard_urls: list[str],
        num_samples: int = 2000,
        audio_key: str = "wav",
        metadata_keys: list[str] | None = None,
        target_sr: SampleRate | None = None,
        processor: DataProcessor | None = None,
        seed: int = _DEFAULT_SEED,
    ) -> None:
        self._urls = list(shard_urls)
        self._audio_key = audio_key
        self._metadata_keys = metadata_keys or []
        self._target_sr = target_sr
        self._processor = processor

        random.seed(seed)
        random.shuffle(self._urls)

        self._samples: list[tuple[Waveform, SampleRate, MetadataDict]] = []
        self._load_samples(num_samples)

    def _load_samples(self, num_samples: int) -> None:
        dataset = wds.WebDataset(  # type: ignore
            self._urls,
            shardshuffle=False,
            nodesplitter=wds.shardlists.split_by_node,
            handler=wds.warn_and_continue,  # type: ignore
        )

        for sample in dataset:
            if len(self._samples) >= num_samples:
                break

            result = _decode_sample(
                sample,
                self._audio_key,
                self._metadata_keys,
                self._target_sr,
            )
            if result is None:
                continue

            waveform, sr, metadata = result
            if self._processor is not None:
                waveform, sr, metadata = self._processor(waveform, sr, metadata)

            self._samples.append((waveform, sr, metadata))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[Waveform, SampleRate, MetadataDict]:
        return self._samples[index]

    def __repr__(self) -> str:
        return (
            f"EvalSpeechDataset(samples={len(self._samples)}, "
            f"shards={len(self._urls)})"
        )
