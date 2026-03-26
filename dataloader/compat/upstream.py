"""Upstream integration shims.

This module defines the **interface contract** between the ``dataloader``
package and upstream training frameworks (metasr-internal / fs2). Each
function documents the expected upstream signature and translates our
types into it.

Once real upstream types are available, replace the placeholder type
annotations with the actual classes and adjust the conversion logic.

Usage::

    from dataloader.compat.upstream import to_upstream_batch, to_upstream_dataset

    # Wrap our DataBatch for the upstream training loop
    upstream_batch = to_upstream_batch(data_batch)

    # Wrap our dataset to match upstream DataLoader expectations
    upstream_dataset = to_upstream_dataset(speech_dataset)
"""

from __future__ import annotations

from typing import Any

from dataloader.batch.data_batch import DataBatch
from dataloader.types import MetadataDict, SampleRate, Waveform


def to_upstream_batch(batch: DataBatch) -> dict[str, Any]:
    """Convert a :class:`DataBatch` to the dict format expected upstream.

    The upstream training loop (metasr-internal ``SpeechCollatorWithMasking``
    output) typically expects a flat dict of tensors. This is a placeholder
    mapping — update when real signatures are known.

    Returns
    -------
    dict[str, Any]
        Flat dict with keys matching upstream expectations.
    """
    result: dict[str, Any] = {
        "waveforms": batch.waveforms,
        "waveform_lengths": batch.waveform_lengths,
        "sample_rate": batch.sample_rate,
        "attention_mask": batch.attention_mask,
    }
    if batch.snr_db is not None:
        result["snr_db"] = batch.snr_db
    if batch.durations_s is not None:
        result["durations_s"] = batch.durations_s
    if batch.labels is not None:
        result["labels"] = batch.labels
    if batch.label_mask is not None:
        result["label_mask"] = batch.label_mask
    if batch.frame_labels is not None:
        result["frame_labels"] = batch.frame_labels
    if batch.wav_ids:
        result["wav_ids"] = batch.wav_ids
    return result


def from_upstream_batch(data: dict[str, Any]) -> DataBatch:
    """Convert an upstream batch dict back to a :class:`DataBatch`.

    This is the inverse of :func:`to_upstream_batch`. Use it when
    receiving data from upstream code that needs to flow through
    ``dataloader`` transforms or logging.
    """
    return DataBatch(
        waveforms=data["waveforms"],
        waveform_lengths=data["waveform_lengths"],
        sample_rate=data["sample_rate"],
        attention_mask=data["attention_mask"],
        snr_db=data.get("snr_db"),
        durations_s=data.get("durations_s"),
        labels=data.get("labels"),
        label_mask=data.get("label_mask"),
        frame_labels=data.get("frame_labels"),
        wav_ids=data.get("wav_ids", []),
        metadata=data.get("metadata", []),
        segments=data.get("segments"),
    )


def to_upstream_sample(
    waveform: Waveform,
    sample_rate: SampleRate,
    metadata: MetadataDict,
) -> dict[str, Any]:
    """Convert a single sample triple to upstream dict format.

    Some upstream datasets expect ``__getitem__`` to return a dict
    rather than a tuple. This provides the translation.
    """
    return {
        "waveform": waveform,
        "sample_rate": sample_rate,
        **{k: v for k, v in metadata.items()},
    }
