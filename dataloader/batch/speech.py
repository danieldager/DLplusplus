"""Speech-specific collator with masking support.

:class:`SpeechCollator` pads waveforms to a uniform length, constructs
attention masks, and extracts metadata into batch-level tensors.
"""

from __future__ import annotations

import torch

from dataloader.batch.base import Collator
from dataloader.batch.data_batch import DataBatch
from dataloader.types import MetadataDict, SampleRate, SegmentList, Waveform


class SpeechCollator(Collator):
    """Pad and collate speech samples into a :class:`DataBatch`.

    Extracts known metadata keys into batch-level tensors (``snr_db``,
    ``durations_s``, etc.) so model code can access them directly without
    unpacking per-sample dicts.

    Parameters
    ----------
    pad_to_multiple_of:
        If set, pad waveforms so that the time dimension is a multiple
        of this value.
    snr_key:
        Metadata key for per-sample SNR in dB.
    label_mask_key:
        Metadata key for a precomputed frame-level label mask.
    segments_key:
        Metadata key for raw segment lists to forward to ``DataBatch``.
    """

    def __init__(
        self,
        pad_to_multiple_of: int | None = None,
        snr_key: str = "snr_db",
        label_mask_key: str = "label_mask",
        segments_key: str | None = None,
    ) -> None:
        self._pad_multiple = pad_to_multiple_of
        self._snr_key = snr_key
        self._label_mask_key = label_mask_key
        self._segments_key = segments_key

    def __call__(
        self,
        samples: list[tuple[Waveform, SampleRate, MetadataDict]],
    ) -> DataBatch:
        waveforms: list[Waveform] = []
        lengths: list[int] = []
        all_metadata: list[MetadataDict] = []
        sample_rate: int | None = None

        for wav, sr, meta in samples:
            if sample_rate is None:
                sample_rate = sr
            waveforms.append(wav)
            lengths.append(wav.shape[-1])
            all_metadata.append(meta)

        assert sample_rate is not None, "Cannot collate an empty batch."

        max_len = max(lengths)
        if self._pad_multiple:
            remainder = max_len % self._pad_multiple
            if remainder != 0:
                max_len += self._pad_multiple - remainder

        batch_size = len(waveforms)
        channels = waveforms[0].shape[0]

        # ── Pad waveforms ─────────────────────────────────────────────────
        padded = torch.zeros(batch_size, channels, max_len, dtype=torch.float32)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

        for i, (wav, length) in enumerate(zip(waveforms, lengths)):
            padded[i, :, :length] = wav
            attention_mask[i, :length] = True

        waveform_lengths = torch.tensor(lengths, dtype=torch.long)

        # ── Extract wav_ids ───────────────────────────────────────────────
        wav_ids = [str(m.get("wav_id", "")) for m in all_metadata]

        # ── Duration tensor ───────────────────────────────────────────────
        durations_s = torch.tensor(
            [l / sample_rate for l in lengths],
            dtype=torch.float32,
        )

        # ── SNR tensor ────────────────────────────────────────────────────
        snr_db = self._collate_scalar(all_metadata, self._snr_key)

        # ── Frame-level label masks ───────────────────────────────────────
        frame_labels = self._collate_frame_masks(
            all_metadata,
            self._label_mask_key,
        )

        # ── Raw segments (optional) ───────────────────────────────────────
        segments: list[SegmentList] | None = None
        if self._segments_key:
            segments = [
                m.get(self._segments_key, [])  # type: ignore[misc]
                for m in all_metadata
            ]

        return DataBatch(
            waveforms=padded,
            waveform_lengths=waveform_lengths,
            sample_rate=sample_rate,
            attention_mask=attention_mask,
            snr_db=snr_db,
            durations_s=durations_s,
            frame_labels=frame_labels,
            wav_ids=wav_ids,
            metadata=all_metadata,
            segments=segments,
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _collate_scalar(
        metadata_list: list[MetadataDict],
        key: str,
    ) -> torch.Tensor | None:
        """Extract a scalar float from each sample into a (B,) tensor."""
        values = [m.get(key) for m in metadata_list]
        if all(v is None for v in values):
            return None
        return torch.tensor(
            [float(v) if v is not None else 0.0 for v in values],
            dtype=torch.float32,
        )

    @staticmethod
    def _collate_frame_masks(
        metadata_list: list[MetadataDict],
        key: str,
    ) -> torch.Tensor | None:
        """Pad and stack frame-level boolean masks from metadata."""
        masks = [m.get(key) for m in metadata_list]
        if all(m is None for m in masks):
            return None

        tensors = [m for m in masks if isinstance(m, torch.Tensor)]
        if not tensors:
            return None

        max_frames = max(t.shape[0] for t in tensors)
        n_labels = tensors[0].shape[-1] if tensors[0].dim() > 1 else 1
        padded = torch.zeros(
            len(metadata_list),
            max_frames,
            n_labels,
            dtype=torch.bool,
        )

        for i, m in enumerate(masks):
            if isinstance(m, torch.Tensor):
                t = m if m.dim() > 1 else m.unsqueeze(-1)
                padded[i, : t.shape[0], : t.shape[1]] = t

        return padded

    def __repr__(self) -> str:
        return f"SpeechCollator(pad_to_multiple_of={self._pad_multiple})"
