"""Typed container for a collated batch of speech samples.

:class:`DataBatch` is the output of a :class:`~dataloader.batch.base.Collator`.
It provides a single, typed object that models consume directly from
``torch.utils.data.DataLoader``.

All named tensor fields are batch-level: they share a leading batch dimension
and are padded/stacked by the collator. The ``metadata`` field is a fallback
for non-tensorizable per-sample data (debugging, provenance, raw segment lists).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from dataloader.types import MetadataDict, SegmentList


@dataclass
class DataBatch:
    """Container for a collated batch of speech samples.

    All tensor fields share the first dimension (``batch_size``).
    Variable-length sequences are right-padded with zeros, and
    ``waveform_lengths`` stores the original (unpadded) sample counts.

    Primary tensor fields (always populated):
        waveforms, waveform_lengths, sample_rate, attention_mask

    Metadata tensor fields (populated when available in the pipeline):
        snr_db, c50_db, durations_s, labels, label_mask, frame_labels

    Non-tensor fields:
        metadata (list[MetadataDict]) — per-sample escape hatch
        segments (list[SegmentList]) — raw segment data
        wav_ids (list[str]) — sample identifiers
    """

    # ── Primary (always present) ──────────────────────────────────────────
    waveforms: torch.Tensor                         # (B, C, T_max)
    waveform_lengths: torch.Tensor                  # (B,) dtype=long
    sample_rate: int
    attention_mask: torch.Tensor                    # (B, T_max) dtype=bool

    # ── Metadata tensors (populated when data exists) ─────────────────────
    snr_db: torch.Tensor | None = None              # (B,) per-sample SNR
    c50_db: torch.Tensor | None = None              # (B,) per-sample C50 clarity
    durations_s: torch.Tensor | None = None         # (B,) duration in seconds
    labels: torch.Tensor | None = None              # (B, max_segments) dtype=long
    label_mask: torch.Tensor | None = None          # (B, max_segments) dtype=bool
    frame_labels: torch.Tensor | None = None        # (B, T_frames, n_labels) dtype=bool

    # ── Non-tensor fields (fallback) ──────────────────────────────────────
    wav_ids: list[str] = field(default_factory=list)  # (B,) sample identifiers
    metadata: list[MetadataDict] = field(default_factory=list)  # len B
    segments: list[SegmentList] | None = None       # len B

    @property
    def batch_size(self) -> int:
        return self.waveforms.shape[0]

    @property
    def max_length(self) -> int:
        """Maximum waveform length in samples (padded dimension)."""
        return self.waveforms.shape[-1]

    def to(self, device: torch.device | str) -> DataBatch:
        """Move all tensors to *device* and return ``self``."""
        self.waveforms = self.waveforms.to(device)
        self.waveform_lengths = self.waveform_lengths.to(device)
        self.attention_mask = self.attention_mask.to(device)
        if self.snr_db is not None:
            self.snr_db = self.snr_db.to(device)
        if self.c50_db is not None:
            self.c50_db = self.c50_db.to(device)
        if self.durations_s is not None:
            self.durations_s = self.durations_s.to(device)
        if self.labels is not None:
            self.labels = self.labels.to(device)
        if self.label_mask is not None:
            self.label_mask = self.label_mask.to(device)
        if self.frame_labels is not None:
            self.frame_labels = self.frame_labels.to(device)
        return self

    def pin_memory(self) -> DataBatch:
        """Pin all tensors for faster host-to-device transfer."""
        if not torch.cuda.is_available():
            return self
        self.waveforms = self.waveforms.pin_memory()
        self.waveform_lengths = self.waveform_lengths.pin_memory()
        self.attention_mask = self.attention_mask.pin_memory()
        if self.snr_db is not None:
            self.snr_db = self.snr_db.pin_memory()
        if self.c50_db is not None:
            self.c50_db = self.c50_db.pin_memory()
        if self.durations_s is not None:
            self.durations_s = self.durations_s.pin_memory()
        if self.labels is not None:
            self.labels = self.labels.pin_memory()
        if self.label_mask is not None:
            self.label_mask = self.label_mask.pin_memory()
        if self.frame_labels is not None:
            self.frame_labels = self.frame_labels.pin_memory()
        return self

    def __len__(self) -> int:
        return self.batch_size

    def __repr__(self) -> str:
        fields = [
            f"batch_size={self.batch_size}",
            f"max_length={self.max_length}",
            f"sample_rate={self.sample_rate}",
        ]
        if self.snr_db is not None:
            fields.append("snr_db=...")
        if self.c50_db is not None:
            fields.append("c50_db=...")
        if self.labels is not None:
            fields.append(f"labels=({self.labels.shape})")
        if self.frame_labels is not None:
            fields.append(f"frame_labels=({self.frame_labels.shape})")
        return f"DataBatch({', '.join(fields)})"
