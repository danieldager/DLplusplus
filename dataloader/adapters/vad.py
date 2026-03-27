"""VAD adapter — exposes ``src/pipeline/vad`` outputs as a FeatureProcessor.

Output directory layout consumed by this adapter::

    {output_dir}/
        vad_meta/metadata.parquet       file_id, duration, speech_ratio, …
        vad_raw/segments.parquet        uid, onset, offset, duration
        vad_merged/segments.parquet     uid, onset, offset, duration
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import polars as pl

from dataloader.processor.base import FeatureProcessor
from dataloader.types import MetadataDict, WavID

_META_FILE = "vad_meta/metadata.parquet"
_RAW_SEGS = "vad_raw/segments.parquet"
_MERGED_SEGS = "vad_merged/segments.parquet"


class VADAdapter(FeatureProcessor):
    """Read-only adapter over existing VAD pipeline outputs.

    The VAD pipeline (``src/pipeline/vad.py``) runs via SLURM and produces
    parquet files under ``output/{dataset}/``.  This adapter exposes those
    outputs through the :class:`FeatureProcessor` interface so downstream
    code (``ManifestJoiner``, transforms, loaders) can consume them
    uniformly.

    Parameters
    ----------
    output_dir:
        Root output directory for the dataset (e.g. ``output/seedlings_1``).
        Must contain ``vad_meta/``, ``vad_raw/``, and ``vad_merged/``
        sub-directories.
    """

    name: ClassVar[str] = "vad"
    version: ClassVar[str] = "1.0.0"

    def __init__(self, output_dir: Path | str) -> None:
        self._root = Path(output_dir)
        self._meta_cache: pl.DataFrame | None = None
        self._raw_cache: pl.DataFrame | None = None
        self._merged_cache: pl.DataFrame | None = None

    # ── Lazy loading ──────────────────────────────────────────────────────

    def _meta_df(self) -> pl.DataFrame:
        if self._meta_cache is None:
            path = self._root / _META_FILE
            if not path.is_file():
                raise FileNotFoundError(f"VAD metadata not found: {path}")
            self._meta_cache = pl.read_parquet(path)
        return self._meta_cache

    def _segments_df(self, merged: bool = True) -> pl.DataFrame:
        if merged:
            if self._merged_cache is None:
                path = self._root / _MERGED_SEGS
                if not path.is_file():
                    raise FileNotFoundError(f"VAD merged segments not found: {path}")
                self._merged_cache = pl.read_parquet(path)
            return self._merged_cache
        if self._raw_cache is None:
            path = self._root / _RAW_SEGS
            if not path.is_file():
                raise FileNotFoundError(f"VAD raw segments not found: {path}")
            self._raw_cache = pl.read_parquet(path)
        return self._raw_cache

    # ── FeatureProcessor interface ────────────────────────────────────────

    def process(self, wav_id: WavID, audio_path: Path) -> MetadataDict:
        """Not supported — run ``sbatch slurm/vad.slurm`` instead."""
        raise NotImplementedError(
            "VADAdapter is read-only. Run the VAD pipeline via SLURM: "
            "sbatch slurm/vad.slurm"
        )

    def save(self, wav_id: WavID, metadata: MetadataDict, output_dir: Path) -> Path:
        """Not supported — the VAD pipeline writes its own outputs."""
        raise NotImplementedError(
            "VADAdapter is read-only. The pipeline saves outputs directly."
        )

    def load(self, wav_id: WavID, output_dir: Path | None = None) -> MetadataDict:
        """Load VAD metadata and segments for a single file.

        Parameters
        ----------
        wav_id:
            Waveform identifier (filename stem).
        output_dir:
            Ignored — uses the root set at construction time.

        Returns
        -------
        MetadataDict
            Keys: ``meta`` (dict of summary stats), ``segments_raw``
            (list of dicts), ``segments_merged`` (list of dicts).
        """
        # Pipeline uses file_id in metadata, uid in segments.
        meta_df = self._meta_df()
        meta_rows = meta_df.filter(pl.col("file_id") == wav_id)
        if meta_rows.is_empty():
            raise FileNotFoundError(
                f"No VAD metadata for wav_id={wav_id!r}"
            )

        raw_segs = self._segments_df(merged=False).filter(
            pl.col("uid") == wav_id
        )
        merged_segs = self._segments_df(merged=True).filter(
            pl.col("uid") == wav_id
        )

        return {
            "wav_id": wav_id,
            "meta": meta_rows.row(0, named=True),
            "segments_raw": raw_segs.to_dicts(),
            "segments_merged": merged_segs.to_dicts(),
        }

    def exists(self, wav_id: WavID, output_dir: Path | None = None) -> bool:
        """Check whether VAD outputs exist for *wav_id*."""
        try:
            meta_df = self._meta_df()
        except FileNotFoundError:
            return False
        return meta_df.filter(
            (pl.col("file_id") == wav_id) & pl.col("success")
        ).height > 0

    # ── Convenience ───────────────────────────────────────────────────────

    def list_ids(self) -> list[WavID]:
        """Return all successfully processed wav_ids."""
        try:
            df = self._meta_df()
        except FileNotFoundError:
            return []
        return (
            df.filter(pl.col("success"))
            .get_column("file_id")
            .unique()
            .sort()
            .to_list()
        )

    def as_manifest(self) -> pl.DataFrame:
        """Return the full metadata DataFrame with ``wav_id`` column.

        Renames ``file_id`` → ``wav_id`` for join compatibility.
        """
        return self._meta_df().rename({"file_id": "wav_id"})
