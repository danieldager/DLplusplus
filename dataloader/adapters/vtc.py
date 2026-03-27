"""VTC adapter — exposes ``src/pipeline/vtc`` outputs as a FeatureProcessor.

Output directory layout consumed by this adapter::

    {output_dir}/
        vtc_meta/shard_*.parquet        uid, vtc_threshold, vtc_speech_dur, …
        vtc_raw/shard_*.parquet         uid, onset, offset, duration, label
        vtc_merged/shard_*.parquet      uid, onset, offset, duration, label
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import polars as pl

from dataloader.processor.base import FeatureProcessor
from dataloader.types import MetadataDict, WavID


class VTCAdapter(FeatureProcessor):
    """Read-only adapter over existing VTC pipeline outputs.

    The VTC pipeline (``src/pipeline/vtc.py``) runs via SLURM and produces
    sharded parquet files under ``output/{dataset}/``.  This adapter exposes
    those outputs through the :class:`FeatureProcessor` interface.

    Parameters
    ----------
    output_dir:
        Root output directory for the dataset (e.g. ``output/seedlings_1``).
    """

    name: ClassVar[str] = "vtc"
    version: ClassVar[str] = "1.0.0"

    def __init__(self, output_dir: Path | str) -> None:
        self._root = Path(output_dir)
        self._meta_cache: pl.DataFrame | None = None
        self._raw_cache: pl.DataFrame | None = None
        self._merged_cache: pl.DataFrame | None = None

    # ── Lazy loading ──────────────────────────────────────────────────────

    def _load_shards(self, subdir: str) -> pl.DataFrame:
        path = self._root / subdir
        pq_files = sorted(path.glob("shard_*.parquet"))
        if not pq_files:
            raise FileNotFoundError(
                f"No shard parquets in {path}"
            )
        return pl.concat(
            [pl.read_parquet(f) for f in pq_files],
            how="diagonal_relaxed",
        )

    def _meta_df(self) -> pl.DataFrame:
        if self._meta_cache is None:
            self._meta_cache = self._load_shards("vtc_meta")
        return self._meta_cache

    def _segments_df(self, merged: bool = True) -> pl.DataFrame:
        subdir = "vtc_merged" if merged else "vtc_raw"
        if merged:
            if self._merged_cache is None:
                self._merged_cache = self._load_shards(subdir)
            return self._merged_cache
        if self._raw_cache is None:
            self._raw_cache = self._load_shards(subdir)
        return self._raw_cache

    # ── FeatureProcessor interface ────────────────────────────────────────

    def process(self, wav_id: WavID, audio_path: Path) -> MetadataDict:
        """Not supported — run ``sbatch slurm/vtc.slurm`` instead."""
        raise NotImplementedError(
            "VTCAdapter is read-only. Run the VTC pipeline via SLURM: "
            "sbatch slurm/vtc.slurm"
        )

    def save(self, wav_id: WavID, metadata: MetadataDict, output_dir: Path) -> Path:
        """Not supported — the VTC pipeline writes its own outputs."""
        raise NotImplementedError(
            "VTCAdapter is read-only. The pipeline saves outputs directly."
        )

    def load(self, wav_id: WavID, output_dir: Path | None = None) -> MetadataDict:
        """Load VTC metadata and segments for a single file.

        Returns
        -------
        MetadataDict
            Keys: ``meta`` (dict), ``segments_raw`` (list of dicts with
            ``label``), ``segments_merged`` (list of dicts with ``label``).
        """
        meta_df = self._meta_df()
        meta_rows = meta_df.filter(pl.col("uid") == wav_id)
        if meta_rows.is_empty():
            raise FileNotFoundError(
                f"No VTC metadata for wav_id={wav_id!r}"
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
        """Check whether VTC outputs exist for *wav_id*."""
        try:
            meta_df = self._meta_df()
        except FileNotFoundError:
            return False
        return meta_df.filter(
            (pl.col("uid") == wav_id) & (pl.col("error") == "")
        ).height > 0

    # ── Convenience ───────────────────────────────────────────────────────

    def list_ids(self) -> list[WavID]:
        """Return all successfully processed wav_ids."""
        try:
            df = self._meta_df()
        except FileNotFoundError:
            return []
        return (
            df.filter(pl.col("error") == "")
            .get_column("uid")
            .unique()
            .sort()
            .to_list()
        )

    def as_manifest(self) -> pl.DataFrame:
        """Return the full metadata DataFrame with ``wav_id`` column.

        Renames ``uid`` → ``wav_id`` for join compatibility.
        """
        return self._meta_df().rename({"uid": "wav_id"})
