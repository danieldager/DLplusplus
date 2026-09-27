"""SNR adapter — exposes ``src/pipeline/snr`` outputs as a FeatureProcessor.

Output directory layout consumed by this adapter::

    {output_dir}/
        snr_meta/shard_*.parquet    uid, snr_status, snr_mean, c50_mean, …
        snr/{uid}.npz               snr, c50, vad (float16 per-frame arrays)
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import polars as pl

from dataloader.processor.base import FeatureProcessor
from dataloader.types import MetadataDict, WavID


class SNRAdapter(FeatureProcessor):
    """Read-only adapter over existing SNR (Brouhaha) pipeline outputs.

    The SNR pipeline (``src/pipeline/snr.py``) runs via SLURM and produces
    sharded metadata parquets + per-file ``.npz`` arrays.

    Parameters
    ----------
    output_dir:
        Root output directory for the dataset (e.g. ``output/seedlings_1``).
    """

    name: ClassVar[str] = "snr"
    version: ClassVar[str] = "1.0.0"

    def __init__(self, output_dir: Path | str) -> None:
        self._root = Path(output_dir)
        self._meta_cache: pl.DataFrame | None = None

    # ── Lazy loading ──────────────────────────────────────────────────────

    def _meta_df(self) -> pl.DataFrame:
        if self._meta_cache is None:
            path = self._root / "snr_meta"
            pq_files = sorted(path.glob("shard_*.parquet"))
            if not pq_files:
                raise FileNotFoundError(f"No shard parquets in {path}")
            self._meta_cache = pl.concat(
                [pl.read_parquet(f) for f in pq_files],
                how="diagonal_relaxed",
            )
        return self._meta_cache

    def _npz_path(self, wav_id: WavID) -> Path:
        return self._root / "snr" / f"{wav_id}.npz"

    # ── FeatureProcessor interface ────────────────────────────────────────

    def process(self, wav_id: WavID, audio_path: Path) -> MetadataDict:
        """Not supported — run ``sbatch slurm/snr.slurm`` instead."""
        raise NotImplementedError(
            "SNRAdapter is read-only. Run the SNR pipeline via SLURM: "
            "sbatch slurm/snr.slurm"
        )

    def save(self, wav_id: WavID, metadata: MetadataDict, output_dir: Path) -> Path:
        """Not supported — the SNR pipeline writes its own outputs."""
        raise NotImplementedError(
            "SNRAdapter is read-only. The pipeline saves outputs directly."
        )

    def load(self, wav_id: WavID, output_dir: Path | None = None) -> MetadataDict:
        """Load SNR metadata and per-frame arrays for a single file.

        Returns
        -------
        MetadataDict
            Keys: ``meta`` (dict of summary stats), ``snr`` (float16 array),
            ``c50`` (float16 array), ``vad`` (float16 array),
            ``step_s`` (float).
        """
        meta_df = self._meta_df()
        meta_rows = meta_df.filter(pl.col("uid") == wav_id)
        if meta_rows.is_empty():
            raise FileNotFoundError(
                f"No SNR metadata for wav_id={wav_id!r}"
            )

        npz_path = self._npz_path(wav_id)
        if not npz_path.is_file():
            raise FileNotFoundError(f"No SNR arrays at {npz_path}")

        with np.load(npz_path, allow_pickle=False) as npz:
            arrays = {
                "snr": npz["snr"],
                "c50": npz["c50"],
                "vad": npz["vad"],
                "step_s": float(npz["step_s"]),
            }

        return {
            "wav_id": wav_id,
            "meta": meta_rows.row(0, named=True),
            **arrays,
        }

    def exists(self, wav_id: WavID, output_dir: Path | None = None) -> bool:
        """Check whether SNR outputs exist for *wav_id*."""
        try:
            meta_df = self._meta_df()
        except FileNotFoundError:
            return False
        has_meta = meta_df.filter(
            (pl.col("uid") == wav_id) & (pl.col("snr_status") == "ok")
        ).height > 0
        return has_meta and self._npz_path(wav_id).is_file()

    # ── Convenience ───────────────────────────────────────────────────────

    def list_ids(self) -> list[WavID]:
        """Return all successfully processed wav_ids."""
        try:
            df = self._meta_df()
        except FileNotFoundError:
            return []
        return (
            df.filter(pl.col("snr_status") == "ok")
            .get_column("uid")
            .unique()
            .sort()
            .to_list()
        )

    def as_manifest(self) -> pl.DataFrame:
        """Return the full metadata DataFrame with ``wav_id`` column."""
        return self._meta_df().rename({"uid": "wav_id"})
