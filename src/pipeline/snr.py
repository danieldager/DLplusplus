#!/usr/bin/env python3
"""
Brouhaha SNR pipeline: extract per-frame signal-to-noise ratio and average-pool
into fixed-width windows.

For each audio file in the manifest:
  1. Load the Brouhaha model (pyannote-based multi-task: VAD + SNR + C50)
  2. Run inference → per-frame SNR at model resolution (~16 ms step)
  3. Average-pool into fixed windows (default 1 s) → compact float16 array
  4. Save pooled SNR as compressed .npz per file

Paths are derived from the dataset name:
    manifests/{dataset}.<ext>              input manifest
    output/{dataset}/snr/                  pooled SNR arrays (.npz)
    output/{dataset}/snr_meta/             per-file metadata  (.parquet)

Usage:
    python -m src.pipeline.snr chunks30
    python -m src.pipeline.snr chunks30 --pool_window 0.5
    python -m src.pipeline.snr chunks30 --sample 500

SLURM:
    sbatch slurm/snr.slurm chunks30
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

from src.core.vad_processing import set_seeds
from src.utils import (
    add_sample_argument,
    atomic_write_parquet,
    get_dataset_paths,
    load_manifest,
    log_benchmark,
    sample_manifest,
    shard_list,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("snr")

# Default Brouhaha checkpoint (shipped with the repo)
_DEFAULT_MODEL = "models/best/checkpoints/best.ckpt"


def _ensure_model(model_path: str) -> str:
    """Return *model_path* after ensuring the checkpoint exists on disk.

    If the default checkpoint is missing, it is automatically downloaded
    from HuggingFace (``ylacombe/brouhaha-best``).  For non-default paths the
    caller is responsible for providing a valid file.
    """
    p = Path(model_path)
    if p.exists():
        return model_path

    if model_path != _DEFAULT_MODEL:
        raise FileNotFoundError(
            f"Brouhaha checkpoint not found: {model_path}"
        )

    logger.info("Brouhaha checkpoint not found — downloading from Hugging Face …")
    from scripts.download_brouhaha import ensure_brouhaha_checkpoint

    ensure_brouhaha_checkpoint(p)
    return model_path


def _hhmmss(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ------------------------------------------------------------------
# SNR extraction helpers
# ------------------------------------------------------------------


def _extract_snr(
    pipeline,
    audio_path: Path,
) -> tuple[np.ndarray, float]:
    """Run Brouhaha on one file and return (raw_snr, step_s).

    Returns
    -------
    raw_snr : 1-D float32 array of per-frame SNR values.
    step_s  : Frame step in seconds (from the model's receptive field).
    """
    file = {"uri": audio_path.stem, "audio": str(audio_path)}
    output = pipeline(file)
    snr: np.ndarray = output["snr"]  # (n_frames,) float

    # Resolve the frame step from the underlying model
    seg = pipeline._segmentation
    if hasattr(seg.model, "receptive_field"):
        step_s = float(seg.model.receptive_field.step)  # type: ignore[union-attr]
    elif hasattr(seg.model, "introspection"):
        step_s = float(seg.model.introspection.frames.step)  # type: ignore[union-attr]
    elif hasattr(seg, "_frames"):
        step_s = float(seg._frames.step)  # type: ignore[union-attr]
    else:
        # Fallback: estimate from output length and audio duration
        import soundfile as sf
        info = sf.info(str(audio_path))
        step_s = info.duration / max(len(snr), 1)

    return snr.astype(np.float32), step_s


def pool_snr(
    raw_snr: np.ndarray,
    step_s: float,
    pool_window_s: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Average-pool raw per-frame SNR into fixed-width windows.

    Parameters
    ----------
    raw_snr : 1-D float32 array, one value per model frame.
    step_s  : Model frame step in seconds.
    pool_window_s : Averaging window in seconds.

    Returns
    -------
    pooled : 1-D float16 array, one value per pool window.
    pool_step_s : Effective step of the pooled array (= pool_window_s).
    """
    frames_per_window = max(1, int(round(pool_window_s / step_s)))

    n = len(raw_snr)
    # Number of full windows
    n_windows = n // frames_per_window
    if n_windows == 0:
        # File shorter than one window — average everything
        return np.array([raw_snr.mean()], dtype=np.float16), pool_window_s

    # Trim to full windows and reshape
    trimmed = raw_snr[: n_windows * frames_per_window]
    pooled = trimmed.reshape(n_windows, frames_per_window).mean(axis=1)

    # If there's a leftover tail, average it into one final bin
    leftover = raw_snr[n_windows * frames_per_window :]
    if len(leftover) > 0:
        pooled = np.append(pooled, leftover.mean())

    return pooled.astype(np.float16), pool_window_s


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main(
    dataset: str,
    model_path: str = _DEFAULT_MODEL,
    pool_window: float = 1.0,
    device: str = "cuda",
    array_id: int | None = None,
    array_count: int | None = None,
    sample: int | float | None = None,
):
    set_seeds(42)

    paths = get_dataset_paths(dataset)
    logger.info(f"Dataset: {dataset}")
    logger.info(f"  manifest  : {paths.manifest}")
    logger.info(f"  output    : {paths.output}")
    logger.info(f"  pool      : {pool_window}s window")

    # ------------------------------------------------------------------
    # Load manifest and shard
    # ------------------------------------------------------------------
    manifest_df = load_manifest(paths.manifest)
    manifest_df = sample_manifest(manifest_df, sample)
    if sample is not None:
        logger.info(f"  sample    : {len(manifest_df)} files")

    resolved_paths = manifest_df["path"].drop_nulls().to_list()
    file_ids = [Path(p).stem for p in resolved_paths]
    uid_to_path = dict(zip(file_ids, resolved_paths))

    if array_id is not None and array_count is not None:
        file_ids = shard_list(file_ids, array_id, array_count)
        logger.info(
            f"Shard {array_id}/{array_count - 1}: {len(file_ids)} files"
        )

    shard_id = array_id if array_id is not None else 0

    # ------------------------------------------------------------------
    # Resume: skip files already processed
    # ------------------------------------------------------------------
    snr_dir = paths.output / "snr"
    snr_dir.mkdir(parents=True, exist_ok=True)

    completed_uids: set[str] = set()
    meta_dir = paths.output / "snr_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_path = meta_dir / f"shard_{shard_id}.parquet"
    prev_meta_df: pl.DataFrame | None = None

    # Check all shard metas for already-processed files
    all_meta_files = sorted(meta_dir.glob("shard_*.parquet"))
    if all_meta_files:
        all_meta = pl.read_parquet(all_meta_files)
        completed_uids = set(all_meta["uid"].to_list())

    if meta_path.exists():
        prev_meta_df = pl.read_parquet(meta_path)

    remaining = [uid for uid in file_ids if uid not in completed_uids]
    if len(remaining) < len(file_ids):
        skipped = len(file_ids) - len(remaining)
        logger.info(f"Resume: {skipped} done, {len(remaining)} remaining")
    file_ids_to_process = remaining

    if not file_ids_to_process and not file_ids:
        logger.info("No files to process.")
        return

    # ------------------------------------------------------------------
    # Load model  (lazy imports — keep module importable on login nodes)
    # ------------------------------------------------------------------
    model_path = _ensure_model(model_path)

    import soundfile as sf
    import torch
    from src.compat import patch_torchaudio
    patch_torchaudio()
    from pyannote.audio import Model
    from brouhaha.pipeline import RegressiveActivityDetectionPipeline

    logger.info(f"Model: {model_path}")
    model = Model.from_pretrained(Path(model_path), strict=False)

    if model.device.type == "cpu" and device != "cpu":
        dev = torch.device(device if torch.cuda.is_available() else "cpu")
        model.to(dev)
        logger.info(f"Device: {dev}")
    else:
        logger.info(f"Device: {model.device}")

    pipeline = RegressiveActivityDetectionPipeline(segmentation=model)

    # Resolve model frame step for logging
    seg = pipeline._segmentation
    if hasattr(seg.model, "receptive_field"):
        model_step = float(seg.model.receptive_field.step)  # type: ignore[union-attr]
    elif hasattr(seg.model, "introspection"):
        model_step = float(seg.model.introspection.frames.step)  # type: ignore[union-attr]
    else:
        model_step = 0.016  # typical default
    logger.info(f"  frame step: {model_step * 1000:.1f} ms")
    logger.info(
        f"  pool ratio: {int(round(pool_window / model_step))}:1 "
        f"({model_step * 1000:.1f} ms → {pool_window * 1000:.0f} ms)"
    )

    # ------------------------------------------------------------------
    # Process files
    # ------------------------------------------------------------------
    meta_rows: list[dict] = []
    n_errors = 0
    total = len(file_ids_to_process)
    t0 = time.time()
    log_every = max(1, total // 20)

    # Pre-stat files for GB-based progress
    file_sizes: dict[str, int] = {}
    for uid in file_ids_to_process:
        try:
            file_sizes[uid] = Path(uid_to_path[uid]).stat().st_size
        except OSError:
            file_sizes[uid] = 0
    total_bytes = sum(file_sizes.values()) or 1
    bytes_done = 0
    print(
        f"Shard {shard_id}: {total} files, {total_bytes / 1e9:.1f} GB",
        flush=True,
    )

    for i, uid in enumerate(file_ids_to_process, 1):
        audio_path = Path(uid_to_path[uid])

        try:
            raw_snr, step_s = _extract_snr(pipeline, audio_path)
            pooled, pool_step = pool_snr(raw_snr, step_s, pool_window)

            # Save pooled SNR
            np.savez_compressed(
                snr_dir / f"{uid}.npz",
                snr=pooled,
                pool_step_s=np.float32(pool_step),
                model_step_s=np.float32(step_s),
                n_raw_frames=np.int32(len(raw_snr)),
            )

            info = sf.info(str(audio_path))
            file_dur = info.duration

            meta_rows.append({
                "uid": uid,
                "snr_status": "ok",
                "duration": round(file_dur, 3),
                "n_raw_frames": len(raw_snr),
                "n_pooled_frames": len(pooled),
                "model_step_s": round(step_s, 6),
                "pool_step_s": round(pool_step, 6),
                "snr_mean": round(float(raw_snr.mean()), 2),
                "snr_std": round(float(raw_snr.std()), 2),
                "snr_min": round(float(raw_snr.min()), 2),
                "snr_max": round(float(raw_snr.max()), 2),
                "error": "",
            })

        except Exception as e:
            n_errors += 1
            logger.warning(f"{uid}: {e}")
            meta_rows.append({
                "uid": uid,
                "snr_status": "error",
                "duration": 0.0,
                "n_raw_frames": 0,
                "n_pooled_frames": 0,
                "model_step_s": 0.0,
                "pool_step_s": 0.0,
                "snr_mean": 0.0,
                "snr_std": 0.0,
                "snr_min": 0.0,
                "snr_max": 0.0,
                "error": str(e),
            })

        bytes_done += file_sizes.get(uid, 0)
        now = time.time()
        elapsed = now - t0
        rate = bytes_done / elapsed if elapsed > 0 else 0
        remaining_bytes = total_bytes - bytes_done
        remaining_s = remaining_bytes / rate if rate > 0 else 0
        eta = (
            f"{remaining_s / 60:.0f}m"
            if remaining_s < 3600
            else f"{remaining_s / 3600:.1f}h"
        )
        pct = 100.0 * bytes_done / total_bytes
        if i % log_every == 0 or i == total:
            print(
                f"  SNR  {i:>4}/{total}"
                f"  {bytes_done / 1e9:.1f}/{total_bytes / 1e9:.1f} GB"
                f" ({pct:.0f}%)  ETA {eta}",
                flush=True,
            )

    # ------------------------------------------------------------------
    # Merge with previous results (resume case)
    # ------------------------------------------------------------------
    new_meta_df = pl.DataFrame(meta_rows) if meta_rows else None

    meta_parts: list[pl.DataFrame] = []
    if prev_meta_df is not None:
        if new_meta_df is not None:
            new_uids = set(new_meta_df["uid"].to_list())
            kept = prev_meta_df.filter(~pl.col("uid").is_in(list(new_uids)))
            if not kept.is_empty():
                meta_parts.append(kept)
        else:
            meta_parts.append(prev_meta_df)
    if new_meta_df is not None:
        meta_parts.append(new_meta_df)

    meta_df = pl.concat(meta_parts) if meta_parts else pl.DataFrame()

    # Deduplicate by uid
    if not meta_df.is_empty():
        before = len(meta_df)
        meta_df = meta_df.unique(subset=["uid"], keep="last")
        if len(meta_df) < before:
            logger.warning(
                f"Dedup: removed {before - len(meta_df)} duplicate "
                f"meta rows (shard {shard_id})"
            )

    # ------------------------------------------------------------------
    # Save metadata
    # ------------------------------------------------------------------
    atomic_write_parquet(meta_df, meta_path)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    processed = total - n_errors
    wall = time.time() - t0
    print(flush=True)
    logger.info(f"{'─' * 50}")
    logger.info(f"Shard {shard_id} complete")
    logger.info(f"  Files     : {processed}/{total}  ({n_errors} errors)")
    if not meta_df.is_empty():
        ok = meta_df.filter(pl.col("snr_status") == "ok")
        if not ok.is_empty():
            logger.info(
                f"  SNR       : mean={ok['snr_mean'].mean():.1f} dB  "
                f"std={ok['snr_std'].mean():.1f} dB"
            )
            logger.info(
                f"  Pooled    : {ok['n_pooled_frames'].sum():,} frames "
                f"({pool_window}s windows)"
            )
    logger.info(f"  Wall time : {_hhmmss(wall)}")
    logger.info(f"{'─' * 50}")

    # ------------------------------------------------------------------
    # Benchmark logging
    # ------------------------------------------------------------------
    wall_seconds = time.time() - t0
    total_bytes_final = sum(
        os.path.getsize(uid_to_path[uid])
        for uid in file_ids_to_process
        if os.path.exists(uid_to_path[uid])
    )
    log_benchmark(
        step="snr",
        dataset=dataset,
        n_files=total,
        wall_seconds=wall_seconds,
        total_bytes=total_bytes_final,
        n_workers=1,
        extra={"device": device, "shard_id": shard_id, "pool_window": pool_window},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Brouhaha SNR pipeline: extract and pool signal-to-noise ratio.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m src.pipeline.snr chunks30\n"
            "  python -m src.pipeline.snr chunks30 --pool_window 0.5\n"
            "  python -m src.pipeline.snr chunks30 --sample 500\n"
        ),
    )
    parser.add_argument(
        "dataset",
        help="Dataset name — used to derive output/, metadata/, and figures/ directories.",
    )
    parser.add_argument(
        "--model_path",
        default=_DEFAULT_MODEL,
        help=f"Brouhaha model checkpoint (default: {_DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--pool_window",
        type=float,
        default=1.0,
        help="Averaging window in seconds for pooling raw SNR (default: 1.0)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for inference (default: cuda)",
    )
    parser.add_argument(
        "--array_id",
        type=int,
        help="SLURM array task ID",
    )
    parser.add_argument(
        "--array_count",
        type=int,
        help="Total SLURM array tasks",
    )
    add_sample_argument(parser)

    args = parser.parse_args()
    main(**vars(args))
