#!/usr/bin/env python3
"""Listen to clips from WebDataset shards — validation tool.

Extracts clips from tar shards and saves them as individual audio files
alongside their metadata JSON, so you can listen and verify quality.

Two sampling modes:
  --random (default): pure random sampling
  --diverse:          stratified sampling across density, child fraction,
                      IoU, turns, and dominant label — gives a representative
                      spread of the dataset for validation.

Usage:
    python -m src.packaging.listener output/seedlings/shards
    python -m src.packaging.listener output/seedlings/shards -n 50 --diverse --wav
    python -m src.packaging.listener output/seedlings/shards --clip_id uid_0003
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Descriptive file naming
# ---------------------------------------------------------------------------


def _descriptive_name(meta: dict) -> str:
    """Build a human-readable filename from clip metadata.

    Format: {density}den_{child_frac}chi_{iou}iou_{turns}trn_{dominant}_{clip_id}
    Examples:
        hi-den_hi-chi_lo-iou_hi-trn_KCHI_uid_0042
        lo-den_lo-chi_hi-iou_lo-trn_FEM_uid_0007
    """

    def _bucket(val: float, lo: float = 0.33, hi: float = 0.66) -> str:
        if val <= lo:
            return "lo"
        elif val >= hi:
            return "hi"
        return "mid"

    density = meta.get("vtc_speech_density", 0)
    child_frac = meta.get("child_fraction", 0)
    iou = meta.get("vad_vtc_iou", 0)
    n_turns = meta.get("n_turns", 0)
    dominant = meta.get("dominant_label", "UNK")
    clip_id = meta.get("clip_id", "unknown")

    # Turns: <10 = lo, 10-50 = mid, >50 = hi
    turns_bucket = "lo" if n_turns < 10 else ("hi" if n_turns > 50 else "mid")

    return (
        f"{_bucket(density)}-den_"
        f"{_bucket(child_frac)}-chi_"
        f"{_bucket(iou)}-iou_"
        f"{turns_bucket}-trn_"
        f"{dominant or 'UNK'}_"
        f"{clip_id}"
    )


# ---------------------------------------------------------------------------
# Diverse sampling strategy
# ---------------------------------------------------------------------------

_BUCKETS = [
    # (name, meta_key, predicate, target_per_bucket)
    # Pick clips that are extreme/representative along each dimension.
    (
        "density_hi",
        "vtc_speech_density",
        lambda m: m.get("vtc_speech_density", 0) >= 0.75,
        4,
    ),
    (
        "density_lo",
        "vtc_speech_density",
        lambda m: m.get("vtc_speech_density", 0) <= 0.25,
        4,
    ),
    ("child_hi", "child_fraction", lambda m: m.get("child_fraction", 0) >= 0.75, 4),
    ("child_lo", "child_fraction", lambda m: m.get("child_fraction", 0) <= 0.15, 4),
    ("iou_hi", "vad_vtc_iou", lambda m: m.get("vad_vtc_iou", 0) >= 0.6, 4),
    ("iou_lo", "vad_vtc_iou", lambda m: m.get("vad_vtc_iou", 0) <= 0.15, 4),
    ("turns_hi", "n_turns", lambda m: m.get("n_turns", 0) >= 80, 3),
    ("turns_lo", "n_turns", lambda m: m.get("n_turns", 0) <= 5, 3),
    ("dom_KCHI", "dominant_label", lambda m: m.get("dominant_label") == "KCHI", 3),
    ("dom_FEM", "dominant_label", lambda m: m.get("dominant_label") == "FEM", 3),
    ("dom_MAL", "dominant_label", lambda m: m.get("dominant_label") == "MAL", 2),
    ("dom_OCH", "dominant_label", lambda m: m.get("dominant_label") == "OCH", 2),
]


def _diverse_select(
    all_samples: list[dict],
    n: int,
    seed: int,
) -> list[dict]:
    """Select a diverse set of clips across multiple dimensions.

    Strategy: fill named buckets first (extremes along each axis),
    then fill remaining quota with random samples not yet selected.
    """
    rng = random.Random(seed)

    # Parse metadata for all samples
    for s in all_samples:
        if "json" in s["sample"]:
            s["meta"] = json.loads(s["sample"]["json"])
        else:
            s["meta"] = {}

    selected_keys: set[str] = set()
    selected: list[dict] = []

    for _bucket_name, _, predicate, target in _BUCKETS:
        candidates = [
            s
            for s in all_samples
            if s["key"] not in selected_keys and predicate(s["meta"])
        ]
        rng.shuffle(candidates)
        pick = candidates[:target]
        for s in pick:
            selected_keys.add(s["key"])
            selected.append(s)

    # Fill remaining slots with random samples
    remaining = n - len(selected)
    if remaining > 0:
        pool = [s for s in all_samples if s["key"] not in selected_keys]
        rng.shuffle(pool)
        for s in pool[:remaining]:
            selected_keys.add(s["key"])
            selected.append(s)

    return selected[:n]


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------


def extract_clips(
    shard_dir: Path,
    output_dir: Path,
    n: int = 10,
    seed: int = 42,
    clip_ids: list[str] | None = None,
    force_wav: bool = False,
    diverse: bool = False,
) -> list[Path]:
    """Extract clips from shards into individual files.

    Returns paths to the extracted audio files.
    """
    import webdataset as wds

    shard_dir = Path(shard_dir)
    output_dir = Path(output_dir)

    # Clean output dir for fresh extraction
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tar_files = sorted(shard_dir.glob("*.tar"))
    if not tar_files:
        print(f"ERROR: No .tar files in {shard_dir}", file=sys.stderr)
        sys.exit(1)

    # Load all samples
    all_samples: list[dict] = []
    for tar_path in tar_files:
        ds = wds.WebDataset(str(tar_path))  # type: ignore[attr-defined]
        for sample in ds:
            all_samples.append({"key": sample["__key__"], "sample": sample})

    print(f"Found {len(all_samples)} clips across {len(tar_files)} shards")

    # Select samples
    if clip_ids:
        selected = [s for s in all_samples if s["key"] in set(clip_ids)]
        if not selected:
            print(f"ERROR: None of {clip_ids} found", file=sys.stderr)
            sys.exit(1)
    elif diverse:
        selected = _diverse_select(all_samples, n, seed)
        print(
            f"Diverse sampling: {len(selected)} clips across "
            f"{len(_BUCKETS)} categories"
        )
    else:
        random.seed(seed)
        selected = random.sample(all_samples, min(n, len(all_samples)))

    # Extract
    extracted: list[Path] = []
    for item in selected:
        key = item["key"]
        sample = item["sample"]

        # Parse metadata
        meta = json.loads(sample["json"]) if "json" in sample else {}

        # Build descriptive filename
        fname = _descriptive_name(meta)

        # Find the audio key
        audio_ext = None
        for ext in ("flac", "wav"):
            if ext in sample:
                audio_ext = ext
                break

        if audio_ext is None:
            print(f"  WARN: no audio in {key}, skipping")
            continue

        # Convert to WAV if requested (VS Code can't play FLAC)
        if force_wav and audio_ext != "wav":
            import io
            import soundfile as sf

            data, sr = sf.read(io.BytesIO(sample[audio_ext]))
            audio_path = output_dir / f"{fname}.wav"
            sf.write(str(audio_path), data, sr, format="WAV", subtype="PCM_16")
        else:
            audio_path = output_dir / f"{fname}.{audio_ext}"
            audio_path.write_bytes(sample[audio_ext])

        # Write metadata JSON
        meta_path = output_dir / f"{fname}.json"
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

        extracted.append(audio_path)

        # Print summary line
        dur = meta.get("duration", "?")
        density = meta.get("vtc_speech_density", "?")
        child_frac = meta.get("child_fraction", "?")
        iou = meta.get("vad_vtc_iou", "?")
        n_turns = meta.get("n_turns", "?")
        dominant = meta.get("dominant_label", "?")
        print(f"  {fname}")
        print(
            f"    dur={dur}s  den={density}  chi={child_frac}  "
            f"iou={iou}  turns={n_turns}  dom={dominant}"
        )

    print(f"\nExtracted {len(extracted)} clips to {output_dir}/")
    return extracted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract clips from WebDataset shards for listening/validation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "shard_dir",
        help="Path to the shards directory (e.g. output/seedlings/shards).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output directory for extracted clips (default: {shard_dir}/samples/).",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=50,
        help="Number of clips to extract (default: 50).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for clip selection.",
    )
    parser.add_argument(
        "--clip_id",
        nargs="+",
        default=None,
        help="Extract specific clip(s) by ID.",
    )
    parser.add_argument(
        "--wav",
        action="store_true",
        help="Convert to WAV on extraction (VS Code can play .wav but not .flac).",
    )
    parser.add_argument(
        "--diverse",
        action="store_true",
        help="Use stratified diverse sampling instead of random.",
    )
    args = parser.parse_args()

    shard_dir = Path(args.shard_dir)
    output_dir = Path(args.output) if args.output else shard_dir / "samples"

    extract_clips(
        shard_dir=shard_dir,
        output_dir=output_dir,
        n=args.n,
        seed=args.seed,
        clip_ids=args.clip_id,
        force_wav=args.wav,
        diverse=args.diverse,
    )


if __name__ == "__main__":
    main()
