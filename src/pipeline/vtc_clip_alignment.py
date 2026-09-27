#!/usr/bin/env python3
"""Compare VTC segments from full-file inference vs re-inference on clips.

Approach
--------
1. Convert clip-relative segments to absolute time using clip metadata.
2. **Coverage analysis**: for each label, compute hours where both systems
   agree, hours only detected by full-file VTC, and hours only detected
   by clip VTC.  Uses interval arithmetic (merge + intersect).
3. **Segment matching**: for each full-file segment, find the clip segment
   (same label, same source file) with highest temporal IoU.  Report match
   rates and onset/offset error distributions.

Requires:
    output/{dataset}/vtc_merged/*.parquet            full-file VTC (merged)
    output/{dataset}/vtc_clips/vtc_merged/*.parquet  clip VTC (merged, clip-relative)
    output/{dataset}/stats/clip_stats.parquet         clip boundaries

Usage (login node):
    uv run python -m src.pipeline.vtc_clip_alignment seedlings_10
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import polars as pl

from src.utils import get_dataset_paths

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("vtc_align")

LABELS = ["KCHI", "OCH", "MAL", "FEM"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_parquet_dir(path: Path) -> pl.DataFrame:
    """Concatenate all shard_*.parquet files in a directory."""
    files = sorted(path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {path}")
    return pl.concat([pl.read_parquet(f) for f in files], how="vertical")


def _to_absolute(clip_df: pl.DataFrame, clip_stats: pl.DataFrame) -> pl.DataFrame:
    """Convert clip-relative segments to absolute time.

    Joins on ``clip_id`` to obtain the clip's absolute onset within the
    source recording, then shifts segment onsets/offsets accordingly.

    Returns a DataFrame with columns: ``uid, onset, offset, label``.
    """
    return (
        clip_df.join(
            clip_stats.select("clip_id", "uid", "abs_onset"),
            on="clip_id",
        )
        .with_columns(
            (pl.col("onset") + pl.col("abs_onset")).alias("onset"),
            (pl.col("offset") + pl.col("abs_onset")).alias("offset"),
        )
        .select("uid", "onset", "offset", "label")
    )


# ---------------------------------------------------------------------------
# Interval helpers
# ---------------------------------------------------------------------------


def _merge_intervals(
    intervals: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Merge overlapping or touching intervals.  Returns sorted, disjoint list."""
    if not intervals:
        return []
    s = sorted(intervals)
    merged: list[list[float]] = [[s[0][0], s[0][1]]]
    for on, off in s[1:]:
        if on <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], off)
        else:
            merged.append([on, off])
    return [(a, b) for a, b in merged]


def _interval_total(ivs: list[tuple[float, float]]) -> float:
    return sum(b - a for a, b in ivs)


def _interval_overlap(
    a: list[tuple[float, float]], b: list[tuple[float, float]]
) -> float:
    """Total overlap duration between two sorted, merged interval lists."""
    total = 0.0
    i = j = 0
    while i < len(a) and j < len(b):
        ov_start = max(a[i][0], b[j][0])
        ov_end = min(a[i][1], b[j][1])
        if ov_end > ov_start:
            total += ov_end - ov_start
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return total


# ---------------------------------------------------------------------------
# Coverage analysis (per-label interval overlap)
# ---------------------------------------------------------------------------


def _group_intervals(
    df: pl.DataFrame,
) -> dict[tuple[str, str], list[tuple[float, float]]]:
    """Group segments into ``{(uid, label): [(onset, offset), ...]}``."""
    groups: dict[tuple[str, str], list[tuple[float, float]]] = {}
    uids = df["uid"].to_list()
    onsets = df["onset"].to_list()
    offsets = df["offset"].to_list()
    labels = df["label"].to_list()
    for uid, on, off, lbl in zip(uids, onsets, offsets, labels):
        groups.setdefault((uid, lbl), []).append((on, off))
    return groups


def compute_coverage(
    full_df: pl.DataFrame,
    clip_abs_df: pl.DataFrame,
) -> dict[str, dict[str, float]]:
    """Per-label and all-speech coverage in hours: both, only_full, only_clip."""
    full_groups = _group_intervals(full_df)
    clip_groups = _group_intervals(clip_abs_df)
    all_uids = {uid for uid, _ in list(full_groups) + list(clip_groups)}

    result: dict[str, dict[str, float]] = {}

    for label in LABELS:
        both = only_full = only_clip = 0.0
        for uid in all_uids:
            f = _merge_intervals(full_groups.get((uid, label), []))
            c = _merge_intervals(clip_groups.get((uid, label), []))
            ov = _interval_overlap(f, c)
            both += ov
            only_full += _interval_total(f) - ov
            only_clip += _interval_total(c) - ov
        result[label] = {
            "both_h": both / 3600,
            "only_full_h": only_full / 3600,
            "only_clip_h": only_clip / 3600,
        }

    # All-speech: union across labels per uid
    both_all = only_full_all = only_clip_all = 0.0
    for uid in all_uids:
        full_all: list[tuple[float, float]] = []
        clip_all: list[tuple[float, float]] = []
        for label in LABELS:
            full_all.extend(full_groups.get((uid, label), []))
            clip_all.extend(clip_groups.get((uid, label), []))
        f = _merge_intervals(full_all)
        c = _merge_intervals(clip_all)
        ov = _interval_overlap(f, c)
        both_all += ov
        only_full_all += _interval_total(f) - ov
        only_clip_all += _interval_total(c) - ov
    result["all_speech"] = {
        "both_h": both_all / 3600,
        "only_full_h": only_full_all / 3600,
        "only_clip_h": only_clip_all / 3600,
    }

    return result


# ---------------------------------------------------------------------------
# Segment matching
# ---------------------------------------------------------------------------


def _iou(a_on: float, a_off: float, b_on: float, b_off: float) -> float:
    overlap = max(0.0, min(a_off, b_off) - max(a_on, b_on))
    union = max(a_off, b_off) - min(a_on, b_on)
    return overlap / union if union > 0 else 0.0


def compute_segment_matches(
    full_df: pl.DataFrame,
    clip_abs_df: pl.DataFrame,
    collar_s: float = 0.5,
) -> dict[str, dict]:
    """Match full-file segments → clip segments (same label, best IoU).

    For each full-file segment, find the clip segment with highest temporal
    IoU.  A match is declared if IoU > 0.  A *well-match* further requires
    both onset and offset to agree within ``collar_s``.

    Returns per-label dict with keys:
        n_full, n_clip, n_matched, n_well_matched,
        onset_errors, offset_errors, ious
    """

    def _build(df: pl.DataFrame) -> dict[tuple[str, str], list[tuple[float, float]]]:
        g: dict[tuple[str, str], list[tuple[float, float]]] = {}
        uids = df["uid"].to_list()
        onsets = df["onset"].to_list()
        offsets = df["offset"].to_list()
        labels = df["label"].to_list()
        for uid, on, off, lbl in zip(uids, onsets, offsets, labels):
            g.setdefault((uid, lbl), []).append((on, off))
        for v in g.values():
            v.sort()
        return g

    full_g = _build(full_df)
    clip_g = _build(clip_abs_df)

    results: dict[str, dict] = {}
    for label in LABELS:
        uids = {uid for uid, lbl in list(full_g) + list(clip_g) if lbl == label}
        onset_errs: list[float] = []
        offset_errs: list[float] = []
        ious_list: list[float] = []
        n_full = n_clip = n_matched = n_well = 0

        for uid in uids:
            f_segs = full_g.get((uid, label), [])
            c_segs = clip_g.get((uid, label), [])
            n_full += len(f_segs)
            n_clip += len(c_segs)

            c_start = 0  # sliding window for sorted clip segments
            for f_on, f_off in f_segs:
                # Advance past clip segments that ended well before this one
                while c_start < len(c_segs) and c_segs[c_start][1] < f_on - 5.0:
                    c_start += 1

                best_iou = 0.0
                best_c: tuple[float, float] | None = None
                j = c_start
                while j < len(c_segs):
                    c_on, c_off = c_segs[j]
                    if c_on > f_off + 5.0:
                        break
                    iou_val = _iou(f_on, f_off, c_on, c_off)
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_c = (c_on, c_off)
                    j += 1

                if best_c is not None and best_iou > 0:
                    n_matched += 1
                    on_e = best_c[0] - f_on
                    off_e = best_c[1] - f_off
                    onset_errs.append(on_e)
                    offset_errs.append(off_e)
                    ious_list.append(best_iou)
                    if abs(on_e) <= collar_s and abs(off_e) <= collar_s:
                        n_well += 1

        results[label] = {
            "n_full": n_full,
            "n_clip": n_clip,
            "n_matched": n_matched,
            "n_well_matched": n_well,
            "onset_errors": np.array(onset_errs),
            "offset_errors": np.array(offset_errs),
            "ious": np.array(ious_list),
        }
    return results


# ---------------------------------------------------------------------------
# Figures (extracted to src.plotting.clip_alignment)
# ---------------------------------------------------------------------------


COLLARS = [0.5, 1.0, 2.0, 3.0, 5.0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(dataset: str) -> None:
    paths = get_dataset_paths(dataset)
    orig_dir = paths.output / "vtc_merged"
    clip_dir = paths.output / "vtc_clips" / "vtc_merged"
    clip_stats_path = paths.output / "stats" / "clip_stats.parquet"

    for p in [orig_dir, clip_dir, clip_stats_path]:
        if not p.exists():
            logger.error(f"Missing: {p}")
            sys.exit(1)

    logger.info(f"Dataset : {dataset}")

    # Load
    logger.info("Loading segments...")
    full_df = _load_parquet_dir(orig_dir)
    clip_df = _load_parquet_dir(clip_dir)
    clip_stats = pl.read_parquet(clip_stats_path)

    logger.info(f"  Full-file segments : {len(full_df):,}")
    logger.info(f"  Clip segments      : {len(clip_df):,}")
    logger.info(f"  Clips              : {len(clip_stats):,}")

    # Convert clip segments to absolute time
    logger.info("Converting clip segments to absolute coordinates...")
    clip_abs = _to_absolute(clip_df, clip_stats)
    logger.info(f"  Clip segments (absolute) : {len(clip_abs):,}")

    # Coverage analysis
    logger.info("Computing coverage overlap per label...")
    coverage = compute_coverage(full_df, clip_abs)

    # Segment matching
    logger.info("Matching segments...")
    matches = compute_segment_matches(full_df, clip_abs, collar_s=max(COLLARS))

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print(f"VTC Clip Alignment — {dataset}")
    print("=" * 72)

    print("\n── Coverage (hours) ──")
    print(
        f"  {'Label':<14} {'Both':>8} {'Only full':>10} {'Only clip':>10} {'Agree%':>8}"
    )
    print("  " + "-" * 54)
    for lbl in LABELS + ["all_speech"]:
        c = coverage[lbl]
        total = c["both_h"] + c["only_full_h"] + c["only_clip_h"]
        agree_pct = 100 * c["both_h"] / total if total > 0 else 0
        print(
            f"  {lbl:<14} {c['both_h']:>8.1f} {c['only_full_h']:>10.1f}"
            f" {c['only_clip_h']:>10.1f} {agree_pct:>7.1f}%"
        )

    print(f"\n── Segment matching (multi-collar) ──")
    header = f"  {'Label':<8} {'N_full':>8} {'Matched':>8}"
    for c in COLLARS:
        header += f" {'±' + str(c) + 's':>8}"
    print(header)
    print("  " + "-" * (32 + 9 * len(COLLARS)))
    for lbl in LABELS:
        m = matches[lbl]
        n = m["n_full"]
        line = f"  {lbl:<8} {n:>8,} {m['n_matched']:>8,}"
        for c in COLLARS:
            nw = int(
                np.sum(
                    (np.abs(m["onset_errors"]) <= c) & (np.abs(m["offset_errors"]) <= c)
                )
            )
            pct = 100 * nw / n if n > 0 else 0
            line += f" {pct:>7.1f}%"
        print(line)

    print("\n── Boundary errors (matched segments, seconds) ──")
    print(
        f"  {'Label':<10} {'Onset med':>10} {'Onset IQR':>10}"
        f" {'Offset med':>11} {'Offset IQR':>11}"
    )
    print("  " + "-" * 56)
    for lbl in LABELS:
        m = matches[lbl]
        if len(m["onset_errors"]) > 0:
            on_med = np.median(m["onset_errors"])
            on_iqr = np.percentile(m["onset_errors"], 75) - np.percentile(
                m["onset_errors"], 25
            )
            off_med = np.median(m["offset_errors"])
            off_iqr = np.percentile(m["offset_errors"], 75) - np.percentile(
                m["offset_errors"], 25
            )
            print(
                f"  {lbl:<10} {on_med:>+10.3f} {on_iqr:>10.3f}"
                f" {off_med:>+11.3f} {off_iqr:>11.3f}"
            )
        else:
            print(f"  {lbl:<10} {'n/a':>10} {'n/a':>10} {'n/a':>11} {'n/a':>11}")

    # ------------------------------------------------------------------
    # Save CSVs
    # ------------------------------------------------------------------
    out_dir = paths.output

    # Coverage
    cov_rows = []
    for lbl in LABELS + ["all_speech"]:
        c = coverage[lbl]
        total = c["both_h"] + c["only_full_h"] + c["only_clip_h"]
        cov_rows.append(
            {
                "label": lbl,
                "both_h": round(c["both_h"], 2),
                "only_full_h": round(c["only_full_h"], 2),
                "only_clip_h": round(c["only_clip_h"], 2),
                "agreement_pct": (
                    round(100 * c["both_h"] / total, 2) if total > 0 else 0
                ),
            }
        )
    cov_csv = out_dir / "vtc_clip_alignment_coverage.csv"
    pl.DataFrame(cov_rows).write_csv(cov_csv)
    logger.info(f"Saved -> {cov_csv}")

    # Matches (multi-collar)
    match_rows = []
    for lbl in LABELS:
        m = matches[lbl]
        row = {
            "label": lbl,
            "n_full": m["n_full"],
            "n_clip": m["n_clip"],
            "n_matched": m["n_matched"],
            "match_pct": (
                round(100 * m["n_matched"] / m["n_full"], 2) if m["n_full"] > 0 else 0
            ),
        }
        for c in COLLARS:
            nw = int(
                np.sum(
                    (np.abs(m["onset_errors"]) <= c) & (np.abs(m["offset_errors"]) <= c)
                )
            )
            row[f"within_{c}s_pct"] = (
                round(100 * nw / m["n_full"], 2) if m["n_full"] > 0 else 0
            )
        row["onset_median_s"] = (
            round(float(np.median(m["onset_errors"])), 4)
            if len(m["onset_errors"]) > 0
            else None
        )
        row["offset_median_s"] = (
            round(float(np.median(m["offset_errors"])), 4)
            if len(m["offset_errors"]) > 0
            else None
        )
        match_rows.append(row)
    match_csv = out_dir / "vtc_clip_alignment_matches.csv"
    pl.DataFrame(match_rows).write_csv(match_csv)
    logger.info(f"Saved -> {match_csv}")

    # Compute totals per label
    totals: dict[str, dict[str, float]] = {}
    for lbl in LABELS:
        full_h = (
            float(
                full_df.filter(pl.col("label") == lbl)
                .select((pl.col("offset") - pl.col("onset")).sum())
                .item()
            )
            / 3600
        )
        clip_h = (
            float(
                clip_df.filter(pl.col("label") == lbl)
                .select((pl.col("offset") - pl.col("onset")).sum())
                .item()
            )
            / 3600
        )
        totals[lbl] = {"full_h": full_h, "clip_h": clip_h}
    totals["all_speech"] = {
        "full_h": sum(t["full_h"] for t in totals.values()),
        "clip_h": sum(t["clip_h"] for t in totals.values()),
    }

    # Print totals
    print("\n── Total speech per label (hours) ──")
    print(f"  {'Label':<14} {'Full-file':>10} {'Clip':>10} {'Diff%':>8}")
    print("  " + "-" * 44)
    for lbl in LABELS + ["all_speech"]:
        t = totals[lbl]
        diff_pct = (
            100 * (t["clip_h"] - t["full_h"]) / t["full_h"] if t["full_h"] > 0 else 0
        )
        print(
            f"  {lbl:<14} {t['full_h']:>10.2f} {t['clip_h']:>10.2f} {diff_pct:>+7.2f}%"
        )

    # Figures
    fig_dir = Path("figures") / dataset / "vtc"
    from src.plotting.clip_alignment import save_clip_alignment_figures

    save_clip_alignment_figures(coverage, matches, totals, fig_dir)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python -m src.pipeline.vtc_clip_alignment <dataset>",
            file=sys.stderr,
        )
        sys.exit(1)
    main(sys.argv[1])
