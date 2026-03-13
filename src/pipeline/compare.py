"""VAD vs VTC comparison helpers.

Computes per-file IoU / Precision / Recall between VAD and VTC segment
outputs, generates a 2x3 matplotlib dashboard, and diagnoses low-IoU files.

These functions are called during the packaging step (``package.py``) to
produce comparison CSVs and figures alongside the packaged shards.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import polars as pl


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_parquet_dir(directory: Path) -> pl.DataFrame:
    """Read all .parquet files in *directory* into one DataFrame."""
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {directory}")
    return pl.read_parquet(files)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def _flatten_intervals(
    df: pl.LazyFrame,
    partition_cols: list[str] | None = None,
) -> pl.LazyFrame:
    """Merge overlapping intervals within each partition."""
    if partition_cols is None:
        partition_cols = ["uid"]
    return (
        df.sort(*partition_cols, "onset")
        .with_columns(
            _prev_max_off=pl.col("offset")
            .shift(1)
            .cum_max()
            .over(*partition_cols)
            .fill_null(0.0)
        )
        .with_columns(
            _new_grp=(pl.col("onset") > pl.col("_prev_max_off")).cast(pl.Int32)
        )
        .with_columns(_grp_id=pl.col("_new_grp").cum_sum().over(*partition_cols))
        .group_by(*partition_cols, "_grp_id")
        .agg(onset=pl.col("onset").min(), offset=pl.col("offset").max())
        .with_columns(duration=(pl.col("offset") - pl.col("onset")).round(3))
        .drop("_grp_id")
    )


def _sweep_tp(
    vad_on: np.ndarray, vad_off: np.ndarray,
    vtc_on: np.ndarray, vtc_off: np.ndarray,
) -> float:
    """Total overlap of two sorted non-overlapping interval sets."""
    i = j = 0
    tp = 0.0
    n_v, n_t = len(vad_on), len(vtc_on)
    while i < n_v and j < n_t:
        s = max(vad_on[i], vtc_on[j])
        e = min(vad_off[i], vtc_off[j])
        if s < e:
            tp += e - s
        if vad_off[i] < vtc_off[j]:
            i += 1
        else:
            j += 1
    return tp


def _iter_uid_groups(
    df: pl.DataFrame,
) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Yield (uid, onsets, offsets) from a DataFrame sorted by uid, onset."""
    if df.is_empty():
        return
    sorted_df = df.sort("uid", "onset")
    uids = sorted_df["uid"].to_numpy()
    onsets = sorted_df["onset"].to_numpy()
    offsets = sorted_df["offset"].to_numpy()
    prev = 0
    for i in range(1, len(uids)):
        if uids[i] != uids[prev]:
            yield str(uids[prev]), onsets[prev:i], offsets[prev:i]
            prev = i
    yield str(uids[prev]), onsets[prev:], offsets[prev:]


def calculate_metrics(
    df_vad: pl.DataFrame,
    df_vtc: pl.DataFrame,
    vad_meta: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Per-file IoU, Precision, Recall between VAD and VTC DataFrames."""
    vad_flat = _flatten_intervals(df_vad.lazy()).collect()
    vtc_flat = _flatten_intervals(df_vtc.lazy()).collect()

    vad_stats = vad_flat.group_by("uid").agg(
        pl.col("duration").sum().alias("vad_dur"),
    )
    vtc_stats = vtc_flat.group_by("uid").agg(
        pl.col("duration").sum().alias("vtc_dur"),
    )

    vtc_map: dict[str, tuple[np.ndarray, np.ndarray]] = {
        uid: (on, off) for uid, on, off in _iter_uid_groups(vtc_flat)
    }

    tp_rows: list[dict] = []
    for uid, vad_on, vad_off in _iter_uid_groups(vad_flat):
        if uid in vtc_map:
            vtc_on, vtc_off = vtc_map[uid]
            tp = _sweep_tp(vad_on, vad_off, vtc_on, vtc_off)
            tp_rows.append({"uid": uid, "TP": tp})

    intersection = pl.DataFrame(tp_rows) if tp_rows else pl.DataFrame(
        {"uid": pl.Series([], dtype=pl.String),
         "TP": pl.Series([], dtype=pl.Float64)}
    )

    results = (
        vad_stats.lazy()
        .join(vtc_stats.lazy(), on="uid", how="full", coalesce=True)
        .join(intersection.lazy(), on="uid", how="left")
        .fill_null(0)
        .with_columns(
            FP=pl.col("vtc_dur") - pl.col("TP"),
            FN=pl.col("vad_dur") - pl.col("TP"),
        )
        .with_columns(
            IoU=(pl.col("TP") / (pl.col("TP") + pl.col("FP") + pl.col("FN"))),
            Precision=(pl.col("TP") / (pl.col("TP") + pl.col("FP"))),
            Recall=(pl.col("TP") / (pl.col("TP") + pl.col("FN"))),
        )
        .collect()
    )

    if vad_meta is not None and "file_total" in vad_meta.columns:
        results = results.join(
            vad_meta.select("uid", "file_total"), on="uid", how="left"
        )
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_summary(results: pl.DataFrame, label: str, low_thresh: float = 0.5) -> dict:
    """Print executive summary and return global aggregates."""
    tot = results.select(
        pl.sum("TP"), pl.sum("FP"), pl.sum("FN"),
        pl.sum("vad_dur"), pl.sum("vtc_dur"),
    ).to_dicts()[0]

    denom = tot["TP"] + tot["FP"] + tot["FN"]
    g_iou = tot["TP"] / denom if denom else 0
    g_prec = tot["TP"] / (tot["TP"] + tot["FP"]) if (tot["TP"] + tot["FP"]) else 0
    g_rec = tot["TP"] / (tot["TP"] + tot["FN"]) if (tot["TP"] + tot["FN"]) else 0

    n = len(results)
    n_low = results.filter(pl.col("IoU") < low_thresh).height
    n_high = results.filter(pl.col("IoU") >= low_thresh).height

    print(f"\n  {label}")
    print(f"  {'─' * 44}")
    print(f"  IoU:  {g_iou:.2%}   P: {g_prec:.2%}   R: {g_rec:.2%}")
    print(f"  VAD: {tot['vad_dur']/3600:.1f}h   VTC: {tot['vtc_dur']/3600:.1f}h")
    print(f"  High IoU (>={low_thresh:.0%}): {n_high:,} ({n_high/n:.0%})   "
          f"Low: {n_low:,} ({n_low/n:.0%})")
    print(f"  {'─' * 44}")

    return {"vad_h": tot["vad_dur"] / 3600, "vtc_h": tot["vtc_dur"] / 3600,
            "iou": g_iou, "precision": g_prec, "recall": g_rec}


def diagnose_low_iou(
    results: pl.DataFrame, vtc_df: pl.DataFrame, low_thresh: float = 0.5,
) -> pl.DataFrame:
    """Categorize low-IoU files and print a summary."""
    low = results.filter(pl.col("IoU") < low_thresh)
    vtc_seg_counts = vtc_df.group_by("uid").agg(pl.len().alias("n_vtc_segs"))

    diag = (
        low.join(vtc_seg_counts, on="uid", how="left")
        .fill_null(0)
        .with_columns(
            category=pl.when(pl.col("n_vtc_segs") == 0)
            .then(pl.lit("vtc_silent"))
            .when(pl.col("vtc_dur") < 0.5 * pl.col("vad_dur"))
            .then(pl.lit("vtc_under_detect"))
            .when(pl.col("vtc_dur") > 2.0 * pl.col("vad_dur"))
            .then(pl.lit("vtc_over_detect"))
            .otherwise(pl.lit("moderate_mismatch"))
        )
    )

    print(f"\n  Low-IoU diagnostics (IoU < {low_thresh:.0%}, n={diag.height})")
    print(f"  {'─' * 36}")
    for row in (
        diag.group_by("category").agg(pl.len().alias("n"))
        .sort("n", descending=True).iter_rows(named=True)
    ):
        print(f"    {row['category']:<20s} {row['n']:>6d}")

    return diag


# ---------------------------------------------------------------------------
# Dashboard figure
# ---------------------------------------------------------------------------


def plot_dashboard(
    results: pl.DataFrame, global_stats: dict, title: str,
    output_path: Path, low_thresh: float = 0.5, target_iou: float = 0.9,
) -> None:
    """Generate and save a 2x3 matplotlib comparison dashboard."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    df_low = results.filter(pl.col("IoU") < low_thresh)
    df_high = results.filter(pl.col("IoU") >= low_thresh)

    # Volume bar chart
    ax = axes[0, 0]
    ax.bar(["VAD", "VTC"], [global_stats["vad_h"], global_stats["vtc_h"]],
           color=["#3498db", "#e74c3c"])
    for i, v in enumerate([global_stats["vad_h"], global_stats["vtc_h"]]):
        ax.text(i, v + 0.02 * v, f"{v:.1f}h", ha="center", fontsize=9)
    ax.set_title("Volume (hours)")
    ax.set_ylabel("Hours")

    # Precision vs Recall scatter
    ax = axes[0, 1]
    ax.scatter(results["Recall"].to_numpy(), results["Precision"].to_numpy(),
               alpha=0.3, s=8, color="#AB63FA")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision vs Recall")
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)

    # IoU histogram
    ax = axes[0, 2]
    iou_vals = results["IoU"].drop_nulls().drop_nans().to_numpy()
    ax.hist(iou_vals, bins=50, color="#636EFA", edgecolor="white", linewidth=0.3)
    ax.axvline(target_iou, color="red", linestyle="--", linewidth=1, label=f"target={target_iou}")
    ax.axvline(low_thresh, color="orange", linestyle=":", linewidth=1, label=f"low={low_thresh}")
    ax.set_xlabel("IoU"); ax.set_title("IoU Distribution"); ax.legend(fontsize=8)

    # Duration split
    ax = axes[1, 0]
    if "file_total" in results.columns:
        p99 = results["file_total"].quantile(0.99)
        for grp, name, color in [(df_high, "High IoU", "#1f77b4"), (df_low, "Low IoU", "#d62728")]:
            if "file_total" in grp.columns and grp.height > 0:
                vals = grp.filter(pl.col("file_total") < p99)["file_total"].to_numpy()
                ax.hist(vals, bins=40, alpha=0.6, label=name, color=color, density=True)
        ax.set_xlabel(f"Duration (0\u2013{p99:.0f}s)"); ax.legend(fontsize=8)
    ax.set_title("Duration (High vs Low IoU)")

    # VAD speech ratio
    ax = axes[1, 1]
    if "file_total" in results.columns:
        for grp, name, color in [(df_high, "High IoU", "#1f77b4"), (df_low, "Low IoU", "#d62728")]:
            if grp.height > 0 and "file_total" in grp.columns:
                ratio = (grp["vad_dur"] / grp["file_total"]).to_numpy()
                ax.hist(ratio, bins=40, alpha=0.6, label=name, color=color, density=True)
        ax.set_xlabel("VAD speech ratio"); ax.legend(fontsize=8)
    ax.set_title("Speech Ratio (High vs Low IoU)")

    # Threshold distribution
    ax = axes[1, 2]
    if "vtc_threshold" in results.columns:
        thresholds = results["vtc_threshold"].drop_nulls().drop_nans().to_numpy()
        ax.hist(thresholds, bins=30, color="#FF6692", edgecolor="white", linewidth=0.3)
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="default (0.5)")
        median_val = results["vtc_threshold"].drop_nulls().drop_nans().mean()
        median_t: float = float(median_val) if isinstance(median_val, (int, float)) else 0.0
        ax.axvline(median_t, color="#19D3F3", linestyle="-", linewidth=1.5, label=f"median ({median_t:.2f})")
        ax.set_xlabel("Threshold"); ax.set_title("Threshold Distribution"); ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No vtc_threshold data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Threshold Distribution")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved figure: {output_path.name}")
