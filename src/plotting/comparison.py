"""Compare pipeline dashboard figure."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl


def plot_dashboard(
    results: pl.DataFrame,
    global_stats: dict,
    title: str,
    output_path: Path,
    low_thresh: float = 0.5,
    target_iou: float = 0.9,
) -> None:
    """Generate and save a 2×3 matplotlib dashboard."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    df_low = results.filter(pl.col("IoU") < low_thresh)
    df_high = results.filter(pl.col("IoU") >= low_thresh)

    # 1. Volume bar chart
    ax = axes[0, 0]
    ax.bar(
        ["VAD", "VTC"],
        [global_stats["vad_h"], global_stats["vtc_h"]],
        color=["#3498db", "#e74c3c"],
    )
    for i, v in enumerate([global_stats["vad_h"], global_stats["vtc_h"]]):
        ax.text(i, v + 0.02 * v, f"{v:.1f}h", ha="center", fontsize=9)
    ax.set_title("Volume (hours)")
    ax.set_ylabel("Hours")

    # 2. Precision vs Recall scatter
    ax = axes[0, 1]
    ax.scatter(
        results["Recall"].to_numpy(),
        results["Precision"].to_numpy(),
        alpha=0.3,
        s=8,
        color="#AB63FA",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision vs Recall")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # 3. IoU histogram
    ax = axes[0, 2]
    iou_vals = results["IoU"].drop_nulls().drop_nans().to_numpy()
    ax.hist(iou_vals, bins=50, color="#636EFA", edgecolor="white", linewidth=0.3)
    ax.axvline(
        target_iou,
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"target={target_iou}",
    )
    ax.axvline(
        low_thresh,
        color="orange",
        linestyle=":",
        linewidth=1,
        label=f"low={low_thresh}",
    )
    ax.set_xlabel("IoU")
    ax.set_title("IoU Distribution")
    ax.legend(fontsize=8)

    # 4. Duration split – high vs low IoU
    ax = axes[1, 0]
    if "file_total" in results.columns:
        p99 = results["file_total"].quantile(0.99)
        for grp, name, color in [
            (df_high, "High IoU", "#1f77b4"),
            (df_low, "Low IoU", "#d62728"),
        ]:
            if "file_total" in grp.columns and grp.height > 0:
                vals = grp.filter(pl.col("file_total") < p99)["file_total"].to_numpy()
                ax.hist(vals, bins=40, alpha=0.6, label=name, color=color, density=True)
        ax.set_xlabel(f"Duration (0–{p99:.0f}s)")
        ax.legend(fontsize=8)
    ax.set_title("Duration (High vs Low IoU)")

    # 5. VAD speech ratio (vad_dur / file_total)
    ax = axes[1, 1]
    if "file_total" in results.columns:
        for grp, name, color in [
            (df_high, "High IoU", "#1f77b4"),
            (df_low, "Low IoU", "#d62728"),
        ]:
            if grp.height > 0 and "file_total" in grp.columns:
                ratio = (grp["vad_dur"] / grp["file_total"]).to_numpy()
                ax.hist(
                    ratio, bins=40, alpha=0.6, label=name, color=color, density=True
                )
        ax.set_xlabel("VAD speech ratio")
        ax.legend(fontsize=8)
    ax.set_title("Speech Ratio (High vs Low IoU)")

    # 6. Adaptive threshold distribution
    ax = axes[1, 2]
    if "vtc_threshold" in results.columns:
        thresholds = results["vtc_threshold"].drop_nulls().drop_nans().to_numpy()
        ax.hist(
            thresholds,
            bins=30,
            color="#FF6692",
            edgecolor="white",
            linewidth=0.3,
        )
        ax.axvline(
            0.5,
            color="gray",
            linestyle="--",
            linewidth=1,
            label="default (0.5)",
        )
        median_t = float(results["vtc_threshold"].median() or 0)  # type: ignore
        ax.axvline(
            median_t,
            color="#19D3F3",
            linestyle="-",
            linewidth=1.5,
            label=f"median ({median_t:.2f})",
        )
        ax.set_xlabel("Threshold")
        ax.set_title("Adaptive Threshold Distribution")
        ax.legend(fontsize=8)
    else:
        ax.text(
            0.5,
            0.5,
            "No vtc_threshold data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Adaptive Threshold Distribution")

    fig.tight_layout(rect=[0, 0, 1, 0.95])  # type: ignore
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved figure: {output_path.name}")
