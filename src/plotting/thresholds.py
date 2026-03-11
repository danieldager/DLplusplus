"""Threshold comparison analysis figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl


def plot_heatmap(grid_df: pl.DataFrame, fig_dir: Path) -> None:
    """Generate IoU + Precision + Recall heatmaps."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vad_vals = sorted(grid_df["vad_threshold"].unique().to_list())
    vtc_vals = sorted(grid_df["vtc_threshold"].unique().to_list())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, metric, title in zip(
        axes,
        ["mean_iou", "mean_precision", "mean_recall"],
        ["Mean IoU", "Mean Precision", "Mean Recall"],
    ):
        matrix = np.zeros((len(vad_vals), len(vtc_vals)))
        for row in grid_df.iter_rows(named=True):
            vi = vad_vals.index(row["vad_threshold"])
            vj = vtc_vals.index(row["vtc_threshold"])
            matrix[vi, vj] = row[metric]

        im = ax.imshow(
            matrix,
            aspect="auto",
            origin="lower",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
        )
        ax.set_xticks(range(len(vtc_vals)))
        ax.set_xticklabels([f"{v:.2f}" for v in vtc_vals], rotation=45)
        ax.set_yticks(range(len(vad_vals)))
        ax.set_yticklabels([f"{v:.2f}" for v in vad_vals])
        ax.set_xlabel("VTC threshold")
        ax.set_ylabel("VAD threshold")
        ax.set_title(title)

        # Annotate cells
        for i in range(len(vad_vals)):
            for j in range(len(vtc_vals)):
                val = matrix[i, j]
                color = "white" if val < 0.4 or val > 0.85 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=color,
                )

        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("VAD × VTC Threshold Grid", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = fig_dir / "threshold_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_volume_sensitivity(vol_df: pl.DataFrame, fig_dir: Path) -> None:
    """Bar chart showing total speech hours vs threshold for each system."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thresholds = vol_df["threshold"].to_list()
    vad_hours = vol_df["vad_hours"].to_list()
    vtc_hours = vol_df["vtc_hours"].to_list()

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(thresholds))
    width = 0.35

    ax.bar(
        x - width / 2,
        vad_hours,
        width,
        label="VAD (TenVAD)",
        color="#2196F3",
        alpha=0.8,
    )
    ax.bar(x + width / 2, vtc_hours, width, label="VTC", color="#FF5722", alpha=0.8)

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Total speech (hours)")
    ax.set_title("Speech volume sensitivity to threshold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.2f}" for t in thresholds])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = fig_dir / "volume_sensitivity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
