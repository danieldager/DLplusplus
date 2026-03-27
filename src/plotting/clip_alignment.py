"""Figures for VTC clip-alignment analysis.

Extracted from ``src.pipeline.vtc_clip_alignment`` so the pipeline module
stays focused on interval logic and CSV reporting.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.plotting.utils import lazy_pyplot, save_figure

logger = logging.getLogger("vtc_align")

LABELS = ["KCHI", "OCH", "MAL", "FEM"]
COLLARS = [0.5, 1.0, 2.0, 3.0, 5.0]


def save_clip_alignment_figures(
    coverage: dict[str, dict[str, float]],
    matches: dict[str, dict],
    totals: dict[str, dict[str, float]],
    fig_dir: Path,
) -> None:
    """Generate coverage + matching figures for VTC clip alignment."""
    plt = lazy_pyplot()

    fig_dir.mkdir(parents=True, exist_ok=True)

    order = LABELS + ["all_speech"]
    display = ["KCHI", "OCH", "MAL", "FEM", "All speech"]

    # ------------------------------------------------------------------
    # Figure 1: Coverage breakdown + total comparison (2 panels)
    # ------------------------------------------------------------------
    fig, (ax, ax_tot) = plt.subplots(
        1, 2, figsize=(16, 4.5), gridspec_kw={"width_ratios": [3, 2]}
    )
    both_vals = [coverage[l]["both_h"] for l in order]
    only_full = [coverage[l]["only_full_h"] for l in order]
    only_clip = [coverage[l]["only_clip_h"] for l in order]

    y = np.arange(len(order))
    h = 0.6

    ax.barh(y, both_vals, h, label="Both agree", color="#2ecc71")
    ax.barh(y, only_full, h, left=both_vals, label="Only full-file", color="#e67e22")
    ax.barh(
        y,
        only_clip,
        h,
        left=[b + f for b, f in zip(both_vals, only_full)],
        label="Only clip",
        color="#9b59b6",
    )

    max_total = max(b + f + c for b, f, c in zip(both_vals, only_full, only_clip))
    for i in range(len(order)):
        total = both_vals[i] + only_full[i] + only_clip[i]
        ax.text(
            total + max_total * 0.01,
            i,
            f"{both_vals[i]:.0f} + {only_full[i]:.0f} + {only_clip[i]:.0f} h",
            va="center",
            fontsize=8,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(display, fontsize=11)
    ax.set_xlabel("Hours", fontsize=11)
    ax.set_xlim(0, max_total * 1.35)
    ax.set_title(
        "Coverage Agreement: Full-file VTC vs Clip VTC",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.invert_yaxis()

    # Panel b: total speech per label
    tot_full = [totals[l]["full_h"] for l in order]
    tot_clip = [totals[l]["clip_h"] for l in order]
    y2 = np.arange(len(order))
    h2 = 0.3
    ax_tot.barh(y2 - h2 / 2, tot_full, h2, label="Full-file", color="#3498db")
    ax_tot.barh(y2 + h2 / 2, tot_clip, h2, label="Clip", color="#e74c3c")
    max_tot = max(max(tot_full), max(tot_clip))
    for i in range(len(order)):
        diff_pct = (
            100 * (tot_clip[i] - tot_full[i]) / tot_full[i] if tot_full[i] > 0 else 0
        )
        ax_tot.text(
            max(tot_full[i], tot_clip[i]) + max_tot * 0.01,
            i,
            f"{tot_full[i]:.1f} vs {tot_clip[i]:.1f}h ({diff_pct:+.2f}%)",
            va="center",
            fontsize=8,
        )
    ax_tot.set_yticks(y2)
    ax_tot.set_yticklabels(display, fontsize=11)
    ax_tot.set_xlabel("Hours", fontsize=11)
    ax_tot.set_xlim(0, max_tot * 1.55)
    ax_tot.set_title("Total Speech per Label", fontsize=11, fontweight="bold")
    ax_tot.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax_tot.invert_yaxis()

    fig.suptitle("VTC Clip Alignment \u2014 Coverage", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, fig_dir / "clip_alignment_coverage.png")

    # ------------------------------------------------------------------
    # Figure 2: Segment matching  (3-panel)
    #   a) multi-collar match rate per label
    #   b) boundary-error CDF
    #   c) onset / offset box plots
    # ------------------------------------------------------------------
    match_labels = LABELS
    match_display = ["KCHI", "OCH", "MAL", "FEM"]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 5))

    # Panel a: multi-collar match rates
    x = np.arange(len(COLLARS))
    w = 0.18
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]
    for k, (lbl, disp) in enumerate(zip(match_labels, match_display)):
        m = matches[lbl]
        n = m["n_full"]
        pcts = []
        for c in COLLARS:
            n_well = int(
                np.sum(
                    (np.abs(m["onset_errors"]) <= c) & (np.abs(m["offset_errors"]) <= c)
                )
            )
            pcts.append(100 * n_well / n if n > 0 else 0)
        offset = (k - 1.5) * w
        bars = ax1.bar(x + offset, pcts, w, label=disp, color=colors[k], alpha=0.85)
        for bar, pct in zip(bars, pcts):
            if pct > 5:
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.8,
                    f"{pct:.0f}",
                    ha="center",
                    fontsize=6,
                )

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"\u00b1{c}s" for c in COLLARS], fontsize=10)
    ax1.set_ylabel("% of full-file segments", fontsize=10)
    ax1.set_ylim(0, 105)
    ax1.set_title("Match Rate by Collar", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    # Panel b: CDF of absolute boundary errors
    for k, (lbl, disp) in enumerate(zip(match_labels, match_display)):
        m = matches[lbl]
        if len(m["onset_errors"]) == 0:
            continue
        abs_errs = np.sort(
            np.maximum(np.abs(m["onset_errors"]), np.abs(m["offset_errors"]))
        )
        cdf = np.arange(1, len(abs_errs) + 1) / m["n_full"]
        ax2.plot(abs_errs, 100 * cdf, color=colors[k], label=disp, linewidth=1.5)

    ax2.set_xlim(0, 5)
    ax2.set_ylim(0, 100)
    ax2.set_xlabel("Max boundary error (s)", fontsize=10)
    ax2.set_ylabel("% of full-file segments", fontsize=10)
    ax2.set_title("Cumulative Match Rate", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8, loc="lower right", framealpha=0.9)
    ax2.grid(alpha=0.3, linestyle="--")
    ax2.axvline(0.5, color="gray", linewidth=0.8, linestyle=":")
    ax2.axvline(2.0, color="gray", linewidth=0.8, linestyle=":")

    # Panel c: onset / offset error box plots
    bp_data_on = []
    bp_data_off = []
    bp_labels = []
    bp_display = []
    for lbl, disp in zip(match_labels, match_display):
        if len(matches[lbl]["onset_errors"]) > 0:
            bp_data_on.append(matches[lbl]["onset_errors"])
            bp_data_off.append(matches[lbl]["offset_errors"])
            bp_labels.append(lbl)
            bp_display.append(disp)

    if bp_data_on:
        positions_on = np.arange(len(bp_labels)) * 2
        positions_off = positions_on + 0.6

        bp1 = ax3.boxplot(
            bp_data_on,
            positions=positions_on,
            widths=0.5,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor="#3498db", alpha=0.6),
            medianprops=dict(color="black"),
        )
        bp2 = ax3.boxplot(
            bp_data_off,
            positions=positions_off,
            widths=0.5,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor="#e67e22", alpha=0.6),
            medianprops=dict(color="black"),
        )

        ax3.set_xticks((positions_on + positions_off) / 2)
        ax3.set_xticklabels(bp_display, fontsize=11)
        ax3.set_ylabel("Error (s): clip \u2212 full-file", fontsize=10)
        ax3.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax3.set_title("Boundary Errors (matched)", fontsize=11, fontweight="bold")
        ax3.legend(
            [bp1["boxes"][0], bp2["boxes"][0]],
            ["Onset error", "Offset error"],
            fontsize=8,
            loc="upper right",
            framealpha=0.9,
        )
        ax3.grid(axis="y", alpha=0.3, linestyle="--")
    else:
        ax3.text(0.5, 0.5, "No matched segments", ha="center", va="center")

    fig.suptitle(
        "VTC Clip Alignment \u2014 Segment Matching", fontsize=12, fontweight="bold"
    )
    fig.tight_layout()
    save_figure(fig, fig_dir / "clip_alignment_matching.png")
