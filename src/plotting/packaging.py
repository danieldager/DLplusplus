"""Packaging pipeline figures (summary + label analysis)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.core import LABEL_COLORS, VTC_LABELS


def save_figure(stats: dict, output_path: Path) -> None:
    """Save a multi-panel summary figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle("Clip Packaging Summary", fontsize=14, fontweight="bold")

    # 1. Clip duration distribution
    ax = axes[0, 0]
    ax.hist(stats["durations"], bins=40, color="#4C72B0", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Clip duration (s)")
    ax.set_ylabel("Count")
    ax.set_title("Clip Duration Distribution")
    ax.axvline(
        np.median(stats["durations"]),
        color="red",
        ls="--",
        lw=1,
        label=f'median={np.median(stats["durations"]):.0f}s',
    )
    ax.legend(fontsize=8)

    # 2. VTC speech density
    ax = axes[0, 1]
    ax.hist(stats["vtc_dens"], bins=40, color="#55A868", edgecolor="white", alpha=0.8)
    ax.set_xlabel("VTC speech density")
    ax.set_ylabel("Count")
    ax.set_title("VTC Speech Density")
    ax.axvline(
        np.median(stats["vtc_dens"]),
        color="red",
        ls="--",
        lw=1,
        label=f'median={np.median(stats["vtc_dens"]):.2f}',
    )
    ax.legend(fontsize=8)

    # 3. VAD vs VTC density scatter
    ax = axes[0, 2]
    ax.scatter(stats["vtc_dens"], stats["vad_dens"], alpha=0.3, s=10, c="#C44E52")
    ax.plot([0, 1], [0, 1], "k--", lw=0.5, alpha=0.5)
    ax.set_xlabel("VTC density")
    ax.set_ylabel("VAD density")
    ax.set_title("VAD vs VTC Density")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # 4. IoU distribution
    ax = axes[1, 0]
    ax.hist(stats["ious"], bins=40, color="#8172B2", edgecolor="white", alpha=0.8)
    ax.set_xlabel("VAD–VTC IoU")
    ax.set_ylabel("Count")
    ax.set_title("VAD–VTC Agreement (IoU)")
    ax.axvline(
        np.median(stats["ious"]),
        color="red",
        ls="--",
        lw=1,
        label=f'median={np.median(stats["ious"]):.2f}',
    )
    ax.legend(fontsize=8)

    # 5. Turns per clip
    ax = axes[1, 1]
    max_turns = int(min(stats["turns"].max(), 50))
    ax.hist(
        stats["turns"],
        bins=range(0, max_turns + 2),
        color="#CCB974",
        edgecolor="white",
        alpha=0.8,
    )
    ax.set_xlabel("Speaker turns")
    ax.set_ylabel("Count")
    ax.set_title("Speaker Turns per Clip")
    ax.axvline(
        np.median(stats["turns"]),
        color="red",
        ls="--",
        lw=1,
        label=f'median={np.median(stats["turns"]):.0f}',
    )
    ax.legend(fontsize=8)

    # 6. Labels per clip
    ax = axes[1, 2]
    ax.hist(
        stats["n_labels"],
        bins=range(0, 6),
        color="#64B5CD",
        edgecolor="white",
        alpha=0.8,
        align="left",
    )
    ax.set_xlabel("Unique labels")
    ax.set_ylabel("Count")
    ax.set_title("Label Diversity per Clip")
    ax.set_xticks(range(0, 5))

    # 7. VTC segment duration
    ax = axes[2, 0]
    clipped = np.clip(stats["vtc_seg_durs"], 0, 30)
    ax.hist(clipped, bins=60, color="#DD8452", edgecolor="white", alpha=0.8)
    ax.set_xlabel("VTC segment duration (s, capped at 30)")
    ax.set_ylabel("Count")
    ax.set_title("VTC Segment Duration")
    ax.axvline(
        np.median(stats["vtc_seg_durs"]),
        color="red",
        ls="--",
        lw=1,
        label=f'median={np.median(stats["vtc_seg_durs"]):.1f}s',
    )
    ax.legend(fontsize=8)

    # 8. Gap between VTC segments
    ax = axes[2, 1]
    clipped_gaps = np.clip(stats["vtc_gaps"], 0, 15)
    ax.hist(clipped_gaps, bins=60, color="#DA8BC3", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Gap between VTC segments (s, capped at 15)")
    ax.set_ylabel("Count")
    ax.set_title("Inter-segment Gap")
    ax.axvline(
        np.median(stats["vtc_gaps"]),
        color="red",
        ls="--",
        lw=1,
        label=f'median={np.median(stats["vtc_gaps"]):.1f}s',
    )
    ax.legend(fontsize=8)

    # 9. Labelled speech vs clip duration
    ax = axes[2, 2]
    ax.scatter(
        stats["durations"],
        stats["vtc_durs"],
        alpha=0.3,
        s=10,
        c="#4C72B0",
        label="VTC (labelled)",
    )
    if stats["vad_durs"].sum() > 0:
        ax.scatter(
            stats["durations"],
            stats["vad_durs"],
            alpha=0.2,
            s=10,
            c="#C44E52",
            label="VAD",
            marker="x",
        )
    ax.plot(
        [0, stats["durations"].max()],
        [0, stats["durations"].max()],
        "k--",
        lw=0.5,
        alpha=0.5,
    )
    ax.set_xlabel("Clip duration (s)")
    ax.set_ylabel("Total detected speech in clip (s)")
    ax.set_title("Detected Speech vs Clip Duration\n(diagonal = 100% speech)")
    ax.legend(fontsize=7)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure: {output_path}")


def save_label_figures(stats: dict, output_path: Path) -> None:
    """Save a multi-panel label analysis figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    LABELS = VTC_LABELS
    COLORS = LABEL_COLORS

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle("Label Analysis", fontsize=14, fontweight="bold")

    # ── Row 1: Per-label overview ──────────────────────────────────

    # 1. Per-label total speech hours
    ax = axes[0, 0]
    hours = [
        stats["label_seg_durs"][l].sum() / 3600 if len(stats["label_seg_durs"][l]) else 0
        for l in LABELS
    ]
    bars = ax.bar(LABELS, hours, color=[COLORS[l] for l in LABELS], edgecolor="white")
    ax.set_ylabel("Total speech (hours)")
    ax.set_title("Speech Duration by Label")
    for bar, h in zip(bars, hours):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h,
            f"{h:.1f}h",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 2. Per-label segment count
    ax = axes[0, 1]
    counts = [stats["label_seg_counts"].get(l, 0) for l in LABELS]
    bars = ax.bar(LABELS, counts, color=[COLORS[l] for l in LABELS], edgecolor="white")
    ax.set_ylabel("Number of segments")
    ax.set_title("Segment Count by Label")
    for bar, c in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            c,
            f"{c:,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 3. Per-label mean segment duration
    ax = axes[0, 2]
    means = []
    stds = []
    for l in LABELS:
        d = stats["label_seg_durs"][l]
        if len(d):
            means.append(d.mean())
            stds.append(d.std())
        else:
            means.append(0)
            stds.append(0)
    ax.bar(
        LABELS,
        means,
        yerr=stds,
        color=[COLORS[l] for l in LABELS],
        edgecolor="white",
        capsize=4,
        alpha=0.8,
    )
    ax.set_ylabel("Mean segment duration (s)")
    ax.set_title("Segment Duration by Label (mean ± std)")

    # ── Row 2: VAD–VTC agreement by label ─────────────────────────

    # 4. VAD coverage by label (KEY)
    ax = axes[1, 0]
    cov_means = []
    cov_stds = []
    for l in LABELS:
        d = stats["label_vad_coverage"][l]
        if len(d):
            cov_means.append(d.mean())
            cov_stds.append(d.std())
        else:
            cov_means.append(0)
            cov_stds.append(0)
    bars = ax.bar(
        LABELS,
        cov_means,
        yerr=cov_stds,
        color=[COLORS[l] for l in LABELS],
        edgecolor="white",
        capsize=4,
        alpha=0.8,
    )
    ax.set_ylabel("VAD coverage (fraction)")
    ax.set_title("VAD Coverage of Each VTC Label\n(how much VAD detects per label)")
    ax.set_ylim(0, 1.05)
    for bar, m in zip(bars, cov_means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            m + 0.02,
            f"{m:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 5. IoU vs child speech fraction (KEY hypothesis test)
    ax = axes[1, 1]
    sc = ax.scatter(
        stats["child_fracs"],
        stats["ious"],
        alpha=0.3,
        s=12,
        c=stats["child_fracs"],
        cmap="RdYlBu_r",
        vmin=0,
        vmax=1,
    )
    ax.set_xlabel("Child speech fraction (KCHI+OCH / total VTC)")
    ax.set_ylabel("VAD–VTC IoU")
    ax.set_title("IoU vs Child Speech Fraction\n(hypothesis: more child → lower IoU)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    plt.colorbar(sc, ax=ax, label="Child fraction", shrink=0.8)
    # Add trend line
    if len(stats["child_fracs"]) > 2:
        z = np.polyfit(stats["child_fracs"], stats["ious"], 1)
        x_line = np.linspace(0, 1, 100)
        ax.plot(
            x_line,
            np.polyval(z, x_line),
            "k--",
            lw=1.5,
            alpha=0.7,
            label=f"slope={z[0]:.2f}",
        )
        ax.legend(fontsize=8)

    # 6. IoU by dominant label (box plot)
    ax = axes[1, 2]
    dom_data = {l: [] for l in LABELS}
    for dl, iou in zip(stats["dominant_labels"], stats["ious"]):
        if dl in dom_data:
            dom_data[dl].append(iou)
    box_data = [dom_data[l] if dom_data[l] else [0] for l in LABELS]
    bp = ax.boxplot(box_data, labels=LABELS, patch_artist=True, widths=0.6)
    for patch, l in zip(bp["boxes"], LABELS):
        patch.set_facecolor(COLORS[l])
        patch.set_alpha(0.7)
    ax.set_ylabel("VAD–VTC IoU")
    ax.set_title("IoU by Dominant Label")
    for i, l in enumerate(LABELS):
        n = len(dom_data[l])
        ax.text(i + 1, -0.08, f"n={n}", ha="center", fontsize=8, color="gray")

    # ── Row 3: Gap analysis ────────────────────────────────────────

    # 7. Same-label vs cross-label gap distribution
    ax = axes[2, 0]
    sl = stats["same_label_gaps"]
    cl = stats["cross_label_gaps"]
    max_gap_plot = 10.0
    if len(sl):
        ax.hist(
            np.clip(sl, 0, max_gap_plot),
            bins=50,
            alpha=0.6,
            color="#C44E52",
            label=f"Same-label ({len(sl)})",
            edgecolor="white",
        )
    if len(cl):
        ax.hist(
            np.clip(cl, 0, max_gap_plot),
            bins=50,
            alpha=0.6,
            color="#4C72B0",
            label=f"Cross-label ({len(cl)})",
            edgecolor="white",
        )
    ax.set_xlabel(f"Gap duration (s, capped at {max_gap_plot:.0f})")
    ax.set_ylabel("Count")
    ax.set_title("Gap Duration: Same vs Cross Label")
    ax.legend(fontsize=8)

    # 8. Per-label segment duration distribution (overlaid)
    ax = axes[2, 1]
    for l in LABELS:
        d = stats["label_seg_durs"][l]
        if len(d):
            ax.hist(
                np.clip(d, 0, 30),
                bins=60,
                alpha=0.5,
                color=COLORS[l],
                label=f"{l} ({len(d)})",
                edgecolor="white",
            )
    ax.set_xlabel("Segment duration (s, capped at 30)")
    ax.set_ylabel("Count")
    ax.set_title("Segment Duration Distribution by Label")
    ax.legend(fontsize=8)

    # 9. Child fraction distribution
    ax = axes[2, 2]
    ax.hist(stats["child_fracs"], bins=40, color="#8172B2", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Child speech fraction per clip")
    ax.set_ylabel("Count")
    ax.set_title("Child Speech Fraction Distribution")
    ax.axvline(
        np.median(stats["child_fracs"]),
        color="red",
        ls="--",
        lw=1,
        label=f'median={np.median(stats["child_fracs"]):.2f}',
    )
    ax.legend(fontsize=8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure: {output_path}")
