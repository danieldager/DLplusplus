"""Dashboard pages: SNR & Recording Quality, Noise Environment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from src.core import LABEL_COLORS, VTC_LABELS


def _setup():
    """Lazy matplotlib setup, returns plt module."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


# Friendly colours for noise categories
_NOISE_COLORS: dict[str, str] = {
    "music":     "#E24A33",
    "crying":    "#348ABD",
    "laughter":  "#FBC15E",
    "singing":   "#8EBA42",
    "tv_radio":  "#988ED5",
    "vehicle":   "#777777",
    "animal":    "#FFB5B8",
    "water":     "#55A868",
    "household": "#C44E52",
    "impact":    "#8C8C8C",
    "alarm":     "#CCB974",
    "silence":   "#64B5CD",
    "other":     "#B0B0B0",
}


def save_snr_figures(
    clip_df: pl.DataFrame,
    segment_df: pl.DataFrame,
    conversation_df: pl.DataFrame,
    output_path: Path,
) -> None:
    """SNR / C50 dashboard — 3×3 panels."""
    plt = _setup()

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle("SNR & Recording Quality", fontsize=14, fontweight="bold")

    snr = clip_df["snr_mean"].drop_nulls().drop_nans().to_numpy()
    has_snr = len(snr) > 0

    # 1. Per-clip mean SNR histogram
    ax = axes[0, 0]
    if has_snr:
        ax.hist(snr, bins=50, color="#4C72B0", edgecolor="white", alpha=0.8)
        ax.axvline(
            np.median(snr),
            color="red",
            ls="--",
            lw=1,
            label=f"median={np.median(snr):.1f} dB",
        )
        ax.legend(fontsize=8)
    ax.set_xlabel("Mean SNR per clip (dB)")
    ax.set_ylabel("Count")
    ax.set_title("Per-clip Mean SNR Distribution")

    # 2. SNR vs speech density
    ax = axes[0, 1]
    if has_snr:
        dens = clip_df.filter(pl.col("snr_mean").is_not_null())[
            "vtc_density"
        ].to_numpy()
        snr_f = clip_df.filter(pl.col("snr_mean").is_not_null())["snr_mean"].to_numpy()
        ax.scatter(dens, snr_f, alpha=0.25, s=8, c="#55A868")
    ax.set_xlabel("VTC speech density")
    ax.set_ylabel("Mean SNR (dB)")
    ax.set_title("SNR vs Speech Density")

    # 3. SNR by dominant label (box)
    ax = axes[0, 2]
    if has_snr:
        box_data = []
        labels_used: list[str] = []
        for l in VTC_LABELS:
            vals = clip_df.filter(
                (pl.col("dominant_label") == l) & pl.col("snr_mean").is_not_null()
            )["snr_mean"].to_numpy()
            if len(vals) > 0:
                box_data.append(vals)
                labels_used.append(l)
        if box_data:
            bp = ax.boxplot(box_data, labels=labels_used, patch_artist=True, widths=0.6)
            for patch, l in zip(bp["boxes"], labels_used):
                patch.set_facecolor(LABEL_COLORS.get(l, "#999"))
                patch.set_alpha(0.7)
    ax.set_ylabel("Mean SNR (dB)")
    ax.set_title("SNR by Dominant Label")

    # 4. SNR vs VAD–VTC IoU
    ax = axes[1, 0]
    if has_snr:
        filt = clip_df.filter(pl.col("snr_mean").is_not_null())
        ax.scatter(
            filt["snr_mean"].to_numpy(),
            filt["vad_vtc_iou"].to_numpy(),
            alpha=0.25,
            s=8,
            c="#C44E52",
        )
    ax.set_xlabel("Mean SNR (dB)")
    ax.set_ylabel("VAD–VTC IoU")
    ax.set_title("SNR vs Model Agreement")

    # 5. Per-label mean SNR during speech (bar ± std)
    ax = axes[1, 1]
    if "snr_during" in segment_df.columns:
        seg_snr = segment_df.filter(pl.col("snr_during").is_not_null())
        if len(seg_snr) > 0:
            means, stds, cols = [], [], []
            for l in VTC_LABELS:
                v = seg_snr.filter(pl.col("label") == l)["snr_during"].to_numpy()
                if len(v) > 0:
                    means.append(float(np.mean(v)))
                    stds.append(float(np.std(v)))
                    cols.append(l)
            if cols:
                ax.bar(
                    cols,
                    means,
                    yerr=stds,
                    capsize=4,
                    alpha=0.8,
                    color=[LABEL_COLORS.get(l, "#999") for l in cols],
                    edgecolor="white",
                )
    ax.set_ylabel("Mean SNR during speech (dB)")
    ax.set_title("SNR During Each Speaker Type")

    # 6. Intra-clip SNR std histogram
    ax = axes[1, 2]
    snr_std = clip_df["snr_std"].drop_nulls().drop_nans().to_numpy()
    if len(snr_std) > 0:
        ax.hist(snr_std, bins=50, color="#DD8452", edgecolor="white", alpha=0.8)
        ax.axvline(
            np.median(snr_std),
            color="red",
            ls="--",
            lw=1,
            label=f"median={np.median(snr_std):.1f}",
        )
        ax.legend(fontsize=8)
    ax.set_xlabel("Intra-clip SNR std (dB)")
    ax.set_ylabel("Count")
    ax.set_title("Intra-clip SNR Variability")

    # 7. Low-SNR fraction by threshold
    ax = axes[2, 0]
    if has_snr:
        thresholds = [0, 5, 10, 15, 20]
        fracs = [float((snr < t).mean()) for t in thresholds]
        ax.bar(
            [str(t) for t in thresholds],
            fracs,
            color=["#d62728", "#e74c3c", "#ff7f0e", "#f1c40f", "#2ecc71"],
            edgecolor="white",
            alpha=0.8,
        )
        for i, f in enumerate(fracs):
            ax.text(i, f + 0.01, f"{f:.1%}", ha="center", fontsize=9)
    ax.set_xlabel("SNR threshold (dB)")
    ax.set_ylabel("Fraction of clips below")
    ax.set_title("Low-SNR Clip Fraction")
    ax.set_ylim(0, 1.05)

    # 8. C50 clarity histogram
    ax = axes[2, 1]
    c50 = clip_df["c50_mean"].drop_nulls().drop_nans().to_numpy()
    if len(c50) > 0:
        ax.hist(c50, bins=50, color="#8172B2", edgecolor="white", alpha=0.8)
        ax.axvline(
            np.median(c50),
            color="red",
            ls="--",
            lw=1,
            label=f"median={np.median(c50):.1f} dB",
        )
        ax.legend(fontsize=8)
    ax.set_xlabel("Mean C50 per clip (dB)")
    ax.set_ylabel("Count")
    ax.set_title("C50 Clarity Distribution")

    # 9. Conversation-level SNR histogram
    ax = axes[2, 2]
    if "snr_mean" in conversation_df.columns:
        conv_snr = conversation_df["snr_mean"].drop_nulls().drop_nans().to_numpy()
        if len(conv_snr) > 0:
            ax.hist(conv_snr, bins=50, color="#64B5CD", edgecolor="white", alpha=0.8)
            ax.axvline(
                np.median(conv_snr),
                color="red",
                ls="--",
                lw=1,
                label=f"median={np.median(conv_snr):.1f} dB",
            )
            ax.legend(fontsize=8)
    ax.set_xlabel("Mean SNR per conversation (dB)")
    ax.set_ylabel("Count")
    ax.set_title("Conversation-Level SNR")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure: {output_path}")


# =========================================================================
# Page 2: Conversational Structure (3×3)
# =========================================================================


def save_noise_figures(
    clip_df: pl.DataFrame,
    segment_df: pl.DataFrame,
    output_path: Path,
) -> None:
    """Noise environment dashboard — 3×3 panels.

    Only rendered when ``noise_*`` columns are present in clip_df.
    """
    plt = _setup()

    # Detect available noise columns
    noise_cols = [c for c in clip_df.columns if c.startswith("noise_")]
    if not noise_cols:
        return  # no noise data — skip

    cat_names = [c.replace("noise_", "") for c in noise_cols]

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle("Noise Environment (PANNs CNN14)", fontsize=14, fontweight="bold")

    # --- 1. Pie chart: dominant noise category distribution ---
    ax = axes[0, 0]
    if "dominant_noise" in clip_df.columns:
        dom = clip_df.filter(pl.col("dominant_noise") != "?")
        if len(dom) > 0:
            counts = dom.group_by("dominant_noise").agg(
                pl.count().alias("n")
            ).sort("n", descending=True)
            labels = counts["dominant_noise"].to_list()
            sizes = counts["n"].to_list()
            colors = [_NOISE_COLORS.get(l, "#B0B0B0") for l in labels]
            wedges, texts, autotexts = ax.pie(
                sizes, labels=labels, colors=colors,
                autopct="%1.1f%%", startangle=90,
                textprops={"fontsize": 8},
            )
            for t in autotexts:
                t.set_fontsize(7)
    ax.set_title("Dominant Noise Type per Clip")

    # --- 2. Mean probability per category (bar chart) ---
    ax = axes[0, 1]
    mean_probs = []
    for col in noise_cols:
        vals = clip_df[col].drop_nulls().to_numpy()
        mean_probs.append(float(np.mean(vals)) if len(vals) > 0 else 0.0)
    sorted_pairs = sorted(zip(cat_names, mean_probs), key=lambda x: -x[1])
    sorted_cats, sorted_vals = zip(*sorted_pairs) if sorted_pairs else ([], [])
    colors = [_NOISE_COLORS.get(c, "#B0B0B0") for c in sorted_cats]
    bars = ax.barh(
        list(reversed(sorted_cats)), list(reversed(sorted_vals)),
        color=list(reversed(colors)), edgecolor="white",
    )
    ax.set_xlabel("Mean probability")
    ax.set_title("Average Noise Category Probability")
    ax.set_xlim(0, max(sorted_vals) * 1.15 if sorted_vals else 1)

    # --- 3. Noise vs speech density scatter ---
    ax = axes[0, 2]
    # Use the max non-speech category prob as "noise intensity"
    noise_max = np.zeros(len(clip_df))
    for col in noise_cols:
        vals = clip_df[col].fill_null(0).to_numpy()
        noise_max = np.maximum(noise_max, vals)
    vtc_dens = clip_df["vtc_density"].to_numpy()
    ax.scatter(vtc_dens, noise_max, alpha=0.25, s=8, c="#E24A33")
    ax.set_xlabel("VTC speech density")
    ax.set_ylabel("Max noise probability")
    ax.set_title("Speech Density vs Noise Level")

    # --- 4. Noise category distributions (box plots) ---
    ax = axes[1, 0]
    box_data = []
    box_labels: list[str] = []
    for cat, col in sorted(
        zip(cat_names, noise_cols),
        key=lambda x: -float(clip_df[x[1]].mean() or 0),  # type: ignore[arg-type]
    ):
        vals = clip_df[col].drop_nulls().to_numpy()
        if len(vals) > 0:
            box_data.append(vals)
            box_labels.append(cat)
    if box_data:
        bp = ax.boxplot(
            box_data[:8],  # top 8 for readability
            labels=box_labels[:8],
            patch_artist=True,
            widths=0.6,
            showfliers=False,
            vert=True,
        )
        for patch, l in zip(bp["boxes"], box_labels[:8]):
            patch.set_facecolor(_NOISE_COLORS.get(l, "#999"))
            patch.set_alpha(0.7)
        ax.tick_params(axis="x", rotation=45)
    ax.set_ylabel("Probability")
    ax.set_title("Noise Category Distributions (Top 8)")

    # --- 5. SNR vs dominant noise category (box) ---
    ax = axes[1, 1]
    if "dominant_noise" in clip_df.columns and "snr_mean" in clip_df.columns:
        filtered = clip_df.filter(
            (pl.col("dominant_noise") != "?") & pl.col("snr_mean").is_not_null()
        )
        if len(filtered) > 0:
            cats_present = (
                filtered.group_by("dominant_noise")
                .agg(pl.count().alias("n"))
                .filter(pl.col("n") >= 5)
                .sort("n", descending=True)
                ["dominant_noise"].to_list()[:8]
            )
            box_data_snr = []
            for cat in cats_present:
                d = filtered.filter(pl.col("dominant_noise") == cat)["snr_mean"].to_numpy()
                box_data_snr.append(d)
            if box_data_snr:
                bp = ax.boxplot(
                    box_data_snr,
                    labels=cats_present,
                    patch_artist=True,
                    widths=0.6,
                    showfliers=False,
                )
                for patch, l in zip(bp["boxes"], cats_present):
                    patch.set_facecolor(_NOISE_COLORS.get(l, "#999"))
                    patch.set_alpha(0.7)
                ax.tick_params(axis="x", rotation=45)
    ax.set_ylabel("Mean SNR (dB)")
    ax.set_title("SNR by Dominant Noise Type")

    # --- 6. Noise category co-occurrence heatmap ---
    ax = axes[1, 2]
    # Find top categories and check how often they co-occur (both > threshold)
    top_cats = [c for c, _ in sorted_pairs[:8]]
    top_cols = [f"noise_{c}" for c in top_cats]
    avail = [c for c in top_cols if c in clip_df.columns]
    if len(avail) >= 2:
        threshold = 0.1
        cooc = np.zeros((len(avail), len(avail)))
        for i, c1 in enumerate(avail):
            v1 = clip_df[c1].fill_null(0).to_numpy()
            for j, c2 in enumerate(avail):
                v2 = clip_df[c2].fill_null(0).to_numpy()
                cooc[i, j] = np.mean((v1 > threshold) & (v2 > threshold))
        labels_hm = [c.replace("noise_", "") for c in avail]
        im = ax.imshow(cooc, cmap="YlOrRd", aspect="auto", vmin=0)
        ax.set_xticks(range(len(labels_hm)))
        ax.set_xticklabels(labels_hm, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(labels_hm)))
        ax.set_yticklabels(labels_hm, fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.8, label="Co-occurrence rate")
    ax.set_title("Noise Co-occurrence (prob > 0.1)")

    # --- 7. Noise level over clip position (are starts/ends noisier?) ---
    ax = axes[2, 0]
    # Bin clips into 3rds of their source file and compare noise levels
    if "abs_onset" in clip_df.columns and len(clip_df) > 10:
        onsets = clip_df["abs_onset"].to_numpy()
        thirds = np.digitize(
            onsets / (onsets.max() + 1), bins=[0, 1 / 3, 2 / 3, 1]
        )
        third_labels = ["Start", "Middle", "End"]
        for cat_name in [c for c, v in sorted_pairs[:4]]:
            col = f"noise_{cat_name}"
            if col not in clip_df.columns:
                continue
            vals = clip_df[col].fill_null(0).to_numpy()
            means = [float(np.mean(vals[thirds == t + 1])) if np.sum(thirds == t + 1) > 0 else 0
                     for t in range(3)]
            ax.plot(third_labels, means, marker="o", label=cat_name,
                    color=_NOISE_COLORS.get(cat_name, "#999"))
        ax.legend(fontsize=7, ncol=2)
    ax.set_ylabel("Mean probability")
    ax.set_title("Noise by Recording Position")

    # --- 8. Noise vs child speech fraction ---
    ax = axes[2, 1]
    if "child_fraction" in clip_df.columns:
        cf = clip_df["child_fraction"].to_numpy()
        ax.scatter(cf, noise_max, alpha=0.25, s=8, c="#348ABD")
        # Trend line
        if len(cf) > 5:
            z = np.polyfit(cf, noise_max, 1)
            p = np.poly1d(z)
            xs = np.linspace(0, 1, 100)
            ax.plot(xs, p(xs), "r--", lw=1, alpha=0.7, label=f"slope={z[0]:.3f}")
            ax.legend(fontsize=8)
    ax.set_xlabel("Child speech fraction")
    ax.set_ylabel("Max noise probability")
    ax.set_title("Child Speech vs Noise Level")

    # --- 9. Stacked bar: noise profile by dominant VTC label ---
    ax = axes[2, 2]
    if "dominant_label" in clip_df.columns:
        for_stack = clip_df.filter(pl.col("dominant_label") != "?")
        vtc_labels_present = sorted(
            for_stack["dominant_label"].unique().to_list()
        )
        # For each VTC label, compute mean noise category probs
        bar_bottom = np.zeros(len(vtc_labels_present))
        render_cats = [c for c, _ in sorted_pairs[:6]]  # top 6 noise cats
        for cat in render_cats:
            col = f"noise_{cat}"
            if col not in clip_df.columns:
                continue
            cat_means = []
            for vtc_l in vtc_labels_present:
                subset = for_stack.filter(pl.col("dominant_label") == vtc_l)
                m = subset[col].fill_null(0).mean()
                cat_means.append(float(m) if m is not None else 0.0)  # type: ignore[arg-type]
            ax.bar(
                vtc_labels_present, cat_means, bottom=bar_bottom,
                color=_NOISE_COLORS.get(cat, "#999"), label=cat,
                edgecolor="white", linewidth=0.5,
            )
            bar_bottom += np.array(cat_means)
        ax.legend(fontsize=7, loc="upper right")
    ax.set_ylabel("Mean noise probability")
    ax.set_title("Noise Profile by Dominant Speaker")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# =========================================================================
# Convenience: render all pages
# =========================================================================

