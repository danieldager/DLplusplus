"""Dashboard pages: Dataset Overview, Correlation, Text Summary."""

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


def save_overview_figures(
    clip_df: pl.DataFrame,
    file_df: pl.DataFrame,
    segment_df: pl.DataFrame,
    tier_counts: dict[str, int],
    output_path: Path,
) -> None:
    """Dataset-level overview + cut-quality combined — 3×3 panels."""
    plt = _setup()
    from matplotlib.patches import FancyBboxPatch

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle("Dataset Overview", fontsize=14, fontweight="bold")

    # ── Row 0: Label-level ──────────────────────────────────────────────

    # [0,0] Per-label total speech hours
    ax = axes[0, 0]
    hours = []
    for l in VTC_LABELS:
        col = f"dur_{l}"
        hours.append(clip_df[col].sum() / 3600 if col in clip_df.columns else 0)
    bars = ax.bar(
        VTC_LABELS, hours,
        color=[LABEL_COLORS[l] for l in VTC_LABELS], edgecolor="white",
    )
    for bar, h in zip(bars, hours):
        ax.text(
            bar.get_x() + bar.get_width() / 2, h, f"{h:.1f}h",
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_ylabel("Total speech (hours)")
    ax.set_title("Speech Volume by Speaker Type")

    # [0,1] Per-label segment duration (box plot)
    ax = axes[0, 1]
    if len(segment_df) > 0:
        box_data: list = []
        box_labels: list[str] = []
        for l in VTC_LABELS:
            d = segment_df.filter(pl.col("label") == l)["duration"].to_numpy()
            if len(d) > 0:
                box_data.append(np.clip(d, 0, 30))
                box_labels.append(l)
        if box_data:
            bp = ax.boxplot(
                box_data, labels=box_labels, patch_artist=True,
                widths=0.6, showfliers=False,
            )
            for patch, l in zip(bp["boxes"], box_labels):
                patch.set_facecolor(LABEL_COLORS.get(l, "#999"))
                patch.set_alpha(0.7)
    ax.set_ylabel("Segment duration (s)")
    ax.set_title("Segment Duration by Label")

    # [0,2] Child speech fraction distribution
    ax = axes[0, 2]
    cf = clip_df["child_fraction"].to_numpy()
    ax.hist(cf, bins=40, color="#8172B2", edgecolor="white", alpha=0.8)
    ax.axvline(
        np.median(cf), color="red", ls="--", lw=1,
        label=f"median={np.median(cf):.2f}",
    )
    ax.legend(fontsize=8)
    ax.set_xlabel("Child speech fraction per clip")
    ax.set_ylabel("Count")
    ax.set_title("Child Speech Fraction")

    # ── Row 1: Cut quality ──────────────────────────────────────────────

    # [1,0] Cut-point tier breakdown
    ax = axes[1, 0]
    _tier_labels = {
        "long_union_gap":    "1. Long silence",
        "short_union_gap":   "2. Short silence",
        "vad_only_gap":      "3. VAD-only gap",
        "vtc_only_gap":      "4. VTC-only gap",
        "speaker_boundary":  "5. Speaker boundary",
        "hard_cut":          "6. Hard cut",
        "degenerate_window": "7. Degenerate",
    }
    _tier_hex = {
        "long_union_gap":    "#2ecc71",
        "short_union_gap":   "#27ae60",
        "vad_only_gap":      "#f1c40f",
        "vtc_only_gap":      "#f39c12",
        "speaker_boundary":  "#e67e22",
        "hard_cut":          "#e74c3c",
        "degenerate_window": "#c0392b",
    }
    total_cuts = sum(tier_counts.values())
    if total_cuts > 0:
        names, vals, colors = [], [], []
        for k in _tier_labels:
            cnt = tier_counts.get(k, 0)
            if cnt > 0:
                names.append(_tier_labels[k])
                vals.append(cnt)
                colors.append(_tier_hex.get(k, "#999"))
        ax.barh(names[::-1], vals[::-1], color=colors[::-1], edgecolor="white", alpha=0.8)
        for i, v in enumerate(vals[::-1]):
            pct = 100 * v / total_cuts
            ax.text(v + total_cuts * 0.01, i, f"{v} ({pct:.1f}%)", va="center", fontsize=8)
    ax.set_xlabel("Number of cuts")
    ax.set_title(f"Cut-Point Tier Breakdown ({total_cuts} total)")

    # [1,1] Clip duration distribution
    ax = axes[1, 1]
    durs = clip_df["duration"].to_numpy()
    ax.hist(durs, bins=40, color="#4C72B0", edgecolor="white", alpha=0.8)
    ax.axvline(
        np.median(durs), color="red", ls="--", lw=1,
        label=f"median={np.median(durs):.0f}s",
    )
    ax.legend(fontsize=8)
    ax.set_xlabel("Clip duration (s)")
    ax.set_ylabel("Count")
    ax.set_title("Clip Duration Distribution")

    # [1,2] Clips per source file
    # Use bins=50 (not range-based) so bars always appear regardless of scale
    ax = axes[1, 2]
    if len(file_df) > 0:
        cpf = file_df["n_clips"].to_numpy()
        ax.hist(cpf, bins=50, color="#CCB974", edgecolor="white", alpha=0.8)
        ax.axvline(
            np.median(cpf), color="red", ls="--", lw=1,
            label=f"median={np.median(cpf):.0f}",
        )
        ax.legend(fontsize=8)
    ax.set_xlabel("Clips per file")
    ax.set_ylabel("Count")
    ax.set_title("Clips per Source File")

    # ── Row 2: Density + summary ────────────────────────────────────────

    # [2,0] VAD vs VTC density scatter
    ax = axes[2, 0]
    ax.scatter(
        clip_df["vtc_density"].to_numpy(), clip_df["vad_density"].to_numpy(),
        alpha=0.25, s=8, c="#C44E52",
    )
    ax.plot([0, 1], [0, 1], "k--", lw=0.5, alpha=0.5)
    ax.set_xlabel("VTC density")
    ax.set_ylabel("VAD density")
    ax.set_title("VAD vs VTC Density")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # [2,1] Speech density per clip histogram
    ax = axes[2, 1]
    vtc_dens = clip_df["vtc_density"].to_numpy()
    ax.hist(vtc_dens, bins=40, color="#55A868", edgecolor="white", alpha=0.8)
    ax.axvline(
        np.median(vtc_dens), color="red", ls="--", lw=1,
        label=f"median={np.median(vtc_dens):.2f}",
    )
    ax.legend(fontsize=8)
    ax.set_xlabel("VTC speech density")
    ax.set_ylabel("Count")
    ax.set_title("Speech Density per Clip")

    # [2,2] Dataset summary — structured card layout
    ax = axes[2, 2]
    ax.axis("off")
    total_clips = len(clip_df)
    total_dur_h = clip_df["duration"].sum() / 3600
    total_vtc_h = clip_df["vtc_speech_dur"].sum() / 3600
    total_vad_h = clip_df["vad_speech_dur"].sum() / 3600
    n_files = len(file_df) if len(file_df) > 0 else clip_df["uid"].n_unique()
    median_dur = float(np.median(clip_df["duration"].to_numpy()))
    median_iou = float(np.median(clip_df["vad_vtc_iou"].to_numpy()))
    median_cf = float(np.median(clip_df["child_fraction"].to_numpy()))
    n_convs = int(clip_df["n_conversations"].sum())
    n_turns = int(clip_df["n_conv_turns"].sum())

    # Build rows as (is_header, label, value)
    sum_rows: list[tuple[bool, str, str]] = [
        (True,  "FILES & CLIPS",   ""),
        (False, "Source files",    f"{n_files:,}"),
        (False, "Total clips",     f"{total_clips:,}"),
        (True,  "AUDIO",           ""),
        (False, "Total duration",  f"{total_dur_h:.1f} h"),
        (False, "VTC speech",      f"{total_vtc_h:.1f} h"),
        (False, "VAD speech",      f"{total_vad_h:.1f} h"),
        (True,  "CLIP QUALITY",    ""),
        (False, "Median duration", f"{median_dur:.0f} s"),
        (False, "Median IoU",      f"{median_iou:.2f}"),
        (False, "Child fraction",  f"{median_cf:.2f}"),
    ]
    snr_vals = clip_df["snr_mean"].drop_nulls().drop_nans()
    c50_vals = clip_df["c50_mean"].drop_nulls().drop_nans()
    if len(snr_vals) > 0 or len(c50_vals) > 0:
        sum_rows.append((True, "RECORDING", ""))
        if len(snr_vals) > 0:
            sum_rows.append((False, "Median SNR", f"{float(snr_vals.median()):.1f} dB"))  # type: ignore
        if len(c50_vals) > 0:
            sum_rows.append((False, "Median C50", f"{float(c50_vals.median()):.1f} dB"))  # type: ignore
    sum_rows += [
        (True,  "CONVERSATIONS",       ""),
        (False, "Total turns",         f"{n_turns:,}"),
        (False, "Total conversations",  f"{n_convs:,}"),
    ]

    # Background card
    ax.add_patch(FancyBboxPatch(
        (0.04, 0.04), 0.92, 0.92,
        boxstyle="round,pad=0.02",
        transform=ax.transAxes,
        facecolor="#f8f9fa", edgecolor="#ced4da", lw=1.5,
        zorder=0,
    ))
    ax.set_title("Dataset Summary")

    # Distribute rows evenly over the card
    y_top, y_bot = 0.96, 0.08
    step = (y_top - y_bot) / len(sum_rows)
    for i, (is_hdr, label, value) in enumerate(sum_rows):
        y = y_top - (i + 0.5) * step
        if is_hdr:
            ax.text(
                0.08, y, label, transform=ax.transAxes, va="center",
                fontsize=8, fontweight="bold", color="#868e96",
            )
        else:
            ax.text(
                0.12, y, label, transform=ax.transAxes, va="center",
                fontsize=9, color="#495057",
            )
            ax.text(
                0.92, y, value, transform=ax.transAxes, va="center", ha="right",
                fontsize=9, fontweight="bold", color="#212529",
            )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure: {output_path}")


# =========================================================================
# Page 6: Correlation Matrix
# =========================================================================


def save_correlation_figure(
    corr_df: pl.DataFrame,
    output_path: Path,
) -> None:
    """Correlation heatmap across clip-level numeric metrics."""
    plt = _setup()

    labels = corr_df["metric"].to_list()
    matrix = corr_df.drop("metric").to_numpy()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto", origin="lower")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)

    # Annotate cells
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = matrix[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(
                j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color
            )

    plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    ax.set_title("Clip-Level Metric Correlations", fontsize=13, fontweight="bold")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure: {output_path}")


# =========================================================================
# Text summary — printed to stdout for log parsing
# =========================================================================


def print_dataset_summary(
    dfs: dict[str, pl.DataFrame],
    tier_counts: dict[str, int],
) -> None:
    """Print key dataset/plot metrics to stdout.

    This makes SLURM logs interpretable without inspecting images.
    """
    clip_df = dfs["clip_stats"]
    segment_df = dfs["segment_stats"]
    turn_df = dfs["turn_stats"]
    conv_df = dfs["conversation_stats"]
    trans_df = dfs["transition_stats"]
    file_df = dfs["file_stats"]

    sep = "━" * 60

    # ── Dataset overview ──────────────────────────────────────────
    print(f"\n{sep}")
    print("DATASET OVERVIEW")
    print(sep)
    n_clips = len(clip_df)
    n_files = clip_df["uid"].n_unique()
    total_dur_h = clip_df["duration"].sum() / 3600
    total_vtc_h = clip_df["vtc_speech_dur"].sum() / 3600
    total_vad_h = clip_df["vad_speech_dur"].sum() / 3600
    print(f"  Files            : {n_files:,}")
    print(f"  Clips            : {n_clips:,}")
    print(f"  Total audio      : {total_dur_h:.2f} h")
    print(f"  VTC speech       : {total_vtc_h:.2f} h")
    print(f"  VAD speech       : {total_vad_h:.2f} h")
    durs = clip_df["duration"].to_numpy()
    print(f"  Clip duration    : median={np.median(durs):.0f}s  "
          f"mean={np.mean(durs):.0f}s  std={np.std(durs):.0f}s  "
          f"min={np.min(durs):.0f}s  max={np.max(durs):.0f}s")
    iou = clip_df["vad_vtc_iou"].to_numpy()
    print(f"  VAD-VTC IoU      : median={np.median(iou):.3f}  mean={np.mean(iou):.3f}")

    # ── Per-label speech hours ────────────────────────────────────
    print(f"\n{sep}")
    print("SPEECH VOLUME BY LABEL")
    print(sep)
    for l in VTC_LABELS:
        col = f"dur_{l}"
        if col in clip_df.columns:
            h = clip_df[col].sum() / 3600
            n_seg = int(segment_df.filter(pl.col("label") == l).height) if len(segment_df) > 0 else 0
            print(f"  {l:6s} : {h:7.2f} h   ({n_seg:,} segments)")
    cf = clip_df["child_fraction"].to_numpy()
    print(f"  Child fraction   : median={np.median(cf):.3f}  mean={np.mean(cf):.3f}")

    # ── SNR & C50 ─────────────────────────────────────────────────
    snr_vals = clip_df["snr_mean"].drop_nulls().drop_nans().to_numpy()
    c50_vals = clip_df["c50_mean"].drop_nulls().drop_nans().to_numpy()
    if len(snr_vals) > 0 or len(c50_vals) > 0:
        print(f"\n{sep}")
        print("SNR & C50 (CLIP-LEVEL)")
        print(sep)
    if len(snr_vals) > 0:
        print(f"  SNR  (n={len(snr_vals):,}) : median={np.median(snr_vals):.1f} dB  "
              f"mean={np.mean(snr_vals):.1f}  std={np.std(snr_vals):.1f}  "
              f"[{np.min(snr_vals):.1f}, {np.max(snr_vals):.1f}]")
        for t in [0, 5, 10, 15, 20]:
            frac = float((snr_vals < t).mean())
            print(f"    SNR < {t:2d} dB : {frac:.1%} of clips")
        # Per-label SNR during speech
        if "snr_during" in segment_df.columns:
            seg_snr = segment_df.filter(pl.col("snr_during").is_not_null())
            if len(seg_snr) > 0:
                print("  Per-label SNR during speech:")
                for l in VTC_LABELS:
                    v = seg_snr.filter(pl.col("label") == l)["snr_during"].to_numpy()
                    if len(v) > 0:
                        print(f"    {l:6s} : mean={np.mean(v):.1f} dB  std={np.std(v):.1f}")
    if len(c50_vals) > 0:
        print(f"  C50  (n={len(c50_vals):,}) : median={np.median(c50_vals):.1f} dB  "
              f"mean={np.mean(c50_vals):.1f}  std={np.std(c50_vals):.1f}  "
              f"[{np.min(c50_vals):.1f}, {np.max(c50_vals):.1f}]")

    # ── Turns ─────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("TURNS (gap merge = 300ms)")
    print(sep)
    n_turns = len(turn_df)
    print(f"  Total turns      : {n_turns:,}")
    if n_turns > 0:
        td = turn_df["duration"].to_numpy()
        print(f"  Turn duration    : median={np.median(td):.2f}s  "
              f"mean={np.mean(td):.2f}s  std={np.std(td):.2f}s")
        for l in VTC_LABELS:
            d = turn_df.filter(pl.col("label") == l)["duration"].to_numpy()
            if len(d) > 0:
                print(f"    {l:6s} (n={len(d):,}) : "
                      f"median={np.median(d):.2f}s  mean={np.mean(d):.2f}s  "
                      f"std={np.std(d):.2f}s")
        td_per_min = clip_df["turn_density_per_min"].drop_nulls().to_numpy()
        if len(td_per_min) > 0:
            print(f"  Turn density     : median={np.median(td_per_min):.1f}/min  "
                  f"mean={np.mean(td_per_min):.1f}/min")

    # ── Conversations ─────────────────────────────────────────────
    print(f"\n{sep}")
    print("CONVERSATIONS (max silence = 10s)")
    print(sep)
    n_convs = len(conv_df)
    print(f"  Total conversations : {n_convs:,}")
    if n_convs > 0:
        cd = conv_df["duration"].to_numpy()
        print(f"  Duration         : median={np.median(cd):.1f}s  "
              f"mean={np.mean(cd):.1f}s  std={np.std(cd):.1f}s  "
              f"[{np.min(cd):.1f}, {np.max(cd):.1f}]")
        nt = conv_df["n_turns"].to_numpy()
        print(f"  Turns/conv       : median={np.median(nt):.0f}  "
              f"mean={np.mean(nt):.1f}  std={np.std(nt):.1f}  "
              f"max={np.max(nt)}")
        multi = int(conv_df["is_multi_speaker"].sum())
        print(f"  Multi-speaker    : {multi:,} ({100*multi/n_convs:.1f}%)")
        print(f"  Single-speaker   : {n_convs - multi:,} ({100*(n_convs-multi)/n_convs:.1f}%)")
        # Inter-conversation gaps
        ic_gaps = conv_df["gap_after"].drop_nulls().to_numpy()
        if len(ic_gaps) > 0:
            print(f"  Inter-conv gap   : median={np.median(ic_gaps):.1f}s  "
                  f"mean={np.mean(ic_gaps):.1f}s  std={np.std(ic_gaps):.1f}s")
        # Conversation-level SNR/C50
        if "snr_mean" in conv_df.columns:
            cs = conv_df["snr_mean"].drop_nulls().drop_nans().to_numpy()
            if len(cs) > 0:
                print(f"  Conv SNR         : median={np.median(cs):.1f} dB  "
                      f"mean={np.mean(cs):.1f}")
        if "c50_mean" in conv_df.columns:
            cc = conv_df["c50_mean"].drop_nulls().drop_nans().to_numpy()
            if len(cc) > 0:
                print(f"  Conv C50         : median={np.median(cc):.1f} dB  "
                      f"mean={np.mean(cc):.1f}")

    # ── Speaker transitions ───────────────────────────────────────
    print(f"\n{sep}")
    print("SPEAKER TRANSITIONS")
    print(sep)
    n_trans = len(trans_df)
    print(f"  Total transitions: {n_trans:,}")
    if n_trans > 0:
        top = (
            trans_df.group_by(["from_label", "to_label"])
            .len()
            .sort("len", descending=True)
            .head(10)
        )
        for row in top.iter_rows(named=True):
            fl, tl = row["from_label"], row["to_label"]
            cnt = row["len"]
            gaps = trans_df.filter(
                (pl.col("from_label") == fl) & (pl.col("to_label") == tl)
            )["gap_s"].to_numpy()
            print(f"    {fl:6s} → {tl:6s} : {cnt:5,}  "
                  f"gap: median={np.median(gaps):.2f}s  mean={np.mean(gaps):.2f}s")

    # ── Cut quality ───────────────────────────────────────────────
    print(f"\n{sep}")
    print("CUT-POINT TIER BREAKDOWN")
    print(sep)
    total_cuts = sum(tier_counts.values())
    tier_labels = {
        "long_union_gap":    "1. Long silence",
        "short_union_gap":   "2. Short silence",
        "vad_only_gap":      "3. VAD-only gap",
        "vtc_only_gap":      "4. VTC-only gap",
        "speaker_boundary":  "5. Speaker boundary",
        "hard_cut":          "6. Hard cut",
        "degenerate_window": "7. Degenerate",
    }
    for k, label in tier_labels.items():
        cnt = tier_counts.get(k, 0)
        pct = 100 * cnt / total_cuts if total_cuts > 0 else 0
        print(f"  {label:24s}: {cnt:5,}  ({pct:5.1f}%)")
    print(f"  {'TOTAL':24s}: {total_cuts:5,}")

    # ── Correlation highlights ────────────────────────────────────
    if "correlation" in dfs:
        corr_df = dfs["correlation"]
        metrics = corr_df["metric"].to_list()
        matrix = corr_df.drop("metric").to_numpy()
        print(f"\n{sep}")
        print("TOP CORRELATIONS (|r| > 0.3, off-diagonal)")
        print(sep)
        pairs: list[tuple[float, str, str]] = []
        for i in range(len(metrics)):
            for j in range(i + 1, len(metrics)):
                r = matrix[i, j]
                if abs(r) > 0.3 and not np.isnan(r):
                    pairs.append((abs(r), metrics[i], metrics[j]))
        pairs.sort(reverse=True)
        for absval, m1, m2 in pairs[:15]:
            r = matrix[metrics.index(m1), metrics.index(m2)]
            print(f"  {m1:25s} ↔ {m2:25s}  r={r:+.3f}")

    print(f"\n{sep}")
    print("END OF DASHBOARD SUMMARY")
    print(sep)


# =========================================================================
# Page 7: Noise Environment (PANNs)
# =========================================================================

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

