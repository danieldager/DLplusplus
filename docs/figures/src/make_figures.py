"""Draw the README figures. No data files needed: the numbers below were read from the
dashboards in figures/seedlings_10/ (the per-file tables behind them are not published).
    uv run --with matplotlib --with pillow python docs/figures/src/make_figures.py"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

OUT = Path(__file__).parent.parent
REPO = OUT.parent.parent

# SEEDLingS subset, 52 daylong recordings (figures/seedlings_10/master_overview.png, vad/coverage.png).
SPEAKERS = [("Key child (the wearer)", 138.1, 93.0), ("Adult women", 106.7, 43.5),
            ("Other children", 45.2, 20.7), ("Adult men", 38.7, 16.0)]  # (label, VTC hours, hours VAD misses)
MISSED_PCT = {"Key child (the wearer)": 67.4, "Adult women": 40.8, "Other children": 45.7, "Adult men": 41.5}
CUTS = [("Long silence (≥ 10 s)", 2432), ("Shorter silence", 2204),
        ("Pause heard by the\nspeech detector only", 7), ("Forced, mid-speech", 0)]
# figures/seedlings_10/vtc/chunk_grid_evidence.png, panels (b) and (c): mean per-clip share of whole-file
# segments reproduced by the clip-level run, by boundary tolerance.
TOL = [0.02, 0.1, 0.5, 2.0, 5.0]
ALIGNED = [99.6, 99.7, 99.8, 99.9, 100.0]      # 52 clips starting on the model's window grid
MISALIGNED = [3.9, 15.7, 40.6, 66.0, 80.1]     # 3,939 clips starting off the grid

# dataviz reference palette (light surface)
BLUE, ORANGE = "#2a78d6", "#eb6834"
TRACK = "#e8e7e1"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
W, DPI = 8.0, 200  # 1600 px wide
plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "axes.titlepad": 12, "axes.labelsize": 11, "axes.labelcolor": INK2, "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5, "xtick.color": INK2, "ytick.color": INK2, "legend.fontsize": 10.5,
    "legend.frameon": False, "text.color": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "figure.facecolor": "white", "savefig.facecolor": "white"})


def frame(ax, grid="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.get_layout_engine().set(w_pad=0.2, h_pad=0.2)
    fig.savefig(OUT / name, dpi=DPI * W / fig.get_figwidth())
    plt.close(fig)


def fig_vad_coverage():
    fig, ax = plt.subplots(figsize=(W, 3.6), layout="constrained")
    rows = SPEAKERS[::-1]
    for y, (lab, total, missed) in enumerate(rows):
        caught = 100 - MISSED_PCT[lab]
        ax.barh(y, 100, height=0.56, color=TRACK)
        ax.barh(y, caught, height=0.56, color=BLUE)
        ax.text(caught + 1.2, y + 0.04, f"{caught:.1f}% caught", va="bottom", fontsize=10.5, color=INK,
                fontweight="bold" if y == len(rows) - 1 else "normal")
        ax.text(caught + 1.2, y - 0.02, f"missed {missed:.1f} h of {total:.1f} h", va="top", fontsize=9.5, color=INK2)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_title("Share of each speaker's speech a generic speech detector catches")
    frame(ax)
    save(fig, "vad_coverage.png")


def fig_overview():
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 3.4), layout="constrained", width_ratios=[1, 1.15])
    rows = SPEAKERS[::-1]
    a.barh(range(4), [r[1] for r in rows], height=0.56, color=BLUE)
    for y, r in enumerate(rows):
        a.text(r[1] + 3, y, f"{r[1]:.1f} h", va="center", fontsize=10.5, color=INK)
    a.set_yticks(range(4), [r[0] for r in rows])
    a.set_xlim(0, 165)
    a.set_xlabel("hours of speech")
    a.set_title("Speech by speaker, 739 h of audio")
    frame(a)
    rows = CUTS[::-1]
    b.barh(range(4), [r[1] for r in rows], height=0.56, color=BLUE)
    total = sum(r[1] for r in CUTS)
    for y, r in enumerate(rows):
        b.text(r[1] + 50, y, f"{r[1]:,}  ({100 * r[1] / total:.1f}%)", va="center", fontsize=10.5, color=INK)
    b.set_yticks(range(4), [r[0] for r in rows])
    b.set_xlim(0, 3300)
    b.set_xlabel("cut points")
    b.set_title(f"Where the {total:,} cut points fall")
    frame(b)
    save(fig, "overview.png")


def fig_grid():
    fig, ax = plt.subplots(figsize=(W, 3.8), layout="constrained")
    for ys, c, lab in ((ALIGNED, BLUE, "Clip starts on the model's window grid (52 clips)"),
                       (MISALIGNED, ORANGE, "Clip starts off the grid (3,939 clips)")):
        ax.plot(TOL, ys, color=c, linewidth=2, label=lab, zorder=3)
        ax.scatter(TOL, ys, s=64, color=c, edgecolor="white", linewidth=2, zorder=4)
        ax.annotate(f"{ys[0]:.1f}%", (TOL[0], ys[0]), xytext=(0, -18 if c == BLUE else 10),
                    textcoords="offset points", ha="center", fontsize=11, fontweight="bold", color=INK)
    for x, y in zip(TOL[1:], MISALIGNED[1:]):
        ax.annotate(f"{y:.1f}%", (x, y), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=10, color=INK2)
    ax.set_xscale("log")
    ax.set_xticks(TOL, ["0.02 s\n(identical)", "0.1 s", "0.5 s", "2 s", "5 s"])
    ax.minorticks_off()
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("boundary tolerance")
    ax.set_title("Whole-file speaker segments reproduced when the model runs on a clip")
    ax.legend(loc="lower right")
    frame(ax, grid="y")
    save(fig, "grid.png")


def fig_spectrum():
    # Top panel of the analysis figure (the 743,566 raw segment durations are not published).
    im = Image.open(REPO / "figures/seedlings_10/chunk_artifact/fig3_spectrum.png").convert("RGB")
    top = im.crop((0, 0, im.width, 790))
    top.resize((1600, round(790 * 1600 / im.width)), Image.LANCZOS).save(OUT / "spectrum.png")


if __name__ == "__main__":
    fig_vad_coverage(); fig_overview(); fig_grid(); fig_spectrum()
