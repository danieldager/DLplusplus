"""Shared plotting utilities — lazy imports, figure saving, common helpers."""

from __future__ import annotations

from pathlib import Path


def lazy_pyplot():
    """Return matplotlib.pyplot with the Agg backend.

    Avoids importing matplotlib at module level (which would block on
    display-less HPC nodes).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_figure(fig, output_path: Path, dpi: int = 150) -> None:
    """Save a matplotlib figure, create parent dirs, close, and log."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    print(f"  Figure: {output_path}")


def hist_with_median(ax, data, title: str, bins: int = 40, **kwargs) -> None:
    """Draw a histogram with a red dashed median line and legend."""
    import numpy as np

    data = np.asarray(data)
    data = data[np.isfinite(data)]
    if len(data) == 0:
        ax.set_title(title)
        return
    ax.hist(data, bins=bins, edgecolor="white", linewidth=0.3, **kwargs)
    med = float(np.median(data))
    ax.axvline(med, color="red", ls="--", label=f"median={med:.1f}")
    ax.legend(fontsize=8)
    ax.set_title(title)
