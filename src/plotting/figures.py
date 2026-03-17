"""Figure orchestrator — renders all plot pages from cached DataFrames.

Sub-modules
-----------
- ``snr_noise``     — SNR & Recording Quality, Noise Environment
- ``speech_turns``  — Conversational Structure, Turns & Conversations
- ``overview``      — Dataset Overview, Correlation, Text Summary
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.plotting.overview import (
    print_dataset_summary,
    save_correlation_figure,
    save_overview_figures,
)
from src.plotting.snr_noise import save_noise_figures, save_snr_figures
from src.plotting.speech_turns import save_boss_figures, save_conversation_figures


def save_all_figures(
    dfs: dict[str, pl.DataFrame],
    tier_counts: dict[str, int],
    fig_dir: Path,
) -> None:
    """Render every figure page from cached DataFrames.

    Parameters
    ----------
    dfs : dict[str, pl.DataFrame]
        Output of ``save_all_stats``.  Expected keys:
        clip_stats, segment_stats, turn_stats, conversation_stats,
        transition_stats, file_stats, correlation.
    tier_counts : dict[str, int]
        Cut-tier breakdown from ``build_clips``.
    fig_dir : Path
        Root figure directory.  Pages are saved as PNG files inside it.
    """
    fig_dir.mkdir(parents=True, exist_ok=True)

    clip_df = dfs["clip_stats"]
    segment_df = dfs["segment_stats"]
    turn_df = dfs["turn_stats"]
    conv_df = dfs["conversation_stats"]
    trans_df = dfs["transition_stats"]
    file_df = dfs["file_stats"]

    # Page 1: SNR & Recording Quality
    save_snr_figures(clip_df, segment_df, conv_df, fig_dir / "snr_quality.png")

    # Page 2: Conversational Structure
    save_conversation_figures(
        clip_df, turn_df, conv_df, trans_df, fig_dir / "conversation_structure.png"
    )

    # Page 3: Boss Page — Turns & Conversations
    save_boss_figures(turn_df, conv_df, fig_dir / "turns_conversations.png")

    # Page 4: Dataset Overview + Cut Quality (combined 3×3)
    save_overview_figures(
        clip_df, file_df, segment_df, tier_counts, fig_dir / "dataset_overview.png"
    )

    # Page 5: Correlation Matrix
    if "correlation" in dfs:
        save_correlation_figure(dfs["correlation"], fig_dir / "correlation_matrix.png")

    # Page 6: Noise Environment (PANNs)
    noise_cols = [c for c in clip_df.columns if c.startswith("noise_")]
    n_pages = 4 + (1 if "correlation" in dfs else 0)
    if noise_cols:
        save_noise_figures(clip_df, segment_df, fig_dir / "noise_environment.png")
        n_pages += 1

    print(f"\n  Dashboard: {fig_dir}/ ({n_pages} pages)")

    # Print text summary for log parsing
    print_dataset_summary(dfs, tier_counts)
