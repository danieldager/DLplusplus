"""Build intermediate DataFrames from packaged clips for plotting / caching.

Every function returns a ``polars.DataFrame`` that can be saved as
parquet and later reloaded for the dashboard without re-running the
packaging pipeline.

DataFrames produced
-------------------
``build_clip_stats``
    One row per clip — duration, densities, IoU, SNR, C50, turn/conv counts.

``build_segment_stats``
    One row per VTC segment — onset, offset, label, SNR during segment.

``build_turn_stats``
    One row per conversational turn.

``build_conversation_stats``
    One row per conversation — duration, #turns, speaker labels, mean SNR/C50.

``build_transition_stats``
    One row per speaker transition — from/to labels, gap, durations.

``build_file_stats``
    One row per source file — aggregated clip-level data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import json

import polars as pl

from src.core import VTC_LABELS
from src.core.conversations import (
    Conversation,
    Turn,
    detect_conversations,
    detect_turns,
    extract_transitions,
    inter_conversation_gaps,
)
from src.packaging.clips import CUT_TIERS, Clip, Segment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _snr_for_time_range(
    snr_array: np.ndarray | None,
    step_s: float,
    clip_onset: float,
    seg_onset: float,
    seg_offset: float,
) -> float | None:
    """Mean SNR during a segment's time range within a clip.

    *seg_onset* and *seg_offset* are in absolute coordinates.
    *snr_array* covers [clip_onset, …] with step *step_s*.
    """
    if snr_array is None or len(snr_array) == 0:
        return None
    rel_start = seg_onset - clip_onset
    rel_end = seg_offset - clip_onset
    i0 = max(0, int(rel_start / step_s))
    i1 = max(i0, min(len(snr_array), int(np.ceil(rel_end / step_s))))
    if i0 >= i1:
        return None
    return float(np.mean(snr_array[i0:i1], dtype=np.float32))


def _c50_for_time_range(
    c50_array: np.ndarray | None,
    step_s: float,
    clip_onset: float,
    seg_onset: float,
    seg_offset: float,
) -> float | None:
    """Mean C50 during a segment's time range within a clip."""
    if c50_array is None or len(c50_array) == 0:
        return None
    rel_start = seg_onset - clip_onset
    rel_end = seg_offset - clip_onset
    i0 = max(0, int(rel_start / step_s))
    i1 = max(i0, min(len(c50_array), int(np.ceil(rel_end / step_s))))
    if i0 >= i1:
        return None
    return float(np.mean(c50_array[i0:i1], dtype=np.float32))


# ---------------------------------------------------------------------------
# Clip-level stats
# ---------------------------------------------------------------------------


def build_clip_stats(
    clips: list[tuple[str, Path, int, Clip]],
    tier_counts: dict[str, int] | None = None,
) -> pl.DataFrame:
    """One row per clip with all scalar metrics."""
    rows: list[dict] = []
    for uid, _audio_path, clip_idx, clip in clips:
        # Conversation analysis per clip
        turns = detect_turns(clip.vtc_segments)
        convs = detect_conversations(turns)
        row = {
            "uid": uid,
            "clip_idx": clip_idx,
            "clip_id": f"{uid}_{clip_idx:04d}",
            "duration": round(clip.duration, 3),
            "abs_onset": round(clip.abs_onset, 3),
            "abs_offset": round(clip.abs_offset, 3),
            # VTC
            "vtc_speech_dur": round(clip.vtc_speech_duration, 3),
            "vtc_density": round(clip.speech_density, 3),
            "n_vtc_segments": len(clip.vtc_segments),
            "mean_vtc_seg_dur": round(clip.mean_vtc_seg_duration, 3),
            "mean_vtc_gap": round(clip.mean_vtc_gap, 3),
            "n_turns_raw": clip.n_turns,
            "n_labels": clip.n_labels,
            "has_adult": clip.has_adult,
            "dominant_label": clip.dominant_label or "?",
            "child_fraction": round(clip.child_fraction, 3),
            "child_speech_dur": round(clip.child_speech_duration, 3),
            "adult_speech_dur": round(clip.adult_speech_duration, 3),
            # VAD
            "vad_speech_dur": round(clip.vad_speech_duration, 3),
            "vad_density": round(clip.vad_density, 3),
            "n_vad_segments": len(clip.vad_segments),
            # Agreement
            "vad_vtc_iou": round(clip.vad_vtc_iou, 3),
            # SNR / C50
            "snr_mean": round(clip.snr_mean, 1) if clip.snr_mean is not None else None,
            "snr_std": round(clip.snr_std, 1) if clip.snr_std is not None else None,
            "snr_min": round(clip.snr_min, 1) if clip.snr_min is not None else None,
            "snr_max": round(clip.snr_max, 1) if clip.snr_max is not None else None,
            "c50_mean": round(clip.c50_mean, 1) if clip.c50_mean is not None else None,
            "c50_std": round(clip.c50_std, 1) if clip.c50_std is not None else None,
            "c50_min": round(clip.c50_min, 1) if clip.c50_min is not None else None,
            "c50_max": round(clip.c50_max, 1) if clip.c50_max is not None else None,
            # Per-label durations
            **{f"dur_{l}": round(clip.label_durations.get(l, 0.0), 3)
               for l in VTC_LABELS},
            # Conversations
            "n_conv_turns": len(turns),
            "n_conversations": len(convs),
            "n_multi_speaker_convs": sum(1 for c in convs if c.is_multi_speaker),
            "turn_density_per_min": round(
                len(turns) / (clip.duration / 60) if clip.duration > 0 else 0, 2
            ),
            # Noise classification (from PANNs)
            "dominant_noise": clip.dominant_noise or "?",
        }
        # Add per-category mean probabilities
        profile = clip.noise_profile
        if profile:
            for cat, prob in profile.items():
                row[f"noise_{cat}"] = round(prob, 4)
        rows.append(row)
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Segment-level stats
# ---------------------------------------------------------------------------


def build_segment_stats(
    clips: list[tuple[str, Path, int, Clip]],
) -> pl.DataFrame:
    """One row per VTC segment — label, duration, SNR during segment."""
    rows: list[dict] = []
    for uid, _audio_path, clip_idx, clip in clips:
        for seg in clip.vtc_segments:
            rows.append({
                "uid": uid,
                "clip_idx": clip_idx,
                "onset": round(seg.onset, 3),
                "offset": round(seg.offset, 3),
                "duration": round(seg.duration, 3),
                "label": seg.label or "?",
                "snr_during": round(_seg_snr, 1) if (_seg_snr := _snr_for_time_range(
                    clip.snr_array, clip.snr_step_s,
                    clip.abs_onset, seg.onset, seg.offset,
                )) is not None else None,
                "c50_during": round(_seg_c50, 1) if (_seg_c50 := _c50_for_time_range(
                    clip.c50_array, clip.snr_step_s,
                    clip.abs_onset, seg.onset, seg.offset,
                )) is not None else None,
            })
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Turn-level stats
# ---------------------------------------------------------------------------


def build_turn_stats(
    clips: list[tuple[str, Path, int, Clip]],
    min_gap_s: float = 0.3,
) -> pl.DataFrame:
    """One row per conversational turn."""
    rows: list[dict] = []
    for uid, _audio_path, clip_idx, clip in clips:
        turns = detect_turns(clip.vtc_segments, min_gap_s=min_gap_s)
        for turn_idx, turn in enumerate(turns):
            snr_val = _snr_for_time_range(
                clip.snr_array, clip.snr_step_s,
                clip.abs_onset, turn.onset, turn.offset,
            )
            c50_val = _c50_for_time_range(
                clip.c50_array, clip.snr_step_s,
                clip.abs_onset, turn.onset, turn.offset,
            )
            rows.append({
                "uid": uid,
                "clip_idx": clip_idx,
                "turn_idx": turn_idx,
                "onset": round(turn.onset, 3),
                "offset": round(turn.offset, 3),
                "duration": round(turn.duration, 3),
                "label": turn.label,
                "n_segments": turn.n_segments,
                "snr_during": round(snr_val, 1) if snr_val is not None else None,
                "c50_during": round(c50_val, 1) if c50_val is not None else None,
            })
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Conversation-level stats
# ---------------------------------------------------------------------------


def build_conversation_stats(
    clips: list[tuple[str, Path, int, Clip]],
    min_gap_s: float = 0.3,
    max_silence_s: float = 10.0,
) -> pl.DataFrame:
    """One row per conversation — duration, turns, labels, SNR/C50."""
    rows: list[dict] = []
    for uid, _audio_path, clip_idx, clip in clips:
        turns = detect_turns(clip.vtc_segments, min_gap_s=min_gap_s)
        convs = detect_conversations(turns, max_silence_s=max_silence_s)
        ic_gaps = inter_conversation_gaps(convs)

        for conv_idx, conv in enumerate(convs):
            snr_val = _snr_for_time_range(
                clip.snr_array, clip.snr_step_s,
                clip.abs_onset, conv.onset, conv.offset,
            )
            c50_val = _c50_for_time_range(
                clip.c50_array, clip.snr_step_s,
                clip.abs_onset, conv.onset, conv.offset,
            )
            rows.append({
                "uid": uid,
                "clip_idx": clip_idx,
                "conv_idx": conv_idx,
                "onset": round(conv.onset, 3),
                "offset": round(conv.offset, 3),
                "duration": round(conv.duration, 3),
                "n_turns": conv.n_turns,
                "is_multi_speaker": conv.is_multi_speaker,
                "labels": ";".join(conv.labels_present),
                "n_transitions": len(conv.transitions()),
                "snr_mean": round(snr_val, 1) if snr_val is not None else None,
                "c50_mean": round(c50_val, 1) if c50_val is not None else None,
                "gap_after": (
                    round(ic_gaps[conv_idx], 3)
                    if conv_idx < len(ic_gaps) else None
                ),
            })
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Transition-level stats
# ---------------------------------------------------------------------------


def build_transition_stats(
    clips: list[tuple[str, Path, int, Clip]],
    min_gap_s: float = 0.3,
    max_silence_s: float = 10.0,
) -> pl.DataFrame:
    """One row per speaker transition — from/to labels, gap, durations."""
    rows: list[dict] = []
    for uid, _audio_path, clip_idx, clip in clips:
        turns = detect_turns(clip.vtc_segments, min_gap_s=min_gap_s)
        convs = detect_conversations(turns, max_silence_s=max_silence_s)
        transitions = extract_transitions(convs)
        for tr in transitions:
            rows.append({
                "uid": uid,
                "clip_idx": clip_idx,
                "from_label": tr.from_label,
                "to_label": tr.to_label,
                "gap_s": round(tr.gap_s, 3),
                "from_duration": round(tr.from_duration, 3),
                "to_duration": round(tr.to_duration, 3),
            })
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# File-level stats
# ---------------------------------------------------------------------------


def build_file_stats(
    clips: list[tuple[str, Path, int, Clip]],
    min_gap_s: float = 0.3,
    max_silence_s: float = 10.0,
) -> pl.DataFrame:
    """One row per source file — aggregated from its clips."""
    # Group clips by uid
    from collections import defaultdict
    by_uid: dict[str, list[Clip]] = defaultdict(list)
    for uid, _audio_path, _clip_idx, clip in clips:
        by_uid[uid].append(clip)

    rows: list[dict] = []
    for uid, file_clips in by_uid.items():
        total_dur = sum(c.duration for c in file_clips)
        total_vtc = sum(c.vtc_speech_duration for c in file_clips)
        total_vad = sum(c.vad_speech_duration for c in file_clips)

        # Per-label total speech
        label_totals: dict[str, float] = {l: 0.0 for l in VTC_LABELS}
        for c in file_clips:
            for l, d in c.label_durations.items():
                if l in label_totals:
                    label_totals[l] += d

        # SNR / C50
        snr_vals = [c.snr_mean for c in file_clips if c.snr_mean is not None]
        c50_vals = [c.c50_mean for c in file_clips if c.c50_mean is not None]

        # Conversations across all clips for this file
        all_turns: list[Turn] = []
        all_convs: list[Conversation] = []
        for c in file_clips:
            turns = detect_turns(c.vtc_segments, min_gap_s=min_gap_s)
            convs = detect_conversations(turns, max_silence_s=max_silence_s)
            all_turns.extend(turns)
            all_convs.extend(convs)

        turn_durs = [t.duration for t in all_turns]
        conv_durs = [c.duration for c in all_convs]
        conv_turns = [c.n_turns for c in all_convs]

        rows.append({
            "uid": uid,
            "n_clips": len(file_clips),
            "total_dur": round(total_dur, 3),
            "total_vtc_speech": round(total_vtc, 3),
            "total_vad_speech": round(total_vad, 3),
            "vtc_density": round(total_vtc / total_dur, 3) if total_dur > 0 else 0,
            **{f"total_dur_{l}": round(v, 3) for l, v in label_totals.items()},
            "snr_mean": (
                round(float(np.mean(snr_vals)), 1) if snr_vals else None
            ),
            "c50_mean": (
                round(float(np.mean(c50_vals)), 1) if c50_vals else None
            ),
            "n_turns": len(all_turns),
            "n_conversations": len(all_convs),
            "mean_turn_dur": (
                round(float(np.mean(turn_durs)), 3) if turn_durs else 0.0
            ),
            "mean_conv_dur": (
                round(float(np.mean(conv_durs)), 3) if conv_durs else 0.0
            ),
            "mean_turns_per_conv": (
                round(float(np.mean(conv_turns)), 2) if conv_turns else 0.0
            ),
        })
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Save all DataFrames
# ---------------------------------------------------------------------------


def save_all_stats(
    clips: list[tuple[str, Path, int, Clip]],
    output_dir: Path,
    tier_counts: dict[str, int] | None = None,
    min_gap_s: float = 0.3,
    max_silence_s: float = 10.0,
) -> dict[str, pl.DataFrame]:
    """Build and save all intermediate DataFrames.

    Writes parquet files to ``output_dir/stats/`` and returns a dict
    of DataFrames keyed by name.
    """
    stats_dir = output_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    print("Building intermediate DataFrames ...", flush=True)

    dfs: dict[str, pl.DataFrame] = {}

    dfs["clip_stats"] = build_clip_stats(clips, tier_counts)
    dfs["clip_stats"].write_parquet(stats_dir / "clip_stats.parquet")
    print(f"  clip_stats      : {len(dfs['clip_stats']):,} rows")

    dfs["segment_stats"] = build_segment_stats(clips)
    dfs["segment_stats"].write_parquet(stats_dir / "segment_stats.parquet")
    print(f"  segment_stats   : {len(dfs['segment_stats']):,} rows")

    dfs["turn_stats"] = build_turn_stats(clips, min_gap_s=min_gap_s)
    dfs["turn_stats"].write_parquet(stats_dir / "turn_stats.parquet")
    print(f"  turn_stats      : {len(dfs['turn_stats']):,} rows")

    dfs["conversation_stats"] = build_conversation_stats(
        clips, min_gap_s=min_gap_s, max_silence_s=max_silence_s,
    )
    dfs["conversation_stats"].write_parquet(stats_dir / "conversation_stats.parquet")
    print(f"  conversation_stats: {len(dfs['conversation_stats']):,} rows")

    dfs["transition_stats"] = build_transition_stats(
        clips, min_gap_s=min_gap_s, max_silence_s=max_silence_s,
    )
    dfs["transition_stats"].write_parquet(stats_dir / "transition_stats.parquet")
    print(f"  transition_stats: {len(dfs['transition_stats']):,} rows")

    dfs["file_stats"] = build_file_stats(
        clips, min_gap_s=min_gap_s, max_silence_s=max_silence_s,
    )
    dfs["file_stats"].write_parquet(stats_dir / "file_stats.parquet")
    print(f"  file_stats      : {len(dfs['file_stats']):,} rows")

    # Correlation matrix across numeric clip-level columns
    numeric_cols = [
        "duration", "vtc_density", "vad_density", "vad_vtc_iou",
        "n_conv_turns", "n_labels", "child_fraction",
        "snr_mean", "snr_std", "c50_mean", "c50_std",
        "turn_density_per_min", "n_conversations",
    ]
    avail_cols = [c for c in numeric_cols if c in dfs["clip_stats"].columns]
    if avail_cols:
        corr_df = dfs["clip_stats"].select(avail_cols).to_pandas().corr()
        corr_pl = pl.from_pandas(corr_df.reset_index().rename(
            columns={"index": "metric"}
        ))
        dfs["correlation"] = corr_pl
        corr_pl.write_parquet(stats_dir / "correlation_matrix.parquet")
        print(f"  correlation     : {len(avail_cols)}×{len(avail_cols)} matrix")

    if tier_counts is not None:
        (stats_dir / "tier_counts.json").write_text(
            json.dumps(tier_counts, indent=2)
        )
        print(f"  tier_counts     : {sum(tier_counts.values()):,} cuts")

    print(f"  Saved to: {stats_dir}/")
    return dfs
