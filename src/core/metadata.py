"""Per-file metadata row constructors for VTC."""

from __future__ import annotations

import json


# ---------------------------------------------------------------------------
# VTC metadata templates
# ---------------------------------------------------------------------------

_EMPTY_VTC_META = dict(
    vtc_threshold=float("nan"),
    vtc_speech_dur=0.0,
    vtc_n_segments=0,
    vtc_label_counts="{}",
    vtc_max_sigmoid=float("nan"),
    vtc_mean_sigmoid=float("nan"),
    error="",
)


def vtc_error_row(uid: str, error: str) -> dict:
    """Metadata row for a file that errored during VTC inference."""
    return {**_EMPTY_VTC_META, "uid": uid, "error": error}


def vtc_meta_row(
    uid: str,
    threshold: float,
    segments: list[dict],
    max_sigmoid: float,
    mean_sigmoid: float,
) -> dict:
    """Build a metadata row from a file's VTC results."""
    label_counts: dict[str, int] = {}
    speech_dur = 0.0
    for s in segments:
        speech_dur += s["duration"]
        label_counts[s["label"]] = label_counts.get(s["label"], 0) + 1
    return {
        "uid": uid,
        "vtc_threshold": threshold,
        "vtc_speech_dur": round(speech_dur, 3),
        "vtc_n_segments": len(segments),
        "vtc_label_counts": json.dumps(label_counts),
        "vtc_max_sigmoid": max_sigmoid,
        "vtc_mean_sigmoid": mean_sigmoid,
        "error": "",
    }
