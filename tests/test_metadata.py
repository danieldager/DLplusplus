"""Tests for src.core.metadata — VTC metadata row builders."""

from __future__ import annotations

import json
import math

import pytest

from src.core.metadata import (
    _EMPTY_VTC_META,
    vtc_error_row,
    vtc_meta_row,
)


class TestVtcErrorRow:
    def test_structure(self):
        row = vtc_error_row("file001", "some error")
        assert row["uid"] == "file001"
        assert row["error"] == "some error"

    def test_has_all_keys(self):
        row = vtc_error_row("x", "e")
        for key in _EMPTY_VTC_META:
            assert key in row


class TestVtcMetaRow:
    def test_basic(self):
        segments = [
            {"label": "KCHI", "duration": 1.0},
            {"label": "KCHI", "duration": 2.0},
            {"label": "FEM", "duration": 3.0},
        ]
        row = vtc_meta_row("uid1", 0.4, segments, 0.9, 0.6)
        assert row["uid"] == "uid1"
        assert row["vtc_threshold"] == 0.4
        assert row["vtc_speech_dur"] == pytest.approx(6.0)
        assert row["vtc_n_segments"] == 3
        counts = json.loads(row["vtc_label_counts"])
        assert counts["KCHI"] == 2
        assert counts["FEM"] == 1

    def test_empty_segments(self):
        row = vtc_meta_row("uid2", 0.5, [], 0.0, 0.0)
        assert row["vtc_speech_dur"] == 0.0
        assert row["vtc_n_segments"] == 0
        assert json.loads(row["vtc_label_counts"]) == {}
