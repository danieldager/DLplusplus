"""Integration tests — end-to-end VAD on real speech fixtures.

Tests the contract of process_vad_file using real LibriVox speech clips.
Assertions check output schemas and directional properties (speech files
have higher ratios than silence files) without requiring specific numeric
thresholds that depend on TenVAD's internal model.

Fixtures (from conftest.py):
  speech_clean.wav        6.0s  dense single-speaker (MAL)
  speech_multi.wav       10.2s  all 4 VTC labels
  speech_sparse.wav       5.8s  mostly silence, faint speech
  speech_long.wav        27.3s  dense single-speaker (FEM)
  speech_long_multi.wav  62.7s  multi-speaker, all 4 VTC labels
  silence.wav             3.3s  pure silence
  short.wav               0.3s  very short edge case
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import soundfile as sf

from tests.conftest import (
    _TENVAD_OK,
    requires_tenvad,
)

# Guard heavy imports — TenVAD may not be available on login nodes.
if TYPE_CHECKING:
    from src.core.vad_processing import process_vad_file

if _TENVAD_OK:
    from src.core.vad_processing import process_vad_file
else:

    def process_vad_file(args: Any) -> Any:  # type: ignore[misc]
        raise RuntimeError("TenVAD unavailable")

# ---------------------------------------------------------------------------
# Fixture validation (no TenVAD needed — just checks files exist & are valid)
# ---------------------------------------------------------------------------


class TestFixtureFiles:
    """Verify the fixture WAV files exist and have expected properties."""

    def test_all_files_exist(self, all_fixture_wavs: list[Path]):
        for wav in all_fixture_wavs:
            assert wav.exists(), f"Missing fixture: {wav.name}"

    def test_audio_properties(self, all_fixture_wavs: list[Path]):
        """All fixtures should be 16 kHz mono PCM_16."""
        for wav in all_fixture_wavs:
            info = sf.info(str(wav))
            assert info.samplerate == 16_000, f"{wav.name}: sr={info.samplerate}"
            assert info.channels == 1, f"{wav.name}: channels={info.channels}"

    def test_speech_clean_duration(self, speech_clean_wav: Path):
        info = sf.info(str(speech_clean_wav))
        assert 5.0 < info.duration < 8.0

    def test_speech_multi_duration(self, speech_multi_wav: Path):
        info = sf.info(str(speech_multi_wav))
        assert 9.0 < info.duration < 12.0

    def test_short_duration(self, short_wav: Path):
        info = sf.info(str(short_wav))
        assert info.duration < 1.0

    def test_speech_long_duration(self, speech_long_wav: Path):
        info = sf.info(str(speech_long_wav))
        assert 25.0 < info.duration < 30.0

    def test_speech_long_multi_duration(self, speech_long_multi_wav: Path):
        info = sf.info(str(speech_long_multi_wav))
        assert 60.0 < info.duration < 65.0


# ---------------------------------------------------------------------------
# VAD integration — output schema and directional checks
# ---------------------------------------------------------------------------

# Required keys in the metadata dict returned by process_vad_file
_REQUIRED_META_KEYS = {
    "success",
    "path",
    "file_id",
    "duration",
    "original_sr",
    "speech_ratio",
    "n_speech_segments",
    "n_silence_segments",
    "speech_max",
    "speech_min",
    "speech_sum",
    "speech_num",
    "speech_avg",
    "nospch_max",
    "nospch_min",
    "nospch_sum",
    "nospch_num",
    "nospch_avg",
    "has_long_segment",
    "error",
}

_REQUIRED_SEG_KEYS = {"file_id", "onset", "offset", "duration"}


@requires_tenvad
class TestVadIntegration:
    """End-to-end: VAD on real speech should return correct schemas and
    directional properties."""

    def test_speech_file_detects_speech(self, speech_clean_wav: Path):
        """Clean speech file → success, some detected speech segments."""
        meta, segs = process_vad_file((speech_clean_wav, 256, 0.5))
        assert meta["success"] is True
        assert (
            meta["speech_ratio"] > 0
        ), f"Expected speech_ratio > 0, got {meta['speech_ratio']:.3f}"
        assert meta["n_speech_segments"] >= 1
        assert len(segs) >= 1

    def test_multi_speaker_detects_speech(self, speech_multi_wav: Path):
        """Multi-speaker file → success with detectable speech."""
        meta, segs = process_vad_file((speech_multi_wav, 256, 0.5))
        assert meta["success"] is True
        assert (
            meta["speech_ratio"] > 0
        ), f"Expected speech_ratio > 0, got {meta['speech_ratio']:.3f}"
        assert meta["n_speech_segments"] >= 1

    def test_silence_file_low_speech(self, silence_wav: Path):
        """Pure silence → success with near-zero speech ratio."""
        meta, segs = process_vad_file((silence_wav, 256, 0.5))
        assert meta["success"] is True
        assert (
            meta["speech_ratio"] < 0.15
        ), f"Expected speech_ratio < 0.15 for silence, got {meta['speech_ratio']:.3f}"

    def test_short_file_no_crash(self, short_wav: Path):
        """Very short file (0.3s) processes without error."""
        meta, segs = process_vad_file((short_wav, 256, 0.5))
        assert meta["success"] is True
        assert meta["duration"] > 0

    def test_speech_ratio_ordering(
        self,
        speech_clean_wav: Path,
        silence_wav: Path,
    ):
        """Speech file should have a higher speech ratio than silence file."""
        speech_meta, _ = process_vad_file((speech_clean_wav, 256, 0.5))
        silence_meta, _ = process_vad_file((silence_wav, 256, 0.5))
        assert speech_meta["success"] is True
        assert silence_meta["success"] is True
        assert speech_meta["speech_ratio"] > silence_meta["speech_ratio"], (
            f"Speech ratio {speech_meta['speech_ratio']:.3f} should exceed "
            f"silence ratio {silence_meta['speech_ratio']:.3f}"
        )

    def test_output_schema(self, speech_clean_wav: Path):
        """Metadata and segment dicts have all required keys."""
        meta, segs = process_vad_file((speech_clean_wav, 256, 0.5))
        assert _REQUIRED_META_KEYS <= set(
            meta.keys()
        ), f"Missing meta keys: {_REQUIRED_META_KEYS - set(meta.keys())}"
        assert len(segs) > 0, "Expected at least one segment"
        for seg in segs:
            assert _REQUIRED_SEG_KEYS <= set(
                seg.keys()
            ), f"Missing seg keys: {_REQUIRED_SEG_KEYS - set(seg.keys())}"
            assert seg["onset"] < seg["offset"]
            assert seg["duration"] > 0
            assert seg["onset"] >= 0

    def test_nonexistent_file(self, tmp_path: Path):
        """Nonexistent file → error metadata, no exception raised."""
        meta, segs = process_vad_file((tmp_path / "nope.wav", 256, 0.5))
        assert meta["success"] is False
        assert meta["error"]
        assert segs == []

    def test_long_file_detects_speech(self, speech_long_wav: Path):
        """27s dense speech file → high speech ratio."""
        meta, segs = process_vad_file((speech_long_wav, 256, 0.5))
        assert meta["success"] is True
        assert (
            meta["speech_ratio"] > 0
        ), f"Expected speech_ratio > 0 for 27s dense speech, got {meta['speech_ratio']:.3f}"
        assert meta["n_speech_segments"] >= 1
        assert len(segs) >= 1

    def test_long_multi_speaker_detects_speech(self, speech_long_multi_wav: Path):
        """63s multi-speaker file → high speech ratio with many segments."""
        meta, segs = process_vad_file((speech_long_multi_wav, 256, 0.5))
        assert meta["success"] is True
        assert (
            meta["speech_ratio"] > 0
        ), f"Expected speech_ratio > 0 for 63s multi-speaker, got {meta['speech_ratio']:.3f}"
        assert meta["n_speech_segments"] >= 2
        assert len(segs) >= 2

    def test_long_vs_short_ratio_ordering(
        self,
        speech_long_wav: Path,
        speech_clean_wav: Path,
        silence_wav: Path,
    ):
        """Long speech should have a clearly higher ratio than silence."""
        long_meta, _ = process_vad_file((speech_long_wav, 256, 0.5))
        silence_meta, _ = process_vad_file((silence_wav, 256, 0.5))
        assert long_meta["speech_ratio"] > silence_meta["speech_ratio"], (
            f"Long speech ratio {long_meta['speech_ratio']:.3f} should exceed "
            f"silence ratio {silence_meta['speech_ratio']:.3f}"
        )
