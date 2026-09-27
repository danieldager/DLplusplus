"""Shared fixtures for the VTC test suite.

Session-scoped fixtures provide paths to real speech WAV files committed
under ``tests/fixtures/``.  These are short LibriVox excerpts (public
domain) chosen to cover the range of edge cases the pipeline must handle:

  speech_clean.wav        6.0s  dense single-speaker speech (MAL)
  speech_multi.wav       10.2s  all four VTC labels (FEM, MAL, KCHI, OCH)
  speech_sparse.wav       5.8s  mostly silence with faint speech (~2.5%)
  speech_long.wav        27.3s  dense single-speaker speech (FEM)
  speech_long_multi.wav  62.7s  multi-speaker, all four VTC labels
  silence.wav             3.3s  pure silence
  short.wav               0.3s  very short clip (edge case)

No external data or stitching is needed — the real audio clips are small
enough to live in the repository (~3.6 MB total).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

# ---------------------------------------------------------------------------
# Paths — all fixtures are committed under tests/fixtures/
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures"

SR = 16_000

# Individual fixture WAVs
SPEECH_CLEAN = FIXTURE_DIR / "speech_clean.wav"
SPEECH_MULTI = FIXTURE_DIR / "speech_multi.wav"
SPEECH_SPARSE = FIXTURE_DIR / "speech_sparse.wav"
SPEECH_LONG = FIXTURE_DIR / "speech_long.wav"
SPEECH_LONG_MULTI = FIXTURE_DIR / "speech_long_multi.wav"
SILENCE = FIXTURE_DIR / "silence.wav"
SHORT = FIXTURE_DIR / "short.wav"

ALL_FIXTURE_WAVS = [
    SPEECH_CLEAN, SPEECH_MULTI, SPEECH_SPARSE,
    SPEECH_LONG, SPEECH_LONG_MULTI,
    SILENCE, SHORT,
]
SPEECH_FIXTURE_WAVS = [
    SPEECH_CLEAN, SPEECH_MULTI, SPEECH_SPARSE,
    SPEECH_LONG, SPEECH_LONG_MULTI,
]

# ---------------------------------------------------------------------------
# Availability checks
# ---------------------------------------------------------------------------


def _torchcodec_available() -> bool:
    """Return True if torchcodec can load (requires FFmpeg libs)."""
    try:
        import torchcodec  # noqa: F401

        return True
    except Exception:
        return False


def _tenvad_available() -> bool:
    """Return True if TenVAD can initialise on this machine."""
    try:
        from ten_vad import TenVad

        v = TenVad(hop_size=256, threshold=0.5)
        del v
        return True
    except Exception:
        return False


def _brouhaha_available() -> bool:
    """Return True if Brouhaha + pyannote can be loaded."""
    try:
        from src.compat import patch_torchaudio
        patch_torchaudio()
        from brouhaha.pipeline import RegressiveActivityDetectionPipeline  # noqa: F401
        from pyannote.audio import Model  # noqa: F401

        return True
    except Exception:
        return False


_TORCHCODEC_OK = _torchcodec_available()
_TENVAD_OK = _tenvad_available()
_BROUHAHA_OK = _brouhaha_available()

requires_torchcodec = pytest.mark.skipif(
    not _TORCHCODEC_OK,
    reason="torchcodec/ffmpeg not available — run via slurm/test.slurm on a compute node",
)

requires_tenvad = pytest.mark.skipif(
    not _TENVAD_OK,
    reason="TenVAD unavailable (needs 'module load llvm' for libc++.so.1) — run via slurm/test.slurm",
)

requires_brouhaha = pytest.mark.skipif(
    not _BROUHAHA_OK,
    reason="Brouhaha/pyannote unavailable — run via slurm/test.slurm on a GPU node",
)


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def fixture_dir() -> Path:
    """Return the path to ``tests/fixtures/``."""
    return FIXTURE_DIR


@pytest.fixture(scope="session")
def speech_clean_wav() -> Path:
    """6.0s dense single-speaker speech (MAL). ~86% speech ratio."""
    if not SPEECH_CLEAN.exists():
        pytest.skip("tests/fixtures/speech_clean.wav not found")
    return SPEECH_CLEAN


@pytest.fixture(scope="session")
def speech_multi_wav() -> Path:
    """10.2s multi-speaker clip with all 4 VTC labels. ~88% speech ratio."""
    if not SPEECH_MULTI.exists():
        pytest.skip("tests/fixtures/speech_multi.wav not found")
    return SPEECH_MULTI


@pytest.fixture(scope="session")
def speech_sparse_wav() -> Path:
    """5.8s mostly-silent clip. ~2.5% speech ratio."""
    if not SPEECH_SPARSE.exists():
        pytest.skip("tests/fixtures/speech_sparse.wav not found")
    return SPEECH_SPARSE


@pytest.fixture(scope="session")
def speech_long_wav() -> Path:
    """27.3s dense single-speaker speech (FEM). 97.7% speech ratio."""
    if not SPEECH_LONG.exists():
        pytest.skip("tests/fixtures/speech_long.wav not found")
    return SPEECH_LONG


@pytest.fixture(scope="session")
def speech_long_multi_wav() -> Path:
    """62.7s multi-speaker clip with all 4 VTC labels. 93.7% speech ratio."""
    if not SPEECH_LONG_MULTI.exists():
        pytest.skip("tests/fixtures/speech_long_multi.wav not found")
    return SPEECH_LONG_MULTI


@pytest.fixture(scope="session")
def silence_wav() -> Path:
    """3.3s pure silence."""
    if not SILENCE.exists():
        pytest.skip("tests/fixtures/silence.wav not found")
    return SILENCE


@pytest.fixture(scope="session")
def short_wav() -> Path:
    """0.3s very short clip — edge case."""
    if not SHORT.exists():
        pytest.skip("tests/fixtures/short.wav not found")
    return SHORT


@pytest.fixture(scope="session")
def all_fixture_wavs() -> list[Path]:
    """All 7 fixture WAVs."""
    missing = [p for p in ALL_FIXTURE_WAVS if not p.exists()]
    if missing:
        pytest.skip(f"Missing fixture WAVs: {[p.name for p in missing]}")
    return list(ALL_FIXTURE_WAVS)


@pytest.fixture(scope="session")
def speech_fixture_wavs() -> list[Path]:
    """The 5 fixture WAVs that contain detectable speech."""
    missing = [p for p in SPEECH_FIXTURE_WAVS if not p.exists()]
    if missing:
        pytest.skip(f"Missing fixture WAVs: {[p.name for p in missing]}")
    return list(SPEECH_FIXTURE_WAVS)


# ---------------------------------------------------------------------------
# Manifest fixture (for pipeline-level tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def test_manifest(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write a minimal manifest CSV pointing to all fixture WAVs.

    Returns the path to the manifest file.
    """
    missing = [p for p in ALL_FIXTURE_WAVS if not p.exists()]
    if missing:
        pytest.skip(f"Missing fixture WAVs: {[p.name for p in missing]}")
    manifest = tmp_path_factory.mktemp("vtc_test") / "test_manifest.csv"
    rows = [{"path": str(p)} for p in ALL_FIXTURE_WAVS]
    pl.DataFrame(rows).write_csv(manifest)
    return manifest


# ---------------------------------------------------------------------------
# Backward-compatibility aliases  (used by existing tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def good_book_wavs() -> list[Path]:
    """Backward-compat: return speech WAVs (replaces old good_book chunks)."""
    missing = [p for p in SPEECH_FIXTURE_WAVS if not p.exists()]
    if missing:
        pytest.skip(f"Missing speech fixture WAVs: {[p.name for p in missing]}")
    return list(SPEECH_FIXTURE_WAVS)
