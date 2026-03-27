"""Tests for the Brouhaha SNR pipeline — correctness, output format, and reproducibility.

Tests are split into two groups:

1. **Pure-numpy tests** for ``pool_snr`` — run anywhere (login node included).
2. **GPU/model tests** for ``_extract_snr`` and full-pipeline behaviour — require
   Brouhaha + pyannote (run via ``sbatch slurm/test.slurm``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.pipeline.snr import pool_snr

from tests.conftest import _BROUHAHA_OK, requires_brouhaha

# Guard heavy imports so the module can be *collected* on login nodes.
if _BROUHAHA_OK:
    from src.pipeline.snr import _extract_snr
    from src.utils import set_seeds


# ======================================================================
# pool_snr — pure numpy, no GPU needed
# ======================================================================


class TestPoolSnr:
    """Unit tests for the average-pooling helper."""

    def test_basic_pooling(self):
        """10 frames at 0.1s step pooled into 1s windows → 1 value."""
        raw = np.arange(10, dtype=np.float32)  # [0,1,...,9]
        pooled, step = pool_snr(raw, step_s=0.1, pool_window_s=1.0)

        assert step == 1.0
        assert len(pooled) == 1
        assert pooled.dtype == np.float16
        np.testing.assert_allclose(float(pooled[0]), 4.5, atol=0.01)

    def test_multiple_windows(self):
        """100 frames at 0.016s step pooled into 1s windows → ~2 values."""
        raw = np.ones(100, dtype=np.float32) * 25.0
        pooled, step = pool_snr(raw, step_s=0.016, pool_window_s=1.0)

        frames_per_win = round(1.0 / 0.016)  # 62 or 63
        expected_full = 100 // frames_per_win
        # May have +1 for leftover tail
        assert len(pooled) in (expected_full, expected_full + 1)
        assert step == 1.0
        # All-constant input → every bin is 25.0
        np.testing.assert_allclose(pooled.astype(np.float32), 25.0, atol=0.1)

    def test_leftover_tail(self):
        """Frames that don't fill a complete window go into a final bin."""
        # 5 frames, 2 per window → 2 full windows + 1 leftover
        raw = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        pooled, step = pool_snr(raw, step_s=1.0, pool_window_s=2.0)

        assert step == 2.0
        assert len(pooled) == 3  # [mean(1,2), mean(3,4), mean(5)]
        np.testing.assert_allclose(float(pooled[0]), 1.5, atol=0.01)
        np.testing.assert_allclose(float(pooled[1]), 3.5, atol=0.01)
        np.testing.assert_allclose(float(pooled[2]), 5.0, atol=0.01)

    def test_shorter_than_one_window(self):
        """File shorter than pool_window → single bin with the overall mean."""
        raw = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        pooled, step = pool_snr(raw, step_s=0.016, pool_window_s=10.0)

        assert len(pooled) == 1
        np.testing.assert_allclose(float(pooled[0]), 20.0, atol=0.1)

    def test_output_dtype_is_float16(self):
        """Pooled output must be float16 for compact storage."""
        raw = np.random.randn(500).astype(np.float32) * 10
        pooled, _ = pool_snr(raw, step_s=0.016, pool_window_s=1.0)
        assert pooled.dtype == np.float16

    def test_different_window_sizes(self):
        """Varying the window size changes the output length proportionally."""
        raw = np.random.randn(1000).astype(np.float32)
        step_s = 0.016

        p1, _ = pool_snr(raw, step_s, pool_window_s=0.5)
        p2, _ = pool_snr(raw, step_s, pool_window_s=1.0)
        p4, _ = pool_snr(raw, step_s, pool_window_s=2.0)

        # Shorter window → more bins
        assert len(p1) > len(p2) > len(p4)

    def test_single_frame_input(self):
        """Edge case: single raw frame."""
        raw = np.array([42.0], dtype=np.float32)
        pooled, step = pool_snr(raw, step_s=0.016, pool_window_s=1.0)

        assert len(pooled) == 1
        np.testing.assert_allclose(float(pooled[0]), 42.0, atol=0.1)


# ======================================================================
# Brouhaha model tests — need GPU / compute node
# ======================================================================

_BROUHAHA_MODEL = Path("models/best/checkpoints/best.ckpt")


def _brouhaha_model_available() -> bool:
    """Return True if the Brouhaha checkpoint exists on disk."""
    return _BROUHAHA_MODEL.exists()


requires_brouhaha_model = pytest.mark.skipif(
    not (_BROUHAHA_OK and _brouhaha_model_available()),
    reason="Brouhaha model checkpoint not found — run via slurm/test.slurm on a GPU node",
)


@requires_brouhaha_model
class TestExtractSnr:
    """Integration tests: run Brouhaha on real audio fixtures."""

    @pytest.fixture(autouse=True)
    def _load_pipeline(self):
        """Load the Brouhaha pipeline once per test class."""
        import torch
        from src.compat import patch_torchaudio
        patch_torchaudio()
        from pyannote.audio import Model
        from brouhaha.pipeline import RegressiveActivityDetectionPipeline

        set_seeds(42)
        model = Model.from_pretrained(_BROUHAHA_MODEL, strict=False)
        if model.device.type == "cpu" and torch.cuda.is_available():
            model.to(torch.device("cuda"))
        self.pipeline = RegressiveActivityDetectionPipeline(segmentation=model)

    def test_returns_float32_array_and_positive_step(self, speech_clean_wav: Path):
        """_extract_snr returns (float32 array, positive float step)."""
        raw_snr, step_s = _extract_snr(self.pipeline, speech_clean_wav)

        assert isinstance(raw_snr, np.ndarray)
        assert raw_snr.dtype == np.float32
        assert raw_snr.ndim == 1
        assert len(raw_snr) > 0
        assert isinstance(step_s, float)
        assert step_s > 0

    def test_snr_values_in_plausible_range(self, speech_clean_wav: Path):
        """SNR values should be finite and within a plausible dB range."""
        raw_snr, _ = _extract_snr(self.pipeline, speech_clean_wav)

        assert np.all(np.isfinite(raw_snr)), "SNR contains NaN or Inf"
        # Brouhaha SNR range is roughly -15 to 80 dB (from model constants)
        # Allow some margin for unusual clips
        assert raw_snr.min() >= -50, f"SNR min={raw_snr.min():.1f} dB too low"
        assert raw_snr.max() <= 120, f"SNR max={raw_snr.max():.1f} dB too high"

    def test_output_length_scales_with_duration(
        self, speech_clean_wav: Path, speech_multi_wav: Path
    ):
        """Longer audio → more frames (proportional to duration)."""
        import soundfile as sf

        snr_a, step_a = _extract_snr(self.pipeline, speech_clean_wav)
        snr_b, step_b = _extract_snr(self.pipeline, speech_multi_wav)

        dur_a = sf.info(str(speech_clean_wav)).duration
        dur_b = sf.info(str(speech_multi_wav)).duration

        # Steps should be the same model regardless of file
        np.testing.assert_allclose(step_a, step_b, rtol=0.01)

        # Frame counts should be roughly proportional to duration
        ratio_frames = len(snr_b) / max(len(snr_a), 1)
        ratio_dur = dur_b / max(dur_a, 0.01)
        np.testing.assert_allclose(ratio_frames, ratio_dur, rtol=0.15)

    def test_silence_has_low_snr(self, silence_wav: Path):
        """Pure silence should have very low SNR (near noise floor)."""
        raw_snr, _ = _extract_snr(self.pipeline, silence_wav)

        # Silence → low SNR is expected; at minimum the mean should be
        # well below clean speech (typically > 15 dB)
        assert raw_snr.mean() < 30, (
            f"Expected low SNR for silence, got mean={raw_snr.mean():.1f} dB"
        )

    def test_short_file_succeeds(self, short_wav: Path):
        """Edge case: very short file (0.3s) should not crash."""
        raw_snr, step_s = _extract_snr(self.pipeline, short_wav)
        assert len(raw_snr) > 0
        assert step_s > 0

    def test_all_fixtures_produce_output(self, all_fixture_wavs: list[Path]):
        """Every fixture WAV must produce a non-empty SNR array."""
        for wav in all_fixture_wavs:
            raw_snr, step_s = _extract_snr(self.pipeline, wav)
            assert len(raw_snr) > 0, f"{wav.name}: empty SNR"
            assert step_s > 0, f"{wav.name}: step_s={step_s}"


# ======================================================================
# End-to-end: pool + save round-trip
# ======================================================================


@requires_brouhaha_model
class TestSnrRoundTrip:
    """Full extract → pool → save → load round-trip."""

    @pytest.fixture(autouse=True)
    def _load_pipeline(self):
        import torch
        from src.compat import patch_torchaudio
        patch_torchaudio()
        from pyannote.audio import Model
        from brouhaha.pipeline import RegressiveActivityDetectionPipeline

        set_seeds(42)
        model = Model.from_pretrained(_BROUHAHA_MODEL, strict=False)
        if model.device.type == "cpu" and torch.cuda.is_available():
            model.to(torch.device("cuda"))
        self.pipeline = RegressiveActivityDetectionPipeline(segmentation=model)

    def test_npz_round_trip(self, speech_clean_wav: Path, tmp_path: Path):
        """Extract → pool → savez_compressed → load preserves data."""
        raw_snr, step_s = _extract_snr(self.pipeline, speech_clean_wav)
        pooled, pool_step = pool_snr(raw_snr, step_s, pool_window_s=1.0)

        out = tmp_path / "test.npz"
        np.savez_compressed(
            out,
            snr=pooled,
            pool_step_s=np.float32(pool_step),
            model_step_s=np.float32(step_s),
            n_raw_frames=np.int32(len(raw_snr)),
        )

        loaded = np.load(out)
        np.testing.assert_array_equal(loaded["snr"], pooled)
        assert float(loaded["pool_step_s"]) == pool_step
        assert float(loaded["model_step_s"]) == pytest.approx(step_s)
        assert int(loaded["n_raw_frames"]) == len(raw_snr)


# ======================================================================
# Reproducibility
# ======================================================================


@requires_brouhaha_model
class TestSnrReproducibility:
    """Two Brouhaha runs on the same audio must produce identical SNR arrays."""

    @pytest.fixture(autouse=True)
    def _load_pipeline(self):
        import torch
        from src.compat import patch_torchaudio
        patch_torchaudio()
        from pyannote.audio import Model
        from brouhaha.pipeline import RegressiveActivityDetectionPipeline

        set_seeds(42)
        model = Model.from_pretrained(_BROUHAHA_MODEL, strict=False)
        if model.device.type == "cpu" and torch.cuda.is_available():
            model.to(torch.device("cuda"))
        self.pipeline = RegressiveActivityDetectionPipeline(segmentation=model)

    def test_extract_deterministic(self, speech_clean_wav: Path):
        """Two _extract_snr calls produce bit-identical raw SNR arrays."""
        set_seeds(42)
        snr_a, step_a = _extract_snr(self.pipeline, speech_clean_wav)

        set_seeds(42)
        snr_b, step_b = _extract_snr(self.pipeline, speech_clean_wav)

        assert step_a == step_b, f"Step mismatch: {step_a} vs {step_b}"
        np.testing.assert_array_equal(
            snr_a, snr_b, err_msg="SNR arrays differ between runs"
        )

    def test_pool_deterministic(self, speech_clean_wav: Path):
        """Full extract + pool pipeline is deterministic end-to-end."""
        set_seeds(42)
        raw_a, step_a = _extract_snr(self.pipeline, speech_clean_wav)
        pooled_a, ps_a = pool_snr(raw_a, step_a, 1.0)

        set_seeds(42)
        raw_b, step_b = _extract_snr(self.pipeline, speech_clean_wav)
        pooled_b, ps_b = pool_snr(raw_b, step_b, 1.0)

        np.testing.assert_array_equal(
            pooled_a, pooled_b, err_msg="Pooled SNR differs between runs"
        )
        assert ps_a == ps_b

    def test_all_fixtures_deterministic(self, all_fixture_wavs: list[Path]):
        """Every fixture file is individually deterministic."""
        for wav in all_fixture_wavs:
            set_seeds(42)
            snr_a, step_a = _extract_snr(self.pipeline, wav)

            set_seeds(42)
            snr_b, step_b = _extract_snr(self.pipeline, wav)

            assert step_a == step_b, f"{wav.name}: step {step_a} vs {step_b}"
            np.testing.assert_array_equal(
                snr_a, snr_b,
                err_msg=f"{wav.name}: SNR arrays differ",
            )
