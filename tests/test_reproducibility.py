"""Tests for pipeline reproducibility — identical inputs must produce identical outputs.

Running the VAD or VTC pipeline twice on the same audio must yield
bit-identical segment boundaries and labels.  These tests guarantee that
no hidden randomness (unseeded RNG, non-deterministic CUDA ops, etc.)
can slip through.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import (
    _TORCHCODEC_OK,
    _TENVAD_OK,
    requires_tenvad,
    requires_torchcodec,
)

# Guard torchcodec-dependent imports so the module can still be *collected*
# on login nodes where FFmpeg/torchcodec is unavailable.
if _TORCHCODEC_OK:
    import torch
    from segma.inference import apply_model_on_audio

if _TENVAD_OK:
    from src.core.vad_processing import process_vad_file, set_seeds


# -----------------------------------------------------------------------
# VAD reproducibility
# -----------------------------------------------------------------------


@requires_tenvad
class TestVadReproducibility:
    """process_vad_file must return identical results across two runs."""

    def test_single_file_deterministic(self, speech_clean_wav: Path):
        """Two calls on the same file produce identical metadata and segments."""
        wav = speech_clean_wav
        args = (wav, 256, 0.5)

        set_seeds(42)
        meta_a, segs_a = process_vad_file(args)

        set_seeds(42)
        meta_b, segs_b = process_vad_file(args)

        assert meta_a["success"] is True and meta_b["success"] is True
        # Metadata fields that must match exactly
        for key in (
            "duration",
            "speech_ratio",
            "n_speech_segments",
            "n_silence_segments",
        ):
            assert (
                meta_a[key] == meta_b[key]
            ), f"Metadata mismatch on '{key}': {meta_a[key]} vs {meta_b[key]}"

        # Segment-level exact match
        assert len(segs_a) == len(
            segs_b
        ), f"Segment count mismatch: {len(segs_a)} vs {len(segs_b)}"
        for i, (sa, sb) in enumerate(zip(segs_a, segs_b)):
            assert sa == sb, f"Segment {i} differs: {sa} vs {sb}"

    def test_all_fixtures_deterministic(self, speech_fixture_wavs: list[Path]):
        """All speech fixture files are individually deterministic."""
        for wav in speech_fixture_wavs:
            args = (wav, 256, 0.5)

            set_seeds(42)
            meta_a, segs_a = process_vad_file(args)

            set_seeds(42)
            meta_b, segs_b = process_vad_file(args)

            assert meta_a["success"] is True
            assert len(segs_a) == len(
                segs_b
            ), f"{wav.stem}: segment count {len(segs_a)} vs {len(segs_b)}"
            for i, (sa, sb) in enumerate(zip(segs_a, segs_b)):
                assert sa == sb, f"{wav.stem} segment {i}: {sa} vs {sb}"

    def test_silence_file_deterministic(self, silence_wav: Path):
        """Silence file is deterministic (edge case: zero or few segments)."""
        args = (silence_wav, 256, 0.5)

        set_seeds(42)
        meta_a, segs_a = process_vad_file(args)

        set_seeds(42)
        meta_b, segs_b = process_vad_file(args)

        assert meta_a["success"] is True
        assert meta_a["speech_ratio"] == meta_b["speech_ratio"]
        assert len(segs_a) == len(segs_b)
        for i, (sa, sb) in enumerate(zip(segs_a, segs_b)):
            assert sa == sb, f"Silence segment {i}: {sa} vs {sb}"

    def test_short_file_deterministic(self, short_wav: Path):
        """Very short file (0.3s) is deterministic."""
        args = (short_wav, 256, 0.5)

        set_seeds(42)
        meta_a, segs_a = process_vad_file(args)

        set_seeds(42)
        meta_b, segs_b = process_vad_file(args)

        assert meta_a["success"] is True
        assert meta_a["speech_ratio"] == meta_b["speech_ratio"]
        assert len(segs_a) == len(segs_b)


# -----------------------------------------------------------------------
# VTC reproducibility
# -----------------------------------------------------------------------


def _vtc_model_available() -> bool:
    """Return True if the segma model + checkpoint can be loaded."""
    try:
        import torch  # noqa: F401
        from segma.config import load_config
        from segma.config.base import Config  # noqa: F401
        from segma.models import Models  # noqa: F401
        from segma.utils.encoders import MultiLabelEncoder  # noqa: F401

        cfg_path = Path("VTC-2.0/model/config.yml")
        ckpt_path = Path("VTC-2.0/model/best.ckpt")
        return cfg_path.exists() and ckpt_path.exists()
    except Exception:
        return False


requires_vtc_model = pytest.mark.skipif(
    not (_TORCHCODEC_OK and _vtc_model_available()),
    reason="VTC model or torchcodec unavailable — run via slurm/test.slurm on a GPU node",
)


@requires_vtc_model
class TestVtcReproducibility:
    """VTC forward pass + thresholding must be deterministic."""

    @pytest.fixture(autouse=True)
    def _load_model(self):
        """Load the segma model once per test class."""
        import torch
        from segma.config import load_config
        from segma.models import Models
        from segma.utils.encoders import MultiLabelEncoder

        from src.core.vad_processing import set_seeds as _set_seeds

        cfg_path = "VTC-2.0/model/config.yml"
        ckpt_path = "VTC-2.0/model/best.ckpt"
        config = load_config(cfg_path)
        l_encoder = MultiLabelEncoder(labels=config.data.classes)
        model = Models[config.model.name].load_from_checkpoint(
            checkpoint_path=ckpt_path,
            label_encoder=l_encoder,
            config=config,
            train=False,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.eval()
        model.to(torch.device(device))

        self.model = model
        self.config = config
        self.l_encoder = l_encoder
        self.conv_settings = model.conv_settings
        self.device = device
        self.chunk_duration_s = config.audio.chunk_duration_s
        self._set_seeds = _set_seeds

    def test_forward_pass_deterministic(self, good_book_wavs: list[Path]):
        """Two forward passes on the same file produce identical logits."""
        import torch

        wav = good_book_wavs[0]

        self._set_seeds(42)
        with torch.no_grad():
            logits_a = apply_model_on_audio(
                audio_path=wav,
                model=self.model,
                conv_settings=self.conv_settings,
                device=self.device,  # type: ignore
                batch_size=128,
                chunk_duration_s=self.chunk_duration_s,
            )

        self._set_seeds(42)
        with torch.no_grad():
            logits_b = apply_model_on_audio(
                audio_path=wav,
                model=self.model,
                conv_settings=self.conv_settings,
                device=self.device,  # type: ignore
                batch_size=128,
                chunk_duration_s=self.chunk_duration_s,
            )

        assert torch.equal(logits_a, logits_b), (
            f"Logit mismatch: max diff = "
            f"{(logits_a - logits_b).abs().max().item():.6e}"
        )

    def test_segments_deterministic(self, good_book_wavs: list[Path]):
        """Full VTC pipeline (forward + threshold) yields identical segments."""
        import torch

        from src.core.intervals import intervals_to_segments
        from src.pipeline.vtc import _apply_threshold

        wav = good_book_wavs[0]

        def _run():
            self._set_seeds(42)
            with torch.no_grad():
                logits_t = apply_model_on_audio(
                    audio_path=wav,
                    model=self.model,
                    conv_settings=self.conv_settings,
                    device=self.device,  # type: ignore
                    batch_size=128,
                    chunk_duration_s=self.chunk_duration_s,
                )
            region_data_cpu = [(0, logits_t.cpu())]
            intervals = _apply_threshold(
                region_data_cpu,
                threshold=0.5,
                conv_settings=self.conv_settings,
                l_encoder=self.l_encoder,
            )
            return intervals_to_segments(intervals, wav.stem)

        segs_a = _run()
        segs_b = _run()

        assert len(segs_a) == len(
            segs_b
        ), f"Segment count: {len(segs_a)} vs {len(segs_b)}"
        for i, (sa, sb) in enumerate(zip(segs_a, segs_b)):
            assert sa == sb, f"VTC segment {i}: {sa} vs {sb}"
