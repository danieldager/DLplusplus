"""Compatibility shims for dependency version mismatches.

Import this module *before* pyannote.audio to patch known issues.

Currently patches:
  - torchaudio ≥2.9 removed ``torchaudio.AudioMetaData``, but
    pyannote-audio 3.4.0 still references it in type annotations.
    We inject a lightweight placeholder so the import doesn't crash.
"""

from __future__ import annotations


def patch_torchaudio() -> None:
    """Add a stub ``torchaudio.AudioMetaData`` if it's missing.

    Safe to call multiple times (idempotent).
    """
    try:
        import torchaudio
    except ImportError:
        return  # torchaudio not installed — nothing to patch

    if not hasattr(torchaudio, "AudioMetaData"):
        from dataclasses import dataclass

        @dataclass
        class AudioMetaData:
            """Minimal stub replacing the removed torchaudio.AudioMetaData."""

            sample_rate: int = 0
            num_frames: int = 0
            num_channels: int = 0
            bits_per_sample: int = 0
            encoding: str = ""

        torchaudio.AudioMetaData = AudioMetaData  # type: ignore[attr-defined]

    if not hasattr(torchaudio, "info"):
        # torchaudio >=2.9 also removed torchaudio.info
        # pyannote.audio.core.io uses torchaudio.info(file, backend=...)
        def _info_stub(filepath, backend=None, format=None):
            """Stub using soundfile as fallback for torchaudio.info."""
            import soundfile as sf

            sf_info = sf.info(str(filepath))
            return torchaudio.AudioMetaData(  # type: ignore[attr-defined]
                sample_rate=sf_info.samplerate,
                num_frames=sf_info.frames,
                num_channels=sf_info.channels,
                bits_per_sample=16,
                encoding=sf_info.subtype or "",
            )

        torchaudio.info = _info_stub  # type: ignore[attr-defined]

    if not hasattr(torchaudio, "list_audio_backends"):

        def _list_backends():
            return ["soundfile"]

        torchaudio.list_audio_backends = _list_backends  # type: ignore[attr-defined]
