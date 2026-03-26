"""Dual-mode waveform processor: online transforms or offline save-and-load.

:class:`WaveformProcessor` wraps a waveform-modifying function and supports
two execution modes:

- **Online**: Applied at dataload time as a standard :class:`DataProcessor`
  in the transform pipeline. No disk I/O.
- **Offline**: Runs the transform once, saves the modified waveform (plus
  provenance metadata) to disk. Subsequent loads skip the transform and
  read the pre-processed file directly.

This lets users experiment with on-the-fly transforms for fast iteration,
then bake them in for production training.
"""

from __future__ import annotations

import json
import logging
from abc import abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf
import torch

from dataloader.transform.base import DataProcessor
from dataloader.types import MetadataDict, SampleRate, WavID, Waveform

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class WaveformProvenance:
    """Records what was done to a waveform and by which processor."""

    processor_name: str
    processor_version: str
    params: dict[str, object]
    source_wav_id: str
    created_utc: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class WaveformProcessor(DataProcessor):
    """Base class for transforms that modify waveform audio.

    Subclasses implement :meth:`transform` — the pure waveform operation.
    The base class adds dual-mode orchestration (online vs. offline) and
    provenance tracking.

    Parameters
    ----------
    output_dir:
        If set, enables **offline mode**: processed waveforms are saved
        here, and future calls load from cache instead of re-processing.
        If ``None``, runs in **online mode** (transform applied in memory
        every time).
    output_sr:
        Sample rate for saved files (offline mode). Defaults to the input
        sample rate.
    """

    name: str = "waveform_processor"
    version: str = "1.0.0"

    def __init__(
        self,
        output_dir: Path | str | None = None,
        output_sr: SampleRate | None = None,
    ) -> None:
        self._output_dir = Path(output_dir) if output_dir is not None else None
        self._output_sr = output_sr

    # ── Subclass interface ────────────────────────────────────────────────

    @abstractmethod
    def transform(
        self,
        waveform: Waveform,
        sample_rate: SampleRate,
        metadata: MetadataDict,
    ) -> tuple[Waveform, SampleRate, MetadataDict]:
        """Apply the waveform transformation.

        This is the pure computation — no I/O, no caching.
        Subclasses must override this method.
        """

    def _get_params(self) -> dict[str, object]:
        """Return a dict of constructor parameters for provenance records.

        Override in subclasses to capture relevant config.
        """
        return {}

    # ── Offline I/O ───────────────────────────────────────────────────────

    def _output_path(self, wav_id: WavID) -> Path:
        assert self._output_dir is not None
        return self._output_dir / f"{wav_id}.wav"

    def _provenance_path(self, wav_id: WavID) -> Path:
        assert self._output_dir is not None
        return self._output_dir / f"{wav_id}.provenance.json"

    def exists(self, wav_id: WavID) -> bool:
        """Check whether a cached processed waveform exists for *wav_id*."""
        if self._output_dir is None:
            return False
        return self._output_path(wav_id).is_file()

    def _save(
        self,
        wav_id: WavID,
        waveform: Waveform,
        sample_rate: SampleRate,
    ) -> Path:
        """Save processed waveform and provenance to disk."""
        assert self._output_dir is not None
        self._output_dir.mkdir(parents=True, exist_ok=True)

        out_path = self._output_path(wav_id)
        # Write as WAV — lossless, widely compatible.
        audio_np = waveform.numpy()
        if audio_np.ndim == 2:
            audio_np = audio_np.T  # (channels, samples) → (samples, channels)
        sf.write(out_path, audio_np, sample_rate)

        provenance = WaveformProvenance(
            processor_name=self.name,
            processor_version=self.version,
            params=self._get_params(),
            source_wav_id=wav_id,
            created_utc=datetime.now(timezone.utc).isoformat(),
        )
        with open(self._provenance_path(wav_id), "w") as f:
            json.dump(provenance.to_dict(), f, indent=2, default=str)

        log.debug("Saved processed waveform to %s", out_path)
        return out_path

    def _load_cached(
        self, wav_id: WavID
    ) -> tuple[Waveform, SampleRate]:
        """Load a previously saved processed waveform."""
        path = self._output_path(wav_id)
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data.T)  # (samples, channels) → (channels, samples)
        return waveform, int(sr)

    # ── DataProcessor interface ───────────────────────────────────────────

    def __call__(
        self,
        waveform: Waveform,
        sample_rate: SampleRate,
        metadata: MetadataDict,
    ) -> tuple[Waveform, SampleRate, MetadataDict]:
        wav_id = str(metadata.get("wav_id", ""))

        # Offline mode: check cache first.
        if self._output_dir is not None and wav_id and self.exists(wav_id):
            waveform, sample_rate = self._load_cached(wav_id)
            metadata = {**metadata, f"{self.name}_cached": True}
            return waveform, sample_rate, metadata

        # Apply the transform.
        waveform, sample_rate, metadata = self.transform(
            waveform, sample_rate, metadata,
        )

        # Offline mode: save to disk for next time.
        if self._output_dir is not None and wav_id:
            self._save(wav_id, waveform, sample_rate)
            metadata = {**metadata, f"{self.name}_cached": False}

        return waveform, sample_rate, metadata

    def __repr__(self) -> str:
        mode = "offline" if self._output_dir else "online"
        return f"{type(self).__name__}(name={self.name!r}, mode={mode!r})"


# ── Concrete implementations ─────────────────────────────────────────────────


class Denoiser(WaveformProcessor):
    """Denoise waveform using spectral gating.

    A simple example of a :class:`WaveformProcessor` that can run
    online (every dataload) or offline (save once, load from cache).

    Parameters
    ----------
    n_fft:
        FFT size for spectral gating.
    noise_reduce_factor:
        Factor by which to reduce noise magnitude.
    output_dir:
        If set, cache denoised waveforms to disk.
    """

    name = "denoiser"
    version = "1.0.0"

    def __init__(
        self,
        n_fft: int = 1024,
        noise_reduce_factor: float = 0.5,
        output_dir: Path | str | None = None,
    ) -> None:
        super().__init__(output_dir=output_dir)
        self._n_fft = n_fft
        self._noise_reduce_factor = noise_reduce_factor

    def _get_params(self) -> dict[str, object]:
        return {
            "n_fft": self._n_fft,
            "noise_reduce_factor": self._noise_reduce_factor,
        }

    def transform(
        self,
        waveform: Waveform,
        sample_rate: SampleRate,
        metadata: MetadataDict,
    ) -> tuple[Waveform, SampleRate, MetadataDict]:
        # Spectral gating: suppress frequencies below noise floor.
        for ch in range(waveform.shape[0]):
            stft = torch.stft(
                waveform[ch],
                n_fft=self._n_fft,
                return_complex=True,
            )
            magnitude = stft.abs()
            noise_floor = magnitude.mean(dim=-1, keepdim=True)
            mask = (magnitude > noise_floor * self._noise_reduce_factor).float()
            stft_clean = stft * mask
            waveform[ch] = torch.istft(
                stft_clean,
                n_fft=self._n_fft,
                length=waveform.shape[-1],
            )
        return waveform, sample_rate, metadata
