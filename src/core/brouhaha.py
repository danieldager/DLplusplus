"""Shared Brouhaha model loading and extraction helpers.

Used by ``src.pipeline.snr`` and ``src.pipeline.segment_snr`` to avoid
duplicating the noisy model-init and inference code.
"""

from __future__ import annotations

import io
import logging
import sys
import warnings
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Default Brouhaha checkpoint (shipped with the repo)
DEFAULT_MODEL = "models/best/checkpoints/best.ckpt"


def ensure_model(model_path: str) -> str:
    """Return *model_path* after ensuring the checkpoint exists on disk.

    If the default checkpoint is missing, it is automatically downloaded
    from HuggingFace (``ylacombe/brouhaha-best``).  For non-default paths the
    caller is responsible for providing a valid file.
    """
    p = Path(model_path)
    if p.exists():
        return model_path

    if model_path != DEFAULT_MODEL:
        raise FileNotFoundError(f"Brouhaha checkpoint not found: {model_path}")

    logger.info("Brouhaha checkpoint not found — downloading from Hugging Face …")
    from scripts.download_brouhaha import ensure_brouhaha_checkpoint

    ensure_brouhaha_checkpoint(p)
    return model_path


def load_brouhaha_pipeline(model_path: str, device: str = "cuda"):
    """Load Brouhaha model and return the pipeline object.

    Handles warning suppression, checkpoint auto-download,
    stdout capture during noisy pyannote/brouhaha init,
    and device placement.

    Call :func:`ensure_model` before this to auto-download if needed.
    """
    import torch
    from pyannote.audio import Model
    from brouhaha.pipeline import RegressiveActivityDetectionPipeline

    for _mod in (
        "speechbrain",
        "pytorch_lightning",
        "lightning",
        "lightning.fabric",
        "lightning_fabric",
    ):
        logging.getLogger(_mod).setLevel(logging.WARNING)

    logger.info(f"Model: {model_path}")

    # Redirect stdout to swallow bare print() from pyannote + brouhaha
    _real_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = Model.from_pretrained(Path(model_path), strict=False)
    finally:
        sys.stdout = _real_stdout

    if model.device.type == "cpu" and device != "cpu":
        dev = torch.device(device if torch.cuda.is_available() else "cpu")
        model.to(dev)
        logger.info(f"Device: {dev}")
    else:
        logger.info(f"Device: {model.device}")

    # Pipeline init triggers "Using default parameters" print — suppress it
    sys.stdout = io.StringIO()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline = RegressiveActivityDetectionPipeline(segmentation=model)
    finally:
        sys.stdout = _real_stdout

    # Suppress recurring runtime warnings from pyannote / torchaudio
    warnings.filterwarnings("ignore", module=r"pyannote\.audio")
    warnings.filterwarnings("ignore", message=".*backend.*parameter.*not used.*")

    return pipeline


def extract_brouhaha(
    pipeline,
    audio_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Run Brouhaha on one file and return (raw_vad, raw_snr, raw_c50, step_s).

    Returns
    -------
    raw_vad : 1-D float32 array of per-frame VAD probability [0, 1].
    raw_snr : 1-D float32 array of per-frame SNR values (dB).
    raw_c50 : 1-D float32 array of per-frame C50 clarity values (dB).
    step_s  : Frame step in seconds (from the model's receptive field).
    """
    file = {"uri": audio_path.stem, "audio": str(audio_path)}

    # Run the underlying segmentation model directly to get all 3 outputs
    # (vad_prob, snr, c50) instead of going through the pipeline's apply()
    # which only returns snr and c50.
    seg = pipeline._segmentation
    segmentations = seg(file)
    data = segmentations.data  # (n_frames, 3): [vad_prob, snr, c50]
    vad: np.ndarray = data[:, 0]
    snr: np.ndarray = data[:, 1]
    c50: np.ndarray = data[:, 2]

    # Resolve the frame step from the underlying model
    if hasattr(seg.model, "receptive_field"):
        step_s = float(seg.model.receptive_field.step)  # type: ignore[union-attr]
    elif hasattr(seg.model, "introspection"):
        step_s = float(seg.model.introspection.frames.step)  # type: ignore[union-attr]
    elif hasattr(seg, "_frames"):
        step_s = float(seg._frames.step)  # type: ignore[union-attr]
    else:
        # Fallback: estimate from output length and audio duration
        import soundfile as sf

        info = sf.info(str(audio_path))
        step_s = info.duration / max(len(snr), 1)

    return (
        vad.astype(np.float32),
        snr.astype(np.float32),
        c50.astype(np.float32),
        step_s,
    )
