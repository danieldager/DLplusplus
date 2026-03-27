"""Shared audio utilities.

Extracted from ``src.core.vad_processing`` to avoid duplication in
``src.packaging.writer``.
"""

from __future__ import annotations

import numpy as np


def resample_block(data: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    """Resample a block of samples using linear interpolation.

    Only called when the file sample rate differs from the target rate.
    """
    if from_sr == to_sr:
        return data
    n_out = int(len(data) * to_sr / from_sr)
    indices = np.linspace(0, len(data) - 1, n_out)
    lo = np.floor(indices).astype(np.int64)
    hi = np.minimum(lo + 1, len(data) - 1)
    frac = (indices - lo).astype(np.float32)
    return (data[lo] * (1 - frac) + data[hi] * frac).astype(data.dtype)
