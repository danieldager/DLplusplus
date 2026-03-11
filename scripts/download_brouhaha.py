#!/usr/bin/env python3
"""Download the Brouhaha model checkpoint from Hugging Face Hub.

Usage (standalone):
    python scripts/download_brouhaha.py            # downloads to models/best/checkpoints/best.ckpt
    python scripts/download_brouhaha.py --force     # re-download even if present
    python scripts/download_brouhaha.py --verify    # only verify existing checkpoint

Usage (as library):
    from scripts.download_brouhaha import ensure_brouhaha_checkpoint
    ckpt = ensure_brouhaha_checkpoint()             # returns Path to checkpoint

The checkpoint is downloaded from:
    https://huggingface.co/ylacombe/brouhaha-best

Licence: MIT (same as the Brouhaha model)
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────
HF_REPO_ID = "ylacombe/brouhaha-best"
HF_FILENAME = "best.ckpt"

DEFAULT_TARGET = Path("models/best/checkpoints/best.ckpt")

EXPECTED_SHA256 = (
    "9c237e4a7b1de8b456dbee25db853342bf374b19d8732b72b61356519e390ae1"
)
EXPECTED_SIZE_BYTES = 47_224_097  # ~47 MB


# ── Helpers ───────────────────────────────────────────────────────────
def _sha256(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def verify_checkpoint(path: Path) -> bool:
    """Return True if the checkpoint at *path* matches the expected hash."""
    if not path.exists():
        logger.error("Checkpoint not found: %s", path)
        return False

    size = path.stat().st_size
    if size != EXPECTED_SIZE_BYTES:
        logger.error(
            "Size mismatch: got %d bytes, expected %d bytes",
            size,
            EXPECTED_SIZE_BYTES,
        )
        return False

    digest = _sha256(path)
    if digest != EXPECTED_SHA256:
        logger.error(
            "SHA-256 mismatch:\n  got      %s\n  expected %s",
            digest,
            EXPECTED_SHA256,
        )
        return False

    logger.info("Checkpoint verified: %s (SHA-256 OK)", path)
    return True


def ensure_brouhaha_checkpoint(
    target: Path = DEFAULT_TARGET,
    *,
    force: bool = False,
) -> Path:
    """Download the Brouhaha checkpoint if it is not already present.

    Parameters
    ----------
    target : Path
        Local file path for the checkpoint.
    force : bool
        If True, re-download even if the file already exists and is valid.

    Returns
    -------
    Path
        Absolute path to the verified checkpoint.
    """
    target = Path(target)

    if target.exists() and not force:
        if verify_checkpoint(target):
            return target.resolve()
        logger.warning(
            "Existing checkpoint failed verification — re-downloading."
        )

    # -- Download from Hugging Face Hub -----------------------------------
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.error(
            "huggingface_hub is required to download the Brouhaha model.\n"
            "Install it with:  uv pip install huggingface-hub"
        )
        raise

    logger.info(
        "Downloading Brouhaha checkpoint from %s/%s …", HF_REPO_ID, HF_FILENAME
    )

    # hf_hub_download caches internally; we copy to the expected target path.
    cached_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_FILENAME,
    )

    target.parent.mkdir(parents=True, exist_ok=True)

    # Prefer hard-link (same filesystem) to avoid doubling disk usage; fall
    # back to copy if the cache lives on another mount.
    import shutil

    target_tmp = target.with_suffix(".ckpt.tmp")
    try:
        target_tmp.unlink(missing_ok=True)
        target_tmp.hardlink_to(cached_path)
    except OSError:
        shutil.copy2(cached_path, target_tmp)

    target_tmp.rename(target)
    logger.info("Saved checkpoint to %s", target)

    if not verify_checkpoint(target):
        raise RuntimeError(
            f"Downloaded checkpoint failed SHA-256 verification.\n"
            f"  Expected: {EXPECTED_SHA256}\n"
            f"  Got:      {_sha256(target)}\n"
            f"Delete {target} and try again, or download manually from:\n"
            f"  https://huggingface.co/{HF_REPO_ID}"
        )

    return target.resolve()


# ── CLI ───────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download / verify the Brouhaha model checkpoint.",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET,
        help=f"Destination path (default: {DEFAULT_TARGET})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the checkpoint already exists.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify the existing checkpoint (do not download).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    if args.verify:
        ok = verify_checkpoint(args.target)
        sys.exit(0 if ok else 1)

    path = ensure_brouhaha_checkpoint(args.target, force=args.force)
    print(f"✓ Brouhaha checkpoint ready: {path}")


if __name__ == "__main__":
    main()
