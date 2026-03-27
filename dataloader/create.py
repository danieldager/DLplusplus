"""End-to-end DataLoader creation from a single config.

The main entry point is :func:`create_dataloader`, which turns a
:class:`~dataloader.config.DatasetConfig` (or a path to one) into a
ready-to-train ``torch.utils.data.DataLoader`` backed by WebDataset shards.

Example
-------
::

    from dataloader import create_dataloader, DatasetConfig

    # From a config object
    config = DatasetConfig(
        dataset_dir="output/seedlings_1",
    )
    loader = create_dataloader(config)

    for batch in loader:
        print(batch.waveforms.shape, batch.batch_size)
        break

    # Or from a saved JSON file
    loader = create_dataloader("configs/seedlings.json")
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch.utils.data

from dataloader.batch.speech import SpeechCollator
from dataloader.build import build_manifest
from dataloader.config import DatasetConfig
from dataloader.dataset.webdataset import WebDatasetSpeechDataset

log = logging.getLogger(__name__)


def create_dataloader(
    config: DatasetConfig | str | Path,
) -> torch.utils.data.DataLoader:
    """Build a ready-to-train PyTorch DataLoader from a dataset config.

    This is the top-level convenience function for the Dataloader++ workflow:

    1. Load config (if given a path).
    2. Discover ``.tar`` shards in ``{dataset_dir}/shards/``.
    3. If any filters are active, build a manifest and compute the set
       of allowed source-file IDs.
    4. Instantiate :class:`WebDatasetSpeechDataset` with shard URLs.
    5. Wrap in ``torch.utils.data.DataLoader`` with
       :class:`SpeechCollator`.

    Parameters
    ----------
    config:
        A :class:`DatasetConfig` instance, or a path to a JSON config file
        (or directory containing ``dataset_config.json``).

    Returns
    -------
    torch.utils.data.DataLoader
        A streaming DataLoader yielding :class:`~dataloader.batch.data_batch.DataBatch`
        objects, ready for model consumption.

    Raises
    ------
    FileNotFoundError
        If no ``.tar`` shards are found, or the config file doesn't exist.
    """
    if isinstance(config, (str, Path)):
        config = DatasetConfig.load(config)

    dataset_dir = Path(config.dataset_dir)
    shard_dir = dataset_dir / "shards"

    # ── Discover shards ───────────────────────────────────────────────────
    shard_urls = sorted(str(p) for p in shard_dir.glob("*.tar"))
    if not shard_urls:
        raise FileNotFoundError(
            f"No .tar shards found in {shard_dir}. "
            f"Has the packaging pipeline been run?"
        )
    log.info("Found %d shard(s) in %s", len(shard_urls), shard_dir)

    # ── Build filter set (if any filters are active) ──────────────────────
    allowed_ids: set[str] | None = None
    if config.filters.is_active:
        manifest = build_manifest(dataset_dir, filters=config.filters)
        allowed_ids = set(manifest.df["wav_id"].to_list())
        log.info(
            "Manifest filtering active: %d source files pass filters",
            len(allowed_ids),
        )

    # ── Create dataset ────────────────────────────────────────────────────
    ldr = config.loader
    dataset = WebDatasetSpeechDataset(
        shard_urls=shard_urls,
        audio_key=ldr.audio_key,
        metadata_keys=ldr.metadata_keys,
        target_sr=config.pipeline.target_sr,
        shuffle_buffer=ldr.shuffle_buffer,
        seed=ldr.seed,
        allowed_ids=allowed_ids,
    )

    # ── Create collator ──────────────────────────────────────────────────
    collator = SpeechCollator(
        pad_to_multiple_of=ldr.pad_to_multiple_of,
    )

    # ── Build DataLoader ─────────────────────────────────────────────────
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=ldr.batch_size,
        num_workers=ldr.num_workers,
        prefetch_factor=ldr.prefetch_factor if ldr.num_workers > 0 else None,
        collate_fn=collator,
        pin_memory=ldr.pin_memory,
        drop_last=ldr.drop_last,
    )
    log.info(
        "Created DataLoader: batch_size=%d, num_workers=%d, shards=%d",
        ldr.batch_size,
        ldr.num_workers,
        len(shard_urls),
    )
    return loader
