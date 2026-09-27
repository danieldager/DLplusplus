"""Feature Processor abstractions for offline metadata extraction."""

from dataloader.processor.base import FeatureProcessor
from dataloader.processor.registry import ProcessorRegistry, default_registry

__all__ = ["FeatureProcessor", "ProcessorRegistry", "default_registry"]
