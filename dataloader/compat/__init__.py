"""Compatibility shim for upstream integration.

This sub-package provides translation layers between the ``dataloader``
package types and upstream framework types (metasr-internal, fs2, etc.).

When upstream signatures become available, implement concrete adapters
here. The core ``dataloader`` package remains framework-agnostic.
"""
