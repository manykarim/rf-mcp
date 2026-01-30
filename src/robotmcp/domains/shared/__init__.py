"""Shared Kernel - Types shared across bounded contexts.

This module contains the minimal set of types that are shared between
the Snapshot Context and Element Registry Context as defined in ADR-001.
"""

from robotmcp.domains.shared.kernel import (
    AriaNode,
    AriaRole,
    ElementRef,
)

__all__ = [
    "AriaNode",
    "AriaRole",
    "ElementRef",
]
