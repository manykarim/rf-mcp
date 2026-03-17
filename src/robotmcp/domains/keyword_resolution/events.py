"""Keyword Resolution Domain Events."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class BddPrefixStripped:
    """Emitted when a BDD prefix is stripped from a keyword name."""
    original_name: str
    stripped_name: str
    prefix: str
    source_tool: str
    session_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": "bdd_prefix_stripped",
            "original_name": self.original_name,
            "stripped_name": self.stripped_name,
            "prefix": self.prefix,
            "source_tool": self.source_tool,
        }


@dataclass(frozen=True)
class EmbeddedArgMatched:
    """Emitted when an embedded argument keyword is matched."""
    call_name: str
    template_name: str
    extracted_args: Tuple[str, ...]
    library: str
    source_tool: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": "embedded_arg_matched",
            "call_name": self.call_name,
            "template_name": self.template_name,
            "extracted_args": list(self.extracted_args),
            "library": self.library,
            "source_tool": self.source_tool,
        }
