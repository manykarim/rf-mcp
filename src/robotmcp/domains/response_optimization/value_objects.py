"""Response Optimization Value Objects."""
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Dict, FrozenSet, Tuple


class Verbosity(Enum):
    """Response verbosity level."""
    VERBOSE = "verbose"    # No compression at all
    STANDARD = "standard"  # Remove empty fields only
    COMPACT = "compact"    # Full abbreviation + truncation


# Mirror FIELD_ABBREVIATIONS from token_efficient_output.py
# so value objects stay self-contained (no runtime import of utils).
_STANDARD_ABBREVIATIONS: Dict[str, str] = {
    "success": "ok",
    "error": "err",
    "message": "msg",
    "result": "res",
    "status": "st",
    "keyword": "kw",
    "arguments": "args",
    "session_id": "sid",
    "execution_time": "time",
    "step_id": "id",
    "output": "out",
    "description": "desc",
    "library": "lib",
    "libraries": "libs",
    "keywords": "kws",
    "count": "n",
    "total": "tot",
    "metadata": "meta",
    "original_type": "type",
    "attributes": "attrs",
    "documentation": "doc",
    "short_doc": "sdoc",
    "assigned_variables": "vars",
    "session_variables": "svars",
    "resolved_arguments": "rargs",
    "active_library": "alib",
    "browser_state": "bstate",
}


@dataclass(frozen=True)
class FieldAbbreviationMap:
    """Maps verbose field names to compact abbreviations.

    Standard mappings (28 total, from token_efficient_output.py).
    """
    enabled: bool
    mappings: FrozenSet[Tuple[str, str]]

    @classmethod
    def standard(cls) -> FieldAbbreviationMap:
        return cls(enabled=True, mappings=frozenset(_STANDARD_ABBREVIATIONS.items()))

    @classmethod
    def disabled(cls) -> FieldAbbreviationMap:
        return cls(enabled=False, mappings=frozenset())

    def abbreviate(self, field_name: str) -> str:
        """Return abbreviated name if enabled and mapping exists."""
        if not self.enabled:
            return field_name
        mapping_dict = dict(self.mappings)
        return mapping_dict.get(field_name, field_name)


@dataclass(frozen=True)
class TruncationPolicy:
    """Controls emergency truncation thresholds."""
    max_string: int = 500
    max_list_items: int = 20
    max_dict_items: int = 15

    @classmethod
    def default(cls) -> TruncationPolicy:
        return cls(max_string=2000, max_list_items=50, max_dict_items=30)

    @classmethod
    def aggressive(cls) -> TruncationPolicy:
        return cls(max_string=300, max_list_items=10, max_dict_items=10)


class SnapshotCompressionMode(Enum):
    """Snapshot compression strategy."""
    NONE = "none"
    FOLDED = "folded"                    # SimHash list folding only
    INCREMENTAL_DIFF = "incremental"     # Diff against previous snapshot
    FOLDED_DIFF = "folded_diff"          # Both: fold first, then diff


@dataclass(frozen=True)
class TokenEstimate:
    """Token count estimate for budget checking."""
    char_count: int
    estimated_tokens: int
    budget: int
    CHARS_PER_TOKEN: ClassVar[float] = 4.0

    @classmethod
    def from_dict(cls, obj: dict, budget: int = 4000) -> TokenEstimate:
        content = json.dumps(obj, default=str)
        char_count = len(content)
        return cls(
            char_count=char_count,
            estimated_tokens=int(char_count / cls.CHARS_PER_TOKEN),
            budget=budget,
        )

    @property
    def within_budget(self) -> bool:
        return self.estimated_tokens <= self.budget

    @property
    def overage(self) -> int:
        return max(0, self.estimated_tokens - self.budget)
