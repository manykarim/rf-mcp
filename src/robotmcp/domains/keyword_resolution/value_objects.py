"""Keyword Resolution Domain Value Objects.

Immutable types that carry no identity. Equality is structural.
All value objects use frozen dataclasses with __post_init__ validation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Dict, Optional, Tuple


class BddPrefixType(str, Enum):
    """Supported BDD prefix types."""
    GIVEN = "Given"
    WHEN = "When"
    THEN = "Then"
    AND = "And"
    BUT = "But"


@dataclass(frozen=True)
class BddPrefix:
    """Result of BDD prefix stripping.

    Attributes:
        original_name: The keyword name as received
        stripped_name: The keyword name without the prefix
        prefix_type: The BDD prefix that was stripped, or None
    """
    original_name: str
    stripped_name: str
    prefix_type: Optional[BddPrefixType] = None

    _BDD_PREFIX_RE: ClassVar[re.Pattern] = re.compile(
        r"^(given|when|then|and|but)\s+", re.IGNORECASE
    )

    def __post_init__(self) -> None:
        if not self.original_name:
            raise ValueError("original_name must not be empty")

    @property
    def has_prefix(self) -> bool:
        return self.prefix_type is not None

    @classmethod
    def from_keyword(cls, keyword_name: str) -> BddPrefix:
        """Strip BDD prefix from a keyword name."""
        m = cls._BDD_PREFIX_RE.match(keyword_name)
        if m:
            prefix_str = m.group(1).capitalize()
            return cls(
                original_name=keyword_name,
                stripped_name=keyword_name[m.end():],
                prefix_type=BddPrefixType(prefix_str),
            )
        return cls(
            original_name=keyword_name,
            stripped_name=keyword_name,
            prefix_type=None,
        )


@dataclass(frozen=True)
class EmbeddedPattern:
    """Parsed embedded argument pattern for a keyword.

    Wraps RF's EmbeddedArguments for use in robotmcp's keyword cache.
    """
    template_name: str
    arg_names: Tuple[str, ...]
    regex_pattern: str

    def __post_init__(self) -> None:
        if "${" not in self.template_name:
            raise ValueError(
                f"template_name must contain embedded args: {self.template_name}"
            )


@dataclass(frozen=True)
class EmbeddedMatch:
    """Result of matching a concrete keyword name against an embedded pattern."""
    template_name: str
    concrete_name: str
    extracted_args: Tuple[str, ...]
    library: str


class DataFormat(str, Enum):
    """Supported data file formats for DataDriver integration."""
    CSV = "csv"
    JSON = "json"
    XLSX = "xlsx"
    XLS = "xls"


@dataclass(frozen=True)
class DataRow:
    """A single row of test data from an external source."""
    __test__ = False
    test_name: str
    arguments: Dict[str, str]
    tags: Tuple[str, ...] = ()
    documentation: str = ""


@dataclass(frozen=True)
class DataSource:
    """Represents a loaded external data source."""
    file_path: str
    format: DataFormat
    rows: Tuple[DataRow, ...]
    column_names: Tuple[str, ...]

    @property
    def count(self) -> int:
        return len(self.rows)
