"""Data models for Robot Framework library and keyword information."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class KeywordInfo:
    """Information about a Robot Framework keyword."""
    name: str
    library: str
    method_name: str
    doc: str = ""
    short_doc: str = ""
    args: List[str] = field(default_factory=list)
    defaults: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    is_builtin: bool = False


@dataclass
class ParsedArguments:
    """Parsed positional and named arguments."""
    positional: List[str] = field(default_factory=list)
    named: Dict[str, str] = field(default_factory=dict)


@dataclass
class ArgumentInfo:
    """Information about a single argument from LibDoc signature parsing."""
    name: str
    type_hint: str
    default_value: Optional[str] = None
    is_varargs: bool = False
    is_kwargs: bool = False


@dataclass 
class LibraryInfo:
    """Information about a Robot Framework library."""
    name: str
    instance: Any
    keywords: Dict[str, KeywordInfo] = field(default_factory=dict)
    doc: str = ""
    version: str = ""
    scope: str = "SUITE"