"""Read / merge / write helpers for the configuration formats used by the
coding-agent MCP config files rf-mcp registers itself into.

Formats: ``json`` (Claude Code, Copilot, opencode, Gemini, Cursor), ``jsonc``
(Kilo Code), ``toml`` (Codex), ``yaml`` (goose). JSON/TOML/YAML round-trip; TOML
preserves comments/formatting via tomlkit. JSONC comments are not preserved on
write (written back as plain JSON, which is valid JSONC) - documented behaviour.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, List, Tuple

SUPPORTED_FORMATS = ("json", "jsonc", "toml", "yaml")


def _strip_jsonc(text: str) -> str:
    """Remove // line and /* */ block comments, ignoring those inside strings."""
    out: List[str] = []
    i, n = 0, len(text)
    in_str = False
    quote = ""
    while i < n:
        ch = text[i]
        if in_str:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(text[i + 1]); i += 2; continue
            if ch == quote:
                in_str = False
            i += 1; continue
        if ch in "\"'":
            in_str = True; quote = ch; out.append(ch); i += 1; continue
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "*":
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2; continue
        out.append(ch); i += 1
    # trailing commas are legal in JSONC but not JSON - drop them
    return re.sub(r",(\s*[}\]])", r"\1", "".join(out))


def load(path: Path, fmt: str) -> Tuple[Any, bool]:
    """Return (data, existed). Missing/empty file -> (empty container, False)."""
    if not path.exists() or path.stat().st_size == 0:
        return ({}, False)
    text = path.read_text(encoding="utf-8")
    if fmt == "json":
        return (json.loads(text), True)
    if fmt == "jsonc":
        try:
            return (json.loads(text), True)
        except json.JSONDecodeError:
            return (json.loads(_strip_jsonc(text)), True)
    if fmt == "toml":
        import tomlkit
        return (tomlkit.parse(text), True)
    if fmt == "yaml":
        import yaml
        return (yaml.safe_load(text) or {}, True)
    raise ValueError(f"unsupported format: {fmt}")


def dump(path: Path, fmt: str, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt in ("json", "jsonc"):
        text = json.dumps(data, indent=2) + "\n"
    elif fmt == "toml":
        import tomlkit
        text = tomlkit.dumps(data)
    elif fmt == "yaml":
        import yaml
        text = yaml.safe_dump(data, sort_keys=False, default_flow_style=False)
    else:
        raise ValueError(f"unsupported format: {fmt}")
    path.write_text(text, encoding="utf-8")


def ensure_container(data: Any, keys: List[str]) -> Any:
    """Walk/create nested dicts along ``keys`` and return the innermost dict."""
    node = data
    for k in keys:
        if k not in node or not isinstance(node.get(k), dict):
            node[k] = {}
        node = node[k]
    return node


def get_container(data: Any, keys: List[str]) -> Any:
    """Return the innermost dict at ``keys`` or None if any level is missing."""
    node = data
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return None
        node = node[k]
    return node if isinstance(node, dict) else None


def prune_empty(data: Any, keys: List[str]) -> None:
    """After a removal, drop now-empty container dicts along ``keys``."""
    for i in range(len(keys), 0, -1):
        parent = get_container(data, keys[: i - 1]) if i > 1 else data
        k = keys[i - 1]
        if isinstance(parent, dict) and isinstance(parent.get(k), dict) and not parent[k]:
            del parent[k]

