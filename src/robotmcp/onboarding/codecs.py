"""Read / merge / write helpers for the configuration formats used by the
coding-agent MCP config files rf-mcp registers itself into.

Formats: ``json`` (Claude Code, Copilot, opencode, Gemini, Cursor), ``jsonc``
(Kilo Code), ``toml`` (Codex), ``yaml`` (goose).

Per-format write fidelity - stated honestly, because "merge in place" previously
implied more than it delivered (change: installer-cli-safety sec 4):

=========  ===========================================================================
 toml      Byte-identical round trip. Comments (including inline ones), key order and
           formatting are preserved via tomlkit. This is the reference implementation.
 json      Data-preserving: other servers, unrelated keys and key ORDER survive.
           Re-indented to 2 spaces, so a 4-space file produces a whitespace-only diff.
 jsonc     As json, but COMMENTS ARE LOST on write (re-emitted as plain JSON, which is
           still valid JSONC). Trailing commas are normalised away. String VALUES are
           never altered.
 yaml      Data-preserving, but COMMENTS ARE LOST, and anchors/aliases/merge keys would
           be expanded inline - so `rewrite_hazard` REFUSES to rewrite such a file
           rather than silently flattening it.
=========  ===========================================================================
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple

SUPPORTED_FORMATS = ("json", "jsonc", "toml", "yaml")


def _strip_jsonc(text: str) -> str:
    """Remove // line and /* */ block comments and trailing commas, never touching
    the contents of string literals.

    The trailing-comma removal used to be a ``re.sub(r",(\\s*[}\\]])", ...)`` over
    the WHOLE joined output, which silently rewrote string VALUES: a config
    containing ``"Close the brace like this: { a, } and [ b, ]"`` came back as
    ``"... { a } and [ b ]"``. The scanner below already tracks string state, so
    the comma handling now happens inside it (change: installer-cli-safety sec 4).
    """
    out: List[str] = []
    # Index in `out` of the last emitted comma that is a candidate for removal,
    # and a flag for whether only whitespace has been emitted since.
    pending_comma: int = -1
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
            # A string starts: any pending comma is a real separator, not trailing.
            pending_comma = -1
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
        if ch == ",":
            pending_comma = len(out)
            out.append(ch); i += 1; continue
        if ch in "}]" and pending_comma >= 0:
            # Only whitespace/comments since the comma -> it was a trailing comma.
            del out[pending_comma]
            pending_comma = -1
            out.append(ch); i += 1; continue
        if not ch.isspace():
            pending_comma = -1
        out.append(ch); i += 1
    return "".join(out)


class ConfigParseError(Exception):
    """An existing agent config could not be parsed.

    Raised instead of letting a raw ``JSONDecodeError`` / tomlkit ``ParseError`` /
    yaml error escape, so the CLI can name the FILE (the tracebacks did not) and
    keep going (change: installer-cli-safety sec 6).
    """

    def __init__(self, path: Path, fmt: str, cause: BaseException):
        self.path, self.fmt, self.cause = path, fmt, cause
        super().__init__(f"cannot parse {fmt} config {path}: {cause}")


# YAML anchors (`&name`), aliases (`*name`) and merge keys (`<<:`) do not survive a
# safe_load/safe_dump round trip: they are expanded inline. The DATA stays equal, so
# nothing looks wrong, but the DRY link is severed - editing the anchor target later
# no longer propagates. Refuse rather than silently flatten (installer-cli-safety sec 4).
_YAML_ANCHOR_RE = re.compile(r"(?m)(?:^|\s)(?:&[A-Za-z0-9_-]+|\*[A-Za-z0-9_-]+|<<\s*:)")


def rewrite_hazard(path: Path, fmt: str) -> Optional[str]:
    """A reason this file must not be rewritten in place, or None when it is safe."""
    if fmt != "yaml" or not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    if _YAML_ANCHOR_RE.search(text):
        return ("this YAML uses anchors/aliases/merge keys, which a safe round trip "
                "would expand inline and silently break; add the rf-mcp entry by hand "
                "(see `robotmcp init`) or remove the anchors first")
    return None


def load(path: Path, fmt: str) -> Tuple[Any, bool]:
    """Return (data, existed). Missing/empty file -> (empty container, False).

    Raises ConfigParseError when the file exists but is malformed.
    """
    if not path.exists() or path.stat().st_size == 0:
        return ({}, False)
    text = path.read_text(encoding="utf-8")
    try:
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
    except ConfigParseError:
        raise
    except Exception as exc:
        raise ConfigParseError(path, fmt, exc) from exc
    raise ValueError(f"unsupported format: {fmt}")


def _detect_json_indent(path: Path, default: int = 2) -> int:
    """Indent width of an existing JSON file, so a merge does not reformat it.

    A 4-space `.mcp.json` was rewritten at 2 spaces, producing a whitespace-only
    diff across a VCS-tracked file (change: installer-cli-safety sec 4).
    """
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.lstrip(" ")
            if stripped and stripped != line and not stripped.startswith("//"):
                return len(line) - len(stripped)
    except OSError:
        pass
    return default


def dump(path: Path, fmt: str, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt in ("json", "jsonc"):
        indent = _detect_json_indent(path) if path.exists() else 2
        text = json.dumps(data, indent=indent) + "\n"
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

