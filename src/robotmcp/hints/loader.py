"""TOML loader for project-custom `.robotframework/hints.toml` rules."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import tomlkit
from tomlkit.exceptions import ParseError

from robotmcp.hints.engine import HintRule

logger = logging.getLogger(__name__)


def _rule_line(raw_text: str, rule_id: str, index: int) -> int | None:
    """Return the 1-based line of the rule's `id = ...` entry, or of its `[[hints]]` header."""
    lines = raw_text.splitlines()
    id_re = re.compile(rf"^\s*id\s*=\s*['\"]{re.escape(rule_id)}['\"]")
    for no, line in enumerate(lines, 1):
        if id_re.match(line):
            return no
    headers = [no for no, line in enumerate(lines, 1) if line.strip() == "[[hints]]"]
    return headers[index] if index < len(headers) else None


def _build_rule(item: dict, idx: int) -> HintRule | None:
    rule_id = str(item.get("id") or f"custom-hint-{idx + 1}")
    title = str(item.get("title") or rule_id)
    raw_err = item.get("error_pattern") or item.get("pattern")
    raw_kw = item.get("keyword_pattern")
    arg_pattern_raw = item.get("arg_pattern")
    if not raw_err and not raw_kw and not arg_pattern_raw:
        return None

    msg_parts: list[str] = []
    if item.get("message"):
        msg_parts.append(str(item["message"]))
    else:
        if item.get("explanation"):
            msg_parts.append(str(item["explanation"]))
        if item.get("suggestion"):
            msg_parts.append(str(item["suggestion"]))

    fix_raw = item.get("fix_examples") or item.get("examples") or []
    skills_raw = item.get("skills") or []
    if isinstance(skills_raw, str):
        skills_raw = [skills_raw]
    skills = [str(s) for s in skills_raw] if isinstance(skills_raw, list) else []
    for uri in skills:
        if not uri.startswith("skill://") or len(uri) <= len("skill://"):
            raise ValueError(f"skills: {uri!r} is not a skill:// link, e.g. 'skill://acme-auth/references/tokens.md'")
    applies_raw = item.get("applies_to") or ["runtime", "static"]
    if isinstance(applies_raw, str):
        applies_raw = [applies_raw]
    return HintRule(
        id=rule_id,
        title=title,
        error_pattern=str(raw_err) if raw_err else ".*",
        keyword_pattern=str(raw_kw) if raw_kw else ".*",
        arg_pattern=str(arg_pattern_raw) if arg_pattern_raw is not None else None,
        message=" ".join(msg_parts).strip(),
        fix_examples=[str(x) for x in fix_raw] if isinstance(fix_raw, list) else [],
        relevance=int(item.get("relevance") or item.get("priority") or 90),
        applies_to=[str(a) for a in applies_raw],
        library_pattern=str(item["library_pattern"]) if item.get("library_pattern") else None,
        skills=skills,
    )


def load_custom_rules_with_errors(hints_file: Path) -> tuple[list[HintRule], list[str]]:
    """Load valid rules from `hints_file`; return them with one message per skipped rule or file error."""
    if not hints_file.is_file():
        return [], []
    try:
        raw_text = hints_file.read_text(encoding="utf-8")
        doc = tomlkit.parse(raw_text)
    except (ParseError, UnicodeDecodeError, OSError) as exc:
        msg = f"{hints_file}: not loaded ({exc}); using built-in rules only"
        logger.warning(msg)
        return [], [msg]

    raw_hints = doc.get("hints", [])
    if not isinstance(raw_hints, list):
        msg = f"{hints_file}: `hints` must be an array of tables ([[hints]])"
        logger.warning(msg)
        return [], [msg]

    rules: list[HintRule] = []
    errors: list[str] = []
    for idx, item in enumerate(raw_hints):
        if not isinstance(item, dict):
            continue
        rule_id = str(item.get("id") or f"custom-hint-{idx + 1}")
        try:
            rule = _build_rule(dict(item), idx)
            if rule is None:
                continue
            rule.validate_patterns()
        except (re.error, ValueError, TypeError) as exc:
            line = _rule_line(raw_text, rule_id, idx)
            where = f"{hints_file}:{line}" if line else str(hints_file)
            msg = f"{where}: rule '{rule_id}' skipped ({exc})"
            logger.warning(msg)
            errors.append(msg)
            continue
        rules.append(rule)
    return rules, errors


def load_custom_rules(hints_file: Path) -> list[HintRule]:
    """Load custom HintRule entries from a `.robotframework/hints.toml` file, skipping invalid ones."""
    return load_custom_rules_with_errors(hints_file)[0]
