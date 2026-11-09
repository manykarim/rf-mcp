"""Tolerant tool argument types for weak models (legacy ADR-010 pattern).

The aliases only add a `BeforeValidator`, so the published JSON schema stays the
plain `array` / `enum` while the server also accepts what weak models send.
"""

from __future__ import annotations

import ast
import json
from typing import Annotated, Any, Literal, get_args

from fastmcp.server.middleware import Middleware, MiddlewareContext
from pydantic import BeforeValidator


def _python_literal(text: str) -> Any:
    """`['Tests.A']` as Python writes it (2026-11-09 Qwen3-Coder: `by_longname="['Tests.Web.Saucedemo.Login']"` became
    one `-bl` with brackets and quotes, and robot found no tests)."""
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None


def coerce_list(value: Any) -> Any:
    """Turn a JSON array string, a comma-separated string or a single string into a list."""
    if value is None or isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = _python_literal(text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        if "," in text:
            return [part.strip() for part in text.split(",") if part.strip()]
        return [text]
    return value


def coerce_dict(value: Any) -> Any:
    """Turn a JSON object string into a dict."""
    if isinstance(value, str) and value.strip().startswith("{"):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value
        return parsed if isinstance(parsed, dict) else value
    return value


def _case_insensitive(choices: tuple[str, ...]):
    lookup = {c.lower(): c for c in choices}

    def _coerce(value: Any) -> Any:
        if isinstance(value, str):
            return lookup.get(value.strip().lower(), value)
        return value

    return _coerce


def tolerant_literal(literal_type: Any) -> Any:
    """Wrap a `Literal[...]` so values match case-insensitively."""
    return Annotated[literal_type, BeforeValidator(_case_insensitive(get_args(literal_type)))]


StrList = Annotated[list[str], BeforeValidator(coerce_list)]
StrDict = Annotated[dict[str, str], BeforeValidator(coerce_dict)]
Target = tolerant_literal(Literal["server", "client"])
Shell = tolerant_literal(Literal["powershell", "bash"])


# Parameter names weak models sent instead of the published one (2026-11-07 Codex/GPT-5.6-Luna:
# `rf_results(path="results/x.xml")` failed with `unexpected_keyword_argument` and the agent gave up on the run).
ARGUMENT_ALIASES: dict[str, dict[str, str]] = {
    "rf_results": {"path": "output_path", "result_file": "output_path", "output_file": "output_path"},
}


# Empty values mean "not set" for every argument. Models behind Claude Code on OpenRouter send every optional
# argument, unset ones as "", [], {} or the text "[]" (2026-11-09: GPT-6-Luna `rf_analyze(shell="")` failed "Input
# should be 'powershell' or 'bash'"; Qwen3-Coder `files=""` and `output=[]` failed validation, `rerunfailed="[]"`
# became `--rerunfailed []`; DeepSeek sent `profile="[]"`).
EMPTY_TEXT = frozenset({"", "[]", "{}", "null"})


def is_unset(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip() in EMPTY_TEXT
    return isinstance(value, (list, dict)) and not value


class ArgumentAliases(Middleware):
    """Rename alias arguments to the published name before validation (only when the published one is absent), and
    drop arguments sent empty ("", "[]", "{}", "null", [] or {})."""

    async def on_call_tool(self, context: MiddlewareContext, call_next) -> Any:
        aliases = ARGUMENT_ALIASES.get(getattr(context.message, "name", ""))
        args = getattr(context.message, "arguments", None)
        if isinstance(args, dict):
            for alias, name in (aliases or {}).items():
                if alias in args and name not in args:
                    args[name] = args.pop(alias)
            for name in [n for n, v in args.items() if is_unset(v)]:
                del args[name]
        return await call_next(context)
