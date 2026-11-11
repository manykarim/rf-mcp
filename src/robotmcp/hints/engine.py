"""Declarative error hinting engine for RobotCode CLI, REPL, and test runs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal, Sequence

from pydantic import BaseModel, Field


class Hint(BaseModel):
    """Actionable recovery hint returned to coding agents on keyword or command failure."""

    id: str
    title: str
    message: str
    fix_examples: list[str] = Field(default_factory=list)
    relevance: int = 50


HintKind = Literal["runtime", "static", "any"]


class HintRule(BaseModel):
    """Declarative pattern rule matching error output, keyword name, or arguments.

    `applies_to` names the output kinds the rule is meant for: `runtime` (runs, REPL and
    debug sessions, CLI errors) and/or `static` (`rf_analyze` diagnostics).
    """

    id: str
    title: str
    error_pattern: str = ".*"
    keyword_pattern: str = ".*"
    arg_pattern: str | None = None
    message: str
    fix_examples: list[str] = Field(default_factory=list)
    relevance: int = 50
    applies_to: list[Literal["runtime", "static"]] = Field(default_factory=lambda: ["runtime"])
    library_pattern: str | None = None
    """Only for this library: tested against the context's libraries, or the error text when none are known."""

    def validate_patterns(self) -> None:
        """Compile every pattern; raise `re.error` naming the field on an invalid one."""
        for field_name in ("error_pattern", "keyword_pattern", "arg_pattern", "library_pattern"):
            pattern = getattr(self, field_name)
            if pattern is None:
                continue
            try:
                re.compile(pattern)
            except re.error as exc:
                raise re.error(f"{field_name}: {exc}") from exc

    def matches(self, ctx: HintContext) -> bool:
        """Return True if this rule matches the given HintContext."""
        if self.error_pattern and self.error_pattern != ".*":
            if not re.search(self.error_pattern, ctx.error_text, re.IGNORECASE | re.DOTALL):
                return False

        if self.keyword_pattern and self.keyword_pattern != ".*":
            kw_target = ctx.keyword or ctx.inferred_keyword()
            if not re.search(self.keyword_pattern, kw_target, re.IGNORECASE):
                return False

        if self.arg_pattern:
            joined_args = "    ".join(ctx.arguments)
            target_text = f"{ctx.keyword}    {joined_args}".strip()
            if not re.search(self.arg_pattern, target_text, re.IGNORECASE | re.DOTALL):
                return False

        if self.library_pattern:
            evidence = " ".join(ctx.libraries) if ctx.libraries else ctx.error_text
            if not re.search(self.library_pattern, evidence, re.IGNORECASE):
                return False

        return True

    def to_hint(self) -> Hint:
        """Convert rule to a Hint response model."""
        return Hint(
            id=self.id,
            title=self.title,
            message=self.message,
            fix_examples=list(self.fix_examples),
            relevance=self.relevance,
        )


class HintContext(BaseModel):
    """Context of a failed Robot Framework / RobotCode operation."""

    keyword: str = ""
    arguments: list[str] = Field(default_factory=list)
    error_text: str = ""
    libraries: list[str] = Field(default_factory=list)

    def inferred_keyword(self) -> str:
        """Infer keyword name from statement or error text when not explicitly provided."""
        if self.keyword:
            parts = re.split(r"\s{2,}|\t", self.keyword.strip())
            if parts:
                first = parts[0]
                if re.match(r"^[\$\@\&]\{[^}]+\}\s*=?", first) and len(parts) > 1:
                    return parts[1]
                return first
        m = re.search(r"No keyword with name '([^']+)' found", self.error_text)
        if m:
            return m.group(1)
        return self.keyword


class HintEngine:
    """Evaluates failed operations against built-in and project-custom `.robotframework/hints.toml` rules."""

    def __init__(
        self,
        custom_rules: Sequence[HintRule] | None = None,
        hints_file: Path | None = None,
    ) -> None:
        from robotmcp.hints.builtin_rules import BUILTIN_HINT_RULES
        from robotmcp.hints.loader import load_custom_rules_with_errors

        rules_by_id: dict[str, HintRule] = {
            rule.id: rule for rule in BUILTIN_HINT_RULES
        }
        self.load_errors: list[str] = []
        if hints_file is not None:
            custom, self.load_errors = load_custom_rules_with_errors(hints_file)
            for rule in custom:
                rules_by_id[rule.id] = rule
        if custom_rules is not None:
            for rule in custom_rules:
                rules_by_id[rule.id] = rule
        self._rules = list(rules_by_id.values())

    @property
    def rules(self) -> list[HintRule]:
        """Return active rules sorted by relevance descending."""
        return sorted(self._rules, key=lambda r: r.relevance, reverse=True)

    def diagnose(
        self,
        ctx: HintContext,
        *,
        max_hints: int = 3,
        kind: HintKind = "any",
    ) -> list[Hint]:
        """Evaluate `ctx` and return up to `max_hints` top-ranked Hint objects.

        `kind` restricts the rules to those meant for that output (`runtime` or `static`).
        """
        if not ctx.error_text.strip() and not ctx.keyword.strip():
            return []

        # Normalize statement when keyword contains two-space separated args
        normalized_ctx = ctx
        if not ctx.arguments and ("  " in ctx.keyword or "\t" in ctx.keyword):
            cells = [c.strip() for c in re.split(r"\s{2,}|\t", ctx.keyword.strip()) if c.strip()]
            if cells:
                if re.match(r"^[\$\@\&]\{[^}]+\}\s*=?", cells[0]) and len(cells) > 1:
                    kw_name = cells[1]
                    kw_args = cells[2:]
                else:
                    kw_name = cells[0]
                    kw_args = cells[1:]
                normalized_ctx = HintContext(
                    keyword=kw_name,
                    arguments=kw_args,
                    error_text=ctx.error_text,
                    libraries=ctx.libraries,
                )

        matched: list[Hint] = []
        seen_ids: set[str] = set()
        for rule in self.rules:
            if kind != "any" and kind not in rule.applies_to:
                continue
            if rule.id not in seen_ids and rule.matches(normalized_ctx):
                matched.append(rule.to_hint())
                seen_ids.add(rule.id)
                if len(matched) >= max_hints:
                    break
        return matched
