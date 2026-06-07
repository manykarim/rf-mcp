"""Centralized library detection from scenario text.

v5 (ADR-024): adds a layered model:
  - MENTION layer (legacy `get_scores`, `detect`): broad pattern scoring used
    for diagnostics and capability hints. Unchanged behaviour for callers.
  - PREFERENCE layer (new `detect_explicit_preference`): conservative,
    evidence-based. Only patterns annotated `explicit=True` contribute.
    Implements sentence-scoped negation/migration with conflict surfacing.

The mention layer keeps its existing test contract:
`_compiled_patterns: Dict[str, List[Tuple[re.Pattern, int]]]` — verified by
`tests/integration/test_nlp_improvements.py:565-580`.
"""

from __future__ import annotations

import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Pattern, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Value objects (per ADR-024 §3.5, §4.1.6 and DDD §4.1)
# =============================================================================


@dataclass(frozen=True)
class PatternRule:
    """A single annotated regex pattern."""

    pattern: str
    weight: int
    explicit: bool
    rationale: str

    compiled: Pattern = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not (1 <= self.weight <= 10):
            raise ValueError(f"weight must be 1-10, got {self.weight}")
        object.__setattr__(self, "compiled", re.compile(self.pattern, re.IGNORECASE))


@dataclass(frozen=True)
class PatternMatch:
    """A single pattern-match record used as evidence."""

    library: str
    pattern: str
    weight: int
    text_span: str
    sentence_index: int = -1  # internal; not surfaced in to_dict

    def to_dict(self) -> Dict[str, Any]:
        return {
            "library": self.library,
            "pattern": self.pattern,
            "weight": self.weight,
            "text_span": self.text_span,
        }


@dataclass(frozen=True)
class DetectionPolicy:
    """Operator-tunable thresholds."""

    default_min_score: int = 5
    conflict_min_score: int = 8
    ambiguity_window: int = 4

    @classmethod
    def from_env(cls) -> "DetectionPolicy":
        return cls(
            default_min_score=int(
                os.getenv("ROBOTMCP_LIBRARY_DETECTION_MIN_SCORE", "5")
            ),
            conflict_min_score=int(
                os.getenv("ROBOTMCP_LIBRARY_DETECTION_CONFLICT_THRESHOLD", "8")
            ),
            ambiguity_window=int(
                os.getenv("ROBOTMCP_LIBRARY_DETECTION_AMBIGUITY_WINDOW", "4")
            ),
        )


@dataclass(frozen=True)
class PreferenceResolution:
    """Public output of detect_explicit_preference."""

    library: Optional[str]
    source: Literal["rule", "sampling"]
    evidence: List[Dict[str, Any]]
    conflicts: Dict[str, List[Dict[str, Any]]]
    all_scores: Dict[str, int]
    sampling_evidence: Optional[str] = None

    @property
    def is_decisive(self) -> bool:
        return self.library is not None

    @property
    def has_conflicts(self) -> bool:
        return bool(self.conflicts)


# =============================================================================
# Module-level constants (CONFLICT_GROUPS promoted from local per ADR-024 §3.3)
# =============================================================================


CONFLICT_GROUPS: Dict[str, Tuple[str, ...]] = {
    "web_automation": ("Browser", "SeleniumLibrary"),
}


# v7 (Codex round-6 C1): sentence delimiter is sentence punctuation
# (.;!?,) OR a paragraph break (2+ consecutive newlines with optional
# whitespace between). v6 dropped `\n` entirely to fix the orphaning bug
# ("do not use\nPlaywright" → None correct), but went too far: paragraph-
# separated text now leaked negation across paragraphs. Example v6 bug:
#   "Do not use this approach\n\nUse Playwright for the test."
#   → v6 single sentence: negation finds "Playwright" in remaining text →
#     subtracts Browser score → returns None (WRONG; user said do-not-use-
#     this-approach, NOT do-not-use-Playwright).
#
# v7 keeps single-newline as NON-boundary (preserves the v6 negation fix)
# while making double-newline (paragraph break) a boundary (restores
# paragraph-scope semantics). Newline boundary inside a paragraph is still
# enforced at the EXPLICIT PATTERN level via `[^\S\n]+` in LIBRARY_RULES_DEFAULT.
#
# Lineage:
#   v5: r"[.;!?,\n]+"         (\n was hard boundary — orphaned negation)
#   v6: r"[.;!?,]+"           (no \n — fixed orphaning; broke paragraphs)
#   v7: r"[.;!?,]+|\n\s*\n+"  (sentence punct OR paragraph break — both work)
_SENTENCE_DELIMITERS = r"[.;!?,]+|\n\s*\n+"


# v5: SINGLE negation regex with alternation, longest-first. Each negation
# span fires exactly ONCE (regex engine's left-to-right alternation guarantees
# longest match wins). v4 list-iteration caused double-deduction (round-3 D1).
# v5 restores standalone `skip` (round-4 regression).
_NEGATION_REGEX: Pattern = re.compile(
    r"\b(?:"
    # Compound phrases (longer first)
    r"do\s+not\s+use|don't\s+use|"
    r"not\s+using|without\s+using|stop\s+using|avoid\s+using|"
    r"skip\s+using|exclude\s+using|"
    # Simple phrases
    r"do\s+not|don't|"
    r"without|stop|avoid|skip|exclude"  # v5: skip restored
    r")\b",
    re.IGNORECASE,
)


# v5: migration patterns. `(?P<src>.+?)` / `(?P<dst>.+?)` plus
# `_first_library_token_in()` to resolve canonical library names.
# v6 (Codex round-5 D-multiword-migration): destination lookahead changed from
# (?=[\s.,;!?]|$) to (?=[.,;!?\n]|$). v5 stopped capture at the first
# whitespace, so "Migrate from Selenium to Requests library" captured
# dst='Requests' — and `_first_library_token_in("Requests")` returned None
# because bare `requests` is deliberately excluded from `_LIBRARY_TOKENS`
# (too generic). Requests/Database/SSH/XML migrations all failed silently.
# v6 captures up to the next sentence-punctuation/newline/EOL, then lets
# `_first_library_token_in()` resolve the canonical library — which DOES
# match the multi-word forms (`requests\s*library`, `database\s*library`, etc.).
# Source capture changes the same way so multi-word source names work too
# ("Migrate from Selenium Library to Browser Library").
MIGRATION_PATTERNS: List[str] = [
    r"\bmigrat(?:e|ion|ing)\b.+?\bfrom\b\s+(?P<src>.+?)\s+\bto\b\s+(?P<dst>.+?)(?=[.,;!?\n]|$)",
    r"\bswitch(?:ing)?\s+from\b\s+(?P<src>.+?)\s+\bto\b\s+(?P<dst>.+?)(?=[.,;!?\n]|$)",
    r"\binstead\s+of\b\s+(?P<src>.+?)\s+\b(?:use|with|via)\b\s+(?P<dst>.+?)(?=[.,;!?\n]|$)",
    r"\breplace\b\s+(?P<src>.+?)\s+\b(?:with|by|for)\b\s+(?P<dst>.+?)(?=[.,;!?\n]|$)",
]


# v5: library-token map for negation/migration target resolution. Deliberately
# EXCLUDES generic English nouns (`browser`, `database`, `ssh`, `requests`,
# `xml`) because users writing "do not use ssh" usually mean the SSH protocol,
# not SSHLibrary. Brand names (`selenium`, `playwright`, `appium`) kept.
_LIBRARY_TOKENS: Dict[str, List[str]] = {
    "SeleniumLibrary": [
        "selenium",
        "seleniumlibrary",
        r"selenium\s*library",
        "selenium2library",
        "selenium3library",
        "webdriver",
        "chromedriver",
        "geckodriver",
        "edgedriver",
        "safaridriver",
    ],
    "Browser": [
        "playwright",
        "browserlibrary",
        r"browser\s*library",
        "rfbrowser",
        r"robotframework[- ]browser",
        r"playwright[- ]core",
        "chromium",
        "webkit",
    ],
    "RequestsLibrary": ["requestslibrary", r"requests\s*library"],
    "AppiumLibrary": ["appium", "appiumlibrary", r"appium\s*library"],
    "DatabaseLibrary": ["databaselibrary", r"database\s*library"],
    "SSHLibrary": ["sshlibrary", r"ssh\s*library"],
    "XML": ["xmllibrary", r"xml\s*library"],
    "PlatynUI.BareMetal": [
        "platynui",
        r"platynui\.baremetal",
        r"platynui\s*baremetal",
        r"platynui[- ]cli",
        r"platynui[- ]native",
    ],
}


def _first_library_token_in(text: str) -> Optional[str]:
    """Return the canonical library name of the FIRST library token in `text`.

    Returns None if no library token matches.
    """
    earliest: Optional[Tuple[str, int]] = None
    for lib, tokens in _LIBRARY_TOKENS.items():
        for token in tokens:
            m = re.search(rf"\b{token}\b", text, re.IGNORECASE)
            if m and (earliest is None or m.start() < earliest[1]):
                earliest = (lib, m.start())
    return earliest[0] if earliest else None


def _build_sentence_spans(text: str) -> List[Tuple[int, int]]:
    """Split `text` on `_SENTENCE_DELIMITERS` while preserving (start, end)
    character positions so PatternMatch entries can record the sentence_index.
    """
    spans: List[Tuple[int, int]] = []
    pos = 0
    delim_re = re.compile(_SENTENCE_DELIMITERS)
    for m in delim_re.finditer(text):
        if pos < m.start():
            spans.append((pos, m.start()))
        pos = m.end()
    if pos < len(text):
        spans.append((pos, len(text)))
    return spans


# =============================================================================
# Pattern table (LIBRARY_RULES_DEFAULT — v5 source of truth)
# =============================================================================

# v5: each entry annotated with explicit/rationale. `explicit=True` patterns
# contribute to detect_explicit_preference; `explicit=False` patterns remain
# in the mention layer (`get_scores`). The classification policy: explicit
# means verbatim library/runtime/driver identifier OR preference-verb idiom
# bound to a library-specific token. Generic domain markers stay non-explicit.
#
# v5 NOTE on bare-token preference verbs: brand names (`selenium`, `playwright`,
# `appium`) stay inside preference-verb patterns. Generic English nouns
# (`browser`, `database`, `ssh`, `requests`, `xml`) are NOT in preference-verb
# patterns — too ambiguous.
#
# v5 NOTE on newlines: preference-verb patterns use `[^\S\n]+` instead of
# `\s+` so multi-line text "do not use\nSelenium" doesn't match `use selenium`
# across the line break (newline acts as sentence boundary).

LIBRARY_RULES_DEFAULT: Dict[str, List[PatternRule]] = {
    "SeleniumLibrary": [
        PatternRule(
            r"\b(use|using|with|via|through|prefer)[^\S\n]+(selenium|seleniumlibrary|selenium[^\S\n]*library)\b",
            10, True, "preference verb + selenium brand token",
        ),
        PatternRule(r"\bseleniumlibrary\b", 9, True, "verbatim library name"),
        PatternRule(r"\bselenium\b", 6, True, "standalone selenium mention"),
        PatternRule(r"\bwebdriver\b", 6, True, "selenium-specific WebDriver term"),
        PatternRule(
            r"\b(chromedriver|geckodriver|edgedriver|safaridriver)\b",
            7, True, "selenium drivers",
        ),
        PatternRule(r"\bselenium\s+grid\b", 8, True, "selenium-specific tech"),
        PatternRule(r"\bselenium\s+standalone\b", 7, True, "selenium-specific tech"),
        PatternRule(r"\bclassic\s+selenium\b", 7, True, "selenium-specific phrasing"),
        PatternRule(r"\bselenium\s+automation\b", 7, True, "selenium-domain phrasing"),
        PatternRule(
            r"\b(selenium\s+(2|3|4)|selenium2library|selenium3library)\b",
            8, True, "selenium version mention",
        ),
        PatternRule(
            r"\b(desired\s+capabilities|driver\s+capabilities)\b",
            7, True, "selenium Capabilities API",
        ),
        PatternRule(
            r"\b(create\s+webdriver|get\s+webelement)\b",
            8, True, "selenium-specific keywords",
        ),
        PatternRule(
            r"\btest\s+automation\s+with\s+selenium\b",
            8, True, "explicit phrasing",
        ),
        # Mention-only (NOT explicit) — generic NL that overlaps with SL keyword names
        PatternRule(
            r"\bopen\s+browser\b", 6, False,
            "REMOVED from explicit — generic NL verb that overlaps with SL keyword",
        ),
        PatternRule(
            r"\b(input\s+text|click\s+element|page\s+should\s+contain)\b", 6, False,
            "REMOVED from explicit — SL keyword names but also generic NL",
        ),
        PatternRule(
            r"\b(implicit|explicit)\s+wait\b", 6, False,
            "REMOVED from explicit — generic concept across web libraries",
        ),
    ],
    "Browser": [
        PatternRule(
            r"\b(use|using|with|via|through|prefer)[^\S\n]+(playwright|browserlibrary|browser[^\S\n]*library|rfbrowser|robotframework[- ]browser|playwright[- ]core)\b",
            10, True, "preference verb + Browser-specific token (no bare 'browser')",
        ),
        PatternRule(r"\bbrowser\s*library\b", 9, True, "verbatim library name"),
        PatternRule(r"\bplaywright\b", 9, True, "verbatim Browser library tech"),
        PatternRule(
            r"\b(rfbrowser|robotframework[- ]browser|playwright[- ]core)\b",
            9, True, "verbatim package names",
        ),
        PatternRule(r"\bchromium\b", 9, True, "Playwright kernel (as specific as 'playwright')"),
        PatternRule(r"\bwebkit\b", 9, True, "Playwright kernel"),
        # Mention-only
        PatternRule(
            r"\bbrowser\b", 4, False,
            "bare 'browser' too generic for explicit; kept in mention layer",
        ),
        PatternRule(r"\bmodern\s+web\s+testing\b", 7, False, "marketing copy"),
        PatternRule(r"\bmodern\s+browser\s+automation\b", 8, False, "marketing copy"),
        PatternRule(r"\bcross[- ]browser\s+testing\b", 6, False, "generic test type"),
        PatternRule(
            r"\bnew\s+(browser|page|context)\b", 8, False,
            "Browser keyword names but also generic NL",
        ),
        PatternRule(r"\bfill\s+(text|secret)\b", 7, False, "Browser keyword name"),
        PatternRule(
            r"\b(headless\s+browser|headless\s+chromium)\b", 6, False,
            "generic test config",
        ),
        PatternRule(
            r"\b(shadow\s+dom|web\s+components?)\b", 6, False,
            "modern web concept; not Browser-exclusive",
        ),
        PatternRule(
            r"\b(SPA|single\s+page\s+app(lication)?)\b", 5, False,
            "describes app architecture",
        ),
        PatternRule(r"\b(e2e|end.to.end)\s+(test|automat)", 5, False, "generic test type"),
    ],
    "RequestsLibrary": [
        PatternRule(
            r"\b(use|using|with|via|through|prefer)[^\S\n]+(requestslibrary|requests[^\S\n]*library)\b",
            10, True, "preference verb + Requests-specific (no bare 'requests')",
        ),
        PatternRule(r"\brequestslibrary\b", 9, True, "verbatim library name"),
        PatternRule(
            r"\b(create\s+session|get\s+on\s+session|post\s+on\s+session)\b",
            8, True, "RL keyword names with 'session' qualifier",
        ),
        PatternRule(
            r"\b(status\s+should\s+be|request\s+should\s+be)\b",
            7, True, "RL-specific keyword phrasing",
        ),
        PatternRule(
            r"\b(GET|POST|PUT|DELETE|PATCH)\s+on\s+session\b",
            7, True, "HTTP-method + RL 'on session' qualifier",
        ),
        # Mention-only
        PatternRule(r"\brequests\b", 4, False, "bare 'requests' too generic"),
        PatternRule(r"\brest\s+api\s+testing\b", 7, False, "domain marker"),
        PatternRule(r"\bapi\s+automation\b", 6, False, "domain marker"),
        PatternRule(r"\bstatus\s+code\b", 5, False, "generic HTTP term"),
        PatternRule(
            r"\b(GET|POST|PUT|DELETE|PATCH)\s+request\b", 5, False,
            "bare HTTP verb + request — domain-generic",
        ),
        PatternRule(r"\bhttp\s+requests?\b", 5, False, "generic HTTP term"),
        PatternRule(r"\b(webservice|web\s+service)\b", 5, False, "generic"),
        PatternRule(r"\bmicroservice\b", 5, False, "generic"),
        PatternRule(r"\b(bearer\s+token|JWT|OAuth2?)\b", 5, False, "auth concepts"),
        PatternRule(r"\b(swagger|openapi)\b", 5, False, "api-doc tooling"),
        PatternRule(r"\b(graphql|gRPC|SOAP)\b", 5, False, "API protocols"),
        PatternRule(r"\b(webhook|callback\s+url)\b", 5, False, "generic"),
    ],
    "AppiumLibrary": [
        PatternRule(
            r"\b(use|using|with|via|through|prefer)[^\S\n]+(appium|appiumlibrary|appium[^\S\n]*library)\b",
            10, True, "preference verb + Appium token",
        ),
        PatternRule(r"\bappium(?:library)?\b", 9, True, "verbatim library / runtime name"),
        PatternRule(
            r"\b(UIAutomator2?|XCUITest|Espresso)\b",
            7, True, "Appium-specific runtime",
        ),
        # Mention-only
        PatternRule(r"\bmobile\s+automation\b", 7, False, "domain marker"),
        PatternRule(r"\bmobile\s+app\s+testing\b", 7, False, "domain marker"),
        PatternRule(r"\bandroid\s+testing\b", 6, False, "platform, not library identifier"),
        PatternRule(r"\bios\s+testing\b", 6, False, "platform, not library identifier"),
        PatternRule(r"\b(open\s+application|close\s+application)\b", 8, False, "keyword names"),
        PatternRule(
            r"\b(tap|swipe|long\s+press|double\s+tap|flick|scroll|pinch)\b",
            6, False, "mobile action verbs",
        ),
        PatternRule(r"\bdevice\b", 5, False, "too generic"),
        PatternRule(r"\b(emulator|simulator)\b", 5, False, "generic"),
        PatternRule(r"\b(native\s+app|hybrid\s+app|webview)\b", 6, False, "generic mobile"),
        PatternRule(r"\b(APK|IPA|bundle\s+id|package\s+name)\b", 6, False, "mobile artifacts"),
        PatternRule(
            r"\b(device\s+farm|BrowserStack|Sauce\s+Labs)\b", 5, False, "cloud services",
        ),
        PatternRule(r"\b(iphone|ipad|tablet|smartphone)\b", 5, False, "device names"),
    ],
    "DatabaseLibrary": [
        PatternRule(
            r"\b(use|using|with|via|through|prefer)[^\S\n]+(databaselibrary|database[^\S\n]*library)\b",
            10, True, "preference verb + Database-specific (no bare 'database')",
        ),
        PatternRule(r"\bdatabaselibrary\b", 9, True, "verbatim library name"),
        PatternRule(
            r"\b(connect\s+to\s+database|execute\s+sql|call\s+stored\s+procedure)\b",
            8, True, "DatabaseLibrary keyword names",
        ),
        # Mention-only
        PatternRule(
            r"\b(SELECT|INSERT|UPDATE|DELETE)\s+(FROM|INTO|SET|\*)\b",
            5, False, "SQL is user's domain, not library identifier",
        ),
        PatternRule(
            r"\b(postgres(?:ql)?|mysql|mariadb|sqlite)\b",
            5, False, "DB engine name, not RF library identifier",
        ),
        PatternRule(
            r"\b(oracle|sql\s+server|mssql|mongodb)\b",
            5, False, "DB engine name, not RF library identifier",
        ),
        PatternRule(r"\bsql\s+testing\b", 6, False, "generic SQL test type"),
        PatternRule(r"\bdatabase\s+validation\b", 6, False, "generic"),
        PatternRule(
            r"\b(row\s+count|check\s+if\s+exists)\b",
            7, False, "DB-keyword names but generic NL",
        ),
        PatternRule(r"\b(connection\s+string|DSN|ODBC)\b", 5, False, "generic DB config"),
        PatternRule(r"\bstored\s+procedure\b", 6, False, "generic SQL concept"),
        PatternRule(r"\b(CRUD|schema\s+migration)\b", 5, False, "generic"),
    ],
    "SSHLibrary": [
        PatternRule(
            r"\b(use|using|with|via|through|prefer)[^\S\n]+(sshlibrary|ssh[^\S\n]*library)\b",
            10, True, "preference verb + SSH-specific (no bare 'ssh')",
        ),
        PatternRule(r"\bsshlibrary\b", 9, True, "verbatim library name"),
        PatternRule(
            r"\blogin\s+with\s+public\s+key\b",
            7, True, "SSHLibrary-specific keyword name",
        ),
        PatternRule(r"\b(sftp|scp)\b", 6, True, "SSH-specific protocols"),
        # Mention-only
        PatternRule(
            r"\bopen\s+connection\b", 7, False,
            "overlaps with Telnet, Database, Browser keyword names",
        ),
        PatternRule(
            r"\b(execute\s+command|start\s+command)\b",
            6, False, "generic command-execution",
        ),
        PatternRule(
            r"\b(get\s+file|put\s+file|get\s+directory|put\s+directory)\b",
            6, False, "generic file ops",
        ),
        PatternRule(
            r"\bssh\s+(into|to)\b", 6, False,
            "action verb, not library identifier",
        ),
        PatternRule(r"\bremote\s+server\s+commands?\b", 5, False, "generic"),
        PatternRule(
            r"\b(remote\s+(server|execution|machine|host))\b", 5, False, "generic",
        ),
        PatternRule(
            r"\b(linux|unix)\s+(server|machine|system)\b", 5, False, "OS-generic",
        ),
    ],
    "XML": [
        PatternRule(
            r"\b(use|using|with|via|through|prefer)[^\S\n]+(xmllibrary|xml[^\S\n]*library)\b",
            10, True, "preference verb + XML library-specific (no bare 'xml')",
        ),
        PatternRule(r"\bxmllibrary\b", 9, True, "verbatim library name"),
        # Mention-only
        PatternRule(r"\bxml\s+parsing\b", 6, False, "XML parsing is user's task"),
        PatternRule(r"\bxml\s+validation\b", 6, False, "domain marker"),
        PatternRule(
            r"\b(parse\s+xml|save\s+xml|log\s+element)\b",
            7, False, "XMLLibrary keyword names but also generic NL",
        ),
        PatternRule(
            r"\b(xslt|dtd|xsd)\b", 6, False,
            "XML standards, not RF library identifier",
        ),
        PatternRule(
            r"\bxpath\s+(expression|query|selector)\b",
            6, False, "XPath also used by SeleniumLibrary locators",
        ),
        PatternRule(
            r"\b(get\s+element\s+text|get\s+element\s+attribute)\b",
            6, False, "overlaps with other libs' keyword names",
        ),
        PatternRule(
            r"\b(namespace|element\s+tree|lxml)\b",
            5, False, "generic XML lib terms",
        ),
        PatternRule(
            r"\bxml\s+(file|document|response|config)\b",
            5, False, "generic XML mention",
        ),
        PatternRule(r"\bxml\b", 4, False, "bare xml could mean file format"),
    ],
    "PlatynUI.BareMetal": [
        PatternRule(
            r"\b(use|using|with|via|through|prefer)[^\S\n]+(platynui(?:\.baremetal)?|platynui[^\S\n]*baremetal)\b",
            10, True, "preference verb + PlatynUI token",
        ),
        PatternRule(r"\bplatynui\.baremetal\b", 9, True, "verbatim library name"),
        PatternRule(r"\bplatynui\b", 8, True, "standalone PlatynUI brand token"),
        PatternRule(r"\bplatynui[- ](cli|native|inspector)\b", 8, True, "PlatynUI tooling"),
        # Mention-only (domain markers, NOT explicit preference)
        PatternRule(
            r"\bdesktop\s+(automation|testing|ui)\b", 7, False, "domain marker",
        ),
        PatternRule(
            r"\bnative\s+(desktop|app(?:lication)?)\s+(test|automation)",
            6, False, "generic domain",
        ),
        PatternRule(
            r"\b(windows\s+uia|at[- ]spi2?|accessibility\s+tree)\b",
            6, False, "desktop accessibility technology",
        ),
    ],
}


# =============================================================================
# LibraryDetector
# =============================================================================


class LibraryDetector:
    """Centralized library detection from scenario text.

    Two-store internal design:
      - _compiled_patterns: legacy Dict[lib, List[Tuple[Pattern, int]]] for
        existing test contract at tests/integration/test_nlp_improvements.py
        (lines 565-580). Mention-layer driver.
      - LIBRARY_RULES: new Dict[lib, List[PatternRule]] for the explicit
        preference layer (detect_explicit_preference).
    """

    # Legacy class attribute kept for backward-compat — derived from LIBRARY_RULES.
    # Older callers may reference LibraryDetector.LIBRARY_PATTERNS.
    LIBRARY_PATTERNS: Dict[str, List[Tuple[str, int]]] = {
        lib: [(rule.pattern, rule.weight) for rule in rules]
        for lib, rules in LIBRARY_RULES_DEFAULT.items()
    }

    DEFAULT_MIN_SCORE = 5

    # Legacy negation patterns — preserved for backward compat in `get_scores`
    # (mention layer). The new explicit-preference path uses _NEGATION_REGEX
    # + sentence-scoped algorithm.
    NEGATION_PATTERNS = [
        re.compile(
            r"\b(not|don't|do\s+not|without|stop|avoid|skip|exclude)\s+(?:using\s+)?",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(instead\s+of|migrate\s+from|replace|replacing|move\s+away\s+from)\s+",
            re.IGNORECASE,
        ),
    ]

    def __init__(
        self,
        min_score: Optional[int] = None,
        rules: Optional[Dict[str, List[PatternRule]]] = None,
        policy: Optional[DetectionPolicy] = None,
    ) -> None:
        """Initialize LibraryDetector.

        Args:
            min_score: Legacy threshold for `detect`/`get_scores` (mention layer).
            rules: Optional override for LIBRARY_RULES (v5 — preference layer).
            policy: Optional DetectionPolicy override.
        """
        self.min_score = min_score or self.DEFAULT_MIN_SCORE
        self.LIBRARY_RULES = rules if rules is not None else LIBRARY_RULES_DEFAULT
        self.policy = policy or DetectionPolicy.from_env()

        # _compiled_patterns: legacy 2-tuple shape — preserves
        # tests/integration/test_nlp_improvements.py:565-580 contract
        # (for p, _ in entries: p.findall(...)).
        self._compiled_patterns: Dict[str, List[Tuple[Pattern, int]]] = {
            lib: [(rule.compiled, rule.weight) for rule in rules_list]
            for lib, rules_list in self.LIBRARY_RULES.items()
        }

    # ------------------------------------------------------------------
    # Mention layer (legacy API — unchanged behaviour)
    # ------------------------------------------------------------------

    def detect(self, text: str, min_score: Optional[int] = None) -> Optional[str]:
        """Detect library preference via mention-layer scoring (legacy).

        New callers should prefer `detect_explicit_preference()`.
        """
        if not text:
            return None
        scores = self.get_scores(text)
        if not scores:
            return None
        threshold = min_score or self.min_score
        best_lib, best_score = max(scores.items(), key=lambda x: x[1])
        if best_score >= threshold:
            logger.debug(
                f"Detected library preference: {best_lib} (score: {best_score})"
            )
            return best_lib
        return None

    def get_scores(self, text: str) -> Dict[str, int]:
        """Mention-layer scores (ALL patterns regardless of explicit flag)."""
        if not text:
            return {}
        text_lower = text.lower()
        scores: Dict[str, int] = defaultdict(int)
        for lib, patterns in self._compiled_patterns.items():
            for pattern, weight in patterns:
                matches = len(pattern.findall(text_lower))
                if matches > 0:
                    scores[lib] += matches * weight
        # Legacy negation handling (unchanged from pre-v5)
        for lib_name, score in list(scores.items()):
            if score > 0:
                for neg_pattern in self.NEGATION_PATTERNS:
                    for pattern_str, weight in self.LIBRARY_PATTERNS.get(lib_name, []):
                        combined = neg_pattern.pattern + r".*?" + pattern_str
                        if re.search(combined, text, re.IGNORECASE):
                            scores[lib_name] = max(0, scores[lib_name] - weight * 2)
                            break
        return dict(scores)

    def detect_all(
        self, text: str, min_score: Optional[int] = None
    ) -> List[Tuple[str, int]]:
        """Mention-layer multi-library detection (legacy)."""
        scores = self.get_scores(text)
        threshold = min_score or self.min_score
        detected = [(lib, score) for lib, score in scores.items() if score >= threshold]
        return sorted(detected, key=lambda x: -x[1])

    def get_conflicting_detections(self, text: str) -> Dict[str, List[str]]:
        """Legacy conflict surface (mention layer)."""
        scores = self.get_scores(text)
        conflicts: Dict[str, List[str]] = {}
        for group_name, group_libs in CONFLICT_GROUPS.items():
            detected = [lib for lib in group_libs if scores.get(lib, 0) > 0]
            if len(detected) > 1:
                conflicts[group_name] = detected
        return conflicts

    # ------------------------------------------------------------------
    # Preference layer (v5 — explicit detection)
    # ------------------------------------------------------------------

    def detect_explicit_preference(
        self, text: str, policy: Optional[DetectionPolicy] = None
    ) -> PreferenceResolution:
        """Conservative explicit-preference detection.

        Algorithm (per ADR-024 §3.3, proposal v5 §3.1 Step 6):
          1. Build sentence spans + compute raw explicit scores + collect matches.
          2. Sentence-scoped negation (single _NEGATION_REGEX, longest-first
             alternation; phrase-based with _first_library_token_in lookup)
             + migration (source/destination resolution).
          3. CONFLICT CHECK on raw scores BEFORE threshold filter.
          4. Threshold filter (conflict_min_score for in-group libs,
             default_min_score for out-of-group).
          5. Pick winner.
        """
        p = policy or self.policy
        empty = PreferenceResolution(
            library=None,
            source="rule",
            evidence=[],
            conflicts={},
            all_scores={},
            sampling_evidence=None,
        )
        if not text:
            return empty

        sentence_spans = _build_sentence_spans(text)

        def sentence_index_for(pos: int) -> int:
            for i, (start, end) in enumerate(sentence_spans):
                if start <= pos < end:
                    return i
            return -1

        # === Step 1: raw scores + matches ===
        raw_scores: Dict[str, int] = defaultdict(int)
        matches_by_lib: Dict[str, List[PatternMatch]] = defaultdict(list)
        for lib, rules in self.LIBRARY_RULES.items():
            for rule in rules:
                if not rule.explicit:
                    continue
                for m in rule.compiled.finditer(text):
                    raw_scores[lib] += rule.weight
                    if len(matches_by_lib[lib]) < 8:
                        matches_by_lib[lib].append(
                            PatternMatch(
                                library=lib,
                                pattern=rule.pattern,
                                weight=rule.weight,
                                text_span=m.group(0),
                                sentence_index=sentence_index_for(m.start()),
                            )
                        )

        # === Step 2: sentence-scoped negation + migration ===
        for sentence_index, (start, end) in enumerate(sentence_spans):
            sent = text[start:end]
            if not sent.strip():
                continue
            # 2a. Migration patterns
            for pattern_src in MIGRATION_PATTERNS:
                for m in re.finditer(pattern_src, sent, re.IGNORECASE):
                    src_lib = _first_library_token_in(m.group("src"))
                    dst_lib = _first_library_token_in(m.group("dst"))
                    if src_lib:
                        self._subtract_sentence_score(
                            raw_scores, matches_by_lib, sent, sentence_index, src_lib
                        )
                    if dst_lib:
                        raw_scores[dst_lib] += 5
            # 2b. Single-library negation (SINGLE regex alternation)
            for m in _NEGATION_REGEX.finditer(sent):
                remaining = sent[m.end():]
                target_lib = _first_library_token_in(remaining)
                if target_lib:
                    self._subtract_sentence_score(
                        raw_scores, matches_by_lib, sent, sentence_index, target_lib
                    )

        # === Step 3: conflict check on RAW scores ===
        conflicts: Dict[str, List[Dict[str, Any]]] = {}
        for group_name, members in CONFLICT_GROUPS.items():
            libs_with_signal = [
                lib for lib in members if raw_scores.get(lib, 0) > 0
            ]
            if len(libs_with_signal) < 2:
                continue
            ranked = sorted(
                libs_with_signal, key=lambda l: raw_scores[l], reverse=True
            )
            if raw_scores[ranked[0]] - raw_scores[ranked[1]] <= p.ambiguity_window:
                conflicts[group_name] = [
                    {
                        "library": lib,
                        "score": raw_scores[lib],
                        "patterns_matched": [pm.pattern for pm in matches_by_lib[lib]],
                    }
                    for lib in ranked
                ]

        if conflicts:
            return PreferenceResolution(
                library=None,
                source="rule",
                evidence=[],
                conflicts=conflicts,
                all_scores=dict(raw_scores),
                sampling_evidence=None,
            )

        # === Step 4: threshold filter ===
        candidates: Dict[str, int] = {}
        for lib, score in raw_scores.items():
            threshold = (
                p.conflict_min_score
                if self._in_conflict_group(lib)
                else p.default_min_score
            )
            if score >= threshold:
                candidates[lib] = score

        if not candidates:
            return PreferenceResolution(
                library=None,
                source="rule",
                evidence=[],
                conflicts={},
                all_scores=dict(raw_scores),
                sampling_evidence=None,
            )

        # === Step 5: winner ===
        winner = max(candidates, key=candidates.get)
        return PreferenceResolution(
            library=winner,
            source="rule",
            evidence=[pm.to_dict() for pm in matches_by_lib[winner]],
            conflicts={},
            all_scores=dict(raw_scores),
            sampling_evidence=None,
        )

    def _in_conflict_group(self, library: str) -> bool:
        return any(library in members for members in CONFLICT_GROUPS.values())

    def _subtract_sentence_score(
        self,
        raw_scores: Dict[str, int],
        matches_by_lib: Dict[str, List[PatternMatch]],
        sentence: str,
        sentence_index: int,
        library: str,
    ) -> None:
        """Subtract sentence-local explicit contributions for `library` and
        strip evidence whose `sentence_index` matches.
        """
        # v6 (Codex round-5 D-repeated-token): count OCCURRENCES, not "did
        # this rule match at all". v5 used `rule.compiled.search(sentence)`
        # which fired once per rule regardless of how many times the pattern
        # appeared in the sentence. Step 1 (raw scoring) uses `finditer` and
        # adds weight per occurrence, so subtraction must mirror that. Without
        # this fix, "do not use Playwright Playwright" scored Browser=28 but
        # only deducted 19 → Browser=9 wins despite the negation.
        deduction = 0
        for rule in self.LIBRARY_RULES.get(library, []):
            if not rule.explicit:
                continue
            occurrences = len(rule.compiled.findall(sentence))
            if occurrences:
                deduction += rule.weight * occurrences
        if deduction:
            raw_scores[library] = max(0, raw_scores[library] - deduction)
            matches_by_lib[library] = [
                pm
                for pm in matches_by_lib[library]
                if pm.sentence_index != sentence_index
            ]


# Module-level singleton for convenience
_detector: Optional[LibraryDetector] = None


def get_library_detector() -> LibraryDetector:
    """Get the global LibraryDetector instance."""
    global _detector
    if _detector is None:
        _detector = LibraryDetector()
    return _detector


def detect_library_preference(text: str, min_score: int = 5) -> Optional[str]:
    """Convenience function (legacy — uses mention-layer detect)."""
    return get_library_detector().detect(text, min_score)


def detect_explicit_library_preference(text: str) -> PreferenceResolution:
    """v5 convenience: returns the full PreferenceResolution for the new
    preference-layer path. Use this instead of `detect()` when you need
    evidence/conflicts/source provenance.
    """
    return get_library_detector().detect_explicit_preference(text)
