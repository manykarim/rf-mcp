"""Benchmark find_keywords + get_keyword_info across a scenario matrix.

Goal: capture how the discovery tools respond to:
- Different strategies (semantic / pattern / catalog)
- With and without a library filter (Browser, SeleniumLibrary, none)
- Precise vs vague vs nonsense queries
- Edge cases (ambiguous names, unknown keywords, fuzzy matches)

For each scenario, record:
- The full response (inline + externalized artifact content)
- Inline-token estimate (what the LLM sees without fetching the artifact)
- Artifact-token estimate (what's hidden behind the externalization link)
- Top match name + library
- Relevance assessment (manual, recorded in the report)

Outputs:
- /tmp/discovery_benchmark/<scenario_id>.json — raw response
- /tmp/discovery_benchmark/<scenario_id>.artifact.txt — externalized content (if any)
- /tmp/discovery_benchmark/summary.json — aggregate metrics

Run: uv run python scripts/benchmark_discovery_tools.py
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from robotmcp.server import find_keywords, get_keyword_info  # type: ignore


OUT = Path("/tmp/discovery_benchmark")


def _approx_tokens(text: str) -> int:
    """~4 chars/token approximation."""
    return max(1, len(text) // 4)


def _unwrap(tool):
    return getattr(tool, "fn", tool)


def _artifact_path_from(response: Any) -> Optional[Path]:
    """If response['result'] points at an externalized artifact, return
    that path. Otherwise None."""
    if not isinstance(response, dict):
        return None
    result_field = response.get("result")
    if not isinstance(result_field, str):
        return None
    # Format: "Content saved to <path> (NNN bytes, ~NNN tokens)."
    if "Content saved to" not in result_field:
        return None
    # Extract the path between "saved to " and " ("
    try:
        after = result_field.split("Content saved to ", 1)[1]
        path = after.split(" (", 1)[0].strip()
        return Path(path)
    except Exception:
        return None


@dataclass
class ScenarioResult:
    scenario_id: str
    description: str
    tool: str
    inputs: Dict[str, Any]
    response: Dict[str, Any]
    artifact_bytes: int = 0
    artifact_tokens: int = 0
    artifact_content: Optional[str] = None
    inline_tokens: int = 0
    error: Optional[str] = None

    # Derived fields filled after analysis
    matches_top: Optional[Dict[str, Any]] = None
    matches_libs: List[str] = field(default_factory=list)
    matches_count: int = 0
    recommendations: List[str] = field(default_factory=list)
    excluded_count: int = 0
    excluded_alternatives_count: int = 0
    library_filter_source: Optional[str] = None


async def run_find_keywords(
    sid: str,
    description: str,
    **kwargs,
) -> ScenarioResult:
    fn = _unwrap(find_keywords)
    try:
        resp = await fn(**kwargs)
    except Exception as e:
        return ScenarioResult(
            scenario_id=sid, description=description, tool="find_keywords",
            inputs=kwargs, response={}, error=str(e),
        )
    # Serialize the response for inline-token estimation BEFORE expanding
    # the artifact (we want what the LLM sees in its first read).
    inline = json.dumps(resp, default=str)
    inline_tok = _approx_tokens(inline)

    artifact_bytes = 0
    artifact_tokens = 0
    artifact_content: Optional[str] = None
    apath = _artifact_path_from(resp)
    if apath and apath.exists():
        artifact_bytes = apath.stat().st_size
        artifact_content = apath.read_text(encoding="utf-8")
        artifact_tokens = _approx_tokens(artifact_content)

    sr = ScenarioResult(
        scenario_id=sid, description=description, tool="find_keywords",
        inputs=kwargs, response=resp,
        artifact_bytes=artifact_bytes, artifact_tokens=artifact_tokens,
        artifact_content=artifact_content,
        inline_tokens=inline_tok,
    )

    # Extract analytical fields
    result = resp.get("result")
    if isinstance(result, dict):
        matches = result.get("matches") or []
        sr.matches_count = len(matches)
        sr.matches_libs = sorted({m.get("library", "?") for m in matches})
        if matches:
            sr.matches_top = {
                "keyword_name": matches[0].get("keyword_name") or matches[0].get("name"),
                "library": matches[0].get("library"),
                "confidence": matches[0].get("confidence"),
            }
        sr.recommendations = result.get("recommendations") or []
    elif isinstance(result, str) and artifact_content:
        # Externalized — parse the artifact
        try:
            parsed = json.loads(artifact_content)
            matches = parsed.get("matches") or []
            sr.matches_count = len(matches)
            sr.matches_libs = sorted({m.get("library", "?") for m in matches})
            if matches:
                sr.matches_top = {
                    "keyword_name": matches[0].get("keyword_name") or matches[0].get("name"),
                    "library": matches[0].get("library"),
                    "confidence": matches[0].get("confidence"),
                }
            sr.recommendations = parsed.get("recommendations") or []
        except Exception:
            pass
    # For pattern strategy, results are under "results"
    if "results" in resp:
        results = resp.get("results") or []
        sr.matches_count = len(results)
        sr.matches_libs = sorted({m.get("library", "?") for m in results})
        if results:
            sr.matches_top = {
                "keyword_name": results[0].get("name"),
                "library": results[0].get("library"),
            }

    lf = resp.get("library_filter")
    if isinstance(lf, dict):
        sr.excluded_count = lf.get("count", 0) or 0
        sr.library_filter_source = lf.get("source")
    sr.excluded_alternatives_count = len(resp.get("excluded_alternatives") or [])
    return sr


async def run_get_keyword_info(
    sid: str,
    description: str,
    **kwargs,
) -> ScenarioResult:
    fn = _unwrap(get_keyword_info)
    try:
        resp = await fn(**kwargs)
    except Exception as e:
        return ScenarioResult(
            scenario_id=sid, description=description, tool="get_keyword_info",
            inputs=kwargs, response={}, error=str(e),
        )
    inline = json.dumps(resp, default=str)
    return ScenarioResult(
        scenario_id=sid, description=description, tool="get_keyword_info",
        inputs=kwargs, response=resp,
        inline_tokens=_approx_tokens(inline),
    )


SCENARIOS: List[Dict[str, Any]] = [
    # ---- find_keywords / semantic ----
    {
        "id": "S01_semantic_precise_browser",
        "description": "Precise semantic query + Browser library filter",
        "tool": "find_keywords",
        "kwargs": {"query": "select dropdown option by visible label", "strategy": "semantic", "library_name": "Browser"},
    },
    {
        "id": "S02_semantic_precise_selenium",
        "description": "Precise semantic query + SeleniumLibrary filter (symmetry check)",
        "tool": "find_keywords",
        "kwargs": {"query": "select dropdown option by visible label", "strategy": "semantic", "library_name": "SeleniumLibrary"},
    },
    {
        "id": "S03_semantic_precise_nolib",
        "description": "Precise semantic query + NO library filter",
        "tool": "find_keywords",
        "kwargs": {"query": "select dropdown option by visible label", "strategy": "semantic"},
    },
    {
        "id": "S04_semantic_vague_browser",
        "description": "Vague semantic query + Browser library",
        "tool": "find_keywords",
        "kwargs": {"query": "do something with form", "strategy": "semantic", "library_name": "Browser"},
    },
    {
        "id": "S05_semantic_vague_nolib",
        "description": "Vague semantic query + NO library",
        "tool": "find_keywords",
        "kwargs": {"query": "do something with form", "strategy": "semantic"},
    },
    {
        "id": "S06_semantic_oneword_click",
        "description": "Single-word semantic query 'click' + Browser",
        "tool": "find_keywords",
        "kwargs": {"query": "click", "strategy": "semantic", "library_name": "Browser"},
    },
    {
        "id": "S07_semantic_oneword_navigate",
        "description": "Single-word semantic query 'navigate' + Browser",
        "tool": "find_keywords",
        "kwargs": {"query": "navigate", "strategy": "semantic", "library_name": "Browser"},
    },
    {
        "id": "S08_semantic_nonsense",
        "description": "Nonsense semantic query 'banana telephone'",
        "tool": "find_keywords",
        "kwargs": {"query": "banana telephone", "strategy": "semantic", "library_name": "Browser"},
    },
    {
        "id": "S09_semantic_empty",
        "description": "Empty-string semantic query",
        "tool": "find_keywords",
        "kwargs": {"query": "", "strategy": "semantic", "library_name": "Browser"},
    },
    {
        "id": "S10_semantic_cross_domain",
        "description": "Cross-domain query 'send http post request' + Browser (mismatch)",
        "tool": "find_keywords",
        "kwargs": {"query": "send http post request with json body", "strategy": "semantic", "library_name": "Browser"},
    },
    {
        "id": "S11_semantic_api_query",
        "description": "API-domain query 'send http post request' + no filter",
        "tool": "find_keywords",
        "kwargs": {"query": "send http post request with json body", "strategy": "semantic"},
    },
    {
        "id": "S12_semantic_long_query",
        "description": "Long descriptive query + Browser",
        "tool": "find_keywords",
        "kwargs": {
            "query": "wait for the modal dialog to appear and then click the confirm button at the bottom of the form",
            "strategy": "semantic",
            "library_name": "Browser",
        },
    },
    {
        "id": "S13_semantic_bdd_prefixed",
        "description": "BDD-prefixed query 'When I click button' (prefix-strip path) + Browser",
        "tool": "find_keywords",
        "kwargs": {"query": "When I click submit button", "strategy": "semantic", "library_name": "Browser"},
    },

    # ---- find_keywords / pattern ----
    {
        "id": "S14_pattern_click_star",
        "description": "Pattern 'Click*' + Browser library",
        "tool": "find_keywords",
        "kwargs": {"query": "Click*", "strategy": "pattern", "library_name": "Browser"},
    },
    {
        "id": "S15_pattern_click_star_nolib",
        "description": "Pattern 'Click*' + no filter",
        "tool": "find_keywords",
        "kwargs": {"query": "Click*", "strategy": "pattern"},
    },
    {
        "id": "S16_pattern_get_star",
        "description": "Pattern 'Get*' + Browser",
        "tool": "find_keywords",
        "kwargs": {"query": "Get*", "strategy": "pattern", "library_name": "Browser"},
    },
    {
        "id": "S17_pattern_exact",
        "description": "Pattern exact 'Go To' + Browser",
        "tool": "find_keywords",
        "kwargs": {"query": "Go To", "strategy": "pattern", "library_name": "Browser"},
    },
    {
        "id": "S18_pattern_unmatchable",
        "description": "Pattern 'XYZNoSuchThing*' (zero matches)",
        "tool": "find_keywords",
        "kwargs": {"query": "XYZNoSuchThing*", "strategy": "pattern", "library_name": "Browser"},
    },

    # ---- find_keywords / catalog ----
    {
        "id": "S19_catalog_browser",
        "description": "Catalog full listing + Browser",
        "tool": "find_keywords",
        "kwargs": {"query": "", "strategy": "catalog", "library_name": "Browser", "limit": 20},
    },
    {
        "id": "S20_catalog_no_session_no_lib",
        "description": "Catalog without session OR library (empty path)",
        "tool": "find_keywords",
        "kwargs": {"query": "", "strategy": "catalog"},
    },
    {
        "id": "S21_catalog_filtered_keyword",
        "description": "Catalog + query filter 'select' + Browser",
        "tool": "find_keywords",
        "kwargs": {"query": "select", "strategy": "catalog", "library_name": "Browser", "limit": 20},
    },

    # ---- get_keyword_info ----
    {
        "id": "K01_known_browser_kw",
        "description": "Known Browser keyword 'Click'",
        "tool": "get_keyword_info",
        "kwargs": {"mode": "keyword", "keyword_name": "Click", "library_name": "Browser"},
    },
    {
        "id": "K02_known_sl_kw",
        "description": "Known SeleniumLibrary keyword 'Select From List By Label'",
        "tool": "get_keyword_info",
        "kwargs": {"mode": "keyword", "keyword_name": "Select From List By Label", "library_name": "SeleniumLibrary"},
    },
    {
        "id": "K03_ambiguous_no_lib",
        "description": "Ambiguous keyword 'Go To' WITHOUT library_name (exists in Browser + SL)",
        "tool": "get_keyword_info",
        "kwargs": {"mode": "keyword", "keyword_name": "Go To"},
    },
    {
        "id": "K04_unknown",
        "description": "Unknown keyword 'XYZNoSuchKeyword'",
        "tool": "get_keyword_info",
        "kwargs": {"mode": "keyword", "keyword_name": "XYZNoSuchKeyword"},
    },
    {
        "id": "K05_typo",
        "description": "Typo'd keyword 'Clikc' (fuzzy/no match)",
        "tool": "get_keyword_info",
        "kwargs": {"mode": "keyword", "keyword_name": "Clikc"},
    },
    {
        "id": "K06_library_mode",
        "description": "Library mode - get full Browser library doc",
        "tool": "get_keyword_info",
        "kwargs": {"mode": "library", "library_name": "Browser"},
    },
    {
        "id": "K07_parse_mode",
        "description": "Parse mode - parse Click(selector='css=button')",
        "tool": "get_keyword_info",
        "kwargs": {
            "mode": "parse",
            "keyword_name": "Click",
            "library_name": "Browser",
            "arguments": ["css=button#submit"],
        },
    },
    {
        "id": "K08_bdd_prefix",
        "description": "BDD-prefixed keyword name 'When Click'",
        "tool": "get_keyword_info",
        "kwargs": {"mode": "keyword", "keyword_name": "When Click", "library_name": "Browser"},
    },
]


async def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    results: List[ScenarioResult] = []
    for s in SCENARIOS:
        print(f"running {s['id']}: {s['description']}")
        if s["tool"] == "find_keywords":
            r = await run_find_keywords(s["id"], s["description"], **s["kwargs"])
        elif s["tool"] == "get_keyword_info":
            r = await run_get_keyword_info(s["id"], s["description"], **s["kwargs"])
        else:
            continue
        results.append(r)
        # Per-scenario JSON dump
        out_json = {
            "id": r.scenario_id,
            "description": r.description,
            "tool": r.tool,
            "inputs": r.inputs,
            "response": r.response,
            "inline_tokens": r.inline_tokens,
            "artifact_bytes": r.artifact_bytes,
            "artifact_tokens": r.artifact_tokens,
            "matches_top": r.matches_top,
            "matches_libs": r.matches_libs,
            "matches_count": r.matches_count,
            "recommendations": r.recommendations,
            "excluded_count": r.excluded_count,
            "excluded_alternatives_count": r.excluded_alternatives_count,
            "library_filter_source": r.library_filter_source,
            "error": r.error,
        }
        (OUT / f"{r.scenario_id}.json").write_text(
            json.dumps(out_json, indent=2, default=str),
            encoding="utf-8",
        )
        if r.artifact_content:
            (OUT / f"{r.scenario_id}.artifact.txt").write_text(
                r.artifact_content, encoding="utf-8",
            )

    summary = {
        "total_scenarios": len(results),
        "scenarios": [
            {
                "id": r.scenario_id,
                "description": r.description,
                "tool": r.tool,
                "inline_tokens": r.inline_tokens,
                "artifact_tokens": r.artifact_tokens,
                "total_tokens": r.inline_tokens + r.artifact_tokens,
                "matches_top": (
                    f"{r.matches_top.get('library')}.{r.matches_top.get('keyword_name')}"
                    if r.matches_top else None
                ),
                "matches_count": r.matches_count,
                "matches_libs": r.matches_libs,
                "recommendations_top": r.recommendations[0] if r.recommendations else None,
                "excluded_count": r.excluded_count,
                "excluded_alternatives_count": r.excluded_alternatives_count,
                "library_filter_source": r.library_filter_source,
                "error": r.error,
            }
            for r in results
        ],
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8",
    )
    print(f"\n=== Benchmark complete: {len(results)} scenarios → {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
