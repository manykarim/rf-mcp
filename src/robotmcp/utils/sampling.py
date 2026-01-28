"""LLM-powered sampling functions for scenario analysis and library recommendation.

This module implements the LLM-based approach that scored 86.9% in our experiment
vs 70.2% for rule-based NLP. Functions use MCP ctx.sample() to enhance scenario
analysis, library recommendation, session type detection, and library preference
detection.

All functions are designed as optional enhancements that gracefully fall back to
None on any failure, allowing the caller to use rule-based results instead.

Feature flag: Set ROBOTMCP_USE_SAMPLING=true|1|yes to enable.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def is_sampling_enabled() -> bool:
    """Check if LLM sampling is enabled via ROBOTMCP_USE_SAMPLING env var.

    Supports values: "true", "1", "yes" (case-insensitive).
    Default: disabled.

    Returns:
        True if sampling is enabled, False otherwise.
    """
    val = os.environ.get("ROBOTMCP_USE_SAMPLING", "").lower().strip()
    return val in ("true", "1", "yes")


def _build_system_prompt() -> str:
    """Build the system prompt for all sampling calls.

    Returns:
        System prompt string instructing the LLM to respond as a Robot Framework
        expert with valid JSON only.
    """
    return (
        "You are a Robot Framework test automation expert. "
        "Always respond with valid JSON only. No markdown formatting."
    )


def _parse_json_response(text: str) -> Optional[Union[dict, list]]:
    """Parse a JSON response from an LLM, stripping markdown code fences if present.

    Handles common LLM output patterns:
    - Raw JSON
    - JSON wrapped in ```json ... ``` fences
    - JSON wrapped in ``` ... ``` fences
    - Leading/trailing whitespace

    Args:
        text: Raw LLM response text.

    Returns:
        Parsed JSON as dict or list, or None if parsing fails.
    """
    if not text:
        return None

    cleaned = text.strip()

    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    fence_pattern = re.compile(
        r"^```(?:json)?\s*\n?(.*?)\n?\s*```$", re.DOTALL
    )
    match = fence_pattern.match(cleaned)
    if match:
        cleaned = match.group(1).strip()

    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning("Failed to parse JSON from LLM response: %s", e)
        logger.debug("Raw response text: %s", text[:500])
        return None


async def sample_analyze_scenario(
    ctx, scenario: str, context: str
) -> Optional[Dict[str, Any]]:
    """Use MCP ctx.sample() to enhance scenario analysis.

    Called AFTER the rule-based analysis runs, to refine session type, detected
    context, primary library, and library preference. Returns None on any failure
    so the caller can fall back to rule-based results.

    Args:
        ctx: FastMCP Context object with sample() method.
        scenario: Natural language test scenario description.
        context: Testing context hint (e.g. "web", "api", "mobile").

    Returns:
        Dict with keys: session_type, detected_context, primary_library,
        library_preference. Or None on failure.
    """
    if not ctx or not hasattr(ctx, "sample"):
        return None

    prompt = f"""Analyze the following Robot Framework test scenario and determine the best configuration.

Scenario: {scenario}
Context hint: {context}

Valid session_type values: "web_automation", "api_testing", "mobile_testing", "system_testing", "xml_processing", "database_testing", "data_processing", "visual_testing", "mixed", "unknown"
Valid detected_context values: "web", "api", "mobile", "system", "database", "generic"

Key Robot Framework libraries:
- Browser (Playwright-based, modern web automation - preferred for new web tests)
- SeleniumLibrary (Selenium-based, traditional web automation)
- RequestsLibrary (HTTP/REST API testing)
- AppiumLibrary (mobile app testing)
- DatabaseLibrary (database testing)
- XML (XML processing)
- SSHLibrary (remote server commands)
- BuiltIn, Collections, String (core utilities)

Rules:
1. Browser and SeleniumLibrary are mutually exclusive - never recommend both
2. Prefer Browser over SeleniumLibrary unless Selenium is explicitly requested
3. If the scenario explicitly mentions a specific library by name, set library_preference to that library
4. If no explicit library is mentioned, set library_preference to null

Respond with a JSON object:
{{
    "session_type": "<one of the valid session_type values>",
    "detected_context": "<one of the valid detected_context values>",
    "primary_library": "<best Robot Framework library name>",
    "library_preference": "<explicitly mentioned library name or null>"
}}"""

    try:
        result = await ctx.sample(
            prompt,
            system_prompt=_build_system_prompt(),
            temperature=0.1,
            max_tokens=300,
        )

        response_text = result.text if hasattr(result, "text") else (result if isinstance(result, str) else str(result))
        parsed = _parse_json_response(response_text)

        if not isinstance(parsed, dict):
            logger.warning(
                "sample_analyze_scenario: expected dict, got %s",
                type(parsed).__name__,
            )
            return None

        # Validate required keys
        required_keys = {
            "session_type",
            "detected_context",
            "primary_library",
        }
        if not required_keys.issubset(parsed.keys()):
            missing = required_keys - parsed.keys()
            logger.warning(
                "sample_analyze_scenario: missing keys: %s", missing
            )
            return None

        return parsed

    except Exception as e:
        logger.warning("sample_analyze_scenario failed: %s", e)
        return None


async def sample_recommend_libraries(
    ctx,
    scenario: str,
    context: str,
    rule_based_recommendations: List[Dict],
) -> Optional[List[Dict]]:
    """Use MCP ctx.sample() to generate better library recommendations.

    This is the full LLM approach (not just refinement). It considers the
    scenario, context, and existing rule-based recommendations as input
    to produce a more accurate set of library recommendations.

    Args:
        ctx: FastMCP Context object with sample() method.
        scenario: Natural language test scenario description.
        context: Testing context hint (e.g. "web", "api", "mobile").
        rule_based_recommendations: List of dicts from the rule-based
            recommender, each with at least "library_name" and "confidence".

    Returns:
        List of recommendation dicts with "library_name", "confidence", and
        "rationale" keys. Or None on failure.
    """
    if not ctx or not hasattr(ctx, "sample"):
        return None

    # Format rule-based recommendations for context
    existing_libs = []
    for rec in rule_based_recommendations:
        name = rec.get("library_name", "unknown")
        conf = rec.get("confidence", 0)
        existing_libs.append(f"- {name} (confidence: {conf:.2f})")

    existing_libs_text = "\n".join(existing_libs) if existing_libs else "None"

    prompt = f"""Recommend the best Robot Framework libraries for this test scenario.

Scenario: {scenario}
Context: {context}

Rule-based system suggested:
{existing_libs_text}

Available Robot Framework libraries:
- Browser: Modern Playwright-based web automation (preferred for web)
- SeleniumLibrary: Traditional Selenium-based web automation
- RequestsLibrary: HTTP/REST API testing
- AppiumLibrary: Mobile app testing (Android/iOS)
- DatabaseLibrary: Database testing (SQL)
- SSHLibrary: Remote server commands via SSH
- XML: XML file processing and validation
- BuiltIn: Core Robot Framework keywords (always available)
- Collections: List and dictionary operations
- String: String manipulation
- DateTime: Date and time operations
- OperatingSystem: File system and process operations
- Process: Process execution and management
- Screenshot: Screen capture

STRICT RULES:
1. NEVER include both Browser and SeleniumLibrary - they are mutually exclusive
2. Prefer Browser over SeleniumLibrary unless Selenium is explicitly requested in the scenario
3. Only include libraries actually needed for the scenario
4. Assign realistic confidence scores (0.0 to 1.0)

Respond with a JSON array of objects:
[
    {{"library_name": "<name>", "confidence": <0.0-1.0>, "rationale": "<why>"}}
]"""

    try:
        result = await ctx.sample(
            prompt,
            system_prompt=_build_system_prompt(),
            temperature=0.1,
            max_tokens=500,
        )

        response_text = result.text if hasattr(result, "text") else (result if isinstance(result, str) else str(result))
        parsed = _parse_json_response(response_text)

        if not isinstance(parsed, list):
            logger.warning(
                "sample_recommend_libraries: expected list, got %s",
                type(parsed).__name__,
            )
            return None

        # Validate and normalize each recommendation
        validated = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            lib_name = item.get("library_name")
            if not lib_name:
                continue
            confidence = item.get("confidence", 0.5)
            if not isinstance(confidence, (int, float)):
                confidence = 0.5
            confidence = max(0.0, min(1.0, float(confidence)))

            validated.append(
                {
                    "library_name": lib_name,
                    "confidence": confidence,
                    "rationale": item.get("rationale", ""),
                }
            )

        if not validated:
            logger.warning(
                "sample_recommend_libraries: no valid recommendations parsed"
            )
            return None

        # Final safety check: ensure Browser and SeleniumLibrary exclusion
        lib_names = [r["library_name"] for r in validated]
        if "Browser" in lib_names and "SeleniumLibrary" in lib_names:
            logger.info(
                "sample_recommend_libraries: enforcing Browser/SeleniumLibrary exclusion"
            )
            validated = [
                r
                for r in validated
                if r["library_name"] != "SeleniumLibrary"
            ]

        return validated

    except Exception as e:
        logger.warning("sample_recommend_libraries failed: %s", e)
        return None


async def sample_detect_library_preference(
    ctx, scenario: str
) -> Optional[str]:
    """Use MCP sampling to detect explicit library preference from scenario text.

    Detects when a user explicitly mentions wanting to use a specific Robot
    Framework library (e.g. "use Selenium", "with Browser library").

    Args:
        ctx: FastMCP Context object with sample() method.
        scenario: Natural language test scenario description.

    Returns:
        Library name string (e.g. "Browser", "SeleniumLibrary",
        "RequestsLibrary") if an explicit preference is detected.
        None if no preference detected or on failure.
    """
    if not ctx or not hasattr(ctx, "sample"):
        return None

    prompt = f"""Analyze this Robot Framework test scenario and determine if the user explicitly requests or prefers a specific library.

Scenario: {scenario}

Known Robot Framework libraries:
- Browser (also: "Playwright", "Browser Library")
- SeleniumLibrary (also: "Selenium", "WebDriver")
- RequestsLibrary (also: "Requests", "HTTP library")
- AppiumLibrary (also: "Appium", "mobile library")
- DatabaseLibrary (also: "Database Library", "SQL library")
- SSHLibrary (also: "SSH Library")
- XML (also: "XML Library")

Rules:
- Only detect EXPLICIT mentions or preferences (e.g. "use Selenium", "with Browser library", "using Playwright")
- Generic terms like "browser", "web", "click" do NOT indicate a library preference
- If no explicit library preference is found, respond with null

Respond with a JSON object:
{{"library_preference": "<library name or null>"}}"""

    try:
        result = await ctx.sample(
            prompt,
            system_prompt=_build_system_prompt(),
            temperature=0.1,
            max_tokens=100,
        )

        response_text = result.text if hasattr(result, "text") else (result if isinstance(result, str) else str(result))
        parsed = _parse_json_response(response_text)

        if not isinstance(parsed, dict):
            logger.warning(
                "sample_detect_library_preference: expected dict, got %s",
                type(parsed).__name__,
            )
            return None

        preference = parsed.get("library_preference")
        if preference is None or preference == "null":
            return None

        if not isinstance(preference, str) or not preference.strip():
            return None

        return preference.strip()

    except Exception as e:
        logger.warning("sample_detect_library_preference failed: %s", e)
        return None


async def sample_detect_session_type(
    ctx, scenario: str, context: str
) -> Optional[str]:
    """Use MCP sampling to detect session type from scenario text.

    Returns one of the SessionType enum values as a string.

    Args:
        ctx: FastMCP Context object with sample() method.
        scenario: Natural language test scenario description.
        context: Testing context hint (e.g. "web", "api", "mobile").

    Returns:
        Session type string (e.g. "web_automation", "api_testing") or None
        on failure.
    """
    if not ctx or not hasattr(ctx, "sample"):
        return None

    prompt = f"""Classify this Robot Framework test scenario into exactly one session type.

Scenario: {scenario}
Context hint: {context}

Valid session types (pick exactly one):
- "web_automation": Web browser testing (clicking, filling forms, navigating pages)
- "api_testing": HTTP/REST API testing (requests, responses, endpoints)
- "mobile_testing": Mobile app testing (Android/iOS, Appium, touch gestures)
- "system_testing": System/OS testing (files, processes, commands, SSH)
- "xml_processing": XML parsing, validation, transformation
- "database_testing": Database operations (SQL queries, records, tables)
- "data_processing": Data manipulation (lists, dictionaries, strings, CSV)
- "visual_testing": Visual comparison, image matching, screenshot testing
- "mixed": Multiple distinct testing types combined
- "unknown": Cannot determine from the scenario

Respond with a JSON object:
{{"session_type": "<one of the valid values above>"}}"""

    try:
        result = await ctx.sample(
            prompt,
            system_prompt=_build_system_prompt(),
            temperature=0.1,
            max_tokens=100,
        )

        response_text = result.text if hasattr(result, "text") else (result if isinstance(result, str) else str(result))
        parsed = _parse_json_response(response_text)

        if not isinstance(parsed, dict):
            logger.warning(
                "sample_detect_session_type: expected dict, got %s",
                type(parsed).__name__,
            )
            return None

        session_type = parsed.get("session_type")
        if not isinstance(session_type, str) or not session_type.strip():
            return None

        # Validate against known session types
        valid_types = {
            "web_automation",
            "api_testing",
            "mobile_testing",
            "system_testing",
            "xml_processing",
            "database_testing",
            "data_processing",
            "visual_testing",
            "mixed",
            "unknown",
        }

        session_type = session_type.strip().lower()
        if session_type not in valid_types:
            logger.warning(
                "sample_detect_session_type: invalid type '%s'",
                session_type,
            )
            return None

        return session_type

    except Exception as e:
        logger.warning("sample_detect_session_type failed: %s", e)
        return None
