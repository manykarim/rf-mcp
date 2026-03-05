#!/usr/bin/env python3
"""ADR-014 Memory Comparison: with vs without memory using opencode CLI.

Runs identical scenarios with ROBOTMCP_MEMORY_ENABLED=false (baseline)
and ROBOTMCP_MEMORY_ENABLED=true (treatment), comparing tool call count,
tokens, cost, duration, and success rate.

Usage:
    uv run python tests/e2e/run_memory_comparison.py
    uv run python tests/e2e/run_memory_comparison.py --scenario D
    OPENCODE_MODELS="openrouter/z-ai/glm-4.7-flash" uv run python tests/e2e/run_memory_comparison.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent.parent

# Reuse existing opencode JSON parsing infrastructure
sys.path.insert(0, str(ROOT))
from tests.e2e.test_intent_action_models import (
    ModelResult,
    ToolCall,
    _parse_opencode_json,
)

# ── Configuration ─────────────────────────────────────────────────

_DEFAULT_MODELS = ["openrouter/qwen/qwen3-coder"]
_env_models = os.getenv("OPENCODE_MODELS", "").strip()
MODELS = (
    [m.strip() for m in _env_models.split(",") if m.strip()]
    if _env_models
    else _DEFAULT_MODELS
)

MEMORY_TOOLS = frozenset(
    {
        "robotmcp_recall_step",
        "robotmcp_recall_fix",
        "robotmcp_recall_locator",
        "robotmcp_store_knowledge",
        "robotmcp_get_memory_status",
    }
)

METRICS_DIR = ROOT / "tests" / "e2e" / "metrics" / "memory_comparison"

# ── Prompts ───────────────────────────────────────────────────────

# Scenario A: Repeated Task (Step Recall)
SCENARIO_A_PROMPT = """\
Use RobotMCP to create and execute a BuiltIn keyword test using execute_step.

1. Call analyze_scenario with scenario="BuiltIn keyword test" and context="generic"
2. Call manage_session with action="init", libraries=["BuiltIn"], using the session_id from step 1
3. Use execute_step to run: Set Variable  ${greeting}  Hello World
4. Use execute_step to run: Should Be Equal  ${greeting}  Hello World
5. Use execute_step to run: Log  Test passed: ${greeting}
6. Call build_test_suite to generate the .robot file

Execute exactly these steps. Use execute_step for steps 3-5.
Pass session_id from step 1 to all subsequent calls.
"""

# Scenario B: Error Recovery
SCENARIO_B_SEED_PROMPT = """\
Use RobotMCP to test assertion error handling with BuiltIn library.

1. Call analyze_scenario with scenario="BuiltIn error handling test" and context="generic"
2. Call manage_session with action="init", libraries=["BuiltIn"], using session_id from step 1
3. Use execute_step: Set Variable  ${value}  42
4. Use execute_step: Should Be Equal As Integers  ${value}  99
   (This WILL fail - that's expected! Use raise_on_failure=false)
5. After the failure, use execute_step: Should Be Equal As Integers  ${value}  42
6. Use execute_step: Log  Recovery successful
7. Call build_test_suite

Execute all steps. The failure in step 4 is intentional - handle it.
Pass session_id from step 1 to all calls.
"""

SCENARIO_B_TEST_PROMPT = """\
Use RobotMCP to test integer assertions with BuiltIn library.

1. Call analyze_scenario with scenario="BuiltIn assertion test" and context="generic"
2. Call manage_session with action="init", libraries=["BuiltIn"], using session_id from step 1
3. Use execute_step: Set Variable  ${count}  10
4. Use execute_step: Should Be Equal As Integers  ${count}  99
   (This will fail - use raise_on_failure=false to handle it)
5. After the failure, verify ${count} equals 10 using Should Be Equal As Integers
6. Use execute_step: Log  Assertion test complete
7. Call build_test_suite

Execute all steps. Handle the expected failure in step 4.
Pass session_id from step 1 to all calls.
"""

# Scenario C: Keyword Discovery
SCENARIO_C_SEED_PROMPT = """\
Use RobotMCP to discover and use String manipulation keywords.

1. Call analyze_scenario with scenario="String manipulation test" and context="generic"
2. Call manage_session with action="init", libraries=["BuiltIn", "String"], using session_id from step 1
3. Call find_keywords with query="convert to uppercase"
4. Call find_keywords with query="concatenate strings"
5. Use execute_step: Convert To Upper Case  hello world  (assign_to="upper_result")
6. Use execute_step: Catenate  Hello  World  separator=
7. Call build_test_suite

Execute all steps. Pass session_id from step 1 to all calls.
"""

SCENARIO_C_TEST_PROMPT = """\
Use RobotMCP to work with text using String and BuiltIn libraries.

1. Call analyze_scenario with scenario="Text processing test" and context="generic"
2. Call manage_session with action="init", libraries=["BuiltIn", "String"], using session_id from step 1
3. Convert "test input" to uppercase using the appropriate keyword
4. Concatenate "Hello" and "World" using the appropriate keyword
5. Call build_test_suite

If you're unsure which keywords to use, try find_keywords to discover them.
You may also use recall_step if available to check for previously successful patterns.
Pass session_id from step 1 to all calls.
"""

# Scenario D: Simple Baseline (Control)
SCENARIO_D_PROMPT = """\
Use RobotMCP to run a single BuiltIn keyword.

1. Call analyze_scenario with scenario="Simple log test" and context="generic"
2. Call manage_session with action="init", libraries=["BuiltIn"], using session_id from step 1
3. Use execute_step: Log  Hello from E2E test
4. Call build_test_suite

Execute these 4 steps exactly. Pass session_id from step 1 to all calls.
"""

# ── Restful-Booker API Scenarios ────────────────────────────────

# Scenario H: API CRUD with Authentication (complex multi-step)
SCENARIO_H_SEED_PROMPT = """\
Use RobotMCP to test the Restful Booker API at https://restful-booker.herokuapp.com

1. Call analyze_scenario with scenario="Restful Booker API CRUD" and context="api"
2. Call manage_session with action="init", libraries=["RequestsLibrary", "BuiltIn", "Collections"], \
   using session_id from step 1
3. Use execute_step: Create Session  booker  https://restful-booker.herokuapp.com
4. Use execute_step to GET /booking and verify status code is 200
5. Use execute_step to POST /auth with body {"username": "admin", "password": "password123"} \
   and Content-Type application/json. Extract the token from the response.
6. Use execute_step to POST /booking with a JSON body containing: \
   {"firstname": "Jim", "lastname": "Brown", "totalprice": 111, "depositpaid": true, \
   "bookingdates": {"checkin": "2025-01-01", "checkout": "2025-01-02"}, "additionalneeds": "Breakfast"} \
   and Content-Type application/json. Extract the bookingid from the response.
7. Use execute_step to GET /booking/{bookingid} and verify the firstname is "Jim"
8. Call build_test_suite

Execute all steps. Pass session_id from step 1 to all calls.
"""

SCENARIO_H_TEST_PROMPT = """\
Use RobotMCP to test the Restful Booker API at https://restful-booker.herokuapp.com

1. Call analyze_scenario with scenario="Restful Booker API test" and context="api"
2. Call manage_session with action="init", libraries=["RequestsLibrary", "BuiltIn", "Collections"], \
   using session_id from step 1
3. Create an HTTP session to https://restful-booker.herokuapp.com
4. GET /booking to list bookings and verify 200 response
5. Authenticate with POST /auth (username: admin, password: password123)
6. Create a new booking via POST /booking with JSON body
7. Read back the created booking with GET /booking/{id} and verify the data
8. Call build_test_suite

If recall_step or recall_fix tools are available, use them first to check \
for previously successful API patterns.
Pass session_id from step 1 to all calls.
"""

# Scenario I: API Keyword Discovery (finding HTTP keywords)
SCENARIO_I_SEED_PROMPT = """\
Use RobotMCP to explore the RequestsLibrary and test various HTTP methods \
against https://restful-booker.herokuapp.com

1. Call analyze_scenario with scenario="HTTP method exploration" and context="api"
2. Call manage_session with action="init", libraries=["RequestsLibrary", "BuiltIn", "Collections"], \
   using session_id from step 1
3. Call find_keywords with query="GET request" to discover GET keywords
4. Call find_keywords with query="POST request" to discover POST keywords
5. Call find_keywords with query="response status" to find assertion keywords
6. Use execute_step: Create Session  booker  https://restful-booker.herokuapp.com
7. Use execute_step to GET /booking/1 and verify status code is 200
8. Use execute_step to POST /auth with {"username": "admin", "password": "password123"}
9. Call build_test_suite

Execute all steps. Pass session_id from step 1 to all calls.
"""

SCENARIO_I_TEST_PROMPT = """\
Use RobotMCP to test HTTP operations against https://restful-booker.herokuapp.com

1. Call analyze_scenario with scenario="API HTTP test" and context="api"
2. Call manage_session with action="init", libraries=["RequestsLibrary", "BuiltIn", "Collections"], \
   using session_id from step 1
3. Create an HTTP session to https://restful-booker.herokuapp.com
4. GET /booking/1 and verify the response status is 200
5. POST to /auth with username=admin, password=password123 and verify you get a token
6. Call build_test_suite

If recall_step tools are available, use them first to check for previously successful patterns.
If you need to discover keywords, use find_keywords.
Pass session_id from step 1 to all calls.
"""

# Scenario J: API Error Recovery (auth failure + retry)
SCENARIO_J_SEED_PROMPT = """\
Use RobotMCP to test error handling in the Restful Booker API.

1. Call analyze_scenario with scenario="API error handling" and context="api"
2. Call manage_session with action="init", libraries=["RequestsLibrary", "BuiltIn", "Collections"], \
   using session_id from step 1
3. Use execute_step: Create Session  booker  https://restful-booker.herokuapp.com
4. Use execute_step to POST /auth with WRONG credentials {"username": "wrong", "password": "wrong"} \
   and Content-Type application/json. The response should contain "Bad credentials" — verify this.
5. Use execute_step to POST /auth with CORRECT credentials {"username": "admin", "password": "password123"} \
   and Content-Type application/json. Verify you get a token.
6. Use execute_step to POST /booking with JSON body: \
   {"firstname": "Test", "lastname": "User", "totalprice": 100, "depositpaid": true, \
   "bookingdates": {"checkin": "2025-06-01", "checkout": "2025-06-02"}, "additionalneeds": "None"} \
   and Content-Type application/json.
7. Call build_test_suite

Execute all steps. Pass session_id from step 1 to all calls.
"""

SCENARIO_J_TEST_PROMPT = """\
Use RobotMCP to test authentication error handling with the Restful Booker API.

1. Call analyze_scenario with scenario="API auth error test" and context="api"
2. Call manage_session with action="init", libraries=["RequestsLibrary", "BuiltIn", "Collections"], \
   using session_id from step 1
3. Create an HTTP session to https://restful-booker.herokuapp.com
4. Try authenticating with wrong credentials (username: wrong, password: wrong) and verify the error
5. Authenticate with correct credentials (username: admin, password: password123) and verify success
6. Create a booking to confirm the auth token works
7. Call build_test_suite

If recall_step or recall_fix tools are available, use them first to check \
for previously successful patterns and known error fixes.
Pass session_id from step 1 to all calls.
"""


# ── CarConfig Web Scenarios ─────────────────────────────────────

# Scenario K: CarConfig Demo Login + Navigation (locator-heavy, German text)
SCENARIO_K_SEED_PROMPT = """\
Use RobotMCP to test the car configurator at https://carconfig.makrocode.de/

1. Call analyze_scenario with scenario="CarConfig demo login" and context="web"
2. Call manage_session with action="init", libraries=["Browser", "BuiltIn"], \
   using session_id from step 1. Use headless=true.
3. Use execute_step: New Browser  browser=chromium  headless=true
4. Use execute_step: New Page  https://carconfig.makrocode.de/auth/login
5. Use get_session_state with sections=["page_source"] and include_reduced_dom=true \
   to inspect the login page and find the form locators
6. Click the "Demo-Modus verwenden" button to log in without credentials
7. Verify the page navigated away from the login page (check URL or page content)
8. Click the navigation link to go to the "Konfigurator" page
9. Verify the configurator page loaded (look for vehicle categories like "Luxus-Limousine")
10. Call build_test_suite

You MUST inspect the DOM (step 5) before clicking. Use the actual locators from the DOM.
Pass session_id from step 1 to all calls.
"""

SCENARIO_K_TEST_PROMPT = """\
Use RobotMCP to test the car configurator at https://carconfig.makrocode.de/

1. Call analyze_scenario with scenario="CarConfig demo login test" and context="web"
2. Call manage_session with action="init", libraries=["Browser", "BuiltIn"], \
   using session_id from step 1. Use headless=true.
3. Open a chromium browser (headless) and navigate to https://carconfig.makrocode.de/auth/login
4. Click the "Demo-Modus verwenden" button to log in
5. Navigate to the Konfigurator page
6. Verify the configurator page shows vehicle categories
7. Call build_test_suite

If recall_step or recall_locator tools are available, use them first to check \
for previously successful locator strategies before inspecting the DOM.
Pass session_id from step 1 to all calls.
"""

# Scenario L: CarConfig Vehicle Configuration (multi-step option selection)
SCENARIO_L_SEED_PROMPT = """\
Use RobotMCP to configure a vehicle at https://carconfig.makrocode.de/

1. Call analyze_scenario with scenario="CarConfig vehicle configuration" and context="web"
2. Call manage_session with action="init", libraries=["Browser", "BuiltIn"], \
   using session_id from step 1. Use headless=true.
3. Use execute_step: New Browser  browser=chromium  headless=true
4. Use execute_step: New Page  https://carconfig.makrocode.de/configurator
5. Use get_session_state with sections=["page_source"] and include_reduced_dom=true \
   to inspect the page and find vehicle category locators
6. Click on the "Luxus-Limousine" vehicle card to select it
7. Inspect the DOM again to find the option category sections
8. Click on the "Motor" option category
9. Select an engine option (toggle a checkbox or click an option card)
10. Click on "Innenausstattung" (Interior) category
11. Select an interior option
12. Verify the price display shows updated totals (look for "Gesamtpreis")
13. Call build_test_suite

You MUST inspect the DOM before each new interaction.
Pass session_id from step 1 to all calls.
"""

SCENARIO_L_TEST_PROMPT = """\
Use RobotMCP to configure a vehicle at https://carconfig.makrocode.de/

1. Call analyze_scenario with scenario="CarConfig vehicle config test" and context="web"
2. Call manage_session with action="init", libraries=["Browser", "BuiltIn"], \
   using session_id from step 1. Use headless=true.
3. Open a chromium browser (headless) and navigate to https://carconfig.makrocode.de/configurator
4. Select the "Luxus-Limousine" vehicle
5. Open the "Motor" category and select an engine option
6. Open the "Innenausstattung" category and select an interior option
7. Verify the price summary shows updated totals
8. Call build_test_suite

If recall_step or recall_locator tools are available, use them first to check \
for previously successful locator strategies and navigation patterns.
Pass session_id from step 1 to all calls.
"""

# Scenario M: CarConfig Registration Form (form-heavy, many locators)
SCENARIO_M_SEED_PROMPT = """\
Use RobotMCP to test the registration form at https://carconfig.makrocode.de/

1. Call analyze_scenario with scenario="CarConfig registration form" and context="web"
2. Call manage_session with action="init", libraries=["Browser", "BuiltIn"], \
   using session_id from step 1. Use headless=true.
3. Use execute_step: New Browser  browser=chromium  headless=true
4. Use execute_step: New Page  https://carconfig.makrocode.de/auth/register
5. Use get_session_state with sections=["page_source"] and include_reduced_dom=true \
   to inspect the registration page and find all form field locators
6. Fill in "Vollständiger Name" with: Test Benutzer
7. Fill in "Email-Adresse" with: test.benutzer@example.com
8. Fill in "Passwort" with: TestPasswort123!
9. Fill in "Passwort bestätigen" with: TestPasswort123!
10. Verify all fields are filled (check values)
11. Do NOT submit the form. Just verify the fields are correctly filled.
12. Call build_test_suite

You MUST inspect the DOM (step 5) before filling any fields. Use the actual locators.
Pass session_id from step 1 to all calls.
"""

SCENARIO_M_TEST_PROMPT = """\
Use RobotMCP to test the registration form at https://carconfig.makrocode.de/

1. Call analyze_scenario with scenario="CarConfig registration test" and context="web"
2. Call manage_session with action="init", libraries=["Browser", "BuiltIn"], \
   using session_id from step 1. Use headless=true.
3. Open a chromium browser (headless) and navigate to https://carconfig.makrocode.de/auth/register
4. Fill in the registration form: name "Test Benutzer", email "test.benutzer@example.com", \
   password "TestPasswort123!", confirm password "TestPasswort123!"
5. Verify all fields contain the correct values
6. Do NOT submit. Just verify the form is filled.
7. Call build_test_suite

If recall_step or recall_locator tools are available, use them first to check \
for previously successful form locators and fill patterns.
Pass session_id from step 1 to all calls.
"""


# ── Demoshop Web Scenarios ───────────────────────────────────────

# Scenario E: Demoshop Add to Cart (web locator discovery)
SCENARIO_E_SEED_PROMPT = """\
Use RobotMCP to test adding a product to the shopping cart on https://demoshop.makrocode.de/

1. Call analyze_scenario with scenario="Demoshop add to cart" and context="web"
2. Call manage_session with action="init", libraries=["Browser", "BuiltIn"], \
   using session_id from step 1. Use headless=true for the browser.
3. Use execute_step: New Browser  browser=chromium  headless=true
4. Use execute_step: New Page  https://demoshop.makrocode.de/
5. Use get_session_state with sections=["page_source"] and include_reduced_dom=true \
   to inspect the page and find product locators
6. Click the "Add to cart" button for the first product you find in the DOM
7. Verify the cart badge shows "1" item
8. Call build_test_suite

You MUST inspect the DOM (step 5) before clicking. Use the actual locators from the DOM.
Pass session_id from step 1 to all calls.
"""

SCENARIO_E_TEST_PROMPT = """\
Use RobotMCP to test adding a product to the shopping cart on https://demoshop.makrocode.de/

1. Call analyze_scenario with scenario="Demoshop add to cart test" and context="web"
2. Call manage_session with action="init", libraries=["Browser", "BuiltIn"], \
   using session_id from step 1. Use headless=true for the browser.
3. Open a chromium browser (headless) and navigate to https://demoshop.makrocode.de/
4. Find a product on the page and click its "Add to cart" button
5. Verify the cart count shows 1 item
6. Call build_test_suite

If recall_step or recall_locator tools are available, use them first to check \
for previously successful locator strategies before inspecting the DOM.
Pass session_id from step 1 to all calls.
"""

# Scenario F: Demoshop Full Checkout (complex multi-step web flow)
SCENARIO_F_SEED_PROMPT = """\
Use RobotMCP to test a complete checkout flow on https://demoshop.makrocode.de/

1. Call analyze_scenario with scenario="Demoshop checkout flow" and context="web"
2. Call manage_session with action="init", libraries=["Browser", "BuiltIn"], \
   using session_id from step 1. Use headless=true.
3. Use execute_step: New Browser  browser=chromium  headless=true
4. Use execute_step: New Page  https://demoshop.makrocode.de/
5. Inspect the DOM with get_session_state (include_reduced_dom=true) to find locators
6. Add the "Cascade Water Bottle" to cart by clicking its Add to cart button
7. Verify cart count is 1
8. Click the Checkout link/button
9. Inspect the checkout form DOM to find input field locators
10. Fill in email: tester@example.com, name: Test User, address: 123 Test St
11. Click "Place order"
12. Verify the order confirmation message appears
13. Call build_test_suite

You MUST inspect the DOM before interacting with elements.
Pass session_id from step 1 to all calls.
"""

SCENARIO_F_TEST_PROMPT = """\
Use RobotMCP to test a complete checkout on https://demoshop.makrocode.de/

1. Call analyze_scenario with scenario="Demoshop checkout test" and context="web"
2. Call manage_session with action="init", libraries=["Browser", "BuiltIn"], \
   using session_id from step 1. Use headless=true.
3. Open a chromium browser (headless) and navigate to https://demoshop.makrocode.de/
4. Add a product to the cart
5. Go to checkout
6. Fill in the checkout form (email, name, address)
7. Place the order
8. Verify the order confirmation
9. Call build_test_suite

If recall_step, recall_locator, or recall_fix tools are available, use them first to check \
for previously successful step sequences and locator strategies.
Pass session_id from step 1 to all calls.
"""

# Scenario G: Demoshop Product Navigation (exploration-heavy)
SCENARIO_G_SEED_PROMPT = """\
Use RobotMCP to test product browsing on https://demoshop.makrocode.de/

1. Call analyze_scenario with scenario="Demoshop product browsing" and context="web"
2. Call manage_session with action="init", libraries=["Browser", "BuiltIn"], \
   using session_id from step 1. Use headless=true.
3. Use execute_step: New Browser  browser=chromium  headless=true
4. Use execute_step: New Page  https://demoshop.makrocode.de/
5. Inspect the DOM to find navigation and product locators
6. Count the number of products visible on the page
7. Click on the first product to open its detail page
8. Verify the product detail page loads (check for product title or price)
9. Go back to the main page
10. Call build_test_suite

Inspect the DOM before each interaction.
Pass session_id from step 1 to all calls.
"""

SCENARIO_G_TEST_PROMPT = """\
Use RobotMCP to test product browsing on https://demoshop.makrocode.de/

1. Call analyze_scenario with scenario="Demoshop browse products" and context="web"
2. Call manage_session with action="init", libraries=["Browser", "BuiltIn"], \
   using session_id from step 1. Use headless=true.
3. Open a chromium browser (headless) and navigate to https://demoshop.makrocode.de/
4. Count the products on the page
5. Open one product's detail page
6. Verify product details are shown
7. Navigate back to the main page
8. Call build_test_suite

If recall_step or recall_locator tools are available, use them first to check \
for previously successful navigation patterns and locators.
Pass session_id from step 1 to all calls.
"""


# Scenario N: Demoshop Multi-Product Cart (repeating same locator pattern)
SCENARIO_N_SEED_PROMPT = """\
Use RobotMCP to add multiple products to the cart on https://demoshop.makrocode.de/

1. Call analyze_scenario with scenario="Demoshop multi-product cart" and context="web"
2. Call manage_session with action="init", libraries=["Browser", "BuiltIn"], \
   using session_id from step 1. Use headless=true.
3. Use execute_step: New Browser  browser=chromium  headless=true
4. Use execute_step: New Page  https://demoshop.makrocode.de/
5. Use get_session_state with sections=["page_source"] and include_reduced_dom=true \
   to inspect the page and find product card locators and "Add to cart" buttons
6. Add "Cascade Water Bottle" to cart by clicking its Add to cart button
7. Add "Focus Loop Timer" to cart by clicking its Add to cart button
8. Add "Pulse Bio Ring" to cart by clicking its Add to cart button
9. Verify the cart badge shows "3" items
10. Click on "Checkout" or navigate to the cart page
11. Call build_test_suite

You MUST inspect the DOM (step 5) before clicking. Use the actual locators.
The "Add to cart" buttons are inside article elements containing the product name.
Pass session_id from step 1 to all calls.
"""

SCENARIO_N_TEST_PROMPT = """\
Use RobotMCP to add multiple products to the cart on https://demoshop.makrocode.de/

1. Call analyze_scenario with scenario="Demoshop multi-product cart test" and context="web"
2. Call manage_session with action="init", libraries=["Browser", "BuiltIn"], \
   using session_id from step 1. Use headless=true.
3. Open a chromium browser (headless) and navigate to https://demoshop.makrocode.de/
4. Add "Cascade Water Bottle" to cart
5. Add "Focus Loop Timer" to cart
6. Add "Pulse Bio Ring" to cart
7. Verify the cart shows 3 items
8. Navigate to checkout
9. Call build_test_suite

If recall_step or recall_locator tools are available, use them first to check \
for previously successful "Add to cart" locator patterns — they use the same pattern \
for all products (article:has-text("Product Name") >> button:has-text("Add to cart")).
Pass session_id from step 1 to all calls.
"""


# ── Data Model ────────────────────────────────────────────────────


@dataclass
class RunMetrics:
    """Metrics from a single opencode run."""

    scenario_id: str
    run_label: str  # "baseline", "seed", "benefit", "control_no_mem", "control_mem"
    memory_enabled: bool
    model: str
    total_tool_calls: int = 0
    tool_counts: dict[str, int] = field(default_factory=dict)
    memory_tool_calls: int = 0
    successful_tool_calls: int = 0
    failed_tool_calls: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    duration_seconds: float = 0.0
    error: str | None = None
    has_recall_calls: bool = False
    has_store_calls: bool = False
    raw_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "run_label": self.run_label,
            "memory_enabled": self.memory_enabled,
            "model": self.model,
            "total_tool_calls": self.total_tool_calls,
            "tool_counts": self.tool_counts,
            "memory_tool_calls": self.memory_tool_calls,
            "successful_tool_calls": self.successful_tool_calls,
            "failed_tool_calls": self.failed_tool_calls,
            "total_tokens": self.total_tokens,
            "cost": self.cost,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
            "has_recall_calls": self.has_recall_calls,
            "has_store_calls": self.has_store_calls,
        }


@dataclass
class ScenarioComparison:
    """Comparison between baseline and benefit runs."""

    scenario_id: str
    description: str
    baseline: RunMetrics
    seed: RunMetrics | None
    benefit: RunMetrics
    tool_call_delta: float = 0.0  # negative = improvement
    token_delta: float = 0.0
    time_delta: float = 0.0
    benefit_detected: bool = False
    memory_tools_used: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "description": self.description,
            "runs": {
                "baseline": self.baseline.to_dict(),
                **({"seed": self.seed.to_dict()} if self.seed else {}),
                "benefit": self.benefit.to_dict(),
            },
            "comparison": {
                "tool_call_delta": self.tool_call_delta,
                "token_delta": self.token_delta,
                "time_delta": self.time_delta,
                "benefit_detected": self.benefit_detected,
                "memory_tools_used": self.memory_tools_used,
            },
        }


@dataclass
class IterationStats:
    """Statistical summary across multiple iterations of a scenario."""

    scenario_id: str
    description: str
    iterations: int
    # Per-iteration raw comparisons
    comparisons: list[ScenarioComparison] = field(default_factory=list)
    # Aggregated stats: mean and stddev
    baseline_calls_mean: float = 0.0
    baseline_calls_std: float = 0.0
    benefit_calls_mean: float = 0.0
    benefit_calls_std: float = 0.0
    baseline_tokens_mean: float = 0.0
    baseline_tokens_std: float = 0.0
    benefit_tokens_mean: float = 0.0
    benefit_tokens_std: float = 0.0
    baseline_time_mean: float = 0.0
    baseline_time_std: float = 0.0
    benefit_time_mean: float = 0.0
    benefit_time_std: float = 0.0
    baseline_cost_mean: float = 0.0
    benefit_cost_mean: float = 0.0
    # Deltas
    call_delta_mean: float = 0.0
    call_delta_std: float = 0.0
    token_delta_mean: float = 0.0
    token_delta_std: float = 0.0
    time_delta_mean: float = 0.0
    time_delta_std: float = 0.0
    # Counts
    benefit_detected_count: int = 0
    memory_recall_used_count: int = 0
    # 95% confidence interval for call delta
    call_delta_ci_low: float = 0.0
    call_delta_ci_high: float = 0.0
    token_delta_ci_low: float = 0.0
    token_delta_ci_high: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "description": self.description,
            "iterations": self.iterations,
            "baseline": {
                "calls": {"mean": self.baseline_calls_mean, "std": self.baseline_calls_std},
                "tokens": {"mean": self.baseline_tokens_mean, "std": self.baseline_tokens_std},
                "time": {"mean": self.baseline_time_mean, "std": self.baseline_time_std},
                "cost_mean": self.baseline_cost_mean,
            },
            "benefit": {
                "calls": {"mean": self.benefit_calls_mean, "std": self.benefit_calls_std},
                "tokens": {"mean": self.benefit_tokens_mean, "std": self.benefit_tokens_std},
                "time": {"mean": self.benefit_time_mean, "std": self.benefit_time_std},
                "cost_mean": self.benefit_cost_mean,
            },
            "deltas": {
                "calls": {"mean": self.call_delta_mean, "std": self.call_delta_std, "ci95": [self.call_delta_ci_low, self.call_delta_ci_high]},
                "tokens": {"mean": self.token_delta_mean, "std": self.token_delta_std, "ci95": [self.token_delta_ci_low, self.token_delta_ci_high]},
                "time": {"mean": self.time_delta_mean, "std": self.time_delta_std},
            },
            "benefit_detected_count": self.benefit_detected_count,
            "memory_recall_used_count": self.memory_recall_used_count,
            "per_iteration": [c.to_dict() for c in self.comparisons],
        }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def _ci95(values: list[float]) -> tuple[float, float]:
    """95% confidence interval using t-distribution approximation."""
    n = len(values)
    if n < 2:
        m = _mean(values)
        return (m, m)
    m = _mean(values)
    s = _stddev(values)
    # t-value for 95% CI (two-tailed) — approximate for small n
    t_values = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}
    t = t_values.get(n, 1.96)
    margin = t * s / math.sqrt(n)
    return (m - margin, m + margin)


def _compute_iteration_stats(
    scenario_id: str,
    description: str,
    comparisons: list[ScenarioComparison],
) -> IterationStats:
    """Compute statistical summary from multiple iteration comparisons."""
    n = len(comparisons)
    stats = IterationStats(
        scenario_id=scenario_id,
        description=description,
        iterations=n,
        comparisons=comparisons,
    )

    base_calls = [float(c.baseline.total_tool_calls) for c in comparisons]
    ben_calls = [float(c.benefit.total_tool_calls) for c in comparisons]
    base_tokens = [float(c.baseline.total_tokens) for c in comparisons]
    ben_tokens = [float(c.benefit.total_tokens) for c in comparisons]
    base_time = [c.baseline.duration_seconds for c in comparisons]
    ben_time = [c.benefit.duration_seconds for c in comparisons]
    call_deltas = [c.tool_call_delta for c in comparisons]
    tok_deltas = [c.token_delta for c in comparisons]
    time_deltas = [c.time_delta for c in comparisons]

    stats.baseline_calls_mean = _mean(base_calls)
    stats.baseline_calls_std = _stddev(base_calls)
    stats.benefit_calls_mean = _mean(ben_calls)
    stats.benefit_calls_std = _stddev(ben_calls)
    stats.baseline_tokens_mean = _mean(base_tokens)
    stats.baseline_tokens_std = _stddev(base_tokens)
    stats.benefit_tokens_mean = _mean(ben_tokens)
    stats.benefit_tokens_std = _stddev(ben_tokens)
    stats.baseline_time_mean = _mean(base_time)
    stats.baseline_time_std = _stddev(base_time)
    stats.benefit_time_mean = _mean(ben_time)
    stats.benefit_time_std = _stddev(ben_time)
    stats.baseline_cost_mean = _mean([c.baseline.cost for c in comparisons])
    stats.benefit_cost_mean = _mean([c.benefit.cost for c in comparisons])
    stats.call_delta_mean = _mean(call_deltas)
    stats.call_delta_std = _stddev(call_deltas)
    stats.token_delta_mean = _mean(tok_deltas)
    stats.token_delta_std = _stddev(tok_deltas)
    stats.time_delta_mean = _mean(time_deltas)
    stats.time_delta_std = _stddev(time_deltas)
    stats.benefit_detected_count = sum(1 for c in comparisons if c.benefit_detected)
    stats.memory_recall_used_count = sum(1 for c in comparisons if c.benefit.has_recall_calls)

    ci_call = _ci95(call_deltas)
    stats.call_delta_ci_low, stats.call_delta_ci_high = ci_call
    ci_tok = _ci95(tok_deltas)
    stats.token_delta_ci_low, stats.token_delta_ci_high = ci_tok

    return stats


# ── Core Functions ────────────────────────────────────────────────


def _run_with_memory(
    model: str,
    prompt: str,
    *,
    memory_enabled: bool,
    db_path: str,
    timeout_seconds: int = 300,
) -> ModelResult:
    """Run opencode with memory toggled via environment variables."""
    env = os.environ.copy()
    env["ROBOTMCP_MEMORY_ENABLED"] = "true" if memory_enabled else "false"
    env["ROBOTMCP_MEMORY_DB_PATH"] = db_path
    env["ROBOTMCP_MEMORY_MODEL"] = "potion-base-8M"

    cmd = [
        "opencode",
        "run",
        "--format",
        "json",
        "-m",
        model,
        "--dir",
        str(ROOT),
        prompt,
    ]

    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=str(ROOT),
            env=env,
        )
        output = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        return ModelResult(
            model=model,
            error=f"Timeout after {timeout_seconds}s",
            duration_seconds=time.time() - start,
        )
    except Exception as e:
        return ModelResult(
            model=model,
            error=str(e),
            duration_seconds=time.time() - start,
        )

    result = _parse_opencode_json(output)
    result.model = model
    result.duration_seconds = time.time() - start
    return result


def _extract_metrics(
    result: ModelResult, scenario_id: str, run_label: str, memory_enabled: bool
) -> RunMetrics:
    """Extract structured metrics from a ModelResult."""
    tool_counts: dict[str, int] = {}
    mem_calls = 0
    recall_calls = False
    store_calls = False
    successful = 0
    failed = 0

    for tc in result.tool_calls:
        tool_counts[tc.name] = tool_counts.get(tc.name, 0) + 1
        if tc.name in MEMORY_TOOLS:
            mem_calls += 1
            if "recall" in tc.name:
                recall_calls = True
            if tc.name == "robotmcp_store_knowledge":
                store_calls = True
        if tc.success:
            successful += 1
        else:
            failed += 1

    return RunMetrics(
        scenario_id=scenario_id,
        run_label=run_label,
        memory_enabled=memory_enabled,
        model=result.model,
        total_tool_calls=len(result.tool_calls),
        tool_counts=tool_counts,
        memory_tool_calls=mem_calls,
        successful_tool_calls=successful,
        failed_tool_calls=failed,
        total_tokens=result.total_tokens,
        cost=result.cost,
        duration_seconds=result.duration_seconds,
        error=result.error,
        has_recall_calls=recall_calls,
        has_store_calls=store_calls,
        raw_text=result.text_output[:500],
    )


def _pct(baseline: float, treatment: float) -> float:
    """Calculate percentage change: negative = improvement."""
    if baseline == 0:
        return 0.0
    return (treatment - baseline) / baseline


def _run_scenario(
    scenario_id: str,
    description: str,
    model: str,
    db_base: Path,
    *,
    test_prompt: str,
    seed_prompt: str | None = None,
    timeout: int = 300,
) -> ScenarioComparison:
    """Run a full comparison scenario: baseline + seed (optional) + benefit."""
    db_path = str(db_base / f"{scenario_id}.db")

    # Clean DB for this scenario
    for ext in ("", "-wal", "-shm"):
        p = db_path + ext
        if os.path.exists(p):
            os.unlink(p)

    print(f"\n  [1/3] Baseline (no memory)...", flush=True)
    base_result = _run_with_memory(
        model, test_prompt, memory_enabled=False, db_path=db_path, timeout_seconds=timeout
    )
    baseline = _extract_metrics(base_result, scenario_id, "baseline", False)
    print(f"        {baseline.total_tool_calls} calls, {baseline.total_tokens} tokens, {baseline.duration_seconds:.1f}s")

    seed_metrics = None
    effective_seed_prompt = seed_prompt or test_prompt
    if seed_prompt is not None or scenario_id != "D_control":
        # Clean DB before seed
        for ext in ("", "-wal", "-shm"):
            p = db_path + ext
            if os.path.exists(p):
                os.unlink(p)

        print(f"  [2/3] Seed (memory ON, empty DB)...", flush=True)
        seed_result = _run_with_memory(
            model, effective_seed_prompt, memory_enabled=True, db_path=db_path, timeout_seconds=timeout
        )
        seed_metrics = _extract_metrics(seed_result, scenario_id, "seed", True)
        print(f"        {seed_metrics.total_tool_calls} calls, {seed_metrics.total_tokens} tokens, {seed_metrics.duration_seconds:.1f}s")

    print(f"  [3/3] Benefit (memory ON, warm DB)...", flush=True)
    benefit_result = _run_with_memory(
        model, test_prompt, memory_enabled=True, db_path=db_path, timeout_seconds=timeout
    )
    benefit = _extract_metrics(benefit_result, scenario_id, "benefit", True)
    print(f"        {benefit.total_tool_calls} calls, {benefit.total_tokens} tokens, {benefit.duration_seconds:.1f}s")

    # Compute deltas
    tc_delta = _pct(baseline.total_tool_calls, benefit.total_tool_calls)
    tok_delta = _pct(baseline.total_tokens, benefit.total_tokens)
    time_delta = _pct(baseline.duration_seconds, benefit.duration_seconds)

    mem_tools_used = sorted(
        {tc.name for tc in benefit_result.tool_calls if tc.name in MEMORY_TOOLS}
    )

    return ScenarioComparison(
        scenario_id=scenario_id,
        description=description,
        baseline=baseline,
        seed=seed_metrics,
        benefit=benefit,
        tool_call_delta=tc_delta,
        token_delta=tok_delta,
        time_delta=time_delta,
        benefit_detected=(
            tc_delta < -0.05 or tok_delta < -0.05 or len(mem_tools_used) > 0
        ),
        memory_tools_used=mem_tools_used,
    )


def _run_control_scenario(
    model: str, db_base: Path, *, timeout: int = 300
) -> ScenarioComparison:
    """Run Scenario D (control): baseline vs memory ON with empty DB."""
    db_path = str(db_base / "D_control.db")

    for ext in ("", "-wal", "-shm"):
        p = db_path + ext
        if os.path.exists(p):
            os.unlink(p)

    print(f"\n  [1/2] Baseline (no memory)...", flush=True)
    base_result = _run_with_memory(
        model, SCENARIO_D_PROMPT, memory_enabled=False, db_path=db_path, timeout_seconds=timeout
    )
    baseline = _extract_metrics(base_result, "D_control", "baseline", False)
    print(f"        {baseline.total_tool_calls} calls, {baseline.total_tokens} tokens, {baseline.duration_seconds:.1f}s")

    # Clean DB
    for ext in ("", "-wal", "-shm"):
        p = db_path + ext
        if os.path.exists(p):
            os.unlink(p)

    print(f"  [2/2] Treatment (memory ON, empty DB)...", flush=True)
    treat_result = _run_with_memory(
        model, SCENARIO_D_PROMPT, memory_enabled=True, db_path=db_path, timeout_seconds=timeout
    )
    treatment = _extract_metrics(treat_result, "D_control", "control_mem", True)
    print(f"        {treatment.total_tool_calls} calls, {treatment.total_tokens} tokens, {treatment.duration_seconds:.1f}s")

    tc_delta = _pct(baseline.total_tool_calls, treatment.total_tool_calls)
    tok_delta = _pct(baseline.total_tokens, treatment.total_tokens)
    time_delta = _pct(baseline.duration_seconds, treatment.duration_seconds)

    return ScenarioComparison(
        scenario_id="D_control",
        description="Simple baseline (no expected memory benefit, measures overhead)",
        baseline=baseline,
        seed=None,
        benefit=treatment,
        tool_call_delta=tc_delta,
        token_delta=tok_delta,
        time_delta=time_delta,
        benefit_detected=False,
        memory_tools_used=[],
    )


# ── Output Formatting ────────────────────────────────────────────


def _print_scenario(comp: ScenarioComparison) -> None:
    """Print a formatted comparison for one scenario."""
    print(f"\n--- {comp.scenario_id}: {comp.description} ---")
    hdr = f"  {'Run':<10} {'Memory':<7} {'Calls':>6} {'Tokens':>8} {'Cost':>9} {'Time':>7} {'recall_*':>9} {'store_*':>8}"
    print(hdr)
    print(f"  {'-'*72}")

    for label, m in [("base", comp.baseline), ("seed", comp.seed), ("benefit", comp.benefit)]:
        if m is None:
            continue
        mem_str = "ON" if m.memory_enabled else "OFF"
        recall = sum(1 for k, v in m.tool_counts.items() if "recall" in k for _ in range(v))
        store = m.tool_counts.get("robotmcp_store_knowledge", 0)
        print(
            f"  {label:<10} {mem_str:<7} {m.total_tool_calls:>6} {m.total_tokens:>8} "
            f"${m.cost:>8.4f} {m.duration_seconds:>6.1f}s {recall:>9} {store:>8}"
        )

    delta_sign = lambda v: f"{v:+.0%}"
    print(
        f"  Delta (benefit vs base): {delta_sign(comp.tool_call_delta)} calls, "
        f"{delta_sign(comp.token_delta)} tokens, {delta_sign(comp.time_delta)} time"
    )
    if comp.memory_tools_used:
        print(f"  Memory tools used: {', '.join(comp.memory_tools_used)}")
    if comp.benefit_detected:
        print(f"  ** BENEFIT DETECTED **")


def _print_aggregate(comparisons: list[ScenarioComparison]) -> None:
    """Print aggregate summary across all scenarios."""
    print(f"\n{'='*72}")
    print("AGGREGATE SUMMARY")
    print(f"{'='*72}")

    non_control = [c for c in comparisons if c.scenario_id != "D_control"]
    control = [c for c in comparisons if c.scenario_id == "D_control"]

    if non_control:
        avg_tc = sum(c.tool_call_delta for c in non_control) / len(non_control)
        avg_tok = sum(c.token_delta for c in non_control) / len(non_control)
        avg_time = sum(c.time_delta for c in non_control) / len(non_control)
        benefit_count = sum(1 for c in non_control if c.benefit_detected)
        recall_count = sum(1 for c in non_control if c.benefit.has_recall_calls)

        print(f"  Avg tool call change (benefit vs base):  {avg_tc:+.0%}")
        print(f"  Avg token change:                        {avg_tok:+.0%}")
        print(f"  Avg time change:                         {avg_time:+.0%}")
        print(f"  Benefit detected:                        {benefit_count}/{len(non_control)} scenarios")
        print(f"  Memory recall used:                      {recall_count}/{len(non_control)} scenarios")

    if control:
        c = control[0]
        print(f"  Control overhead (mem ON, empty DB):")
        print(f"    Tokens: {c.token_delta:+.0%}, Time: {c.time_delta:+.0%}")

    print(f"{'='*72}")


def _print_iteration_stats(stats: IterationStats) -> None:
    """Print statistical summary for a scenario across iterations."""
    print(f"\n{'='*72}")
    print(f"STATS: {stats.scenario_id} ({stats.iterations} iterations)")
    print(f"{'='*72}")

    print(f"  {'Metric':<20} {'Baseline':>20} {'Benefit':>20} {'Delta':>15}")
    print(f"  {'-'*75}")

    def _fmt(mean: float, std: float, is_int: bool = False) -> str:
        if is_int:
            return f"{mean:>7.1f} +/- {std:>5.1f}"
        return f"{mean:>9.0f} +/- {std:>7.0f}"

    print(f"  {'Tool calls':<20} {_fmt(stats.baseline_calls_mean, stats.baseline_calls_std, True):>20} "
          f"{_fmt(stats.benefit_calls_mean, stats.benefit_calls_std, True):>20} "
          f"{stats.call_delta_mean:>+7.0%} +/-{stats.call_delta_std:>5.0%}")
    print(f"  {'Tokens':<20} {_fmt(stats.baseline_tokens_mean, stats.baseline_tokens_std):>20} "
          f"{_fmt(stats.benefit_tokens_mean, stats.benefit_tokens_std):>20} "
          f"{stats.token_delta_mean:>+7.0%} +/-{stats.token_delta_std:>5.0%}")
    print(f"  {'Time (s)':<20} {stats.baseline_time_mean:>8.1f} +/- {stats.baseline_time_std:>5.1f}  "
          f"  {stats.benefit_time_mean:>8.1f} +/- {stats.benefit_time_std:>5.1f}  "
          f"{stats.time_delta_mean:>+7.0%} +/-{stats.time_delta_std:>5.0%}")
    print(f"  {'Cost ($)':<20} {stats.baseline_cost_mean:>18.4f}   {stats.benefit_cost_mean:>18.4f}")

    print(f"\n  95% CI (calls delta): [{stats.call_delta_ci_low:+.0%}, {stats.call_delta_ci_high:+.0%}]")
    print(f"  95% CI (token delta): [{stats.token_delta_ci_low:+.0%}, {stats.token_delta_ci_high:+.0%}]")
    print(f"  Benefit detected: {stats.benefit_detected_count}/{stats.iterations} iterations")
    print(f"  Memory recall used: {stats.memory_recall_used_count}/{stats.iterations} iterations")

    # Significance check
    if stats.iterations >= 3:
        if stats.call_delta_ci_high < 0:
            print(f"  => Tool call reduction is STATISTICALLY SIGNIFICANT (CI entirely < 0)")
        elif stats.call_delta_ci_low > 0:
            print(f"  => Tool calls INCREASED (CI entirely > 0)")
        else:
            print(f"  => Tool call change is NOT statistically significant (CI spans 0)")
        if stats.token_delta_ci_high < 0:
            print(f"  => Token reduction is STATISTICALLY SIGNIFICANT (CI entirely < 0)")
        elif stats.token_delta_ci_low > 0:
            print(f"  => Tokens INCREASED (CI entirely > 0)")
        else:
            print(f"  => Token change is NOT statistically significant (CI spans 0)")


def _print_multi_aggregate(all_stats: list[IterationStats]) -> None:
    """Print overall aggregate across all scenarios with multiple iterations."""
    print(f"\n{'='*72}")
    print("MULTI-ITERATION AGGREGATE SUMMARY")
    print(f"{'='*72}")

    non_control = [s for s in all_stats if s.scenario_id != "D_control"]
    control = [s for s in all_stats if s.scenario_id == "D_control"]

    if non_control:
        print(f"\n  {'Scenario':<25} {'Iters':>5} {'Calls Delta':>15} {'Token Delta':>15} {'Benefit':>10}")
        print(f"  {'-'*72}")
        for s in non_control:
            print(f"  {s.scenario_id:<25} {s.iterations:>5} "
                  f"{s.call_delta_mean:>+7.0%}+/-{s.call_delta_std:>4.0%}  "
                  f"{s.token_delta_mean:>+7.0%}+/-{s.token_delta_std:>4.0%}  "
                  f"{s.benefit_detected_count}/{s.iterations}")

        all_call_deltas = [c.tool_call_delta for s in non_control for c in s.comparisons]
        all_tok_deltas = [c.token_delta for s in non_control for c in s.comparisons]
        print(f"\n  Overall (all non-control iterations):")
        print(f"    Call delta:  {_mean(all_call_deltas):+.0%} +/- {_stddev(all_call_deltas):.0%}  (n={len(all_call_deltas)})")
        print(f"    Token delta: {_mean(all_tok_deltas):+.0%} +/- {_stddev(all_tok_deltas):.0%}  (n={len(all_tok_deltas)})")
        ci_call = _ci95(all_call_deltas)
        ci_tok = _ci95(all_tok_deltas)
        print(f"    95% CI calls:  [{ci_call[0]:+.0%}, {ci_call[1]:+.0%}]")
        print(f"    95% CI tokens: [{ci_tok[0]:+.0%}, {ci_tok[1]:+.0%}]")

    if control:
        s = control[0]
        print(f"\n  Control overhead ({s.iterations} iterations):")
        print(f"    Token delta: {s.token_delta_mean:+.0%} +/- {s.token_delta_std:.0%}")
        print(f"    Time delta:  {s.time_delta_mean:+.0%} +/- {s.time_delta_std:.0%}")

    print(f"{'='*72}")


def _save_results(
    comparisons: list[ScenarioComparison],
    model: str,
    total_time: float,
    *,
    iteration_stats: list[IterationStats] | None = None,
) -> Path:
    """Save results as JSON."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    iterations = iteration_stats[0].iterations if iteration_stats else 1
    filepath = METRICS_DIR / f"memory_comparison_{timestamp}_x{iterations}.json"

    total_cost = sum(
        c.baseline.cost + (c.seed.cost if c.seed else 0) + c.benefit.cost
        for c in comparisons
    )

    data: dict[str, Any] = {
        "timestamp": timestamp,
        "model": model,
        "iterations": iterations,
        "total_time_seconds": total_time,
        "total_cost": total_cost,
    }

    if iteration_stats:
        data["statistics"] = {s.scenario_id: s.to_dict() for s in iteration_stats}
    else:
        data["scenarios"] = {c.scenario_id: c.to_dict() for c in comparisons}
        data["aggregate"] = {
            "scenarios_run": len(comparisons),
            "benefit_detected_count": sum(1 for c in comparisons if c.benefit_detected),
        }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)

    print(f"\nResults saved to: {filepath}")
    return filepath


# ── Preflight ─────────────────────────────────────────────────────


def _preflight_check(model: str, db_path: str) -> bool:
    """Verify opencode + memory system is functional."""
    print("Preflight: Checking opencode + memory system...", flush=True)

    result = _run_with_memory(
        model,
        'Call get_memory_status to check if memory is active. Just call get_memory_status and report the result. Nothing else.',
        memory_enabled=True,
        db_path=db_path,
        timeout_seconds=120,
    )

    if result.error:
        print(f"  FAIL: {result.error}")
        return False

    memory_called = any(
        tc.name == "robotmcp_get_memory_status" for tc in result.tool_calls
    )
    if memory_called:
        print(f"  OK: get_memory_status called ({len(result.tool_calls)} total tool calls, {result.duration_seconds:.1f}s)")
        return True

    # Fallback: check if any tool was called at all
    if result.tool_calls:
        tool_names = [tc.name for tc in result.tool_calls]
        print(f"  WARN: get_memory_status not called, but model used: {tool_names}")
        print(f"  Continuing anyway (memory tools may still work)...")
        return True

    print(f"  FAIL: No tool calls made. Model output: {result.text_output[:200]}")
    return False


# ── Main ──────────────────────────────────────────────────────────


SCENARIOS = {
    "A": ("A_repeated_task", "Repeated BuiltIn task (step recall)", SCENARIO_A_PROMPT, None),
    "B": ("B_error_recovery", "Error recovery (fix recall)", SCENARIO_B_TEST_PROMPT, SCENARIO_B_SEED_PROMPT),
    "C": ("C_keyword_discovery", "Keyword discovery (keyword recall)", SCENARIO_C_TEST_PROMPT, SCENARIO_C_SEED_PROMPT),
    "D": ("D_control", "Simple baseline (overhead measurement)", SCENARIO_D_PROMPT, None),
    "E": ("E_demoshop_cart", "Demoshop add-to-cart (web locator discovery)", SCENARIO_E_TEST_PROMPT, SCENARIO_E_SEED_PROMPT),
    "F": ("F_demoshop_checkout", "Demoshop full checkout (complex web flow)", SCENARIO_F_TEST_PROMPT, SCENARIO_F_SEED_PROMPT),
    "G": ("G_demoshop_browse", "Demoshop product browsing (exploration-heavy)", SCENARIO_G_TEST_PROMPT, SCENARIO_G_SEED_PROMPT),
    "H": ("H_api_crud", "API CRUD with auth (restful-booker)", SCENARIO_H_TEST_PROMPT, SCENARIO_H_SEED_PROMPT),
    "I": ("I_api_keyword_discovery", "API keyword discovery (HTTP methods)", SCENARIO_I_TEST_PROMPT, SCENARIO_I_SEED_PROMPT),
    "J": ("J_api_error_recovery", "API error recovery (auth failure)", SCENARIO_J_TEST_PROMPT, SCENARIO_J_SEED_PROMPT),
    "K": ("K_carconfig_login", "CarConfig demo login + navigation (German locators)", SCENARIO_K_TEST_PROMPT, SCENARIO_K_SEED_PROMPT),
    "L": ("L_carconfig_configure", "CarConfig vehicle configuration (multi-step options)", SCENARIO_L_TEST_PROMPT, SCENARIO_L_SEED_PROMPT),
    "M": ("M_carconfig_register", "CarConfig registration form (form-heavy locators)", SCENARIO_M_TEST_PROMPT, SCENARIO_M_SEED_PROMPT),
    "N": ("N_demoshop_multi_cart", "Demoshop multi-product cart (repeating locator pattern)", SCENARIO_N_TEST_PROMPT, SCENARIO_N_SEED_PROMPT),
}


def main():
    parser = argparse.ArgumentParser(description="ADR-014 Memory Comparison")
    parser.add_argument(
        "--scenario",
        choices=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "all", "web", "api", "carconfig"],
        default="all",
        help="Which scenario to run (default: all, web=E+F+G+N, api=H+I+J, carconfig=K+L+M)",
    )
    parser.add_argument("--iterations", "-n", type=int, default=1, help="Iterations per scenario (default: 1)")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per run in seconds")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip preflight check")
    args = parser.parse_args()

    model = MODELS[0]
    db_base = Path(tempfile.mkdtemp(prefix="rfmcp_e2e_"))
    iterations = max(1, args.iterations)

    print(f"{'='*72}")
    print("ADR-014 MEMORY COMPARISON: rf-mcp with vs without memory")
    print(f"{'='*72}")
    print(f"Model:      {model}")
    print(f"Iterations: {iterations}")
    print(f"DB dir:     {db_base}")
    print(f"Timeout:    {args.timeout}s per run")

    # Preflight
    if not args.skip_preflight:
        preflight_db = str(db_base / "preflight.db")
        if not _preflight_check(model, preflight_db):
            print("\nPreflight failed. Aborting.")
            sys.exit(1)

    # Determine which scenarios to run
    if args.scenario == "all":
        scenario_keys = list(SCENARIOS.keys())
    elif args.scenario == "web":
        scenario_keys = ["E", "F", "G", "N"]
    elif args.scenario == "api":
        scenario_keys = ["H", "I", "J"]
    elif args.scenario == "carconfig":
        scenario_keys = ["K", "L", "M"]
    else:
        scenario_keys = [args.scenario]

    all_comparisons: list[ScenarioComparison] = []
    all_stats: list[IterationStats] = []
    total_start = time.time()

    for key in scenario_keys:
        sid, desc, test_prompt, seed_prompt = SCENARIOS[key]
        scenario_comparisons: list[ScenarioComparison] = []

        for i in range(iterations):
            iter_label = f" (iter {i+1}/{iterations})" if iterations > 1 else ""
            print(f"\n{'='*72}")
            print(f"Scenario {key}: {desc}{iter_label}")
            print(f"{'='*72}")

            # Each iteration gets its own DB subdirectory
            iter_db_base = db_base / f"iter_{i}" if iterations > 1 else db_base
            iter_db_base.mkdir(parents=True, exist_ok=True)

            if key == "D":
                comp = _run_control_scenario(model, iter_db_base, timeout=args.timeout)
            else:
                comp = _run_scenario(
                    sid, desc, model, iter_db_base,
                    test_prompt=test_prompt,
                    seed_prompt=seed_prompt,
                    timeout=args.timeout,
                )

            scenario_comparisons.append(comp)
            all_comparisons.append(comp)
            _print_scenario(comp)

        # Compute stats for this scenario
        stats = _compute_iteration_stats(sid, desc, scenario_comparisons)
        all_stats.append(stats)

        if iterations > 1:
            _print_iteration_stats(stats)

    total_time = time.time() - total_start

    # Aggregate + save
    if iterations > 1:
        _print_multi_aggregate(all_stats)
        _save_results(all_comparisons, model, total_time, iteration_stats=all_stats)
    else:
        _print_aggregate(all_comparisons)
        _save_results(all_comparisons, model, total_time)

    total_cost = sum(c.baseline.cost + (c.seed.cost if c.seed else 0) + c.benefit.cost for c in all_comparisons)
    print(f"\nTotal time: {total_time:.1f}s")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Total runs: {len(all_comparisons) * 3} opencode invocations")


if __name__ == "__main__":
    main()
