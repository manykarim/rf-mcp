"""Test Builder component for generating Robot Framework test suites from executed steps."""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    from robot.api import TestSuite
except ImportError:
    TestSuite = None

try:
    from robot.running.model import Keyword as RunningKeyword
except ImportError:
    RunningKeyword = None

# Import shared library detection utility
from robotmcp.core.event_bus import FrontendEvent, event_bus
from robotmcp.utils.library_detector import detect_library_from_keyword

logger = logging.getLogger(__name__)


@dataclass
class TestCaseStep:
    """Represents a test case step."""

    __test__ = False  # Not a pytest test class

    keyword: str
    arguments: List[str]
    comment: Optional[str] = None
    # Variable assignment tracking for test suite generation
    assigned_variables: List[str] = field(default_factory=list)
    assignment_type: Optional[str] = None  # "single", "multiple", "none"
    # BDD step grouping
    bdd_group: Optional[str] = None
    bdd_intent: Optional[str] = None


@dataclass
class GeneratedTestCase:
    """Represents a generated test case."""

    name: str
    steps: List[TestCaseStep]
    documentation: str = ""
    tags: List[str] = None
    template: Optional[str] = None  # ADR-019: data-driven template
    setup: Optional[TestCaseStep] = None
    teardown: Optional[TestCaseStep] = None


@dataclass
class BddKeywordGroup:
    """A group of steps that form a behavioral keyword."""

    name: str  # Human-readable keyword name
    steps: List[TestCaseStep]  # Implementation steps
    bdd_intent: str = "when"  # "given", "when", "then"


@dataclass
class BddKeyword:
    """A behavioral keyword for BDD-style test suites."""

    name: str  # e.g., 'the user adds "${product}" to the cart'
    steps: List[TestCaseStep]  # Implementation steps
    bdd_intent: str = "when"  # "given", "when", "then", "and", "but"
    embedded_args: Dict[str, str] = field(
        default_factory=dict
    )  # arg_name -> example value
    is_embedded: bool = False  # True if uses embedded arguments
    arg_vars: List[str] = field(default_factory=list)  # [Arguments] variables


@dataclass
class GeneratedTestSuite:
    """Represents a generated test suite."""

    name: str
    test_cases: List[GeneratedTestCase]
    documentation: str = ""
    tags: List[str] = None
    setup: Optional[TestCaseStep] = None
    teardown: Optional[TestCaseStep] = None
    imports: List[str] = None
    resources: List[str] = None
    # Optional: preserved high-level flow blocks recorded during execution
    flow_blocks: List[Dict[str, Any]] | None = None
    # ADR-005: per-test flow blocks (test_name → List[Dict]). When set, overrides
    # suite.flow_blocks for individual test case rendering.
    per_test_flow_blocks: Dict[str, List[Dict[str, Any]]] | None = None
    # Suite-level variables from session (set via manage_session or execute_step assignments)
    variables: Dict[str, Any] = None
    # Track variable files imported via manage_session(action="import_variables")
    variable_files: List[str] = None
    # BDD behavioral keywords (generated when bdd_style=True)
    bdd_keywords: Optional[List[BddKeyword]] = None
    # Suite-level Test Template (all test cases share the same template)
    suite_template: Optional[str] = None
    # Column header comments for Test Cases line (e.g., ["USERNAME", "PASSWORD"])
    column_headers: Optional[List[str]] = None


class TestBuilder:
    """Builds Robot Framework test suites from successful execution steps."""

    def __init__(self, execution_engine=None):
        self.execution_engine = execution_engine
        self.optimization_rules = {
            "combine_waits": True,
            "remove_redundant_verifications": True,
            "group_similar_actions": True,
            "add_meaningful_comments": True,
            "generate_variables": True,
        }

    def _convert_to_evaluation_namespace_syntax(self, expression: str) -> str:
        """Convert ${var} syntax to $var syntax for Robot Framework evaluation namespace.

        In Robot Framework's evaluation namespace (used by Evaluate keyword, IF conditions,
        and other Python evaluation contexts), variables should use $var syntax instead of
        ${var}. The ${var} syntax causes string substitution BEFORE evaluation, while $var
        provides direct access to the actual Python object in the evaluation namespace.

        This method also fixes common quoting issues:
        - '$var'.method() -> $var.method() (quoted variable with method call)
        - Arithmetic with string variables: auto-detects and adds int()/float() wrappers

        Reference: https://robotframework.org/robotframework/latest/RobotFrameworkUserGuide.html#evaluation-namespaces

        Args:
            expression: Expression that may contain ${var} syntax

        Returns:
            Expression with ${var} converted to $var and quoting issues fixed
        """
        if not expression:
            return expression

        result = expression

        # Convert ${var.suffix} to $var.suffix
        # The regex handles nested variables by being non-greedy within braces
        result = re.sub(r"\$\{([A-Za-z_]\w*)([^}]*)\}", r"$\1\2", result)

        # Fix quoted variable method calls: '$var'.method() -> $var.method()
        # Pattern: single or double quoted $var followed by .method()
        # This is wrong because '$var' is treated as literal string, not variable
        result = re.sub(r"['\"](\$[A-Za-z_]\w*)['\"]\.(\w+\([^)]*\))", r"\1.\2", result)

        # Also handle quoted variables without method calls that should be unquoted
        # Pattern: '$var' at word boundaries in comparison/arithmetic contexts
        # e.g., '$var' + something or '$var' == something
        result = re.sub(r"['\"](\$[A-Za-z_]\w*)['\"](\s*[\+\-\*/%<>=!&|])", r"\1\2", result)
        result = re.sub(r"([\+\-\*/%<>=!&|]\s*)['\"](\$[A-Za-z_]\w*)['\"]", r"\1\2", result)

        # Handle quoted variables inside function arguments: func('$var') -> func($var)
        # Common cases: len('$var'), int('$var'), str('$var'), upper('$var'), etc.
        # Pattern: function_name('$var' or "$var")
        result = re.sub(r"(\w+\s*\(\s*)['\"](\$[A-Za-z_]\w*)['\"](\s*\))", r"\1\2\3", result)

        # Handle quoted variables as first argument in multi-arg functions: func('$var', ...)
        result = re.sub(r"(\w+\s*\(\s*)['\"](\$[A-Za-z_]\w*)['\"](\s*,)", r"\1\2\3", result)

        # Handle quoted variables as middle/last arguments: func(..., '$var') or func(..., '$var', ...)
        result = re.sub(r"(,\s*)['\"](\$[A-Za-z_]\w*)['\"](\s*[,)])", r"\1\2\3", result)

        return result



    def _is_arithmetic_expression(self, value: str) -> bool:
        """Check if a value contains an arithmetic expression that can't be used in VAR.

        VAR keyword cannot evaluate expressions like ${count + 1} or ${price * 2}.
        These must use the Evaluate keyword instead.

        Args:
            value: The value string to check

        Returns:
            True if the value contains arithmetic that VAR can't handle
        """
        if not value:
            return False

        # Pattern: ${var + num} or ${var - num} or ${var * num} etc.
        # These patterns fail with VAR but work with Evaluate
        arithmetic_pattern = r'\$\{[^}]*[\+\-\*/%][^}]*\}'
        return bool(re.search(arithmetic_pattern, value))

    def _extract_expression(self, value: str) -> Optional[str]:
        """Extract the expression from a ${...} wrapper.

        Args:
            value: String like "${count + 1}" or "${price * 2}"

        Returns:
            The inner expression without ${}, or None if not wrapped
        """
        if not value:
            return None

        # Match ${...} and extract the content
        match = re.match(r'\$\{([^}]+)\}', value.strip())
        if match:
            return match.group(1)
        return None

    def _add_type_conversion_if_needed(self, expression: str) -> str:
        """Add int() or float() wrappers to variables in arithmetic expressions.

        When variables come from external sources (input, API responses), they are
        often strings. Arithmetic operations on strings fail, so we wrap them
        with int() to ensure numeric operations work.

        Args:
            expression: An expression like "$count + 1" or "$price * 2"

        Returns:
            Expression with type conversion wrappers where needed
        """
        if not expression:
            return expression

        # Check if the expression has arithmetic operators
        if not re.search(r'[\+\-\*/%]', expression):
            return expression

        # Pattern: $var followed by arithmetic (without existing int()/float() wrapper)
        # Replace $var with int($var) when in arithmetic context

        result = expression

        # Find variables that need wrapping (not already wrapped)
        # Pattern: $var that's not inside int() or float()
        var_pattern = r'(?<!\w)(\$[A-Za-z_]\w*)(?!\s*\))'

        def wrap_var(match):
            var = match.group(1)
            # Check if already wrapped by looking at what comes before
            start = match.start()
            prefix = result[:start]
            # If preceded by 'int(' or 'float(', don't wrap
            if prefix.rstrip().endswith('int(') or prefix.rstrip().endswith('float('):
                return var
            return f'int({var})'

        result = re.sub(var_pattern, wrap_var, result)

        return result

    async def build_suite(
        self,
        session_id: str = "default",
        test_name: str = "",
        tags: List[str] = None,
        documentation: str = "",
        remove_library_prefixes: bool = True,
        bdd_style: bool = False,
        data_driven_mode: str = "auto",
    ) -> Dict[str, Any]:
        """
        Generate Robot Framework test suite from successful execution steps.

        Args:
            session_id: Session with executed steps
            test_name: Name for the test case
            tags: Test tags
            documentation: Test documentation
            remove_library_prefixes: Remove library prefixes from keywords (e.g., "Browser.Click" -> "Click")
            bdd_style: Generate BDD-style suite with Keywords section. When True,
                steps are grouped into behavioral keywords (Given/When/Then) and a
                ``*** Keywords ***`` section is appended to the generated RF text.

        Returns:
            Generated test suite with RF API objects and text representation
        """
        try:
            if tags is None:
                tags = []

            # ── ADR-005: Check for multi-test mode ────────────────────────
            session = None
            if self.execution_engine:
                session = self.execution_engine.sessions.get(session_id)

            if session and session.test_registry.is_multi_test_mode():
                # MULTI-TEST MODE: build from TestRegistry
                # Auto-end any running test
                if session.test_registry.current_test_name:
                    session.test_registry.end_test(status="pass")
                    try:
                        from robotmcp.components.execution.rf_native_context_manager import (
                            get_rf_native_context_manager,
                        )
                        mgr = get_rf_native_context_manager()
                        mgr.end_test_in_context(session_id)
                    except Exception:
                        pass

                test_cases = []
                for name, test_info in session.test_registry.tests.items():
                    # Fix 1: Adopt legacy steps as template body when test has
                    # data_rows but no steps (agent executed before start_test)
                    if test_info.template and (test_info.data_rows or test_info.named_data_rows):
                        if not test_info.steps and hasattr(session, 'steps') and session.steps:
                            test_info.steps = list(session.steps)

                    if not test_info.steps and not test_info.data_rows and not test_info.named_data_rows:
                        continue
                    # Expand named_data_rows into separate test cases
                    if test_info.named_data_rows and test_info.template:
                        impl_steps = self._build_impl_steps_from_test_info(test_info)
                        for row_name, row_values in test_info.named_data_rows:
                            row_tc = GeneratedTestCase(
                                name=row_name,
                                steps=impl_steps + [TestCaseStep(keyword="", arguments=list(row_values))],
                                template=test_info.template,
                                documentation=test_info.documentation,
                                tags=test_info.tags,
                            )
                            test_cases.append(row_tc)
                    else:
                        tc = self._build_test_case_from_test_info(test_info)
                        test_cases.append(tc)

                if not test_cases:
                    return {
                        "success": False,
                        "error": "No tests with steps found in multi-test session",
                        "suite": None,
                    }

                suite = await self._build_test_suite(test_cases, session_id)

                # Add suite setup/teardown from session metadata
                if session.suite_setup:
                    suite.setup = TestCaseStep(
                        keyword=session.suite_setup["keyword"],
                        arguments=session.suite_setup.get("arguments", []),
                    )
                if session.suite_teardown:
                    suite.teardown = TestCaseStep(
                        keyword=session.suite_teardown["keyword"],
                        arguments=session.suite_teardown.get("arguments", []),
                    )

                # Populate per-test flow blocks for rendering
                per_test_fb = {}
                for name, ti in session.test_registry.tests.items():
                    if ti.flow_blocks:
                        per_test_fb[name] = ti.flow_blocks
                if per_test_fb:
                    suite.per_test_flow_blocks = per_test_fb

                # Override suite name if test_name provided
                if test_name:
                    suite.name = test_name
            else:
                # ── LEGACY MODE: unchanged single-test behavior ──────────

                # First, check if session is ready for test suite generation
                if self.execution_engine:
                    readiness_check = await self.execution_engine.validate_test_readiness(
                        session_id
                    )
                    if not readiness_check.get("ready_for_suite_generation", False):
                        return {
                            "success": False,
                            "error": "Session not ready for test suite generation",
                            "guidance": readiness_check.get("guidance", []),
                            "validation_summary": readiness_check.get(
                                "validation_summary", {}
                            ),
                            "recommendation": "Use validate_step_before_suite() to validate individual steps first",
                        }

                # Get session steps from execution engine
                steps = await self._get_session_steps(session_id)

                if not steps:
                    return {
                        "success": False,
                        "error": f"No steps found for session '{session_id}'",
                        "suite": None,
                    }

                # Filter successful steps
                successful_steps = [step for step in steps if step.get("status") == "pass"]

                if not successful_steps:
                    return {
                        "success": False,
                        "error": "No successful steps to build suite from",
                        "suite": None,
                    }

                # Build test case from steps
                test_case = await self._build_test_case(
                    successful_steps,
                    test_name or f"Test_{session_id}",
                    tags,
                    documentation,
                    session_id,
                )

                # Create test suite
                suite = await self._build_test_suite([test_case], session_id)

            # Apply library prefix removal if requested
            if remove_library_prefixes:
                suite = self._apply_prefix_removal(suite)

            # Apply data-driven mode promotion BEFORE BDD (suite_template
            # skips BDD transformation for template test cases)
            self._apply_data_driven_mode(suite, data_driven_mode)

            # Apply BDD transformation if requested
            if bdd_style:
                suite = self._transform_to_bdd_style(suite)

            # Generate Robot Framework API objects
            rf_suite = await self._create_rf_suite(suite)

            # Generate text representation
            rf_text = await self._generate_rf_text(suite)

            # Generate execution statistics
            # In multi-test mode, collect all steps as dicts for stats
            if session and session.test_registry.is_multi_test_mode():
                successful_steps = [
                    {
                        "keyword": s.keyword,
                        "arguments": s.arguments,
                        "status": s.status,
                        "result": s.result,
                    }
                    for s in session.test_registry.all_steps_flat()
                    if s.is_successful
                ]
            stats = await self._generate_statistics(successful_steps, suite)

            # Check for untracked variables (referenced in steps but not in Variables section)
            # This provides early warning when generated tests may be incomplete
            variable_warnings = self._check_untracked_variables(suite, session_id)

            # Build structured steps (keywords + control blocks) for consumers
            structured_cases = []
            for tc in suite.test_cases:
                structured_cases.append({
                    "name": tc.name,
                    "structured_steps": self._build_structured_steps(tc, suite.flow_blocks),
                })

            result = {
                "success": True,
                "session_id": session_id,
                # Include warnings about potentially missing variables
                "warnings": variable_warnings if variable_warnings else None,
                "suite": {
                        "name": suite.name,
                        "documentation": suite.documentation,
                        "tags": suite.tags or [],
                        # Expose flow blocks so callers can reconstruct control structures
                        "flow_blocks": suite.flow_blocks or [],
                        "test_cases": [
                            {
                                "name": tc.name,
                                "documentation": tc.documentation,
                                "tags": tc.tags or [],
                            # steps omitted in favor of structured_steps
                            "setup": {
                                "keyword": tc.setup.keyword,
                                "arguments": [
                                    self._escape_robot_argument(arg)
                                    for arg in (tc.setup.arguments or [])
                                ],
                            }
                            if tc.setup
                            else None,
                            "teardown": {
                                "keyword": tc.teardown.keyword,
                                "arguments": [
                                    self._escape_robot_argument(arg)
                                    for arg in (tc.teardown.arguments or [])
                                ],
                            }
                            if tc.teardown
                            else None,
                            # Provide structured rendering alongside linear steps
                            "structured_steps": next((c["structured_steps"] for c in structured_cases if c["name"] == tc.name), []),
                        }
                        for tc in suite.test_cases
                    ],
                    "imports": suite.imports or [],
                    # Include session variables in suite output so consumers know what's defined
                    "variables": suite.variables or {},
                    "variable_files": suite.variable_files or [],
                    "setup": {
                        "keyword": suite.setup.keyword,
                        "arguments": [
                            self._escape_robot_argument(arg)
                            for arg in (suite.setup.arguments or [])
                        ],
                    }
                    if suite.setup
                    else None,
                    "teardown": {
                        "keyword": suite.teardown.keyword,
                        "arguments": [
                            self._escape_robot_argument(arg)
                            for arg in (suite.teardown.arguments or [])
                        ],
                    }
                    if suite.teardown
                    else None,
                },
                "rf_text": rf_text,
                "statistics": stats,
            }
            event_bus.publish_sync(
                FrontendEvent(
                    event_type="suite_built",
                    session_id=session_id,
                    payload={
                        "test_cases": len(suite.test_cases),
                        "step_count": sum(len(tc.steps) for tc in suite.test_cases),
                    },
                )
            )
            return result

        except Exception as e:
            logger.error(f"Error building test suite: {e}")
            return {"success": False, "error": str(e), "suite": None}

    # ── BDD transformation methods ─────────────────────────────────────

    # Keywords whose presence signals a "setup / given" group
    _SETUP_KEYWORDS = frozenset({
        "new browser", "new context", "new page", "open browser", "go to",
    })

    # Keywords whose presence signals an "assertion / then" group
    _ASSERTION_KEYWORDS_PREFIXES = (
        "should be", "should contain", "should not", "should match",
        "should start with", "should end with", "element should",
        "page should", "checkbox should", "list should", "title should",
        "location should", "wait until element",
    )

    # Phase 2.1: Semantic phase indicators from locator/argument content
    _PHASE_INDICATORS = {
        "cart": ("add to cart", "add .* to cart", "remove from cart",
                 "cart", "basket", "bag"),
        "checkout": ("checkout", "proceed", "continue", "shipping",
                     "payment", "billing", "place order"),
        "order": ("finish", "complete order", "submit order", "confirm"),
        "login": ("login", "sign in", "log in", "username", "password"),
        "navigate": ("products", "home", "categories", "menu", "back"),
    }

    def _transform_to_bdd_style(self, suite: GeneratedTestSuite) -> GeneratedTestSuite:
        """Transform a flat test suite into BDD-style with a Keywords section.

        Three-pass architecture (Phase 3.1):
        Pass 1: Filter instrumentation steps, group into BddKeyword objects.
        Pass 2: Detect & merge embedded args — keyword names may change.
        Pass 3: Assign test case references using final merged names.

        Data-driven BDD (Phase 4):
        Template test cases are wrapped in BDD wrapper keywords.
        """
        # ── Pass 1: Group steps into preliminary keywords ────────────
        all_bdd_keywords: List[BddKeyword] = []
        seen_keyword_names: Dict[str, BddKeyword] = {}
        # Track which groups map to which test cases for Pass 3
        tc_group_map: List[Tuple[GeneratedTestCase, List[BddKeywordGroup]]] = []

        for tc in suite.test_cases:
            # Skip suite_template TCs — they're data rows, not keyword steps
            if suite.suite_template:
                continue

            if getattr(tc, "template", None):
                self._transform_template_to_bdd(
                    tc, all_bdd_keywords, seen_keyword_names, suite
                )
                continue

            # Phase 1.2: Filter instrumentation steps before grouping
            filtered = self._filter_instrumentation_steps(tc.steps)
            groups = self._group_steps_into_bdd_keywords(filtered)

            for group in groups:
                if group.name not in seen_keyword_names:
                    kw = BddKeyword(
                        name=group.name,
                        steps=list(group.steps),
                        bdd_intent=group.bdd_intent,
                    )
                    all_bdd_keywords.append(kw)
                    seen_keyword_names[kw.name] = kw

            tc_group_map.append((tc, groups))

        # ── Pass 2: Detect embedded args (may rename keywords) ───────
        all_bdd_keywords = self._detect_embedded_args(all_bdd_keywords)

        # Rebuild name lookup after potential merges
        final_name_lookup: Dict[str, BddKeyword] = {kw.name: kw for kw in all_bdd_keywords}

        # Build old-name → new-name mapping for merged keywords.
        # Only map names whose keywords were structurally identical to the
        # merged keyword (same keyword sequence + arg count).
        old_to_new: Dict[str, str] = {}
        for kw in all_bdd_keywords:
            old_to_new[kw.name] = kw.name  # identity for non-merged

        for kw in all_bdd_keywords:
            if kw.is_embedded and kw.embedded_args:
                # Build the structural key for this merged keyword
                merged_key = tuple(
                    (s.keyword, len(s.arguments or []))
                    for s in kw.steps
                )
                for old_name, old_kw in list(seen_keyword_names.items()):
                    if old_name in final_name_lookup or old_name in old_to_new:
                        continue
                    # Only map if structurally matching
                    old_key = tuple(
                        (s.keyword, len(s.arguments or []))
                        for s in old_kw.steps
                    )
                    if old_key == merged_key:
                        old_to_new[old_name] = kw.name

        # ── Pass 3: Assign BDD references using final names ──────────
        prefix_map = {"given": "Given", "when": "When", "then": "Then"}

        for tc, groups in tc_group_map:
            bdd_refs: List[TestCaseStep] = []
            for group in groups:
                # Resolve name through merge mapping
                final_name = old_to_new.get(group.name, group.name)
                resolved_kw = final_name_lookup.get(final_name)
                if resolved_kw:
                    intent = resolved_kw.bdd_intent
                else:
                    # Fallback to original group intent
                    intent = group.bdd_intent
                    final_name = group.name

                prefix = prefix_map.get(intent, "When")

                # For embedded keywords, substitute the actual argument values
                if resolved_kw and resolved_kw.is_embedded and resolved_kw.embedded_args:
                    # Use the original group's step arguments to fill in embedded values
                    call_name = self._build_embedded_call(
                        resolved_kw, group.steps
                    )
                    bdd_refs.append(
                        TestCaseStep(keyword=f"{prefix} {call_name}", arguments=[])
                    )
                else:
                    bdd_refs.append(
                        TestCaseStep(keyword=f"{prefix} {final_name}", arguments=[])
                    )
            tc.steps = bdd_refs

        suite.bdd_keywords = all_bdd_keywords

        # Phase 4.2: Extract URLs into Variables section
        self._extract_bdd_variables(suite)

        return suite

    def _extract_bdd_variables(self, suite: GeneratedTestSuite) -> None:
        """Extract hardcoded URLs from BDD keyword steps into Variables section."""
        if not suite.bdd_keywords:
            return
        if suite.variables is None:
            suite.variables = {}

        url_pattern = re.compile(r'https?://[^\s]+')
        for kw in suite.bdd_keywords:
            for step in kw.steps:
                for i, arg in enumerate(step.arguments or []):
                    arg_str = str(arg)
                    if url_pattern.match(arg_str) and "://" in arg_str:
                        # Generate a variable name from the URL
                        try:
                            from urllib.parse import urlparse
                            parsed = urlparse(arg_str)
                            domain = (parsed.hostname or "site").replace("www.", "")
                            domain = domain.split(".")[0].upper()
                            var_name = f"{domain}_URL"
                        except Exception:
                            var_name = "BASE_URL"

                        if var_name not in suite.variables:
                            suite.variables[var_name] = arg_str
                            step.arguments[i] = f"${{{var_name}}}"

    def _build_embedded_call(
        self, kw: BddKeyword, original_steps: List[TestCaseStep]
    ) -> str:
        """Build the call name for an embedded keyword using original step values.

        For a keyword named ``the user clicks ${value}``,
        and original steps containing ``Click role=button[name="Add Speaker to cart"]``,
        produce ``the user clicks "add speaker to cart"``.
        """
        name = kw.name
        if not kw.embedded_args or not original_steps:
            return name

        # Find the actual argument values from the original steps
        for arg_name, _example in kw.embedded_args.items():
            placeholder = f"${{{arg_name}}}"
            if placeholder not in name:
                continue
            # Find the differing value from the original steps vs the template
            for orig_step, tmpl_step in zip(original_steps, kw.steps):
                for oi, (orig_arg, tmpl_arg) in enumerate(
                    zip(orig_step.arguments or [], tmpl_step.arguments or [])
                ):
                    if placeholder in str(tmpl_arg) and str(orig_arg) != str(tmpl_arg):
                        # Humanize the locator for the call name
                        actual = self._humanize_locator(str(orig_arg))
                        if actual == "the element":
                            actual = str(orig_arg)  # Fallback to raw
                        name = name.replace(placeholder, f'"{actual}"', 1)
                        break
                if placeholder not in name:
                    break
        return name

    # Phase 1.2: Instrumentation step filtering
    def _filter_instrumentation_steps(
        self, steps: List[TestCaseStep]
    ) -> List[TestCaseStep]:
        """Remove LLM instrumentation steps that pollute BDD output.

        Steps like ``Get Text body`` (whole-page DOM read) and bare
        ``Get Url``/``Get Title`` without assertion operators are agent
        observation artifacts, not real test assertions.
        """
        result = []
        for step in steps:
            kw = (step.keyword or "").strip().lower()
            args = step.arguments or []

            # Get Text body (no assertion) — LLM reads DOM for navigation
            if kw == "get text" and args and str(args[0]).lower() == "body":
                if len(args) <= 1:
                    continue  # No assertion operator — skip

            # Bare Get Url / Get Title without assertion
            if kw in ("get url", "get title") and len(args) <= 1:
                if not any(s.assigned_variables for s in [step] if s.assigned_variables):
                    continue

            # Get Property without assertion operator (observation only)
            if kw == "get property" and len(args) <= 2:
                if not any(s.assigned_variables for s in [step] if s.assigned_variables):
                    continue

            result.append(step)
        return result if result else steps  # Never return empty

    def _transform_template_to_bdd(
        self,
        tc: GeneratedTestCase,
        all_bdd_keywords: List[BddKeyword],
        seen_keyword_names: Dict[str, BddKeyword],
        suite: GeneratedTestSuite,
    ) -> None:
        """Transform a data-driven (template) test case into BDD style.

        C3 fix: Instead of creating a wrapper that delegates to sub-keywords
        (which breaks variable forwarding), we create a single template keyword
        with BDD-prefixed steps directly inside. This ensures [Arguments] are
        in scope for all steps.

        Example output:
            Verify Login
                [Arguments]    ${username}    ${password}    ${expected}
                Given the user fills in "${username}" and "${password}"
                When the user clicks login
                Then the message should show "${expected}"
        """
        # Extract implementation steps (have keyword) vs data rows (empty keyword)
        template_steps = [s for s in tc.steps if s.keyword and s.keyword.strip()]
        data_rows = [s for s in tc.steps if not (s.keyword and s.keyword.strip())]

        if not template_steps:
            # No implementation steps — leave the test case as-is
            return

        # Filter instrumentation from template body
        template_steps = self._filter_instrumentation_steps(template_steps)

        # Group template steps into BDD phases
        groups = self._group_steps_into_bdd_keywords(template_steps)

        # Build BDD-prefixed steps for the template keyword body.
        # These steps reference the template [Arguments] directly — no wrapper
        # indirection needed.
        prefix_map = {"given": "Given", "when": "When", "then": "Then"}
        bdd_body_steps: List[TestCaseStep] = []

        for group in groups:
            prefix = prefix_map.get(group.bdd_intent, "When")
            if len(group.steps) == 1:
                # Single step: add BDD prefix directly
                step = group.steps[0]
                bdd_body_steps.append(
                    TestCaseStep(
                        keyword=f"{prefix} {step.keyword}",
                        arguments=list(step.arguments or []),
                    )
                )
            else:
                # Multi-step group: create a sub-keyword and reference it
                kw_name = group.name
                if kw_name not in seen_keyword_names:
                    kw = BddKeyword(
                        name=kw_name,
                        steps=list(group.steps),
                        bdd_intent=group.bdd_intent,
                    )
                    all_bdd_keywords.append(kw)
                    seen_keyword_names[kw_name] = kw
                bdd_body_steps.append(
                    TestCaseStep(
                        keyword=f"{prefix} {kw_name}",
                        arguments=[],
                    )
                )

        # Extract [Arguments] from template steps
        arg_vars = self._extract_template_arg_vars(template_steps)

        # Create the template keyword with BDD body
        template_kw_name = tc.template or f"Verify {tc.name}"
        template_kw = BddKeyword(
            name=template_kw_name,
            steps=bdd_body_steps,
            bdd_intent="when",
            arg_vars=arg_vars,
        )
        all_bdd_keywords.append(template_kw)
        seen_keyword_names[template_kw_name] = template_kw

        # Update the test case: keep only data rows
        tc.template = template_kw_name
        if data_rows:
            tc.steps = data_rows

    # M3: RF built-in variables that should NOT appear in [Arguments]
    _RF_BUILTIN_VARS = frozenset({
        "EMPTY", "SPACE", "TRUE", "FALSE", "NONE", "NULL",
        "CURDIR", "TEMPDIR", "EXECDIR", "OUTPUT_DIR", "OUTPUT_FILE",
        "LOG_FILE", "LOG_LEVEL", "REPORT_FILE", "TEST_NAME",
        "SUITE_NAME", "SUITE_SOURCE", "PREV_TEST_STATUS",
    })

    def _extract_template_arg_vars(self, steps: List[TestCaseStep]) -> List[str]:
        """Extract ${var} references from template steps for [Arguments] line.

        M3 fix: Filters out RF built-in variables (${EMPTY}, ${SPACE}, etc.)
        that should not appear in [Arguments].
        """
        var_pattern = re.compile(r'\$\{([^}]+)\}')
        found: List[str] = []
        seen: set = set()
        for step in steps:
            for arg in (step.arguments or []):
                for match in var_pattern.finditer(str(arg)):
                    var_name = match.group(1)
                    if var_name not in seen and var_name.upper() not in self._RF_BUILTIN_VARS:
                        seen.add(var_name)
                        found.append(f"${{{var_name}}}")
        return found

    def _group_steps_into_bdd_keywords(
        self, steps: List[TestCaseStep]
    ) -> List[BddKeywordGroup]:
        """Group flat steps into BDD keyword clusters.

        Priority:
        1. Explicit ``bdd_group`` annotations on steps.
        2. Heuristic grouping based on keyword semantics.
        """
        if not steps:
            return []

        # ── Strategy 1: explicit groups ──────────────────────────────
        has_explicit = any(s.bdd_group for s in steps)
        if has_explicit:
            return self._group_by_explicit_annotations(steps)

        # ── Strategy 2: heuristic grouping ───────────────────────────
        return self._group_by_heuristics(steps)

    def _group_by_explicit_annotations(
        self, steps: List[TestCaseStep]
    ) -> List[BddKeywordGroup]:
        """Group steps that share the same ``bdd_group`` value."""
        groups: List[BddKeywordGroup] = []
        current_group_name: Optional[str] = None
        current_steps: List[TestCaseStep] = []
        current_intent: str = "when"

        for step in steps:
            group_name = step.bdd_group or step.keyword
            intent = step.bdd_intent or "when"

            if group_name != current_group_name:
                if current_steps:
                    groups.append(
                        BddKeywordGroup(
                            name=current_group_name or self._generate_bdd_keyword_name(
                                current_steps, current_intent
                            ),
                            steps=current_steps,
                            bdd_intent=current_intent,
                        )
                    )
                current_group_name = group_name
                current_steps = [step]
                current_intent = intent
            else:
                current_steps.append(step)

        if current_steps:
            groups.append(
                BddKeywordGroup(
                    name=current_group_name or self._generate_bdd_keyword_name(
                        current_steps, current_intent
                    ),
                    steps=current_steps,
                    bdd_intent=current_intent,
                )
            )

        return groups

    def _group_by_heuristics(
        self, steps: List[TestCaseStep]
    ) -> List[BddKeywordGroup]:
        """Group steps by semantic heuristics with phase-aware boundaries.

        Phase 2.1: Detects semantic phases (cart, checkout, etc.) from locator
        content and forces group splits at phase transitions.
        Phase 2.2: Enforces max group size (7 steps) to prevent monolithic keywords.
        """
        groups: List[BddKeywordGroup] = []
        current_steps: List[TestCaseStep] = []
        current_intent: str = "when"
        current_phase: Optional[str] = None

        def _flush():
            if current_steps:
                name = self._generate_bdd_keyword_name(current_steps, current_intent)
                groups.append(
                    BddKeywordGroup(
                        name=name,
                        steps=list(current_steps),
                        bdd_intent=current_intent,
                    )
                )
                current_steps.clear()

        for step in steps:
            kw_lower = (step.keyword or "").strip().lower()
            step_intent = self._classify_step_intent(kw_lower, step.arguments)
            step_phase = self._detect_phase(step)

            # Phase 2.1: Phase change forces a new group
            phase_changed = (
                step_phase is not None
                and current_phase is not None
                and step_phase != current_phase
            )

            # Phase 2.2: Max group size constraint
            size_exceeded = len(current_steps) >= 7

            if step_intent != current_intent or phase_changed or size_exceeded:
                _flush()
                current_intent = step_intent

            if step_phase is not None:
                current_phase = step_phase

            current_steps.append(step)

        _flush()
        return groups

    def _classify_step_intent(
        self, keyword_lower: str, arguments: Optional[List[str]] = None
    ) -> str:
        """Classify a single keyword into given/when/then.

        Improved: considers arguments to distinguish assertions from observations.
        ``Get Text`` with an assertion operator (==, contains, should) is "then";
        without one it's "when" (observation/data extraction).
        """
        if keyword_lower in self._SETUP_KEYWORDS:
            return "given"
        for prefix in self._ASSERTION_KEYWORDS_PREFIXES:
            if keyword_lower.startswith(prefix):
                return "then"

        # Get keywords: only "then" if they have an assertion operator
        if keyword_lower.startswith("get ") and keyword_lower in (
            "get text", "get element count", "get title", "get url",
            "get attribute", "get property",
        ):
            args = arguments or []
            # Check for assertion operators in arguments
            assertion_ops = {"==", "!=", ">=", "<=", ">", "<",
                             "contains", "should", "matches", "starts",
                             "ends", "validate", "then", "equal"}
            for arg in args:
                if str(arg).strip().lower() in assertion_ops:
                    return "then"
            # No assertion operator → observation, treat as "when"
            return "when"

        return "when"

    def _detect_phase(self, step: TestCaseStep) -> Optional[str]:
        """Detect the semantic phase of a step from its locator content."""
        args = step.arguments or []
        if not args:
            return None
        humanized = self._humanize_locator(args[0]).lower()
        for phase, indicators in self._PHASE_INDICATORS.items():
            for indicator in indicators:
                if re.search(indicator, humanized):
                    return phase
        return None

    def _generate_bdd_keyword_name(
        self, steps: List[TestCaseStep], intent: str
    ) -> str:
        """Generate a natural-language BDD keyword name from steps.

        Phase 2.3: Improved naming with locator content analysis and
        domain-aware multi-step names.
        """
        if len(steps) == 1:
            return self._name_single_step(steps[0], intent)

        # Multi-step groups
        if intent == "given":
            return self._name_given_group(steps)

        if intent == "then":
            return self._name_then_group(steps)

        # "when" multi-step
        return self._name_when_group(steps)

    def _name_single_step(self, step: TestCaseStep, intent: str) -> str:
        """Name a single-step BDD keyword."""
        kw = (step.keyword or "").strip().lower()
        args = step.arguments or []

        # Setup keywords
        if kw in ("new browser", "new page", "open browser", "go to"):
            target = args[0] if args else "the page"
            if "://" in str(target):
                try:
                    from urllib.parse import urlparse
                    domain = urlparse(str(target)).hostname or "the page"
                    domain = domain.replace("www.", "").split(".")[0]
                    return f"the {domain} is open"
                except Exception:
                    pass
            return "the page is open"

        if kw == "new context":
            return "a browser context is created"

        if kw == "click":
            target = self._humanize_locator(args[0]) if args else "the element"
            return f"the user clicks {target}"

        if kw in ("fill text", "fill", "type text"):
            target = self._humanize_locator(args[0]) if args else "the field"
            return f"the user fills in {target}"

        if kw in ("select from list by value", "select from list by label",
                   "select from list by index"):
            target = self._humanize_locator(args[0]) if args else "the dropdown"
            return f"the user selects from {target}"

        if kw in ("check checkbox", "select checkbox"):
            target = self._humanize_locator(args[0]) if args else "the checkbox"
            return f"the user checks {target}"

        if kw in ("get text", "get element count"):
            target = self._humanize_locator(args[0]) if args else "the element"
            # Phase 1.3: Avoid "the the element" double article
            target = self._strip_leading_article(target)
            # Check if we have an assertion value to make name more specific
            if len(args) >= 3:
                return f"the {target} should show the expected value"
            return f"the {target} shows the expected value"

        if "should" in kw:
            return "the result is correct"

        # Hover keywords
        if kw == "hover":
            target = self._humanize_locator(args[0]) if args else "the element"
            return f"the user hovers over {self._strip_leading_article(target)}"

        # Generic single-step
        return f"the user performs {kw}"

    def _strip_leading_article(self, text: str) -> str:
        """Strip leading 'the ' to avoid 'the the X' double-article."""
        if text.startswith("the "):
            return text[4:]
        return text

    def _name_given_group(self, steps: List[TestCaseStep]) -> str:
        """Name a multi-step 'given' group."""
        for step in steps:
            for arg in (step.arguments or []):
                if "://" in str(arg):
                    try:
                        from urllib.parse import urlparse
                        domain = urlparse(str(arg)).hostname or "the application"
                        domain = domain.replace("www.", "").split(".")[0]
                        return f"the {domain} is open"
                    except Exception:
                        pass
        return "the application is set up"

    def _name_then_group(self, steps: List[TestCaseStep]) -> str:
        """Name a multi-step 'then' group with specificity."""
        # Extract targets from assertion steps
        targets = []
        for step in steps:
            if step.arguments:
                target = self._humanize_locator(step.arguments[0])
                target = self._strip_leading_article(target)
                if target != "element":
                    targets.append(target)

        unique_targets = list(dict.fromkeys(targets))

        if len(unique_targets) == 1:
            return f"the {unique_targets[0]} should show the expected value"
        if len(unique_targets) == 2:
            return f"the {unique_targets[0]} and {unique_targets[1]} are verified"
        if unique_targets:
            return f"the {unique_targets[0]} results are verified"
        return "the expected results are verified"

    def _name_when_group(self, steps: List[TestCaseStep]) -> str:
        """Name a multi-step 'when' group with action awareness."""
        # Classify each step's action type and target
        fills = []
        clicks = []
        for step in steps:
            kw_lower = (step.keyword or "").strip().lower()
            target = (
                self._humanize_locator(step.arguments[0])
                if step.arguments else "element"
            )
            target = self._strip_leading_article(target)
            if kw_lower in ("fill text", "fill", "type text"):
                fills.append(target)
            elif kw_lower == "click":
                clicks.append(target)

        # Pure fill group
        if fills and not clicks:
            if len(fills) <= 2:
                return f"the user fills in {' and '.join(fills)}"
            return "the user fills in the form"

        # Pure click group
        if clicks and not fills:
            if len(clicks) == 1:
                return f"the user clicks {clicks[0]}"
            # Use the last click (often the submit/confirm action)
            return f"the user clicks {clicks[-1]}"

        # Mixed — describe the dominant action
        if fills and clicks:
            # Fill + click pattern = form submission
            if len(clicks) == 1:
                return f"the user fills in the form and clicks {clicks[0]}"
            return "the user completes the form"

        return "the user performs the action"

    def _humanize_locator(self, locator: Any) -> str:
        """Convert a CSS/ARIA locator to human-readable text."""
        loc = str(locator)
        # aria-label extraction
        if "aria-label=" in loc:
            match = re.search(r'aria-label="([^"]+)"', loc)
            if match:
                return match.group(1).lower()
        # role=button[name="..."] extraction
        if 'name="' in loc:
            match = re.search(r'name="([^"]+)"', loc)
            if match:
                return match.group(1).lower()
        # text= extraction
        if loc.startswith("text="):
            return loc[5:].strip('"').strip("'").lower()
        # id= extraction
        if loc.startswith("id=") or loc.startswith("#"):
            name = loc.replace("id=", "").replace("#", "").replace("-", " ").replace("_", " ")
            return f"the {name} field"
        # data-testid extraction
        if "data-testid=" in loc or "data-test=" in loc:
            match = re.search(r'data-test(?:id)?="([^"]+)"', loc)
            if match:
                return match.group(1).replace("-", " ").replace("_", " ")
        # CSS class extraction — Phase 1.4: strip pseudo-selectors
        if loc.startswith("."):
            name = loc[1:]
            name = re.sub(r':[\w-]+\([^)]*\)', '', name)  # :nth-child(1) etc.
            name = re.sub(r':[\w-]+', '', name)  # :hover, :focus etc.
            return name.replace("-", " ").replace("_", " ").strip()
        # xpath: extract meaningful text from XPath expressions
        if loc.startswith("//") or loc.startswith("xpath="):
            match = re.search(r"@(?:Name|name|text\(\))='([^']+)'", loc)
            if not match:
                match = re.search(r'@(?:Name|name|text\(\))="([^"]+)"', loc)
            if match:
                return match.group(1).lower()
        # Fallback
        return "the element"

    def _detect_embedded_args(
        self, keywords: List[BddKeyword]
    ) -> List[BddKeyword]:
        """Detect repeated patterns and create embedded argument keywords.

        When multiple keywords differ only in a literal argument value, merge them
        into a single keyword with an embedded ``${arg}`` placeholder.
        """
        if len(keywords) < 2:
            return keywords

        # Group keywords by structural similarity: same number of steps,
        # same keyword sequence, differing only in argument values.
        from collections import defaultdict

        def _structural_key(kw: BddKeyword) -> Tuple:
            return tuple(
                (s.keyword, len(s.arguments or []))
                for s in kw.steps
            )

        struct_groups: Dict[Tuple, List[BddKeyword]] = defaultdict(list)
        for kw in keywords:
            key = _structural_key(kw)
            struct_groups[key].append(kw)

        result: List[BddKeyword] = []
        merged_names: set = set()

        for key, group in struct_groups.items():
            if len(group) < 2:
                result.extend(group)
                continue

            # Find differing argument positions
            first = group[0]
            diff_positions: List[Tuple[int, int]] = []  # (step_idx, arg_idx)
            for step_idx, first_step in enumerate(first.steps):
                for arg_idx in range(len(first_step.arguments or [])):
                    values = set()
                    for kw in group:
                        val = str(kw.steps[step_idx].arguments[arg_idx])
                        values.add(val)
                    if len(values) > 1:
                        diff_positions.append((step_idx, arg_idx))

            if not diff_positions:
                # No differences — keep all
                result.extend(group)
                continue

            # Create a merged keyword with embedded args
            merged_steps: List[TestCaseStep] = []
            embedded_args: Dict[str, str] = {}
            arg_counter = 0

            for step_idx, step in enumerate(first.steps):
                new_args = list(step.arguments or [])
                for arg_idx in range(len(new_args)):
                    if (step_idx, arg_idx) in diff_positions:
                        arg_name = f"arg{arg_counter + 1}" if arg_counter > 0 else "value"
                        embedded_args[arg_name] = str(new_args[arg_idx])
                        new_args[arg_idx] = f"${{{arg_name}}}"
                        arg_counter += 1
                merged_steps.append(
                    TestCaseStep(
                        keyword=step.keyword,
                        arguments=new_args,
                        assigned_variables=step.assigned_variables,
                        assignment_type=step.assignment_type,
                    )
                )

            # Build merged keyword name with embedded args
            base_name = first.name
            # Try to substitute the varying part in the name as well
            for kw in group:
                merged_names.add(kw.name)

            # Create the merged keyword with embedded arg placeholders in name
            # Find common prefix/suffix in names to build template
            merged_name = self._build_embedded_name(
                [kw.name for kw in group], embedded_args
            )

            merged_kw = BddKeyword(
                name=merged_name,
                steps=merged_steps,
                bdd_intent=first.bdd_intent,
                embedded_args=embedded_args,
                is_embedded=True,
            )
            result.append(merged_kw)

        return result

    def _build_embedded_name(
        self, names: List[str], embedded_args: Dict[str, str]
    ) -> str:
        """Build an embedded-argument keyword name from a set of similar names.

        Phase 1.1: Uses WORD-level prefix/suffix matching instead of
        character-level, avoiding malformed names like ``the c ${value}``.
        """
        if not names or len(names) < 2:
            return names[0] if names else "the user performs the action"

        arg_name = list(embedded_args.keys())[0] if embedded_args else "value"

        # Split all names into word lists
        word_lists = [n.split() for n in names]
        first_words = word_lists[0]

        # Find common word prefix
        prefix_len = 0
        for i in range(len(first_words)):
            word = first_words[i]
            if all(
                len(wl) > i and wl[i] == word
                for wl in word_lists[1:]
            ):
                prefix_len = i + 1
            else:
                break

        # Find common word suffix
        suffix_len = 0
        for i in range(1, len(first_words) + 1):
            word = first_words[-i]
            if all(
                len(wl) >= i and wl[-i] == word
                for wl in word_lists[1:]
            ):
                suffix_len = i
            else:
                break

        # Avoid overlap
        if prefix_len + suffix_len > len(first_words):
            suffix_len = max(0, len(first_words) - prefix_len)

        prefix_words = first_words[:prefix_len] if prefix_len else []
        suffix_words = first_words[len(first_words) - suffix_len:] if suffix_len else []

        if not prefix_words and not suffix_words:
            # No common words — use first name's leading words + ${arg}
            return f'{" ".join(first_words[:-1])} ${{{arg_name}}}' if len(first_words) > 1 else f'${{{arg_name}}}'

        parts = []
        if prefix_words:
            parts.append(" ".join(prefix_words))
        parts.append(f'${{{arg_name}}}')
        if suffix_words:
            parts.append(" ".join(suffix_words))

        return " ".join(parts)

    # ── End BDD transformation methods ─────────────────────────────

    def _apply_data_driven_mode(
        self, suite: GeneratedTestSuite, mode: str
    ) -> None:
        """Apply data-driven mode promotion to the suite.

        Modes:
        - "auto": auto-detect — if all TCs share the same template AND
          each TC has exactly 1 data row, promote to suite_template.
        - "per_test": no promotion (current [Template] per test case).
        - "suite_template": force promotion — set Test Template in Settings,
          suppress per-test [Template].
        """
        if mode == "per_test":
            return

        # Collect templates from all test cases
        templates = {tc.template for tc in suite.test_cases if tc.template}

        if not templates:
            return  # No template test cases

        if len(templates) != 1:
            return  # Mixed templates — can't promote

        common_template = templates.pop()

        # Check if each TC has exactly 1 data row (named-row pattern)
        all_single_row = all(
            len([s for s in tc.steps if not (s.keyword and s.keyword.strip())]) <= 1
            for tc in suite.test_cases
            if tc.template
        )

        should_promote = False
        if mode == "suite_template":
            should_promote = True
        elif mode == "auto":
            # Auto-promote when: all TCs share template AND have ≤1 data row each
            # AND there are at least 2 template TCs (otherwise per_test is fine)
            template_tcs = [tc for tc in suite.test_cases if tc.template]
            should_promote = all_single_row and len(template_tcs) >= 2

        if should_promote:
            suite.suite_template = common_template

            # Extract column headers from multiple sources:
            # 1. BDD keyword arg_vars (when bdd_style=True)
            headers_found = False
            for kw in (suite.bdd_keywords or []):
                if kw.name == common_template and kw.arg_vars:
                    suite.column_headers = [
                        v.strip("${}") for v in kw.arg_vars
                    ]
                    headers_found = True
                    break

            # 2. Template implementation steps (extract ${var} names)
            if not headers_found:
                var_pattern = re.compile(r'\$\{([^}]+)\}')
                col_names: List[str] = []
                seen_vars: set = set()
                for tc in suite.test_cases:
                    for step in tc.steps:
                        if step.keyword and step.keyword.strip():
                            for arg in (step.arguments or []):
                                for m in var_pattern.finditer(str(arg)):
                                    vn = m.group(1)
                                    if vn not in seen_vars and vn.upper() not in self._RF_BUILTIN_VARS:
                                        seen_vars.add(vn)
                                        col_names.append(vn)
                    if col_names:
                        break  # First TC with impl steps is enough
                if col_names:
                    suite.column_headers = col_names

            # Suppress per-test [Template] — suite_template mode handles it
            for tc in suite.test_cases:
                if tc.template == common_template:
                    tc.template = None

    def _build_impl_steps_from_test_info(self, test_info) -> List[TestCaseStep]:
        """Extract implementation steps (keyword-bearing) from a TestInfo."""
        steps = []
        for exec_step in test_info.steps:
            steps.append(TestCaseStep(
                keyword=exec_step.keyword,
                arguments=exec_step.arguments,
                assigned_variables=getattr(exec_step, "assigned_variables", []),
                assignment_type=getattr(exec_step, "assignment_type", None),
                bdd_group=getattr(exec_step, "bdd_group", None),
                bdd_intent=getattr(exec_step, "bdd_intent", None),
            ))
        return steps

    def _build_test_case_from_test_info(self, test_info) -> "GeneratedTestCase":
        """Build a GeneratedTestCase from a TestInfo (ADR-005 multi-test mode)."""
        steps = []
        for exec_step in test_info.steps:
            tc_step = TestCaseStep(
                keyword=exec_step.keyword,
                arguments=exec_step.arguments,
                assigned_variables=getattr(exec_step, "assigned_variables", []),
                assignment_type=getattr(exec_step, "assignment_type", None),
                bdd_group=getattr(exec_step, "bdd_group", None),
                bdd_intent=getattr(exec_step, "bdd_intent", None),
            )
            steps.append(tc_step)

        setup = None
        if test_info.setup:
            setup = TestCaseStep(
                keyword=test_info.setup["keyword"],
                arguments=test_info.setup.get("arguments", []),
            )

        teardown = None
        if test_info.teardown:
            teardown = TestCaseStep(
                keyword=test_info.teardown["keyword"],
                arguments=test_info.teardown.get("arguments", []),
            )

        # Append data rows from TestInfo (added via add_data_row action)
        data_rows = getattr(test_info, 'data_rows', [])
        if data_rows:
            for row in data_rows:
                steps.append(TestCaseStep(keyword="", arguments=list(row)))

        return GeneratedTestCase(
            name=test_info.name,
            steps=steps,
            documentation=test_info.documentation,
            tags=test_info.tags,
            template=getattr(test_info, 'template', None),  # ADR-019
            setup=setup,
            teardown=teardown,
        )

    async def _get_session_steps(self, session_id: str) -> List[Dict[str, Any]]:
        """Get executed steps from session."""
        if not self.execution_engine:
            logger.warning("No execution engine provided, returning empty steps list")
            return []

        try:
            # Get session from execution engine
            session = self.execution_engine.sessions.get(session_id)
            if not session:
                logger.warning(f"Session '{session_id}' not found")
                return []

            # Update session activity to prevent cleanup during suite building
            from datetime import datetime

            session.last_activity = datetime.now()
            logger.debug(f"Updated session {session_id} activity during suite building")

            # Convert ExecutionStep objects to dictionary format
            steps = []
            for step in session.steps:
                step_dict = {
                    "keyword": step.keyword,
                    "arguments": step.arguments,
                    "status": step.status,
                    "step_id": step.step_id,
                    # Include variable assignment information for test suite generation
                    "assigned_variables": step.assigned_variables,
                    "assignment_type": step.assignment_type,
                }

                # Add optional fields if available
                if step.error:
                    step_dict["error"] = step.error
                if step.result:
                    step_dict["result"] = step.result
                if step.start_time and step.end_time:
                    step_dict["duration"] = (
                        step.end_time - step.start_time
                    ).total_seconds()

                steps.append(step_dict)

            logger.info(f"Retrieved {len(steps)} steps from session '{session_id}'")
            return steps

        except Exception as e:
            logger.error(f"Error retrieving session steps: {e}")
            return []

    def _reorder_steps_by_variable_dependencies(
        self, steps: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Reorder steps so variable definitions precede their usages.

        This fixes issues caused by parallel tool calls arriving in execution order
        rather than logical order. Uses topological sort based on variable dependencies.

        Args:
            steps: List of step dictionaries

        Returns:
            Reordered list of steps with definitions before usages
        """
        import re

        if not steps:
            return steps

        # Variable pattern: ${VAR}, $VAR, @{LIST}, %{ENV}
        var_pattern = re.compile(r'[\$@%]\{?(\w+)\}?')

        # Build mapping of which steps define which variables
        step_defines: Dict[int, set] = {}  # step_index -> set of variable names defined
        step_uses: Dict[int, set] = {}     # step_index -> set of variable names used

        for i, step in enumerate(steps):
            step_defines[i] = set()
            step_uses[i] = set()

            # Variables defined by this step
            assigned = step.get("assigned_variables", [])
            for var in assigned:
                # Extract variable name from ${VAR} or $VAR
                match = var_pattern.search(str(var))
                if match:
                    step_defines[i].add(match.group(1))

            # For Set Variable keywords, the first argument defines the variable
            keyword = step.get("keyword", "").lower()
            if keyword in ["set suite variable", "set global variable", "set test variable"]:
                args = step.get("arguments", [])
                if args:
                    match = var_pattern.search(str(args[0]))
                    if match:
                        step_defines[i].add(match.group(1))

            # Variables used by this step (in arguments)
            for arg in step.get("arguments", []):
                for match in var_pattern.finditer(str(arg)):
                    var_name = match.group(1)
                    # Don't count a variable as "used" if this step defines it
                    if var_name not in step_defines[i]:
                        step_uses[i].add(var_name)

        # Build dependency graph: step i depends on step j if j defines a var that i uses
        dependencies: Dict[int, set] = {i: set() for i in range(len(steps))}
        var_to_definer: Dict[str, int] = {}  # variable name -> step index that defines it

        # First pass: record which step defines each variable
        for i, defines in step_defines.items():
            for var in defines:
                var_to_definer[var] = i

        # Second pass: build dependencies
        for i, uses in step_uses.items():
            for var in uses:
                if var in var_to_definer:
                    definer = var_to_definer[var]
                    if definer != i:  # Don't depend on self
                        dependencies[i].add(definer)

        # Topological sort using Kahn's algorithm
        # Count incoming edges for each node
        in_degree = {i: 0 for i in range(len(steps))}
        for deps in dependencies.values():
            for dep in deps:
                in_degree[dep] = in_degree.get(dep, 0)

        # Actually count incoming edges correctly
        in_degree = {i: 0 for i in range(len(steps))}
        for i, deps in dependencies.items():
            for dep in deps:
                # dep -> i means i depends on dep, so in_degree[i] should increase
                pass
        # Recalculate: for each dependency i -> dep, dep must come before i
        # So we want: i has dependencies, i's in_degree = len(dependencies[i])
        in_degree = {i: len(deps) for i, deps in dependencies.items()}

        # Queue of steps with no dependencies (in_degree = 0)
        queue = [i for i, deg in in_degree.items() if deg == 0]
        result = []

        while queue:
            # Pick the one with smallest original index to maintain order stability
            queue.sort()
            current = queue.pop(0)
            result.append(current)

            # Remove this node and update in_degrees
            for i, deps in dependencies.items():
                if current in deps:
                    in_degree[i] -= 1
                    if in_degree[i] == 0 and i not in result and i not in queue:
                        queue.append(i)

        # If we couldn't order all steps, there's a cycle - return original order
        if len(result) != len(steps):
            return steps

        # Return steps in new order
        return [steps[i] for i in result]

    async def _build_test_case(
        self,
        steps: List[Dict[str, Any]],
        test_name: str,
        tags: List[str],
        documentation: str,
        session_id: str = None,
    ) -> GeneratedTestCase:
        """Build a test case from execution steps."""

        # Reorder steps so variable definitions precede usages
        # This fixes issues caused by parallel tool calls arriving out of order
        steps = self._reorder_steps_by_variable_dependencies(steps)

        # Convert steps to test case steps
        test_steps = []
        imports = set()

        # Get suite_level_variables from session to filter out duplicate Set Variable steps
        # These variables are already rendered in the *** Variables *** section
        suite_level_var_names: set = set()
        # Collect IF/WHILE conditions from flow_blocks to filter orphan Evaluate steps
        # When an AI Agent calls Evaluate with the same expression as an IF condition,
        # the Evaluate is redundant - the IF block already handles the condition evaluation
        if_conditions: set = set()
        if session_id and self.execution_engine:
            try:
                sess = self.execution_engine.sessions.get(session_id)
                if sess:
                    suite_level_var_names = getattr(sess, "suite_level_variables", set()) or set()
                    # Collect IF conditions from flow_blocks
                    for block in getattr(sess, "flow_blocks", []) or []:
                        if block.get("type") == "if":
                            cond = block.get("condition", "")
                            if cond:
                                # Normalize condition for comparison
                                # Convert both ${var} and $var forms to canonical form
                                norm_cond = self._convert_to_evaluation_namespace_syntax(cond).strip()
                                if_conditions.add(norm_cond)
            except Exception:
                pass

        for step in steps:
            keyword = step.get("keyword", "")
            arguments = step.get("arguments", [])
            assigned_variables = step.get("assigned_variables", [])
            assignment_type = step.get("assignment_type")

            # Handle import statements separately
            if keyword.lower() == "import library":
                if arguments:
                    imports.add(arguments[0])
                continue

            # Filter out Set Variable steps for variables already in suite_level_variables
            # These variables are rendered in *** Variables *** section, so we don't want duplicates
            if keyword.lower() in ["set suite variable", "set global variable", "set test variable"]:
                if arguments:
                    # Extract variable name from first argument (e.g., "${VAR}" -> "VAR")
                    var_arg = str(arguments[0])
                    var_name = var_arg.strip("${}")
                    if var_name in suite_level_var_names:
                        # Skip this step - variable is already in *** Variables *** section
                        continue

            # Filter out orphan Evaluate steps that match IF conditions
            # When an AI Agent calls Evaluate with the same expression as an IF condition,
            # without assigning the result to a variable, it's redundant
            if keyword.lower() == "evaluate" and if_conditions:
                # Only filter if there's no assignment (orphan Evaluate)
                if not assigned_variables:
                    if arguments:
                        expr = str(arguments[0]).strip()
                        # Normalize expression for comparison
                        norm_expr = self._convert_to_evaluation_namespace_syntax(expr).strip()
                        if norm_expr in if_conditions:
                            # Skip this step - it's an orphan Evaluate matching an IF condition
                            continue

            # Extract BDD fields from step dict
            bdd_group = step.get("bdd_group")
            bdd_intent = step.get("bdd_intent")

            # Apply optimizations with assignment information
            optimized_step = await self._optimize_step(
                keyword,
                arguments,
                test_steps,
                session_id,
                assigned_variables,
                assignment_type,
            )

            if optimized_step:  # Only add if not filtered out by optimization
                # Carry BDD annotations through to the TestCaseStep
                if bdd_group:
                    optimized_step.bdd_group = bdd_group
                if bdd_intent:
                    optimized_step.bdd_intent = bdd_intent
                test_steps.append(optimized_step)

        # Generate meaningful documentation if not provided
        if not documentation:
            documentation = await self._generate_documentation(test_steps, test_name)

        # Add setup and teardown if needed
        setup, teardown = await self._generate_setup_teardown(test_steps)

        return GeneratedTestCase(
            name=test_name,
            steps=test_steps,
            documentation=documentation,
            tags=tags or [],
            setup=setup,
            teardown=teardown,
        )

    async def _build_test_suite(
        self, test_cases: List[GeneratedTestCase], session_id: str
    ) -> GeneratedTestSuite:
        """Build a test suite from test cases."""

        # Determine libraries strictly from keywords used in the generated steps
        # Prefer RF context namespace mapping; fall back to pattern detection
        all_imports: set = set()
        keyword_to_lib: Dict[str, str] = {}
        try:
            from robotmcp.components.execution.rf_native_context_manager import (
                get_rf_native_context_manager,
            )

            mgr = get_rf_native_context_manager()
            lst = mgr.list_available_keywords(session_id)
            if lst.get("success"):
                for item in lst.get("library_keywords", []) or []:
                    # Map lowercased keyword name to library
                    name = str(item.get("name", "")).lower()
                    lib = item.get("library")
                    if name and lib:
                        keyword_to_lib[name] = lib
        except Exception:
            pass

        for test_case in test_cases:
            for step in test_case.steps:
                kw = step.keyword or ""
                # Explicit prefix (Library.Keyword)
                if "." in kw:
                    prefix = kw.split(".", 1)[0].strip()
                    if prefix and prefix != "BuiltIn":
                        all_imports.add(prefix)
                        continue
                # RF namespace mapping
                lib = keyword_to_lib.get(kw.lower())
                if lib and lib != "BuiltIn":
                    all_imports.add(lib)
                    continue
                # Pattern-based fallback
                library = await self._detect_library_from_keyword(kw, session_id)
                if library and library != "BuiltIn":
                    all_imports.add(library)

        # Resolve conflicting web automation libraries
        # If both Browser and SeleniumLibrary are detected, determine which one to use
        # based on library-specific keywords
        all_imports = self._resolve_web_library_conflict(all_imports, test_cases)

        # Validate library exclusion rules for test suite generation
        # Only validate if we don't have a session with execution history
        if not self._session_has_execution_history(session_id):
            self._validate_suite_library_exclusions(all_imports, session_id)

        # BuiltIn is automatically available in Robot Framework, so we don't import it explicitly

        # Generate suite documentation
        suite_docs = await self._generate_suite_documentation(test_cases, session_id)

        # Generate common tags
        common_tags = await self._extract_common_tags(test_cases)

        # Pull resources imported into the RF context for this session
        resources: List[str] = []
        try:
            from robotmcp.components.execution.rf_native_context_manager import (
                get_rf_native_context_manager,
            )

            mgr = get_rf_native_context_manager()
            ctx = getattr(mgr, "_session_contexts", {}).get(session_id)
            if ctx and ctx.get("resources"):
                resources = list(ctx.get("resources"))
        except Exception:
            resources = []

        # Include flow blocks recorded during execution (if any)
        flow_blocks = None
        try:
            sess = self.execution_engine.sessions.get(session_id)
            if sess and hasattr(sess, "flow_blocks") and sess.flow_blocks:
                flow_blocks = list(sess.flow_blocks)
        except Exception:
            flow_blocks = None

        # Collect session variables for inclusion in *** Variables *** section
        # IMPORTANT: Only variables explicitly set via manage_session(action="set_variables", scope="suite")
        # should be included in the *** Variables *** section.
        # Variables created via RF keywords (Set Variable, Set Suite Variable, Set Test Variable,
        # Set Global Variable) or VAR syntax should remain INLINE in the test cases.
        # Variable files are imported via *** Settings *** section (Variables keyword).
        suite_variables: Dict[str, Any] = {}
        variable_files: List[str] = []
        try:
            sess = self.execution_engine.sessions.get(session_id)
            if sess:
                # Only include variables that were explicitly set via manage_session
                # These are tracked in the suite_level_variables set
                suite_level_var_names = getattr(sess, "suite_level_variables", set()) or set()

                if suite_level_var_names and hasattr(sess, "variables") and sess.variables:
                    for var_name in suite_level_var_names:
                        # Look up value in session variables
                        if var_name in sess.variables:
                            suite_variables[var_name] = sess.variables[var_name]
                        # Also check decorated form ${var_name}
                        decorated_name = f"${{{var_name}}}"
                        if decorated_name in sess.variables and var_name not in suite_variables:
                            suite_variables[var_name] = sess.variables[decorated_name]

                # Get imported variable files from session for *** Settings *** section
                if hasattr(sess, "loaded_variable_files") and sess.loaded_variable_files:
                    variable_files = [
                        vf.get("path") for vf in sess.loaded_variable_files if vf.get("path")
                    ]

                logger.debug(
                    f"Collected {len(suite_variables)} suite-level variables and {len(variable_files)} "
                    f"variable files from session {session_id}"
                )
        except Exception as e:
            logger.warning(f"Could not collect session variables: {e}")
            suite_variables = {}
            variable_files = []

        return GeneratedTestSuite(
            name=f"Generated_Suite_{session_id}",
            test_cases=test_cases,
            documentation=suite_docs,
            tags=common_tags,
            imports=list(all_imports),
            resources=resources,
            flow_blocks=flow_blocks,
            variables=suite_variables if suite_variables else None,
            variable_files=variable_files if variable_files else None,
        )

    async def _optimize_step(
        self,
        keyword: str,
        arguments: List[str],
        existing_steps: List[TestCaseStep],
        session_id: str = None,
        assigned_variables: List[str] = None,
        assignment_type: str = None,
    ) -> Optional[TestCaseStep]:
        """Apply optimization rules to a step."""

        # Rule: Combine consecutive waits
        if self.optimization_rules.get("combine_waits") and keyword.lower() in [
            "sleep",
            "wait",
        ]:
            if existing_steps and existing_steps[-1].keyword.lower() in [
                "sleep",
                "wait",
            ]:
                # Skip this wait step as it's redundant
                return None

        # Rule: Remove redundant verifications
        if self.optimization_rules.get("remove_redundant_verifications"):
            if keyword.lower().startswith("page should contain"):
                # Check if we already have the same verification
                for step in existing_steps:
                    if (
                        step.keyword.lower().startswith("page should contain")
                        and step.arguments == arguments
                    ):
                        return None  # Skip redundant verification

        # Use original arguments - they already worked during execution
        processed_arguments = arguments

        # Rule: Add meaningful comments
        comment = None
        if self.optimization_rules.get("add_meaningful_comments"):
            comment = await self._generate_step_comment(keyword, processed_arguments)

        return TestCaseStep(
            keyword=keyword,
            arguments=processed_arguments,
            comment=comment,
            assigned_variables=assigned_variables or [],
            assignment_type=assignment_type,
        )

    async def _generate_step_comment(
        self, keyword: str, arguments: List[str]
    ) -> Optional[str]:
        """Generate a meaningful comment for a step."""

        keyword_lower = keyword.lower()

        if "open browser" in keyword_lower:
            url = arguments[0] if arguments else "default"
            browser = arguments[1] if len(arguments) > 1 else "default browser"
            return f"# Open {browser} and navigate to {url}"

        elif "input text" in keyword_lower:
            element = arguments[0] if arguments else "element"
            value = arguments[1] if len(arguments) > 1 else "value"
            return f"# Enter '{value}' into {element}"

        elif "click" in keyword_lower:
            element = arguments[0] if arguments else "element"
            return f"# Click on {element}"

        elif "should contain" in keyword_lower:
            text = arguments[0] if arguments else "text"
            return f"# Verify page contains '{text}'"

        return None

    async def _generate_documentation(
        self, steps: List[TestCaseStep], test_name: str
    ) -> str:
        """Generate documentation for a test case."""

        # Analyze steps to understand the test flow
        flow_description = []

        for step in steps:
            keyword_lower = step.keyword.lower()

            if "open browser" in keyword_lower:
                flow_description.append("Opens browser")
            elif "go to" in keyword_lower or "navigate" in keyword_lower:
                flow_description.append("Navigates to page")
            elif "input" in keyword_lower:
                flow_description.append("Enters data")
            elif "click" in keyword_lower:
                flow_description.append("Performs click action")
            elif "should" in keyword_lower or "verify" in keyword_lower:
                flow_description.append("Verifies result")
            elif "close" in keyword_lower:
                flow_description.append("Cleans up")

        if flow_description:
            description = ", ".join(flow_description)
            return f"Test case that {description.lower()}."

        return f"Automated test case: {test_name}"

    async def _generate_setup_teardown(
        self, steps: List[TestCaseStep]
    ) -> Tuple[Optional[TestCaseStep], Optional[TestCaseStep]]:
        """Generate setup and teardown steps if needed."""

        setup = None
        teardown = None

        # Check if we need browser cleanup
        has_browser_actions = any(
            "browser" in step.keyword.lower()
            or "click" in step.keyword.lower()
            or "fill" in step.keyword.lower()
            or "get text" in step.keyword.lower()
            or "input" in step.keyword.lower()
            for step in steps
        )

        # Determine if using Browser Library or SeleniumLibrary
        has_browser_lib = any(
            "new browser" in step.keyword.lower()
            or "new page" in step.keyword.lower()
            or "fill" in step.keyword.lower()
            for step in steps
        )

        if has_browser_actions:
            # Check if we already have close browser
            has_close = any("close browser" in step.keyword.lower() for step in steps)

            if not has_close:
                teardown = TestCaseStep(
                    keyword="Close Browser",
                    arguments=[],
                    comment="# Cleanup: Close browser",
                )

        return setup, teardown

    async def _detect_library_from_keyword(
        self, keyword: str, session_id: str = None
    ) -> Optional[str]:
        """
        Detect which library a keyword belongs to, respecting session library choice.

        Args:
            keyword: Keyword name to detect library for
            session_id: Session ID to check for library preference

        Returns:
            Library name or None
        """
        # First check if we have a session with a specific web automation library
        if (
            session_id
            and self.execution_engine
            and hasattr(self.execution_engine, "sessions")
        ):
            session = self.execution_engine.sessions.get(session_id)
            if session:
                session_web_lib = self._get_session_web_library(session)
                if session_web_lib:
                    # Check if this keyword could belong to the session's library
                    keyword_lower = keyword.lower().strip()

                    # For Browser Library keywords
                    if session_web_lib == "Browser" and any(
                        kw in keyword_lower
                        for kw in [
                            "click",
                            "fill text",
                            "get text",
                            "wait for elements state",
                            "check checkbox",
                            "select options by",
                            "hover",
                            "new browser",
                            "new page",
                        ]
                    ):
                        return "Browser"

                    # For SeleniumLibrary keywords
                    elif session_web_lib == "SeleniumLibrary" and any(
                        kw in keyword_lower
                        for kw in [
                            "click element",
                            "input text",
                            "select from list",
                            "wait until element",
                            "open browser",
                            "close browser",
                            "select checkbox",
                            "get text",
                        ]
                    ):
                        return "SeleniumLibrary"

        # Fallback to shared library detection utility with dynamic keyword discovery
        keyword_discovery = None
        if self.execution_engine and hasattr(
            self.execution_engine, "keyword_discovery"
        ):
            keyword_discovery = self.execution_engine.keyword_discovery

        return detect_library_from_keyword(keyword, keyword_discovery)

    def _get_session_libraries(self, session_id: str) -> set:
        """
        Get libraries that were actually used in session execution.

        This method prioritizes the session's imported_libraries over keyword pattern matching
        because the session has already proven these libraries work together successfully.

        Args:
            session_id: Session ID to get libraries from

        Returns:
            Set of library names from session execution history
        """
        if not self.execution_engine or not hasattr(self.execution_engine, "sessions"):
            return set()

        session = self.execution_engine.sessions.get(session_id)
        if not session:
            return set()

        # Use imported_libraries from session (these are known to work together)
        session_libraries = set(session.imported_libraries)

        # Filter out BuiltIn as it's automatically available
        session_libraries.discard("BuiltIn")

        logger.debug(
            f"Session {session_id} libraries from execution history: {session_libraries}"
        )
        return session_libraries

    def _session_has_execution_history(self, session_id: str) -> bool:
        """
        Check if session has execution history (successful steps).

        Args:
            session_id: Session ID to check

        Returns:
            True if session has executed steps, False otherwise
        """
        if not self.execution_engine or not hasattr(self.execution_engine, "sessions"):
            return False

        session = self.execution_engine.sessions.get(session_id)
        if not session:
            return False

        # Check if session has any successful steps
        has_history = len(session.steps) > 0
        logger.debug(
            f"Session {session_id} has execution history: {has_history} ({len(session.steps)} steps)"
        )
        return has_history

    def _get_session_web_library(self, session) -> Optional[str]:
        """
        Safely get the web automation library from a session.

        Handles both new sessions (with get_web_automation_library method)
        and older sessions (without the method).

        Args:
            session: ExecutionSession object

        Returns:
            Web automation library name or None
        """
        # Try the new method first
        if hasattr(session, "get_web_automation_library"):
            return session.get_web_automation_library()

        # Fallback for older sessions - check imported_libraries directly
        if hasattr(session, "imported_libraries"):
            web_automation_libs = ["Browser", "SeleniumLibrary"]
            for lib in session.imported_libraries:
                if lib in web_automation_libs:
                    return lib

        # Final fallback - check browser_state.active_library
        if hasattr(session, "browser_state") and hasattr(
            session.browser_state, "active_library"
        ):
            active_lib = session.browser_state.active_library
            if active_lib == "browser":
                return "Browser"
            elif active_lib == "selenium":
                return "SeleniumLibrary"

        return None

    def _resolve_web_library_conflict(
        self, imports: set, test_cases: List[GeneratedTestCase]
    ) -> set:
        """Resolve conflicts between Browser and SeleniumLibrary.

        When both libraries are detected (due to ambiguous keywords like Click, Get Text),
        this method determines which library to keep based on library-specific keywords.

        Args:
            imports: Set of detected library names
            test_cases: List of test cases to analyze for library-specific keywords

        Returns:
            Updated imports set with conflicting library removed
        """
        web_libs = {"Browser", "SeleniumLibrary"}
        detected_web_libs = imports & web_libs

        # Only resolve if both are detected
        if len(detected_web_libs) != 2:
            return imports

        # Keywords unique to each library (not shared)
        browser_unique = {
            "new browser", "new context", "new page", "close context", "close page",
            "fill text", "fill", "get viewport size", "set viewport size",
            "wait for elements state", "get element count", "get elements",
            "select options by", "check checkbox", "get property"
        }
        selenium_unique = {
            "open browser", "input text", "click element", "click button",
            "select from list", "wait until element", "page should contain",
            "element should be visible", "capture page screenshot",
            "maximize browser window", "select checkbox"
        }

        # Collect all keywords from test cases
        all_keywords = set()
        for tc in test_cases:
            for step in tc.steps:
                if step.keyword:
                    all_keywords.add(step.keyword.lower().strip())

        # Check for library-specific keywords
        has_browser_specific = any(kw in all_keywords for kw in browser_unique)
        has_selenium_specific = any(kw in all_keywords for kw in selenium_unique)

        # Resolve conflict based on unique keywords
        if has_browser_specific and not has_selenium_specific:
            logger.info(
                "Resolved library conflict: Keeping Browser (found Browser-specific keywords)"
            )
            imports.discard("SeleniumLibrary")
        elif has_selenium_specific and not has_browser_specific:
            logger.info(
                "Resolved library conflict: Keeping SeleniumLibrary (found SeleniumLibrary-specific keywords)"
            )
            imports.discard("Browser")
        elif has_browser_specific and has_selenium_specific:
            # Both specific keywords found - this is a real conflict, log warning
            logger.warning(
                "Both Browser and SeleniumLibrary specific keywords detected. "
                "Keeping Browser as default. Consider using separate sessions for each library."
            )
            imports.discard("SeleniumLibrary")
        else:
            # Only ambiguous keywords - default to Browser (more modern)
            logger.info(
                "No library-specific keywords found. Defaulting to Browser library."
            )
            imports.discard("SeleniumLibrary")

        return imports

    async def _generate_suite_documentation(
        self, test_cases: List[GeneratedTestCase], session_id: str
    ) -> str:
        """Generate documentation for the test suite."""

        case_count = len(test_cases)

        # Analyze test types
        test_types = set()
        for test_case in test_cases:
            for step in test_case.steps:
                if "browser" in step.keyword.lower():
                    test_types.add("web automation")
                elif "request" in step.keyword.lower():
                    test_types.add("API testing")
                elif "database" in step.keyword.lower():
                    test_types.add("database testing")

        type_description = ", ".join(test_types) if test_types else "automation"

        # Create a single-line documentation that won't break Robot Framework syntax
        doc = f"Test suite generated from session {session_id} containing {case_count} test case{'s' if case_count != 1 else ''} for {type_description}."

        return doc

    def _format_rf_documentation(self, documentation: str) -> List[str]:
        """Format documentation for Robot Framework syntax.

        Handles both single-line and multi-line documentation properly.
        Multi-line documentation uses '...' continuation markers.

        Args:
            documentation: Documentation string (can contain newlines)

        Returns:
            List of formatted documentation lines
        """
        if not documentation:
            return []

        # Split documentation into lines and clean them
        doc_lines = [
            line.strip() for line in documentation.strip().split("\n") if line.strip()
        ]

        if not doc_lines:
            return []

        formatted_lines = []

        if len(doc_lines) == 1:
            # Single line documentation
            formatted_lines.append(f"Documentation    {doc_lines[0]}")
        else:
            # Multi-line documentation with continuation markers
            formatted_lines.append(f"Documentation    {doc_lines[0]}")
            for line in doc_lines[1:]:
                formatted_lines.append(f"...              {line}")

        return formatted_lines

    def _format_rf_test_case_documentation(self, documentation: str) -> List[str]:
        """Format test case documentation for Robot Framework syntax.

        Similar to suite documentation but uses [Documentation] format with proper indentation.

        Args:
            documentation: Documentation string (can contain newlines)

        Returns:
            List of formatted test case documentation lines
        """
        if not documentation:
            return []

        # Split documentation into lines and clean them
        doc_lines = [
            line.strip() for line in documentation.strip().split("\n") if line.strip()
        ]

        if not doc_lines:
            return []

        formatted_lines = []

        if len(doc_lines) == 1:
            # Single line test case documentation
            formatted_lines.append(f"    [Documentation]    {doc_lines[0]}")
        else:
            # Multi-line test case documentation with continuation markers
            formatted_lines.append(f"    [Documentation]    {doc_lines[0]}")
            for line in doc_lines[1:]:
                formatted_lines.append(f"    ...                {line}")

        return formatted_lines

    async def _extract_common_tags(
        self, test_cases: List[GeneratedTestCase]
    ) -> List[str]:
        """Extract common tags across test cases."""

        if not test_cases:
            return []

        # Find tags that appear in all test cases
        common_tags = set(test_cases[0].tags or [])

        for test_case in test_cases[1:]:
            case_tags = set(test_case.tags or [])
            common_tags = common_tags.intersection(case_tags)

        # Add generated tags based on content analysis
        generated_tags = ["automated", "generated"]

        # Analyze test content for additional tags
        has_web = any(
            any("browser" in step.keyword.lower() for step in tc.steps)
            for tc in test_cases
        )
        if has_web:
            generated_tags.append("web")

        has_api = any(
            any("request" in step.keyword.lower() for step in tc.steps)
            for tc in test_cases
        )
        if has_api:
            generated_tags.append("api")

        return list(common_tags) + generated_tags

    async def _create_rf_suite(self, suite: GeneratedTestSuite) -> TestSuite:
        """Create Robot Framework API suite object."""

        rf_suite = TestSuite(name=suite.name)
        rf_suite.doc = suite.documentation

        # Add imports
        for library in suite.imports or []:
            rf_suite.resource.imports.library(library)
        for res in suite.resources or []:
            try:
                rf_suite.resource.imports.resource(res)
            except Exception:
                pass

        # Add test cases
        for test_case in suite.test_cases:
            rf_test = rf_suite.tests.create(
                name=test_case.name, doc=test_case.documentation
            )

            # ADR-019: Set template for data-driven tests
            if getattr(test_case, 'template', None):
                rf_test.template = test_case.template

            # Add tags
            if test_case.tags:
                rf_test.tags.add(test_case.tags)

            # Add setup
            if test_case.setup:
                escaped_setup_args = [
                    self._escape_robot_argument(arg)
                    for arg in (test_case.setup.arguments or [])
                ]
                rf_test.setup.config(
                    name=test_case.setup.keyword, args=escaped_setup_args
                )

            # Add steps
            for step in test_case.steps:
                escaped_step_args = [
                    self._escape_robot_argument(arg)
                    for arg in (step.arguments or [])
                ]
                rf_test.body.create_keyword(name=step.keyword, args=escaped_step_args)

            # Add teardown
            if test_case.teardown:
                escaped_teardown_args = [
                    self._escape_robot_argument(arg)
                    for arg in (test_case.teardown.arguments or [])
                ]
                rf_test.teardown.config(
                    name=test_case.teardown.keyword, args=escaped_teardown_args
                )

        return rf_suite

    async def _generate_rf_text(self, suite: GeneratedTestSuite) -> str:
        """Generate Robot Framework text representation."""

        lines = []

        # Suite header
        lines.append("*** Settings ***")

        # Format documentation properly for Robot Framework
        if suite.documentation:
            doc_lines = self._format_rf_documentation(suite.documentation)
            lines.extend(doc_lines)

        # Imports
        # Resources first, then libraries
        if suite.resources:
            for res in suite.resources:
                lines.append(f"Resource        {self._format_path_for_rf(res)}")
        if suite.imports:
            for library in suite.imports:
                # If library looks like a path, format it for RF portability
                if any(
                    ch in library
                    for ch in [
                        "\\\
",
                        "/",
                    ]
                ) or (":" in library and len(library) >= 2):
                    lib_line = self._format_path_for_rf(library)
                else:
                    lib_line = library
                lines.append(f"Library         {lib_line}")

        # Variable files go in Settings section per RF documentation:
        # https://robotframework.org/robotframework/latest/RobotFrameworkUserGuide.html#variable-files
        if suite.variable_files:
            for var_file in suite.variable_files:
                lines.append(f"Variables       {self._format_path_for_rf(var_file)}")

        # ADR-005: Suite Setup / Suite Teardown in Settings
        if suite.setup:
            escaped_args = [
                self._escape_robot_argument(a) for a in (suite.setup.arguments or [])
            ]
            args_str = ("    " + "    ".join(escaped_args)) if escaped_args else ""
            lines.append(f"Suite Setup     {suite.setup.keyword}{args_str}")
        if suite.teardown:
            escaped_args = [
                self._escape_robot_argument(a) for a in (suite.teardown.arguments or [])
            ]
            args_str = ("    " + "    ".join(escaped_args)) if escaped_args else ""
            lines.append(f"Suite Teardown  {suite.teardown.keyword}{args_str}")

        if suite.tags:
            lines.append(f"Test Tags       {'    '.join(suite.tags)}")

        # Suite-level Test Template (named-row data-driven pattern)
        if suite.suite_template:
            lines.append(f"Test Template    {suite.suite_template}")

        lines.append("")

        # Variables section - ONLY include variables explicitly set via manage_session
        # Variables created via RF keywords (Set Variable, Set Suite Variable, etc.)
        # or VAR syntax should remain inline in test cases, not in this section
        if suite.variables:
            lines.append("*** Variables ***")

            # Add inline variables (only suite-level variables from manage_session)
            for var_name, var_value in sorted(suite.variables.items()):
                formatted_line = self._format_variable_for_rf(var_name, var_value)
                lines.append(formatted_line)

            lines.append("")

        # Test cases — column headers for suite_template mode
        if suite.suite_template and suite.column_headers:
            header = "    ".join(suite.column_headers)
            lines.append(f"*** Test Cases ***    {header}")
        else:
            lines.append("*** Test Cases ***")

        for test_case in suite.test_cases:
            # Suite-template mode: name + args on SAME line (no [Template], no indented body)
            if suite.suite_template:
                data_steps = [s for s in test_case.steps
                              if not (s.keyword and s.keyword.strip())]
                if data_steps and data_steps[0].arguments:
                    escaped = [
                        self._escape_robot_argument(str(a))
                        for a in data_steps[0].arguments
                    ]
                    escaped = [a if a else "${EMPTY}" for a in escaped]
                    lines.append(f"{test_case.name}    {'    '.join(escaped)}")
                else:
                    lines.append(f"{test_case.name}")
                continue  # Skip normal per-test rendering

            lines.append(f"{test_case.name}")

            if test_case.documentation:
                # Format test case documentation properly
                test_doc_lines = self._format_rf_test_case_documentation(
                    test_case.documentation
                )
                lines.extend(test_doc_lines)

            if test_case.tags:
                lines.append(f"    [Tags]    {'    '.join(test_case.tags)}")

            # ADR-019: Template support for data-driven tests
            if getattr(test_case, 'template', None):
                lines.append(f"    [Template]    {test_case.template}")

            if test_case.setup:
                escaped_setup_args = [
                    self._escape_robot_argument(arg)
                    for arg in test_case.setup.arguments
                ]
                lines.append(
                    f"    [Setup]    {test_case.setup.keyword}    {'    '.join(escaped_setup_args)}"
                )

            # C2 fix: Template check BEFORE flow_blocks — template test cases
            # must always use data-row rendering regardless of flow_blocks.
            if getattr(test_case, 'template', None):
                # C1 fix: Only render DATA ROWS (empty keyword), not template body steps
                for step in test_case.steps:
                    # Data rows have empty or whitespace-only keyword
                    if step.keyword and step.keyword.strip():
                        continue  # Skip template body steps
                    escaped_args = [
                        self._escape_robot_argument(str(arg))
                        for arg in (step.arguments or [])
                    ]
                    if escaped_args:
                        # M2 fix: Replace empty values with ${EMPTY}
                        escaped_args = [a if a else '${EMPTY}' for a in escaped_args]
                        lines.append(f"    {'    '.join(escaped_args)}")
            else:
                # Flow-aware rendering: if structured flow blocks exist, merge them
                # with surrounding linear steps so that keywords before/after the
                # block are kept.
                # ADR-005: prefer per-test flow blocks over suite-level
                tc_flow_blocks = None
                if suite.per_test_flow_blocks and test_case.name in suite.per_test_flow_blocks:
                    tc_flow_blocks = suite.per_test_flow_blocks[test_case.name]
                elif hasattr(suite, 'flow_blocks') and suite.flow_blocks:
                    tc_flow_blocks = suite.flow_blocks
                if tc_flow_blocks:
                    # Map each flow block to its covered index range in linear steps
                    block_ranges, used_indices = self._map_blocks_to_ranges(test_case.steps, tc_flow_blocks)
                    cur = 0
                    for block, start_idx, end_idx in block_ranges:
                        # Emit linear steps before this block
                        while cur < start_idx:
                            if cur not in used_indices:
                                line = await self._render_linear_step(test_case.steps[cur])
                                lines.append(line)
                            cur += 1
                        # Emit this flow block in RF syntax
                        lines.extend(self._render_flow_blocks([block], indent="    "))
                        # Advance cur past the block
                        cur = max(cur, end_idx + 1)
                    # Emit remaining linear steps after last block
                    while cur < len(test_case.steps):
                        if cur not in used_indices:
                            line = await self._render_linear_step(test_case.steps[cur])
                            lines.append(line)
                        cur += 1
                else:
                    # Test steps (legacy linear rendering)
                    for step in test_case.steps:
                        line = await self._render_linear_step(step)
                        lines.append(line)

            if test_case.teardown:
                escaped_teardown_args = [
                    self._escape_robot_argument(arg)
                    for arg in test_case.teardown.arguments
                ]
                lines.append(
                    f"    [Teardown]    {test_case.teardown.keyword}    {'    '.join(escaped_teardown_args)}"
                )

            lines.append("")

        # BDD Keywords section
        if suite.bdd_keywords:
            lines.append("*** Keywords ***")
            for kw in suite.bdd_keywords:
                lines.append(kw.name)
                # Phase 4: Emit [Arguments] for template wrapper keywords
                arg_vars = kw.arg_vars if kw.arg_vars else None
                if arg_vars:
                    lines.append(f"    [Arguments]    {'    '.join(arg_vars)}")
                for step in kw.steps:
                    escaped_args = [
                        self._escape_robot_argument(str(a))
                        for a in (step.arguments or [])
                    ]
                    args_str = ("    " + "    ".join(escaped_args)) if escaped_args else ""
                    lines.append(f"    {step.keyword}{args_str}")
                lines.append("")

        return "\n".join(lines)

    async def _render_linear_step(self, step: TestCaseStep) -> str:
        """Render a single linear keyword step with proper escaping and assignments."""
        # Check if this is a Set Variable keyword that should be converted to VAR syntax
        # This applies to Set Test/Suite/Global Variable keywords that remain after filtering
        # (i.e., variables NOT in suite_level_variables - test-case-only variables)
        keyword_lower = (step.keyword or "").strip().lower()
        var_scope_mapping = {
            "set test variable": "TEST",
            "set suite variable": "SUITE",
            "set global variable": "GLOBAL",
        }

        if keyword_lower in var_scope_mapping and step.arguments:
            # Convert to VAR syntax: VAR    ${VAR}    value    scope=SCOPE
            scope = var_scope_mapping[keyword_lower]
            var_name = str(step.arguments[0])  # e.g., "${VAR}"
            # Remaining arguments are values
            if len(step.arguments) > 1:
                # Check if the value contains arithmetic expression (${var + 1} pattern)
                # VAR keyword cannot evaluate expressions - must use Evaluate keyword instead
                value_str = str(step.arguments[1])
                if self._is_arithmetic_expression(value_str):
                    # Convert to Evaluate: ${var_name} =    Evaluate    expression
                    # Extract expression from ${...} if present
                    expr = self._extract_expression(value_str)
                    if expr:
                        # Convert to evaluation namespace syntax and add int() wrapper if needed
                        converted_expr = self._convert_to_evaluation_namespace_syntax(expr)
                        # Add type conversion for arithmetic
                        converted_expr = self._add_type_conversion_if_needed(converted_expr)
                        return f"    {var_name} =    Evaluate    {converted_expr}"

                escaped_values = [self._escape_robot_argument(str(arg)) for arg in step.arguments[1:]]
                values_str = "    ".join(escaped_values)
                return f"    VAR    {var_name}    {values_str}    scope={scope}"
            else:
                # Variable with no value
                return f"    VAR    {var_name}    scope={scope}"

        # Generate variable assignment syntax if applicable
        if step.assigned_variables and step.assignment_type:
            if step.assignment_type == "single" and len(step.assigned_variables) == 1:
                var_assignment = step.assigned_variables[0]
                step_line = f"    {var_assignment} =    {step.keyword}"
            elif step.assignment_type == "multiple" and len(step.assigned_variables) > 1:
                var_assignments = "    ".join(step.assigned_variables)
                step_line = f"    {var_assignments} =    {step.keyword}"
            else:
                step_line = f"    {step.keyword}"
        else:
            step_line = f"    {step.keyword}"

        if step.arguments:
            processed_args = list(step.arguments)
            # Normalize Evaluate expressions to use $var syntax (Robot Framework requirement)
            # Reference: https://robotframework.org/robotframework/latest/RobotFrameworkUserGuide.html#evaluation-namespaces
            try:
                if (step.keyword or "").strip().lower() == "evaluate" and processed_args:
                    expr = str(processed_args[0])
                    # Convert ${var.suffix} -> $var.suffix inside the expression
                    processed_args[0] = self._convert_to_evaluation_namespace_syntax(expr)
            except Exception:
                pass
            if (
                self.execution_engine
                and hasattr(self.execution_engine, "_convert_locator_for_library")
                and step.arguments
            ):
                # Detect library from keyword
                library = await self._detect_library_from_keyword(step.keyword)
                if library and any(
                    kw in step.keyword.lower()
                    for kw in ["click", "fill", "get text", "wait", "select"]
                ):
                    try:
                        converted = self.execution_engine._convert_locator_for_library(
                            step.arguments[0], library
                        )
                        if converted != step.arguments[0]:
                            processed_args[0] = converted
                    except Exception:
                        pass
            escaped_args = [self._escape_robot_argument(arg) for arg in processed_args]
            args_str = "    ".join(escaped_args)
            step_line += f"    {args_str}"

        if step.comment:
            step_line += f"    {step.comment}"
        return step_line

    def _collect_flow_steps(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Flatten flow nodes into a list of {keyword, arguments} dictionaries."""
        steps: List[Dict[str, Any]] = []
        for n in nodes or []:
            ntype = n.get("type")
            if ntype == "for_each":
                steps.extend(n.get("body") or [])
            elif ntype == "if":
                steps.extend(n.get("then") or [])
                steps.extend(n.get("else") or [])
            elif ntype == "try":
                steps.extend(n.get("try") or [])
                steps.extend(n.get("except") or [])
                steps.extend(n.get("finally") or [])
            else:
                # Unknown node; best-effort treat as single step
                if n.get("keyword"):
                    steps.append({"keyword": n.get("keyword"), "arguments": n.get("arguments") or []})
        return steps

    def _matches_any_flow_step(self, step: TestCaseStep, flow_steps: List[Dict[str, Any]]) -> bool:
        """Return True if a linear step matches any step in flow_steps by keyword and arguments."""
        skw = (step.keyword or "").strip().lower()
        sargs = [str(a) for a in (step.arguments or [])]
        for fs in flow_steps:
            fkw = (fs.get("keyword", "") or "").strip().lower()
            fargs = [str(a) for a in (fs.get("arguments") or [])]
            if skw == fkw and sargs == fargs:
                return True
        return False

    def _map_blocks_to_ranges(
        self,
        steps: List[TestCaseStep],
        flow_blocks: List[Dict[str, Any]],
    ) -> tuple[list[tuple[Dict[str, Any], int, int]], set[int]]:
        """Map each flow block to the [start,end] index range it covers in linear steps.

        Returns a sorted list of (block, start_idx, end_idx) and a set of used indices.
        """
        ranges: list[tuple[Dict[str, Any], int, int]] = []
        used: set[int] = set()
        # Build per-block flattened steps
        def flatten_block(b: Dict[str, Any]) -> List[Dict[str, Any]]:
            if b.get("type") == "for_each":
                return list(b.get("body") or [])
            if b.get("type") == "if":
                return list(b.get("then") or []) + list(b.get("else") or [])
            if b.get("type") == "try":
                return list(b.get("try") or []) + list(b.get("except") or []) + list(b.get("finally") or [])
            if b.get("keyword"):
                return [{"keyword": b.get("keyword"), "arguments": b.get("arguments") or []}]
            return []
        # Helper to match
        def matches(step: TestCaseStep, flat: Dict[str, Any]) -> bool:
            skw = (step.keyword or "").strip().lower()
            sargs = [str(a) for a in (step.arguments or [])]
            fkw = (flat.get("keyword", "") or "").strip().lower()
            fargs = [str(a) for a in (flat.get("arguments") or [])]
            return skw == fkw and sargs == fargs

        for block in flow_blocks:
            flats = flatten_block(block)
            idxs: list[int] = []
            if flats:
                for i, s in enumerate(steps):
                    if any(matches(s, f) for f in flats):
                        idxs.append(i)
            if idxs:
                start, end = min(idxs), max(idxs)
                for i in range(start, end + 1):
                    used.add(i)
                ranges.append((block, start, end))
            else:
                # No match found; place block at current end
                ranges.append((block, len(steps), len(steps) - 1))
        # Sort by start index
        ranges.sort(key=lambda t: t[1])
        return ranges, used

    def _build_structured_steps(
        self,
        test_case: GeneratedTestCase,
        flow_blocks: List[Dict[str, Any]] | None,
    ) -> List[Dict[str, Any]]:
        """Produce a structured steps list combining pre/post linear keywords with control blocks.

        Shape examples:
        - {"type": "keyword", "keyword": "Log", "arguments": ["hello"], ...}
        - {"type": "control", "control": "TRY"}
        - {"type": "control", "control": "EXCEPT", "args": ["*Error*"]}
        - {"type": "control", "control": "FOR", "args": ["${item}", "IN", "a", "b"]}
        - {"type": "control", "control": "END"}
        """
        struct: List[Dict[str, Any]] = []

        # If no flow, return linear keywords only
        if not flow_blocks:
            for s in test_case.steps:
                struct.append(self._structured_from_step(s))
            return struct

        # Interleave linear steps around and between flow blocks
        block_ranges, used_indices = self._map_blocks_to_ranges(test_case.steps, flow_blocks)
        cur = 0
        for block, start_idx, end_idx in block_ranges:
            # Linear steps before block
            while cur < start_idx:
                if cur not in used_indices:
                    struct.append(self._structured_from_step(test_case.steps[cur]))
                cur += 1
            # The block itself
            struct.extend(self._structure_from_flow_blocks([block]))
            cur = max(cur, end_idx + 1)
        # Remainder
        while cur < len(test_case.steps):
            if cur not in used_indices:
                struct.append(self._structured_from_step(test_case.steps[cur]))
            cur += 1

        return struct

    def _structured_from_step(self, step: TestCaseStep) -> Dict[str, Any]:
        return {
            "type": "keyword",
            "keyword": step.keyword,
            "arguments": [
                self._escape_robot_argument(arg) for arg in (step.arguments or [])
            ],
            "assigned_variables": list(step.assigned_variables or []),
            "assignment_type": step.assignment_type,
        }

    def _build_step_dict_from_flow(self, s: Dict[str, Any]) -> Dict[str, Any]:
        """Build a structured step dictionary from a flow block step.

        Args:
            s: Flow step dictionary with keyword, arguments, and optional assign_to

        Returns:
            Structured step dictionary for structured_steps output
        """
        step_dict = {
            "type": "keyword",
            "keyword": s.get("keyword", ""),
            "arguments": [
                self._escape_robot_argument(arg)
                for arg in (s.get("arguments") or [])
            ],
        }
        # Include assign_to if present
        if s.get("assign_to"):
            step_dict["assign_to"] = s.get("assign_to")
        return step_dict

    def _structure_from_flow_blocks(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for n in nodes or []:
            ntype = n.get("type")
            if ntype == "for_each":
                item_var = n.get("item_var", "item")
                items = list(n.get("items") or [])
                escaped_items = [
                    self._escape_robot_argument(item) for item in items
                ]
                out.append({
                    "type": "control",
                    "control": "FOR",
                    "args": [f"${{{item_var}}}", "IN", *escaped_items],
                })
                for s in n.get("body") or []:
                    out.append(self._build_step_dict_from_flow(s))
                out.append({"type": "control", "control": "END"})
            elif ntype == "if":
                # Convert ${var} to $var for IF conditions (evaluation namespace syntax)
                cond = self._convert_to_evaluation_namespace_syntax(n.get("condition", ""))
                out.append({"type": "control", "control": "IF", "args": [cond]})
                for s in n.get("then") or []:
                    out.append(self._build_step_dict_from_flow(s))
                else_body = n.get("else") or []
                if else_body:
                    out.append({"type": "control", "control": "ELSE"})
                    for s in else_body:
                        out.append(self._build_step_dict_from_flow(s))
                out.append({"type": "control", "control": "END"})
            elif ntype == "try":
                out.append({"type": "control", "control": "TRY"})
                for s in n.get("try") or []:
                    out.append(self._build_step_dict_from_flow(s))
                patterns = list(n.get("except_patterns") or [])
                if n.get("except"):
                    # Use a single EXCEPT node with all patterns when sharing one handler
                    if patterns:
                        out.append({"type": "control", "control": "EXCEPT", "args": [
                            self._escape_robot_argument(p) for p in patterns
                        ]})
                    else:
                        out.append({"type": "control", "control": "EXCEPT"})
                    for s in n.get("except") or []:
                        out.append(self._build_step_dict_from_flow(s))
                if n.get("finally"):
                    out.append({"type": "control", "control": "FINALLY"})
                    for s in n.get("finally") or []:
                        out.append(self._build_step_dict_from_flow(s))
                out.append({"type": "control", "control": "END"})
            else:
                # Unknown node: best-effort as a keyword
                if n.get("keyword"):
                    out.append(self._build_step_dict_from_flow(n))
        return out

    def _render_flow_blocks(self, nodes: List[Dict[str, Any]], indent: str = "") -> List[str]:
        lines: List[str] = []
        for n in nodes:
            ntype = n.get("type")
            if ntype == "for_each":
                item_var = n.get("item_var", "item")
                items = [self._escape_robot_argument(v) for v in (n.get("items") or [])]
                header = f"{indent}FOR    ${{{item_var}}}    IN"
                if items:
                    header += "    " + "    ".join(items)
                lines.append(header)
                body = n.get("body") or []
                lines.extend(self._render_flow_body(body, indent + "    "))
                lines.append(f"{indent}END")
            elif ntype == "if":
                cond = n.get("condition", "")
                # Convert ${var} to $var for IF conditions (evaluation namespace syntax)
                cond = self._convert_to_evaluation_namespace_syntax(cond)
                lines.append(f"{indent}IF    {cond}")
                then_body = n.get("then") or []
                lines.extend(self._render_flow_body(then_body, indent + "    "))
                else_body = n.get("else") or []
                if else_body:
                    lines.append(f"{indent}ELSE")
                    lines.extend(self._render_flow_body(else_body, indent + "    "))
                lines.append(f"{indent}END")
            elif ntype == "try":
                lines.append(f"{indent}TRY")
                try_body = n.get("try") or []
                lines.extend(self._render_flow_body(try_body, indent + "    "))
                patterns = n.get("except_patterns") or []
                if n.get("except"):
                    # If a single handler is provided with multiple patterns, put all on one EXCEPT line.
                    if patterns:
                        joined = "    ".join([str(p) for p in patterns])
                        lines.append(f"{indent}EXCEPT    {joined}")
                    else:
                        lines.append(f"{indent}EXCEPT")
                    lines.extend(self._render_flow_body(n.get("except"), indent + "    "))
                if n.get("finally"):
                    lines.append(f"{indent}FINALLY")
                    lines.extend(self._render_flow_body(n.get("finally"), indent + "    "))
                lines.append(f"{indent}END")
            else:
                # Fallback: plain step
                lines.extend(self._render_flow_body([n], indent))
        return lines

    def _render_flow_body(self, steps: List[Dict[str, Any]], indent: str) -> List[str]:
        out: List[str] = []
        for s in steps or []:
            kw = s.get("keyword", "")
            args = list(s.get("arguments", []) or [])
            assign_to = s.get("assign_to")  # Variable assignment (e.g., "${LENGTH}")

            # Convert Evaluate expressions to use $var syntax (evaluation namespace)
            if kw.strip().lower() == "evaluate" and args:
                args[0] = self._convert_to_evaluation_namespace_syntax(str(args[0]))

            # Build the line with optional variable assignment
            if assign_to:
                # Handle both string and list forms of assign_to
                if isinstance(assign_to, list):
                    # Wrap each variable in ${} if not already wrapped
                    wrapped_vars = []
                    for v in assign_to:
                        v_str = str(v).strip()
                        if not (v_str.startswith("${") or v_str.startswith("@{") or v_str.startswith("&{")):
                            v_str = f"${{{v_str}}}"
                        wrapped_vars.append(v_str)
                    var_assignment = "    ".join(wrapped_vars)
                else:
                    var_assignment = str(assign_to).strip()
                    # Wrap in ${} if not already wrapped
                    if not (var_assignment.startswith("${") or var_assignment.startswith("@{") or var_assignment.startswith("&{")):
                        var_assignment = f"${{{var_assignment}}}"
                line = f"{indent}{var_assignment} =    {self._remove_library_prefix(kw)}"
            else:
                line = f"{indent}{self._remove_library_prefix(kw)}"

            if args:
                esc = [self._escape_robot_argument(a) for a in args]
                line += "    " + "    ".join(esc)
            out.append(line)
        return out

    def _format_path_for_rf(self, path: str) -> str:
        """Format a filesystem path into OS-independent Robot Framework syntax.

        Converts separators to the RF variable ${/} and preserves drive letters
        (e.g., 'C:${/}path${/}to${/}file'). Works for both Windows and Posix inputs.
        """
        if not path:
            return path
        # Normalize all separators to '/'
        s = path.replace("\\\\", "\\")  # collapse escaped backslashes
        parts = re.split(r"[\\/]+", s.strip())
        if not parts:
            return path
        sep = "${/}"
        # Detect Windows drive letter like 'C:'
        drive = None
        if re.match(r"^[A-Za-z]:$", parts[0]):
            drive = parts[0]
            parts = parts[1:]
        # If original path started with a separator (absolute posix), add leading ${/}
        leading = ""
        if s.startswith("/") or s.startswith("\\"):
            leading = sep
        formatted = (drive + sep if drive else leading) + sep.join(
            p for p in parts if p
        )
        return formatted or path

    def _format_variable_for_rf(self, var_name: str, var_value: Any) -> str:
        """Format a variable for Robot Framework *** Variables *** section.

        Handles different variable types:
        - Scalars: ${VAR}    value
        - Lists: @{VAR}    item1    item2    item3
        - Dictionaries: &{VAR}    key1=value1    key2=value2

        Args:
            var_name: Variable name (without ${}/@{}/&{} syntax)
            var_value: Variable value (can be string, number, list, dict)

        Returns:
            Formatted variable line for RF Variables section
        """
        # Remove any existing ${} syntax from name
        clean_name = var_name.strip()
        if clean_name.startswith("${") and clean_name.endswith("}"):
            clean_name = clean_name[2:-1]
        elif clean_name.startswith("@{") and clean_name.endswith("}"):
            clean_name = clean_name[2:-1]
        elif clean_name.startswith("&{") and clean_name.endswith("}"):
            clean_name = clean_name[2:-1]
        elif clean_name.startswith("$") or clean_name.startswith("@") or clean_name.startswith("&"):
            clean_name = clean_name[1:]

        # Calculate column width for proper alignment
        column_width = max(20, len(clean_name) + 4)  # At least 20 chars for variable column

        # Handle different value types
        if isinstance(var_value, dict):
            # Dictionary variable: &{VAR}    key1=value1    key2=value2
            items = [f"{k}={self._escape_robot_argument(v)}" for k, v in var_value.items()]
            return f"&{{{clean_name}}}".ljust(column_width) + "    ".join(items)

        elif isinstance(var_value, (list, tuple)):
            # List variable: @{VAR}    item1    item2    item3
            items = [self._escape_robot_argument(item) for item in var_value]
            return f"@{{{clean_name}}}".ljust(column_width) + "    ".join(items)

        else:
            # Scalar variable: ${VAR}    value
            # Handle special value types
            if var_value is None:
                str_value = "${NONE}"
            elif isinstance(var_value, bool):
                str_value = "${TRUE}" if var_value else "${FALSE}"
            elif isinstance(var_value, (int, float)):
                # Numeric values - keep as-is for RF to interpret
                str_value = str(var_value)
            else:
                # String values
                str_value = self._escape_robot_argument(str(var_value))

            return f"${{{clean_name}}}".ljust(column_width) + str_value

    def _scan_variable_references(self, suite: GeneratedTestSuite) -> set:
        """Scan all test steps for variable references (${VAR}, @{VAR}, &{VAR}).

        This method extracts all variable names referenced in test step arguments,
        setup/teardown, and flow blocks to identify which variables are used
        by the generated test suite.

        Note: This method skips ${{ }} evaluation namespace expressions, which
        are Python evaluation blocks and not variable references.

        Args:
            suite: The generated test suite to scan

        Returns:
            Set of variable names (without ${}/@ {}/&{} syntax) found in the suite
        """
        # Pattern for ${var}, @{var}, &{var} - but NOT ${{ }} evaluation namespace
        # We use negative lookahead (?!\{) to skip ${{ patterns
        variable_pattern = re.compile(r'[\$@&]\{(?!\{)([^}]+)\}')
        found_variables = set()

        # Built-in RF variables that don't need to be defined
        builtin_vars = {
            "CURDIR", "TEMPDIR", "EXECDIR", "OUTPUT_DIR", "OUTPUTDIR",
            "OUTPUT_FILE", "LOG_FILE", "REPORT_FILE", "DEBUG_FILE",
            "LOG_LEVEL", "OPTIONS", "TEST_NAME", "TEST_DOCUMENTATION",
            "TEST_TAGS", "SUITE_NAME", "SUITE_SOURCE", "SUITE_DOCUMENTATION",
            "PREV_TEST_NAME", "PREV_TEST_STATUS", "PREV_TEST_MESSAGE",
            "TEST_STATUS", "TEST_MESSAGE", "KEYWORD_STATUS", "KEYWORD_MESSAGE",
            "LOG_NAME", "REPORT_NAME", "EMPTY", "SPACE", "TRUE", "FALSE",
            "NONE", "NULL", "/", ":"
        }

        def extract_vars_from_string(s: str) -> None:
            """Extract variable names from a string."""
            if not s:
                return
            text = str(s)
            # First, remove ${{ ... }} evaluation namespace blocks to avoid false matches
            # This handles nested braces by finding matching pairs
            cleaned = re.sub(r'\$\{\{[^}]*\}\}', '', text)
            # Also handle triple-brace (malformed) patterns like ${{{ }}}
            cleaned = re.sub(r'\$\{\{\{[^}]*\}\}\}', '', cleaned)

            for match in variable_pattern.finditer(cleaned):
                var_name = match.group(1)
                # Handle nested access like ${var.attr} or ${var}[0]
                base_name = var_name.split('.')[0].split('[')[0].strip()
                if base_name.upper() not in builtin_vars and not base_name.startswith("_"):
                    found_variables.add(base_name)

        def scan_step(step: TestCaseStep) -> None:
            """Scan a single step for variable references."""
            extract_vars_from_string(step.keyword)
            for arg in (step.arguments or []):
                extract_vars_from_string(arg)

        # Scan all test cases
        for test_case in suite.test_cases:
            # Scan setup
            if test_case.setup:
                scan_step(test_case.setup)
            # Scan steps
            for step in test_case.steps:
                scan_step(step)
            # Scan teardown
            if test_case.teardown:
                scan_step(test_case.teardown)

        # Scan suite-level setup/teardown
        if suite.setup:
            scan_step(suite.setup)
        if suite.teardown:
            scan_step(suite.teardown)

        # Scan flow blocks for variable references in conditions
        if suite.flow_blocks:
            self._scan_flow_blocks_for_vars(suite.flow_blocks, extract_vars_from_string)

        return found_variables

    def _scan_flow_blocks_for_vars(self, blocks: List[Dict[str, Any]], extract_fn) -> None:
        """Recursively scan flow blocks for variable references.

        Args:
            blocks: List of flow block dictionaries
            extract_fn: Function to extract variables from strings
        """
        for block in blocks:
            block_type = block.get("type", "")

            if block_type == "if":
                # Scan condition
                extract_fn(block.get("condition", ""))
                # Scan body
                for step in block.get("body", []):
                    if isinstance(step, dict) and "keyword" in step:
                        extract_fn(step.get("keyword", ""))
                        for arg in step.get("arguments", []):
                            extract_fn(arg)
                # Scan else_if branches
                for branch in block.get("else_if", []):
                    extract_fn(branch.get("condition", ""))
                    for step in branch.get("body", []):
                        if isinstance(step, dict) and "keyword" in step:
                            extract_fn(step.get("keyword", ""))
                            for arg in step.get("arguments", []):
                                extract_fn(arg)
                # Scan else body
                for step in block.get("else_body", []):
                    if isinstance(step, dict) and "keyword" in step:
                        extract_fn(step.get("keyword", ""))
                        for arg in step.get("arguments", []):
                            extract_fn(arg)

            elif block_type == "for_each":
                extract_fn(block.get("variable", ""))
                extract_fn(block.get("collection", ""))
                for step in block.get("body", []):
                    if isinstance(step, dict) and "keyword" in step:
                        extract_fn(step.get("keyword", ""))
                        for arg in step.get("arguments", []):
                            extract_fn(arg)

            elif block_type == "try":
                for step in block.get("body", []):
                    if isinstance(step, dict) and "keyword" in step:
                        extract_fn(step.get("keyword", ""))
                        for arg in step.get("arguments", []):
                            extract_fn(arg)
                for step in block.get("except_body", []):
                    if isinstance(step, dict) and "keyword" in step:
                        extract_fn(step.get("keyword", ""))
                        for arg in step.get("arguments", []):
                            extract_fn(arg)

    def _check_untracked_variables(
        self, suite: GeneratedTestSuite, session_id: str
    ) -> List[Dict[str, str]]:
        """Check for variables used in steps but not defined in Variables section.

        This provides warnings when generated tests reference variables that weren't
        explicitly set via manage_session, which may indicate incomplete test suites.

        Args:
            suite: The generated test suite
            session_id: Session ID for context

        Returns:
            List of warning dictionaries with variable name and context
        """
        warnings = []

        # Get all variable references in the suite
        referenced_vars = self._scan_variable_references(suite)

        # Get variables defined in the suite's Variables section
        defined_vars = set()
        if suite.variables:
            for var_name in suite.variables.keys():
                # Normalize name (remove ${} if present)
                clean_name = var_name.strip()
                if clean_name.startswith("${") and clean_name.endswith("}"):
                    clean_name = clean_name[2:-1]
                defined_vars.add(clean_name)

        # Find untracked variables (referenced but not defined)
        untracked = referenced_vars - defined_vars

        # Group warnings into a compact summary instead of per-variable verbose messages
        if untracked:
            warnings.append({
                "type": "untracked_variables",
                "count": len(untracked),
                "variables": sorted(untracked),
                "fix": "manage_session(action='set_variables', variables={...})",
            })

        if warnings:
            var_names = warnings[0].get("variables", []) if warnings else []
            logger.warning(
                f"build_test_suite: Found {len(var_names)} untracked variable(s) in session "
                f"{session_id}: {', '.join(var_names)}"
            )

        return warnings

    async def _generate_statistics(
        self, steps: List[Dict[str, Any]], suite: GeneratedTestSuite
    ) -> Dict[str, Any]:
        """Generate execution statistics."""

        total_original_steps = len(steps)
        total_optimized_steps = sum(len(tc.steps) for tc in suite.test_cases)

        # Count step types
        step_types = {}
        for test_case in suite.test_cases:
            for step in test_case.steps:
                step_type = self._categorize_step(step.keyword)
                step_types[step_type] = step_types.get(step_type, 0) + 1

        optimization_ratio = (
            (total_original_steps - total_optimized_steps) / total_original_steps
            if total_original_steps > 0
            else 0
        )

        return {
            "original_steps": total_original_steps,
            "optimized_steps": total_optimized_steps,
            "optimization_ratio": optimization_ratio,
            "test_cases_generated": len(suite.test_cases),
            "libraries_required": len(suite.imports or []),
            "step_types": step_types,
            "estimated_execution_time": total_optimized_steps
            * 2,  # 2 seconds per step estimate
        }

    def _categorize_step(self, keyword: str) -> str:
        """Categorize a step by its type."""
        keyword_lower = keyword.lower()

        if any(kw in keyword_lower for kw in ["open", "go to", "navigate"]):
            return "navigation"
        elif any(kw in keyword_lower for kw in ["click", "press", "select"]):
            return "interaction"
        elif any(kw in keyword_lower for kw in ["input", "type", "enter", "fill"]):
            return "input"
        elif any(kw in keyword_lower for kw in ["should", "verify", "assert", "check"]):
            return "verification"
        elif any(kw in keyword_lower for kw in ["wait", "sleep", "pause"]):
            return "synchronization"
        elif any(kw in keyword_lower for kw in ["close", "quit", "cleanup"]):
            return "cleanup"
        else:
            return "other"

    def _fix_malformed_evaluation_namespace(self, arg: str) -> str:
        """Fix malformed ${{ }} evaluation namespace expressions.

        Fixes common issues:
        1. Triple-brace ${{{ followed by empty set/dict: ${{{}...}} -> ${{...}}
        2. Missing space after ${{ that creates confusion with dict literals
        3. Spurious empty set at start: ${{{}}' -> ${{ '

        Args:
            arg: String that may contain malformed evaluation namespace

        Returns:
            String with corrected evaluation namespace syntax
        """
        if not arg or "${{" not in arg:
            return arg

        result = arg

        # Fix: ${{{}}'... pattern - spurious {} followed by closing brace
        # This catches ${{{}}'Hello'... -> ${{ 'Hello'...
        # The pattern is: ${{ {} }} 'expr' but written as ${{{}}' which is malformed
        result = re.sub(r'\$\{\{\{\}\}([\'"\$\w])', r"${{ \1", result)

        # Fix: ${{{} at start of expression -> ${{ (remove spurious empty set/dict)
        # Pattern: ${{{}' or ${{{}" or ${{{$var
        result = re.sub(r'\$\{\{\{\}([\'"\$])', r"${{ \1", result)

        # Fix: ${{{{ (4 braces) -> ${{ (probably double-escaped)
        result = re.sub(r'\$\{\{\{\{', r"${{", result)

        # Fix: ${{{' (3 braces followed by quote) -> ${{ '
        # This handles cases like ${{{'-'.join(...)}}} -> ${{ '-'.join(...) }}
        result = re.sub(r'\$\{\{\{([\'"])', r"${{ \1", result)

        # Fix: }}} at end -> }} (extra closing brace from malformed input)
        result = re.sub(r'\}\}\}(?!\})', r"}}", result)

        return result

    def _escape_robot_argument(self, arg: Any) -> str:
        """Escape Robot Framework arguments that start with special characters.

        Accepts non-string args (e.g., int/bool) and converts to string safely.
        """
        if arg is None:
            return ""
        if not isinstance(arg, str):
            try:
                arg = str(arg)
            except Exception:
                arg = f"<{type(arg).__name__}>"
        if not arg:
            return ""

        # Fix malformed evaluation namespace expressions first
        arg = self._fix_malformed_evaluation_namespace(arg)

        # Escape literal newlines/tabs so they don't break the .robot line structure.
        # RF uses \n / \t as escape sequences, so we emit \\n / \\t in the text
        # which RF interprets back as the literal characters at runtime.
        arg = arg.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")

        # Escape arguments starting with # (treated as comments in RF)
        if arg.startswith("#"):
            return f"\\{arg}"

        return arg

    def _remove_library_prefix(self, keyword: str) -> str:
        """Remove library prefix from keyword name for cleaner test suites.

        Converts "LibraryName.KeywordName" -> "KeywordName"
        Leaves keywords without prefixes unchanged.

        Args:
            keyword: Keyword name potentially with library prefix

        Returns:
            Keyword name without library prefix
        """
        if "." in keyword:
            return keyword.split(".", 1)[1]  # Return everything after first dot
        return keyword

    def _apply_prefix_removal(self, suite: GeneratedTestSuite) -> GeneratedTestSuite:
        """Apply library prefix removal to all keywords in the test suite.

        Args:
            suite: Test suite with potentially prefixed keywords

        Returns:
            Test suite with library prefixes removed from keywords
        """
        # Process test cases
        for test_case in suite.test_cases:
            # Process test steps
            for step in test_case.steps:
                step.keyword = self._remove_library_prefix(step.keyword)

            # Process setup
            if test_case.setup:
                test_case.setup.keyword = self._remove_library_prefix(
                    test_case.setup.keyword
                )

            # Process teardown
            if test_case.teardown:
                test_case.teardown.keyword = self._remove_library_prefix(
                    test_case.teardown.keyword
                )

        # Process suite-level setup and teardown
        if suite.setup:
            suite.setup.keyword = self._remove_library_prefix(suite.setup.keyword)

        if suite.teardown:
            suite.teardown.keyword = self._remove_library_prefix(suite.teardown.keyword)

        return suite

    def _get_session_libraries(self, session_id: str) -> set:
        """
        Get libraries that were actually used in session execution.

        This method prioritizes the session's imported_libraries over keyword pattern matching
        because the session has already proven these libraries work together successfully.

        Args:
            session_id: Session ID to get libraries from

        Returns:
            Set of library names from session execution history
        """
        if not self.execution_engine or not hasattr(self.execution_engine, "sessions"):
            return set()

        session = self.execution_engine.sessions.get(session_id)
        if not session:
            return set()

        # Use imported_libraries from session (these are known to work together)
        session_libraries = set(session.imported_libraries)

        # Filter out BuiltIn as it's automatically available
        session_libraries.discard("BuiltIn")

        logger.debug(
            f"Session {session_id} libraries from execution history: {session_libraries}"
        )
        return session_libraries

    def _session_has_execution_history(self, session_id: str) -> bool:
        """
        Check if session has execution history (successful steps).

        Args:
            session_id: Session ID to check

        Returns:
            True if session has executed steps, False otherwise
        """
        if not self.execution_engine or not hasattr(self.execution_engine, "sessions"):
            return False

        session = self.execution_engine.sessions.get(session_id)
        if not session:
            return False

        # Check if session has any successful steps
        has_history = len(session.steps) > 0
        logger.debug(
            f"Session {session_id} has execution history: {has_history} ({len(session.steps)} steps)"
        )
        return has_history

    def _validate_suite_library_exclusions(self, imports: set, session_id: str) -> None:
        """
        Validate that the test suite doesn't violate library exclusion rules.

        Only applies strict validation for new/empty sessions. Sessions with execution
        history have already proven their library combinations work successfully.

        Args:
            imports: Set of library names to be imported
            session_id: Session ID for error reporting

        Raises:
            ValueError: If conflicting libraries are detected for new sessions
        """
        # If session has execution history, trust its library choices
        if self._session_has_execution_history(session_id):
            logger.info(
                f"Session {session_id} has execution history - skipping library exclusion validation"
            )
            return

        # Only apply strict validation for new/empty sessions
        web_automation_libs = ["Browser", "SeleniumLibrary"]
        detected_web_libs = [lib for lib in imports if lib in web_automation_libs]

        if len(detected_web_libs) > 1:
            raise ValueError(
                f"Test suite for session '{session_id}' contains conflicting web automation libraries: "
                f"{detected_web_libs}. Browser Library and SeleniumLibrary are mutually exclusive. "
                f"Please use separate sessions for different libraries."
            )

        # For new sessions, also check session consistency if execution engine is available
        if self.execution_engine and hasattr(self.execution_engine, "sessions"):
            session = self.execution_engine.sessions.get(session_id)
            if session:
                # Use safe method to get web automation library (handles older sessions)
                session_web_lib = self._get_session_web_library(session)
                if session_web_lib and detected_web_libs:
                    suite_web_lib = detected_web_libs[0]
                    if session_web_lib != suite_web_lib:
                        logger.warning(
                            f"Session '{session_id}' uses '{session_web_lib}' but suite "
                            f"detected '{suite_web_lib}' from keywords. Using session library."
                        )
