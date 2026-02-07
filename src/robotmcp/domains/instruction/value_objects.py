"""Instruction Domain Value Objects.

This module contains immutable value objects for the Instruction bounded context.
Value objects are identified by their attributes rather than by identity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Dict, FrozenSet, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class InstructionMode:
    """Instruction configuration mode.

    Modes:
    - OFF: No instructions are applied
    - DEFAULT: Built-in default instructions are used
    - CUSTOM: User-provided custom instructions from file

    Security: Mode values are validated against whitelist.

    Examples:
        >>> mode = InstructionMode.default()
        >>> mode.is_enabled
        True
        >>> mode = InstructionMode.off()
        >>> mode.is_enabled
        False
    """

    value: str

    # Valid mode constants
    OFF: ClassVar[str] = "off"
    DEFAULT: ClassVar[str] = "default"
    CUSTOM: ClassVar[str] = "custom"

    VALID_MODES: ClassVar[FrozenSet[str]] = frozenset({OFF, DEFAULT, CUSTOM})

    def __post_init__(self) -> None:
        """Validate mode value on creation."""
        if self.value not in self.VALID_MODES:
            raise ValueError(
                f"Invalid instruction mode: '{self.value}'. "
                f"Must be one of: {', '.join(sorted(self.VALID_MODES))}"
            )

    @classmethod
    def off(cls) -> "InstructionMode":
        """Create OFF mode (instructions disabled)."""
        return cls(value=cls.OFF)

    @classmethod
    def default(cls) -> "InstructionMode":
        """Create DEFAULT mode (built-in instructions)."""
        return cls(value=cls.DEFAULT)

    @classmethod
    def custom(cls) -> "InstructionMode":
        """Create CUSTOM mode (file-based instructions)."""
        return cls(value=cls.CUSTOM)

    @classmethod
    def from_string(cls, value: str) -> "InstructionMode":
        """Create mode from string value.

        Args:
            value: Mode string (case-insensitive).

        Returns:
            InstructionMode instance.

        Raises:
            ValueError: If value is not a valid mode.
        """
        return cls(value=value.lower())

    @property
    def is_enabled(self) -> bool:
        """Check if instructions are enabled."""
        return self.value != self.OFF

    @property
    def uses_custom_file(self) -> bool:
        """Check if mode uses a custom file."""
        return self.value == self.CUSTOM

    @property
    def uses_default_template(self) -> bool:
        """Check if mode uses the default template."""
        return self.value == self.DEFAULT

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"InstructionMode({self.value!r})"


@dataclass(frozen=True)
class InstructionContent:
    """The actual instruction text content.

    Content is validated for:
    - Non-empty value
    - Maximum length constraints
    - Minimum length constraints

    Attributes:
        value: The instruction text
        source: Origin of the content (default, custom:path, template:id)

    Examples:
        >>> content = InstructionContent(
        ...     value="Use discovery tools before actions.",
        ...     source="default"
        ... )
        >>> content.token_estimate
        8  # Approximate
    """

    value: str
    source: str = "default"

    # Constraints
    MAX_LENGTH: ClassVar[int] = 50000  # 50KB max
    MIN_LENGTH: ClassVar[int] = 10

    def __post_init__(self) -> None:
        """Validate content on creation."""
        if not self.value or len(self.value.strip()) < self.MIN_LENGTH:
            raise ValueError(
                f"Instruction content too short (min {self.MIN_LENGTH} chars)"
            )
        if len(self.value) > self.MAX_LENGTH:
            raise ValueError(
                f"Instruction content too long (max {self.MAX_LENGTH} chars)"
            )

    @property
    def is_from_file(self) -> bool:
        """Check if content was loaded from a file."""
        return self.source.startswith("custom:")

    @property
    def is_from_template(self) -> bool:
        """Check if content was rendered from a template."""
        return self.source.startswith("template:")

    @property
    def is_default(self) -> bool:
        """Check if content is from the default source."""
        return self.source == "default"

    @property
    def token_estimate(self) -> int:
        """Rough estimate of tokens (words * 1.3).

        This is a simple heuristic, not exact tokenization.
        For Claude/GPT models, actual tokens may vary.
        """
        word_count = len(self.value.split())
        return int(word_count * 1.3)

    @property
    def char_count(self) -> int:
        """Get character count."""
        return len(self.value)

    @property
    def line_count(self) -> int:
        """Get line count."""
        return len(self.value.splitlines())

    def __str__(self) -> str:
        return self.value

    def __len__(self) -> int:
        return len(self.value)

    def __repr__(self) -> str:
        preview = self.value[:50] + "..." if len(self.value) > 50 else self.value
        return f"InstructionContent(source={self.source!r}, preview={preview!r})"


@dataclass(frozen=True)
class InstructionPath:
    """File path to custom instruction file.

    Security: Path is validated to prevent directory traversal
    and restricted to allowed extensions.

    Attributes:
        value: The file path string.

    Examples:
        >>> path = InstructionPath("./instructions/custom.txt")
        >>> path.exists
        False  # Depends on filesystem
    """

    value: str

    ALLOWED_EXTENSIONS: ClassVar[FrozenSet[str]] = frozenset(
        {".txt", ".md", ".instruction", ".instructions"}
    )

    def __post_init__(self) -> None:
        """Validate path on creation."""
        if not self.value:
            raise ValueError("Instruction path cannot be empty")

        # Validate path format
        path = Path(self.value)

        # Prevent directory traversal
        if ".." in path.parts:
            raise ValueError("Path traversal (..) not allowed in instruction path")

        # Check extension
        if path.suffix.lower() not in self.ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Invalid file extension: {path.suffix}. "
                f"Allowed: {', '.join(sorted(self.ALLOWED_EXTENSIONS))}"
            )

    @property
    def path(self) -> Path:
        """Get as Path object."""
        return Path(self.value)

    @property
    def exists(self) -> bool:
        """Check if the file exists."""
        return self.path.exists() and self.path.is_file()

    @property
    def extension(self) -> str:
        """Get the file extension."""
        return self.path.suffix.lower()

    @property
    def filename(self) -> str:
        """Get the filename without path."""
        return self.path.name

    def resolve(self, base_path: Optional[Path] = None) -> Path:
        """Resolve path relative to base, with security checks.

        Args:
            base_path: Optional base directory for relative paths.

        Returns:
            Resolved absolute Path.

        Raises:
            ValueError: If resolved path escapes base directory.
        """
        if self.path.is_absolute():
            return self.path

        if base_path:
            resolved = (base_path / self.path).resolve()
            base_resolved = base_path.resolve()
            # Ensure resolved path is under base_path
            try:
                resolved.relative_to(base_resolved)
            except ValueError:
                raise ValueError(
                    f"Path '{self.value}' escapes base directory '{base_path}'"
                )
            return resolved

        return self.path.resolve()

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"InstructionPath({self.value!r})"


@dataclass(frozen=True)
class InstructionTemplate:
    """Predefined instruction template with placeholders.

    Templates support variable substitution for dynamic content:
    - {available_tools}: List of MCP tools available
    - {discovery_keywords}: Keywords for discovery operations
    - {session_context}: Current session information

    Security: Template content is sanitized on creation.

    Attributes:
        template_id: Unique identifier for the template.
        content: Template text with {placeholder} markers.
        description: Human-readable description.
        placeholders: Tuple of expected placeholder names.

    Examples:
        >>> template = InstructionTemplate.discovery_first()
        >>> content = template.render({"available_tools": "get_aria_snapshot"})
        >>> "get_aria_snapshot" in content.value
        True
    """

    template_id: str
    content: str
    description: str
    placeholders: Tuple[str, ...] = field(default_factory=tuple)

    # Template pattern for placeholder detection
    PLACEHOLDER_PATTERN: ClassVar[re.Pattern] = re.compile(r"\{(\w+)\}")

    def __post_init__(self) -> None:
        """Validate template on creation."""
        if not self.template_id or not self.template_id.strip():
            raise ValueError("Template ID cannot be empty")
        if not self.content or not self.content.strip():
            raise ValueError("Template content cannot be empty")

        # Extract and validate placeholders
        found_placeholders = frozenset(self.PLACEHOLDER_PATTERN.findall(self.content))
        declared_placeholders = frozenset(self.placeholders)

        if found_placeholders != declared_placeholders:
            raise ValueError(
                f"Placeholder mismatch: found {sorted(found_placeholders)}, "
                f"declared {sorted(declared_placeholders)}"
            )

    def render(self, context: Dict[str, str]) -> InstructionContent:
        """Render template with provided context values.

        Args:
            context: Dictionary mapping placeholder names to values.

        Returns:
            InstructionContent with rendered template.

        Note:
            Missing placeholders are left as-is in the output.
        """
        rendered = self.content
        for placeholder in self.placeholders:
            if placeholder in context:
                rendered = rendered.replace(f"{{{placeholder}}}", context[placeholder])
        return InstructionContent(
            value=rendered, source=f"template:{self.template_id}"
        )

    @classmethod
    def discovery_first(cls) -> "InstructionTemplate":
        """Default template encouraging discovery before action.

        This is the primary template used when instructions are enabled
        in default mode. It guides LLMs to:
        1. Use discovery tools first
        2. Never guess keywords or libraries
        3. Use proper keyword discovery workflow
        4. Verify test execution results
        """
        return cls(
            template_id="discovery_first",
            content="""IMPORTANT: Before executing any Robot Framework test automation:

1. DISCOVERY FIRST
   - ALWAYS use discovery tools first to understand available keywords and libraries
   - Use 'find_keywords' to search for keywords by name or description
   - Use 'get_keyword_info' to get detailed information about a specific keyword
   - Use 'recommend_libraries' to identify which libraries are needed for your scenario
   - Never proceed with actions until you have confirmed keyword availability

2. NO GUESSING
   - NEVER guess, invent, or fabricate keyword names or argument patterns
   - NEVER assume library names or keyword signatures
   - Only use keywords discovered through the find_keywords or get_keyword_info tools

3. KEYWORD DISCOVERY
   - Use 'find_keywords' with relevant search terms to find available keywords
   - Use 'get_keyword_info' with the exact keyword name to get argument details
   - Use 'check_library_availability' to verify libraries before using them

4. SCENARIO ANALYSIS
   - Use 'analyze_scenario' to understand what a test scenario requires
   - Use 'recommend_libraries' to get library recommendations for your use case
   - Use 'get_locator_guidance' for locator syntax help (CSS, XPath, text strategies)

5. SESSION MANAGEMENT & PAGE INSPECTION
   - Use 'manage_session' with action="init" to create and configure sessions
   - Use 'get_session_state' to check session status and inspect page content
   - Use 'set_library_search_order' to configure library loading priority
   - For web testing, inspect the page BEFORE interacting with elements:
     * get_session_state(sections=["page_source"], include_reduced_dom=True) for ARIA snapshot
     * get_session_state(sections=["page_source"], page_source_filtered=True) for compact DOM
   - Read the snapshot to identify elements, then use real locators (id=, css=, text=, etc.)

6. TEST EXECUTION
   - Use 'execute_step' to run individual Robot Framework keywords
   - Use 'execute_flow' to run structured control flow (if/for/try)
   - Use assign_to in execute_step to capture return values for later steps
   - Use 'build_test_suite' to generate .robot file from executed steps
   - Use 'run_test_suite' to execute a .robot file from disk

Available discovery tools: {available_tools}""",
            description="Encourages discovery-first approach to prevent guessing",
            placeholders=("available_tools",),
        )

    @classmethod
    def locator_prevention(cls) -> "InstructionTemplate":
        """Template focused specifically on preventing keyword guessing."""
        return cls(
            template_id="locator_prevention",
            content="""CRITICAL: Keyword and Library Usage Rules

You MUST NOT:
- Invent keyword names that you have not verified exist
- Guess argument names or patterns for keywords
- Assume library names without checking availability
- Fabricate Robot Framework syntax or structures
- Guess element locators without inspecting the DOM

You MUST:
- Use find_keywords to discover available keywords
- Use get_keyword_info to get exact argument details
- Use check_library_availability to verify libraries exist
- Use recommend_libraries to get appropriate library suggestions
- Report what keywords ARE available if you cannot find expected ones
- Use get_session_state with sections=["page_source"] to inspect page content
- Use include_reduced_dom=True for ARIA snapshot (semantic DOM structure)
- Use page_source_filtered=True for compact/sanitized DOM

If a keyword is not found via find_keywords, it does not exist or is not loaded.""",
            description="Strict keyword and library verification rules",
            placeholders=(),
        )

    @classmethod
    def minimal(cls) -> "InstructionTemplate":
        """Minimal template with essential rules only (~500 chars).

        For capable LLMs like Claude Opus, GPT-4.
        """
        return cls(
            template_id="minimal",
            content="""Use discovery tools (find_keywords, get_keyword_info) before executing keywords.
Verify keywords exist via discovery, never guess keyword names or arguments.
For DOM inspection: get_session_state(sections=["page_source"], include_reduced_dom=True).""",
            description="Minimal instruction set for capable LLMs",
            placeholders=(),
        )

    @classmethod
    def standard(cls) -> "InstructionTemplate":
        """Standard template for general use (~1200 chars).

        This is the default template providing balanced guidance.
        """
        return cls(
            template_id="standard",
            content="""rf-mcp WORKFLOW GUIDE:

1. DISCOVER before EXECUTE:
   - Call find_keywords to see available Robot Framework keywords
   - Call check_library_availability or recommend_libraries to see loaded libraries

2. GET LOCATOR GUIDANCE before INTERACT:
   - Call get_locator_guidance to get locator syntax help for your target library
   - Use real locators (id=, css=, text=, xpath=) based on actual page content

3. SESSION MANAGEMENT & DOM INSPECTION:
   - Call manage_session(action="init", scenario="...", libraries=["Browser", "BuiltIn"]) first
   - Use session_id from response in subsequent calls
   - To inspect DOM/ARIA: get_session_state(sections=["page_source"])
     * include_reduced_dom=True -> ARIA snapshot (lightweight semantic DOM)
     * page_source_filtered=True -> sanitized/compact DOM
   - Read the ARIA snapshot to find real locator values (ids, roles, text)

4. ERROR RECOVERY:
   - If element not found: call get_session_state for fresh DOM/ARIA snapshot
   - If keyword unknown: call find_keywords with library filter

5. MULTI-TEST SUITES (optional):
   - manage_session(action="start_test", test_name="Login Test") to begin a named test
   - manage_session(action="end_test") to finish it, then start another
   - build_test_suite generates a multi-test .robot file

RULES:
- NEVER fabricate locators - inspect the DOM first with get_session_state
- ALWAYS discover keywords before executing unfamiliar ones
- For web: manage_session(action="init") -> get_session_state(ARIA) -> execute_step
- For API: manage_session(action="init") -> find_keywords -> execute_step
- Use assign_to in execute_step to capture return values into variables

Available tools: {available_tools}""",
            description="Standard guidance for general use",
            placeholders=("available_tools",),
        )

    @classmethod
    def detailed(cls) -> "InstructionTemplate":
        """Detailed template for smaller LLMs (~1800 chars).

        For LLMs like Claude Haiku, GPT-4o-mini, Gemini Flash.
        """
        return cls(
            template_id="detailed",
            content="""rf-mcp STEP-BY-STEP GUIDE FOR TEST AUTOMATION:

STEP 1: START SESSION
- Always call manage_session first with action="init"
- Specify libraries you need (e.g., libraries=["Browser", "BuiltIn"])
- Save the session_id from the response
- Example: manage_session(action="init", scenario="Login to web app", libraries=["Browser", "BuiltIn"])

STEP 2: DISCOVER KEYWORDS
- Call find_keywords to see what actions are available
- Filter by library: find_keywords(library="Browser")
- Call get_keyword_info for argument details: get_keyword_info(keyword_name="Fill Text")

STEP 3: FOR WEB TESTING - INSPECT THE PAGE
- Call get_session_state(sections=["page_source"], include_reduced_dom=True) for ARIA snapshot
- ARIA snapshot shows semantic structure with roles, names, and element IDs
- Read the snapshot to find real locator values (id=login-btn, text=Submit, css=.nav-link)
- Call get_locator_guidance for locator syntax help for your target library

STEP 4: EXECUTE STEPS
- Use real locators from the DOM/ARIA snapshot, NEVER guess locators
- Wrong: execute_step(keyword="Click", arguments=["css=#submit"]) (guessed locator)
- Right: execute_step(keyword="Click", arguments=["id=submit-btn"]) (from ARIA snapshot)
- Use assign_to to capture return values: execute_step(keyword="Get Text", arguments=["id=msg"], assign_to="message")

STEP 5: HANDLE ERRORS
- "Element not found" -> call get_session_state for fresh DOM/ARIA snapshot
- "Keyword not found" -> call find_keywords to verify spelling

CRITICAL RULES:
1. NEVER guess locators - always inspect the page first with get_session_state
2. ALWAYS verify keyword exists with find_keywords before first use
3. ALWAYS include session_id in execute_step calls
4. Use get_session_state with include_reduced_dom=True for semantic DOM inspection
5. Use arguments=["arg1", "arg2"] (list of strings) for keyword arguments

Available tools: {available_tools}""",
            description="Detailed step-by-step guide for smaller LLMs",
            placeholders=("available_tools",),
        )

    @classmethod
    def browser_focused(cls) -> "InstructionTemplate":
        """Browser-focused template for web automation (~1000 chars)."""
        return cls(
            template_id="browser-focused",
            content="""rf-mcp WEB AUTOMATION WORKFLOW:

1. manage_session(action="init", scenario="...", libraries=["Browser", "BuiltIn"])
2. get_session_state(sections=["page_source"], include_reduced_dom=True) -> read ARIA snapshot
3. execute_step(keyword="Click", arguments=["id=submit-btn"]) -> use real locator from snapshot

FINDING ELEMENTS:
- ARIA snapshot: get_session_state(sections=["page_source"], include_reduced_dom=True)
  Shows roles, names, states - find real IDs and text for locators
- Full DOM: get_session_state(sections=["page_source"], page_source_filtered=True)
  Sanitized/compact HTML - for detailed DOM inspection
- After page changes, call get_session_state again for fresh DOM
- Call get_locator_guidance for locator syntax help (css=, id=, text=, xpath=)

COMMON BROWSER KEYWORDS:
- "New Browser" / "New Page" - start browser
- "Go To" - navigate to URL
- "Click" - click element
- "Fill Text" - type into input
- "Get Text" - read element text
- "Wait For Elements State" - wait for element

NEVER guess locators. ALWAYS inspect the DOM/ARIA snapshot first.

Available tools: {available_tools}""",
            description="Web automation focused template",
            placeholders=("available_tools",),
        )

    @classmethod
    def api_focused(cls) -> "InstructionTemplate":
        """API-focused template for API testing (~800 chars)."""
        return cls(
            template_id="api-focused",
            content="""rf-mcp API TESTING WORKFLOW:

1. manage_session(action="init", scenario="...", libraries=["RequestsLibrary", "BuiltIn"])
2. find_keywords(library="RequestsLibrary") -> see HTTP keywords
3. execute_step(keyword="GET", arguments=["https://api.example.com/users"])

COMMON API KEYWORDS (RequestsLibrary):
- "Create Session" - establish base URL
- "GET" / "POST" / "PUT" / "DELETE" - HTTP methods
- "GET On Session" - GET with session
- "Status Should Be" - assert status code

RESPONSE HANDLING:
- HTTP responses returned in result.output
- Parse JSON with Collections library keywords
- Assert with BuiltIn keywords (Should Be Equal, etc.)

SESSION STATE INSPECTION:
- Use get_session_state(sections=["variables"]) to see current session variables
- Use get_session_state(sections=["libraries"]) to check loaded libraries
- For hybrid API+web testing: get_session_state(sections=["page_source"], include_reduced_dom=True)

Available tools: {available_tools}""",
            description="API testing focused template",
            placeholders=("available_tools",),
        )

    @classmethod
    def get_by_name(cls, name: str) -> "InstructionTemplate":
        """Get a template by its name.

        Args:
            name: Template name (minimal, standard, detailed, browser-focused, api-focused).

        Returns:
            The corresponding InstructionTemplate.

        Raises:
            ValueError: If the template name is not recognized.
        """
        templates = {
            "minimal": cls.minimal,
            "standard": cls.standard,
            "detailed": cls.detailed,
            "browser-focused": cls.browser_focused,
            "api-focused": cls.api_focused,
            # Alias for discovery_first as default
            "discovery_first": cls.discovery_first,
            "locator_prevention": cls.locator_prevention,
        }
        factory = templates.get(name.lower().strip())
        if factory is None:
            raise ValueError(
                f"Unknown template: '{name}'. "
                f"Valid templates: {', '.join(sorted(templates.keys()))}"
            )
        return factory()

    def __str__(self) -> str:
        return f"Template({self.template_id})"

    def __repr__(self) -> str:
        return (
            f"InstructionTemplate(id={self.template_id!r}, "
            f"placeholders={self.placeholders!r})"
        )
