"""AILibrary - AI-powered keywords for Robot Framework.

Provides three essential keywords that cover all use cases:
- Do: Execute any action described in natural language
- Check: Verify/Assert state described in natural language
- Ask: Query for information described in natural language

Also provides recording and export capabilities:
- Start Recording: Begin capturing executed keywords
- Stop Recording: Stop capturing
- Export Test Suite: Generate .robot file from recording
- Get Recorded Steps: Get steps as list for inspection
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from robot.api import logger as rf_logger
from robot.api.deco import keyword, library

from robotmcp.lib.agent import RFAgent, parse_keyword_call
from robotmcp.lib.context import RFContextBridge
from robotmcp.lib.exporter import TestSuiteExporter
from robotmcp.lib.providers import ProviderConfig
from robotmcp.lib.recorder import ExecutionRecorder, get_recorder
from robotmcp.lib.listener import RecordingListener
from robotmcp.lib.retry import ExecutionResult, RetryContext, RetryHandler

logger = logging.getLogger(__name__)


@library(scope="GLOBAL", version="1.0.0", doc_format="ROBOT")
class AILibrary:
    """AI-powered keywords for Robot Framework.

    This library provides natural language keywords (Do, Check, Ask) that can be
    called directly from Robot Framework tests, bridging AI capabilities with
    executable RF keywords.

    = Installation =

    The AILibrary is an optional extra for the rf-mcp package:
    | pip install rf-mcp[lib]

    = Configuration =

    The library can be configured via import arguments or a YAML config file:

    | *** Settings ***
    | Library    AILibrary
    | ...    provider=anthropic
    | ...    api_key=%{ANTHROPIC_API_KEY}
    | ...    model=claude-sonnet-4-20250514

    Or using a config file:
    | Library    AILibrary    config=${CURDIR}/ai_config.yaml

    = Keywords =

    The library provides three core keywords:

    | Keyword | Purpose | Returns |
    | Do | Execute any action | None |
    | Check | Verify/Assert state | Pass/Fail |
    | Ask | Query for information | String/Data |

    = Examples =

    | *** Test Cases ***
    | Purchase Product
    |     Do    Login as standard_user with password secret_sauce
    |     Do    Add the backpack to cart
    |     Check    Cart shows 1 item
    |     ${order_id}=    Ask    What is the order confirmation number?
    """

    ROBOT_LIBRARY_SCOPE = "GLOBAL"
    ROBOT_LIBRARY_VERSION = "1.0.0"
    # ROBOT_LIBRARY_LISTENER is set in __init__ to capture keyword data (e.g., variable assignments)

    def __init__(
        self,
        provider: str = "anthropic",
        api_key: str = None,
        model: str = "claude-sonnet-4-20250514",
        retries: int = 3,
        retry_delay: str = "1s",
        log_level: str = "INFO",
        config: str = None,
        auto_record: bool = False,
        export_path: str = None,
        **kwargs,
    ):
        """Initialize AILibrary.

        Args:
            provider: AI provider (anthropic, openai, ollama, azure)
            api_key: API key (supports %{ENV_VAR} syntax)
            model: Model name
            retries: Default retry count
            retry_delay: Delay between retries (supports RF time format)
            log_level: Logging verbosity
            config: Path to YAML config file (overrides other args)
            auto_record: Automatically record all executions
            export_path: Path for auto-export after tests
            **kwargs: Additional provider-specific options
        """
        # Load config from file or arguments
        if config:
            self.config = ProviderConfig.from_yaml(config)
        else:
            self.config = ProviderConfig.from_kwargs(
                provider=provider,
                api_key=api_key,
                model=model,
                retries=retries,
                retry_delay=retry_delay,
                log_level=log_level,
                **kwargs,
            )

        # Initialize components
        self._agent = RFAgent(self.config)
        self._context = RFContextBridge()
        self._recorder = get_recorder()
        self._retry_handler = RetryHandler(
            max_retries=self.config.retries,
            retry_delay=self.config.retry_delay,
        )
        self._exporter = TestSuiteExporter(self._recorder)
        # Create listener that shares the same recorder
        self._listener = RecordingListener(recorder=self._recorder)
        # Register listener with Robot Framework (needed to capture data.assign)
        # Setting ROBOT_LIBRARY_LISTENER attribute enables RF to call listener methods
        self.ROBOT_LIBRARY_LISTENER = self._listener

        # Auto-record settings
        self._auto_record = auto_record
        self._export_path = export_path

        # Configure logging
        logging.getLogger("robotmcp.lib").setLevel(
            getattr(logging, self.config.log_level, logging.INFO)
        )

        if self._auto_record:
            self.start_recording()

        rf_logger.info(f"AILibrary initialized with provider: {self.config.provider}")

    # ==========================================================================
    # Core Keywords
    # ==========================================================================

    def _get_pending_variable_assignment(self) -> Optional[str]:
        """Get variable assignment detected by listener for current AI keyword.

        Returns:
            Variable name (without ${}) or None
        """
        pending = self._recorder.get_and_clear_pending_assignment()
        if not pending:
            return None

        var = pending[0] if pending else None
        if var:
            # Normalize: remove trailing = if present (RF includes it in data.assign)
            if var.endswith("="):
                var = var[:-1]
            # Normalize: remove ${} wrapper if present
            if var.startswith("${") and var.endswith("}"):
                var = var[2:-1]
        return var

    @keyword("Do")
    def do(
        self,
        prompt: str,
        retries: int = None,
        retry_delay: str = None,
    ) -> None:
        """Execute an action described in natural language.

        This keyword translates a natural language prompt into appropriate
        Robot Framework keyword calls and executes them.

        | =Arguments= | =Description= |
        | prompt | Natural language description of the action to perform |
        | retries | Number of retry attempts (default: library setting) |
        | retry_delay | Delay between retries (default: library setting) |

        = Examples =
        | Do | Login as standard_user with password secret_sauce |
        | Do | Add the backpack to the shopping cart |
        | Do | Fill checkout form with valid test data |
        | Do | Click the dynamically loaded button | retries=5 | retry_delay=2s |
        """
        # Get any pending variable assignment from listener
        assigned_var = self._get_pending_variable_assignment()

        # Resolve variables in prompt
        resolved_prompt = self._context.resolve_variables(prompt)
        rf_logger.info(f"AILibrary Do: {resolved_prompt}")

        # Start AI step recording (with variable assignment if detected)
        self._recorder.start_ai_step(resolved_prompt, "Do", assigned_variable=assigned_var)

        try:
            result = self._execute_with_ai(
                prompt=resolved_prompt,
                keyword_type="do",
                retries=retries or self.config.retries,
                retry_delay=retry_delay,
            )

            if not result.success:
                self._recorder.end_ai_step(success=False, error=result.error)
                raise AssertionError(f"Do failed: {result.error}")

            self._recorder.end_ai_step(success=True, attempts=result.attempt)

        except Exception as e:
            self._recorder.end_ai_step(success=False, error=str(e))
            raise

    @keyword("Check")
    def check(
        self,
        prompt: str,
        retries: int = None,
        retry_delay: str = None,
    ) -> None:
        """Verify a condition described in natural language.

        This keyword translates a natural language assertion into appropriate
        Robot Framework verification keywords and executes them.

        | =Arguments= | =Description= |
        | prompt | Natural language description of the condition to verify |
        | retries | Number of retry attempts (default: library setting) |
        | retry_delay | Delay between retries (default: library setting) |

        = Examples =
        | Check | Cart badge displays 1 item |
        | Check | User is logged in successfully |
        | Check | Error message is visible |
        | Check | Order confirmation page is displayed |
        """
        # Get any pending variable assignment from listener
        assigned_var = self._get_pending_variable_assignment()

        # Resolve variables in prompt
        resolved_prompt = self._context.resolve_variables(prompt)
        rf_logger.info(f"AILibrary Check: {resolved_prompt}")

        # Start AI step recording (with variable assignment if detected)
        self._recorder.start_ai_step(resolved_prompt, "Check", assigned_variable=assigned_var)

        try:
            result = self._execute_with_ai(
                prompt=resolved_prompt,
                keyword_type="check",
                retries=retries or self.config.retries,
                retry_delay=retry_delay,
            )

            if not result.success:
                self._recorder.end_ai_step(success=False, error=result.error)
                raise AssertionError(f"Check failed: {result.error}")

            self._recorder.end_ai_step(success=True, attempts=result.attempt)

        except Exception as e:
            self._recorder.end_ai_step(success=False, error=str(e))
            raise

    @keyword("Ask")
    def ask(
        self,
        prompt: str,
        retries: int = None,
        retry_delay: str = None,
        assign_to: str = None,
    ) -> str:
        """Query for information described in natural language.

        This keyword translates a natural language query into appropriate
        Robot Framework keywords to extract and return the requested data.

        | =Arguments= | =Description= |
        | prompt | Natural language description of the information to retrieve |
        | retries | Number of retry attempts (default: library setting) |
        | retry_delay | Delay between retries (default: library setting) |
        | assign_to | Variable name to assign result (for recording/export) - usually auto-detected |

        = Examples =
        | ${order_id}= | Ask | What is the order confirmation number? |
        | ${count}= | Ask | How many items are in the cart? |
        | ${title}= | Ask | What is the page title? |
        | ${price}= | Ask | What is the price of the first product? |

        = Variable Assignment =
        Variable assignments are automatically detected from Robot Framework syntax:
        | ${product_name}=    Ask    What is the name of the first product?
        |
        | # Exported as:
        | ${product_name}=    Get Text    .inventory_item_name >> nth=0

        The `assign_to` parameter is only needed if auto-detection doesn't work.

        = Return Value =
        Returns the extracted data as a string.
        """
        # Get any pending variable assignment from listener (RF-detected)
        assigned_var = self._get_pending_variable_assignment()

        # Fall back to explicit assign_to parameter if RF detection didn't find anything
        if not assigned_var and assign_to:
            assigned_var = assign_to.strip()
            if assigned_var.startswith("${") and assigned_var.endswith("}"):
                assigned_var = assigned_var[2:-1]

        # Resolve variables in prompt
        resolved_prompt = self._context.resolve_variables(prompt)
        rf_logger.info(f"AILibrary Ask: {resolved_prompt}")
        if assigned_var:
            rf_logger.debug(f"Variable assignment detected: ${{{assigned_var}}}")

        # Start AI step recording with variable assignment info
        self._recorder.start_ai_step(resolved_prompt, "Ask", assigned_variable=assigned_var)

        try:
            result = self._execute_with_ai(
                prompt=resolved_prompt,
                keyword_type="ask",
                retries=retries or self.config.retries,
                retry_delay=retry_delay,
            )

            if not result.success:
                self._recorder.end_ai_step(success=False, error=result.error)
                raise AssertionError(f"Ask failed: {result.error}")

            self._recorder.end_ai_step(
                success=True, result=result.result, attempts=result.attempt
            )

            return result.result

        except Exception as e:
            self._recorder.end_ai_step(success=False, error=str(e))
            raise

    # ==========================================================================
    # Recording Keywords
    # ==========================================================================

    @keyword("Start Recording")
    def start_recording(self) -> None:
        """Begin recording all keyword executions.

        Records all keywords executed by AI keywords (Do, Check, Ask).
        The actual executed keywords (Fill Text, Click, etc.) are captured
        and stored as sub-steps of the AI keywords for later export.

        = Example =
        | Start Recording |
        | Do | Login as standard_user |
        | Do | Add product to cart |
        | Export Test Suite | ${OUTPUT_DIR}/recorded_test.robot |
        """
        self._recorder.start_recording()
        rf_logger.info("AILibrary: Recording started")

    @keyword("Stop Recording")
    def stop_recording(self) -> None:
        """Stop recording keyword executions.

        = Example =
        | Start Recording |
        | Do | Login as standard_user |
        | Stop Recording |
        | ${steps}= | Get Recorded Steps |
        """
        self._recorder.stop_recording()
        rf_logger.info("AILibrary: Recording stopped")

    @keyword("Export Test Suite")
    def export_test_suite(
        self,
        path: str,
        suite_name: str = "Generated Suite",
        test_name: str = "Generated Test",
        include_comments: bool = True,
        include_imports: bool = True,
        format: str = "robot",
        **kwargs,
    ) -> str:
        """Export recorded keywords as a complete Robot Framework test suite.

        | =Arguments= | =Description= |
        | path | Output file path |
        | suite_name | Suite name in generated file |
        | test_name | Test case name |
        | include_comments | Include original prompts as comments |
        | include_imports | Auto-detect and include Library imports |
        | format | Output format: robot, json, yaml |

        = Additional Options =
        | group_by_prompt | Group keywords under original prompt |
        | exclude_libraries | Libraries to skip (comma-separated) |
        | exclude_keywords | Keywords to skip (comma-separated) |
        | flatten_ai_only | Expand AI keywords into sub-keywords |
        | include_setup_teardown | Include [Setup]/[Teardown] |

        = Example =
        | Export Test Suite | ${OUTPUT_DIR}/purchase_flow.robot |
        | ... | suite_name=Purchase Flow Tests |
        | ... | test_name=Complete Purchase E2E |

        = Returns =
        Path to the exported file.
        """
        # Resolve path
        resolved_path = self._context.resolve_variables(path)

        # Parse exclude options
        if "exclude_libraries" in kwargs and isinstance(kwargs["exclude_libraries"], str):
            kwargs["exclude_libraries"] = [
                lib.strip() for lib in kwargs["exclude_libraries"].split(",")
            ]
        if "exclude_keywords" in kwargs and isinstance(kwargs["exclude_keywords"], str):
            kwargs["exclude_keywords"] = [
                kw.strip() for kw in kwargs["exclude_keywords"].split(",")
            ]

        result_path = self._exporter.export(
            path=resolved_path,
            suite_name=suite_name,
            test_name=test_name,
            include_comments=include_comments,
            include_imports=include_imports,
            format=format,
            **kwargs,
        )

        rf_logger.info(f"AILibrary: Exported test suite to {result_path}")
        return result_path

    @keyword("Get Recorded Steps")
    def get_recorded_steps(self) -> List[Dict[str, Any]]:
        """Returns list of recorded step dictionaries for inspection.

        = Example =
        | ${steps}= | Get Recorded Steps |
        | Log | Recorded ${steps.__len__()} steps |
        | FOR | ${step} | IN | @{steps} |
        |     | Log | ${step}[keyword] |
        | END |

        = Returns =
        List of dictionaries, each containing:
        - keyword: Keyword name
        - args: Positional arguments
        - kwargs: Named arguments
        - library: Library name
        - type: Step type (regular, ai, setup, teardown)
        - timestamp: Execution timestamp
        - success: Whether execution succeeded
        """
        steps = self._recorder.get_steps()
        return [step.to_dict() for step in steps]

    # ==========================================================================
    # Internal Methods
    # ==========================================================================

    def _execute_with_ai(
        self,
        prompt: str,
        keyword_type: str,  # "do", "check", "ask"
        retries: int,
        retry_delay: str = None,
    ) -> ExecutionResult:
        """Execute a prompt using AI with retry logic.

        Args:
            prompt: Natural language prompt
            keyword_type: Type of keyword (do, check, ask)
            retries: Number of retry attempts
            retry_delay: Delay between retries

        Returns:
            ExecutionResult with success status and result/error
        """
        # Configure retry handler for this execution
        if retry_delay:
            delay = ProviderConfig._parse_time_string(retry_delay)
        else:
            delay = self.config.retry_delay

        retry_handler = RetryHandler(max_retries=retries, retry_delay=delay)

        # Run async execution
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                retry_handler.execute_with_retry(
                    prompt=prompt,
                    execute_fn=lambda kw_call: self._execute_keyword_call(kw_call, keyword_type),
                    generate_correction_fn=lambda ctx: self._generate_keyword(
                        prompt, keyword_type, ctx
                    ),
                    get_page_state_fn=self._context.get_page_state,
                    get_keywords_fn=lambda: [
                        kw["name"] for kw in self._context.get_available_keywords()
                    ],
                )
            )
            return result
        finally:
            loop.close()

    def _generate_keyword(
        self,
        prompt: str,
        keyword_type: str,
        retry_context: RetryContext,
    ) -> str:
        """Generate keyword call using AI agent.

        Args:
            prompt: Natural language prompt
            keyword_type: Type of keyword (do, check, ask)
            retry_context: Retry context with error information

        Returns:
            Generated keyword call string
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Use page state from retry context (already fetched)
            context = retry_context.page_state.copy() if retry_context.page_state else {}
            context["available_keywords"] = retry_context.available_keywords[:50] if retry_context.available_keywords else []

            # Add execution history from MCP adapter for better context
            execution_history = self._context.get_execution_history(limit=5)
            if execution_history:
                context["execution_history"] = execution_history

            if keyword_type == "do":
                return loop.run_until_complete(
                    self._agent.generate_keyword_for_do(
                        prompt, context=context, retry_context=retry_context
                    )
                )
            elif keyword_type == "check":
                return loop.run_until_complete(
                    self._agent.generate_keyword_for_check(
                        prompt, context=context, retry_context=retry_context
                    )
                )
            elif keyword_type == "ask":
                return loop.run_until_complete(
                    self._agent.generate_keyword_for_ask(
                        prompt, context=context, retry_context=retry_context
                    )
                )
            else:
                raise ValueError(f"Unknown keyword type: {keyword_type}")
        finally:
            loop.close()

    def _execute_keyword_call(
        self,
        keyword_call: str,
        keyword_type: str,
    ) -> ExecutionResult:
        """Execute a generated keyword call.

        Args:
            keyword_call: Keyword call string (e.g., "Click    button#submit")
            keyword_type: Type of keyword (do, check, ask)

        Returns:
            ExecutionResult with success status and result/error
        """
        # Handle multi-line keyword calls (multiple keywords)
        if "\n" in keyword_call:
            lines = [l.strip() for l in keyword_call.split("\n") if l.strip()]
            last_result = None

            for line in lines:
                result = self._execute_single_keyword(line, keyword_type)
                last_result = result

                if not result.success:
                    return result  # Stop on first failure

            return last_result or ExecutionResult(success=True)

        return self._execute_single_keyword(keyword_call, keyword_type)

    def _execute_single_keyword(
        self,
        keyword_call: str,
        keyword_type: str,
    ) -> ExecutionResult:
        """Execute a single keyword call.

        Args:
            keyword_call: Keyword call string
            keyword_type: Type of keyword (do, check, ask)

        Returns:
            ExecutionResult with success status and result/error
        """
        try:
            # Parse the keyword call
            keyword_name, args = parse_keyword_call(keyword_call)

            if not keyword_name:
                return ExecutionResult(
                    success=False,
                    error="Empty keyword call",
                    keyword="",
                    args=[],
                )

            # Execute the keyword
            success, result, error = self._context.run_keyword_and_return_status(
                keyword_name, *args
            )

            # Record step in execution history for future context (MCP adapter)
            self._context.record_step(
                keyword=keyword_name,
                args=list(args),
                success=success,
                result=result,
                error=error,
            )

            # Record step to ExecutionRecorder for test suite export
            # This will add the keyword as a sub-step to the current AI step
            # (if we're inside a Do/Check/Ask keyword)
            if success:
                library = self._infer_library(keyword_name)
                self._recorder.record_step(
                    keyword=keyword_name,
                    args=list(args),
                    library=library,
                    result=result,
                    success=True,
                )

            return ExecutionResult(
                success=success,
                result=result,
                error=error,
                keyword=keyword_name,
                args=args,
            )

        except Exception as e:
            logger.error(f"Keyword execution error: {e}")
            return ExecutionResult(
                success=False,
                error=str(e),
                keyword=keyword_call,
                args=[],
            )

    def _infer_library(self, keyword_name: str) -> str:
        """Infer the library name from a keyword name.

        Args:
            keyword_name: Keyword name

        Returns:
            Inferred library name or None
        """
        keyword_lower = keyword_name.lower()

        # Browser Library patterns
        browser_keywords = {
            "new browser", "new context", "new page", "go to",
            "click", "fill text", "get text", "take screenshot",
            "wait for elements state", "get url", "get title",
            "get page source", "close browser", "close context",
            "close page", "fill", "type text", "press keys",
            "check checkbox", "uncheck checkbox", "select options by",
        }
        if keyword_lower in browser_keywords:
            return "Browser"

        # SeleniumLibrary patterns
        selenium_keywords = {
            "open browser", "close browser", "go to", "click element",
            "input text", "get text", "wait until element is visible",
            "get location", "get title", "click button", "click link",
        }
        if keyword_lower in selenium_keywords:
            return "SeleniumLibrary"

        # BuiltIn patterns
        builtin_keywords = {
            "log", "should be equal", "should contain", "set variable",
            "run keyword if", "run keywords", "sleep", "should not be empty",
        }
        if keyword_lower in builtin_keywords:
            return "BuiltIn"

        return None

    # ==========================================================================
    # Lifecycle Methods
    # ==========================================================================

    def _end_test(self, name: str, attrs: Dict[str, Any]) -> None:
        """Called when a test ends (for auto-export).

        Args:
            name: Test name
            attrs: Test attributes
        """
        if self._auto_record and self._export_path:
            try:
                export_dir = Path(self._export_path)
                export_dir.mkdir(parents=True, exist_ok=True)
                safe_name = name.replace(" ", "_").replace("/", "_")
                export_file = export_dir / f"{safe_name}.robot"
                self.export_test_suite(str(export_file), test_name=name)
            except Exception as e:
                rf_logger.warn(f"Auto-export failed: {e}")
