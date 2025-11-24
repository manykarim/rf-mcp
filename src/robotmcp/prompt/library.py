"""Robot Framework library exposing MCP prompt execution."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from contextlib import contextmanager
from dataclasses import asdict
from typing import Any, Dict, List, Optional

# Ensure attach mode defaults to off *before* importing server modules
os.environ.setdefault("ROBOTMCP_ATTACH_DEFAULT", "off")

from robot.libraries.BuiltIn import BuiltIn
from robot.conf import RobotSettings
from robot.output import Output, LOGGER, logger as rf_logger

from robotmcp.prompt.config import PromptRuntimeConfig, load_prompt_config
from robotmcp.prompt.runner import PromptRunner


class McpPromptLibrary:
    """Robot Framework library that runs MCP prompts inside the active suite."""

    ROBOT_LIBRARY_SCOPE = "SUITE"

    def __init__(
        self,
        default_prompt: str = "automate",
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float | None = None,
        max_iterations: int | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self._default_prompt = default_prompt
        self._base_config = load_prompt_config(
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_iterations=max_iterations,
            max_tokens=max_tokens,
        )
        self._runner = PromptRunner()
        self._session_results: Dict[str, Dict[str, Any]] = {}

    # --- Public Keywords ---
    def prompt(
        self,
        scenario: str,
        prompt: Optional[str] = None,
        session_id: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        max_iterations: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Run the configured MCP prompt using the provided scenario text."""

        resolved_session = session_id or self._current_robot_session_id()
        self._disable_browser_pause_on_failure()
        self._ensure_robot_logger()
        prelude = self._build_prelude_messages()
        config = self._base_config.with_overrides(
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_iterations=max_iterations,
            max_tokens=max_tokens,
        )

        with self._suppress_python_logging():
            result = self._runner.run(
                scenario=scenario,
                prompt_key=prompt or self._default_prompt,
                session_id=resolved_session,
                config=config,
                prelude_messages=prelude,
            )

        self._store_result(resolved_session, result)
        if not result.success:
            raise AssertionError(result.final_response)
        return result.final_response

    def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        max_iterations: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Send a free-form chat instruction to the MCP agent."""

        resolved_session = session_id or self._current_robot_session_id()
        self._disable_browser_pause_on_failure()
        self._ensure_robot_logger()
        prelude = self._build_prelude_messages()
        config = self._base_config.with_overrides(
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_iterations=max_iterations,
            max_tokens=max_tokens,
        )

        with self._suppress_python_logging():
            result = self._runner.run_chat(
                message=message,
                session_id=resolved_session,
                config=config,
                prelude_messages=prelude,
            )

        self._store_result(resolved_session, result)
        if not result.success:
            raise AssertionError(result.final_response)
        return result.final_response

    def set_prompt_model(self, model: str, base_url: Optional[str] = None) -> None:
        """Override the default model/base URL for subsequent prompts."""

        self._base_config = self._base_config.with_overrides(model=model, base_url=base_url)

    def get_prompt_history(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Return the stored transcript for the specified session."""

        session = session_id or self._current_robot_session_id()
        return self._session_results.get(session, {})

    def reset_prompt_session(self, session_id: Optional[str] = None) -> None:
        """Clear cached result data for the given session."""

        session = session_id or self._current_robot_session_id()
        self._session_results.pop(session, None)

    # --- Internal helpers ---
    def _current_robot_session_id(self) -> str:
        builtin = BuiltIn()
        test_name = builtin.get_variable_value("${TEST NAME}", "robotmcp_prompt")
        return str(test_name or "robotmcp_prompt")

    def _log_execution(self, session_id: str, result) -> None:
        prefix = f"[MCP Prompt:{session_id}]"
        log_enabled = getattr(LOGGER, "_output_file", None) is not None
        if not log_enabled:
            return
        for call in result.executed_calls:
            status = "PASS" if call.success else "FAIL"
            summary = f"{prefix} {status} {call.name} args={json.dumps(call.arguments, ensure_ascii=False)}"
            BuiltIn().log(summary, "INFO" if call.success else "WARN")
            if call.response_text:
                BuiltIn().log(f"{prefix} output: {call.response_text[:500]}", "INFO")
        BuiltIn().log(f"{prefix} final: {result.final_response}", "INFO")

    def _store_result(self, session: str, result) -> None:
        self._session_results[session] = {
            "success": result.success,
            "final_response": result.final_response,
            "transcript": result.transcript,
            "executed_calls": [asdict(call) for call in result.executed_calls],
            "iterations": result.iterations,
        }
        self._log_execution(session, result)

    def _disable_browser_pause_on_failure(self) -> None:
        """Ensure Browser library won't block on failures with interactive prompts."""

        try:
            browser = BuiltIn().get_library_instance("Browser")
        except RuntimeError:
            return
        except Exception:
            return

        if browser is None:
            return

        try:
            if getattr(browser, "pause_on_failure", False):
                browser.pause_on_failure = False
        except Exception:
            # Fall back to keyword if attribute assignment fails
            try:
                BuiltIn().run_keyword("Set Pause On Failure", False)
            except Exception:
                pass

    def _build_prelude_messages(self) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        browser_msg = self._get_browser_context_message()
        if browser_msg:
            messages.append({"role": "system", "content": browser_msg})
        selenium_msg = self._get_selenium_availability_message()
        if selenium_msg:
            messages.append({"role": "system", "content": selenium_msg})
        return messages

    def _get_browser_context_message(self) -> Optional[str]:
        try:
            BuiltIn().get_library_instance("Browser")
        except RuntimeError:
            return None
        except Exception:
            return None

        notes = [
            "Browser library is already imported in this Robot Framework run.",
            "Reuse the existing Browser session instead of calling 'Open Browser'.",
            "Use Browser keywords such as 'Go To', 'Fill Text', 'Click', 'Wait For Elements State'.",
            "Playwright accepts 'chromium', 'firefox', or 'webkit' as browser names.",
        ]
        try:
            current_url = BuiltIn().run_keyword("Get Url")
            if current_url:
                notes.append(f"Current page URL: {current_url}")
        except Exception:
            pass
        return " ".join(notes)

    def _get_selenium_availability_message(self) -> Optional[str]:
        try:
            BuiltIn().get_library_instance("SeleniumLibrary")
            return None
        except RuntimeError:
            return (
                "SeleniumLibrary is not imported in this suite, so Selenium-only keywords "
                "such as 'Open Browser' or 'Input Text' are unavailable. Prefer Browser library keywords."
            )
        except Exception:
            return None

    @contextmanager
    def _suppress_python_logging(self):
        previous = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        try:
            yield
        finally:
            logging.disable(previous)

    def _ensure_robot_logger(self) -> None:
        try:
            if getattr(rf_logger.LOGGER, "_output_file", None):
                return
            temp_output_dir = tempfile.mkdtemp(prefix="rf_mcp_prompt_")
            settings = RobotSettings(outputdir=temp_output_dir, output=None)
            new_output = Output(settings)
            try:
                new_output.library_listeners.new_suite_scope()
            except Exception:
                pass
        except Exception:
            pass
