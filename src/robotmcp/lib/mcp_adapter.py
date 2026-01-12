"""MCP Services Adapter for AILibrary.

Provides integration with rf-mcp MCP Server tools to leverage existing
functionality instead of reimplementing similar features:

- Keyword discovery with rich metadata (args, types, descriptions)
- Session state and execution history tracking
- Page context extraction with element selectors
- Semantic keyword matching

If MCP services are not accessible (e.g., running outside MCP context),
provides necessary context directly to the AI agent.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from robot.api import logger as rf_logger

logger = logging.getLogger(__name__)


@dataclass
class KeywordInfo:
    """Rich keyword information for AI context."""
    name: str
    library: str
    args: List[str] = field(default_factory=list)
    arg_types: List[str] = field(default_factory=list)
    short_doc: str = ""
    full_doc: str = ""
    tags: List[str] = field(default_factory=list)

    def to_context_string(self) -> str:
        """Format keyword info for AI prompt context."""
        parts = [f"{self.name}"]
        if self.args:
            args_str = ", ".join(self.args)
            parts[0] += f"    {args_str}"
        if self.short_doc:
            parts.append(f"  # {self.short_doc[:80]}")
        return "".join(parts)


@dataclass
class ExecutedStep:
    """Record of an executed step for history tracking."""
    keyword: str
    args: List[str]
    success: bool
    result: Any = None
    error: Optional[str] = None
    timestamp: Optional[str] = None


class MCPServicesAdapter:
    """Adapter for accessing MCP Server tools in AILibrary.

    This class provides a unified interface to:
    1. Get available keywords from imported libraries (with descriptions)
    2. Track execution history (previously executed steps)
    3. Get page/session state for context
    4. Find relevant keywords for specific actions

    Falls back to direct RF integration when MCP tools are not available.
    """

    def __init__(self):
        """Initialize the MCP Services Adapter."""
        self._execution_history: List[ExecutedStep] = []
        self._keyword_cache: Dict[str, KeywordInfo] = {}
        self._libraries_loaded: List[str] = []
        self._mcp_available = False

        # Try to initialize MCP components
        self._init_mcp_components()

    def _init_mcp_components(self) -> None:
        """Initialize MCP Server components if available.

        Note: When running inside a Robot Framework test, we skip the
        RFNativeContextManager as it creates test structures that conflict
        with the running test. Instead, we rely on the BuiltIn fallback
        which integrates properly with the active RF execution context.
        """
        # Check if we're running inside a Robot Framework test
        self._inside_rf_test = False
        try:
            from robot.libraries.BuiltIn import BuiltIn
            builtin = BuiltIn()
            # This will raise if not in RF context
            builtin.get_variable_value("${SUITE_NAME}", None)
            self._inside_rf_test = True
            logger.debug("Running inside RF test - using BuiltIn fallback for keywords")
        except Exception:
            pass

        # Skip RFNativeContextManager when inside RF test to avoid conflicts
        if self._inside_rf_test:
            self._rf_context_manager = None
            self._keyword_matcher = None
            self._mcp_available = True  # Mark as available for history tracking
            return

        try:
            # Try to import MCP components (for standalone MCP execution)
            from robotmcp.components.execution.rf_native_context_manager import (
                get_rf_native_context_manager,
            )
            from robotmcp.components.keyword_matcher import KeywordMatcher

            self._rf_context_manager = get_rf_native_context_manager()
            self._keyword_matcher = KeywordMatcher()
            self._mcp_available = True
            logger.info("MCP components initialized successfully")
        except ImportError as e:
            logger.debug(f"MCP components not available: {e}")
            self._rf_context_manager = None
            self._keyword_matcher = None
            self._mcp_available = False

    def record_step(
        self,
        keyword: str,
        args: List[str],
        success: bool,
        result: Any = None,
        error: Optional[str] = None,
    ) -> None:
        """Record an executed step for history tracking.

        Args:
            keyword: Keyword name that was executed
            args: Arguments passed to the keyword
            success: Whether execution succeeded
            result: Result of the execution (if any)
            error: Error message (if failed)
        """
        import datetime

        step = ExecutedStep(
            keyword=keyword,
            args=args,
            success=success,
            result=result,
            error=error,
            timestamp=datetime.datetime.now().isoformat(),
        )
        self._execution_history.append(step)

        # Keep only last 20 steps to limit context size
        if len(self._execution_history) > 20:
            self._execution_history = self._execution_history[-20:]

    def get_execution_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent execution history for context.

        Args:
            limit: Maximum number of steps to return

        Returns:
            List of step dictionaries with keyword, args, and status
        """
        steps = self._execution_history[-limit:] if limit else self._execution_history
        return [
            {
                "keyword": step.keyword,
                "args": step.args,
                "success": step.success,
            }
            for step in steps
        ]

    def get_available_keywords_with_docs(
        self,
        libraries: List[str] = None,
        limit: int = 50,
    ) -> List[KeywordInfo]:
        """Get available keywords with rich documentation.

        Uses MCP Server's keyword discovery if available, otherwise
        falls back to BuiltIn library inspection.

        Args:
            libraries: Optional list of specific libraries to include
            limit: Maximum number of keywords to return

        Returns:
            List of KeywordInfo objects with full metadata
        """
        keywords = []

        # Check if we're inside an RF test (lazy check for better detection)
        if self._is_inside_rf_test():
            # Use BuiltIn fallback when inside RF test
            keywords = self._get_keywords_from_builtin(libraries, limit)
            return keywords

        # Try MCP-based keyword discovery first (for standalone MCP execution)
        if self._rf_context_manager and self._mcp_available:
            try:
                keywords = self._get_keywords_from_mcp(libraries, limit)
                if keywords:
                    return keywords
            except Exception as e:
                logger.debug(f"MCP keyword discovery failed: {e}")

        # Fallback to BuiltIn-based discovery
        keywords = self._get_keywords_from_builtin(libraries, limit)
        return keywords

    def _is_inside_rf_test(self) -> bool:
        """Check if we're running inside a Robot Framework test.

        Returns:
            True if running inside RF test, False otherwise
        """
        try:
            from robot.libraries.BuiltIn import BuiltIn
            builtin = BuiltIn()
            # This will raise if not in RF context
            builtin.get_variable_value("${SUITE_NAME}", None)
            return True
        except Exception:
            return False

    def _get_keywords_from_mcp(
        self,
        libraries: List[str] = None,
        limit: int = 50,
    ) -> List[KeywordInfo]:
        """Get keywords using MCP Server's RFNativeContextManager.

        Args:
            libraries: Optional list of specific libraries
            limit: Maximum number of keywords

        Returns:
            List of KeywordInfo objects
        """
        keywords = []

        try:
            # Try to get session ID from current context
            session_id = "ailib_session"

            # Ensure context exists
            self._rf_context_manager.create_context_for_session(
                session_id,
                libraries=libraries or [],
            )

            # Get keywords from the session
            result = self._rf_context_manager.list_available_keywords(session_id)

            if not result.get("success"):
                return []

            for lib_kw in result.get("library_keywords", [])[:limit]:
                # Get additional documentation
                doc_result = self._rf_context_manager.get_keyword_documentation(
                    session_id, lib_kw.get("name", "")
                )

                args = doc_result.get("args", []) if doc_result.get("success") else []
                doc = doc_result.get("doc", "") if doc_result.get("success") else ""

                kw_info = KeywordInfo(
                    name=lib_kw.get("name", ""),
                    library=lib_kw.get("library", ""),
                    args=args,
                    short_doc=doc[:100] if doc else "",
                    full_doc=doc,
                )
                keywords.append(kw_info)

            # Also include resource keywords
            for res_kw in result.get("resource_keywords", [])[:10]:
                kw_info = KeywordInfo(
                    name=res_kw.get("name", ""),
                    library=res_kw.get("resource", "resource"),
                )
                keywords.append(kw_info)

        except Exception as e:
            logger.debug(f"Error getting keywords from MCP: {e}")

        return keywords

    def _get_keywords_from_builtin(
        self,
        libraries: List[str] = None,
        limit: int = 50,
    ) -> List[KeywordInfo]:
        """Get keywords using BuiltIn library inspection.

        This is the fallback when MCP components are not available.

        Args:
            libraries: Optional list of specific libraries
            limit: Maximum number of keywords

        Returns:
            List of KeywordInfo objects
        """
        keywords = []

        try:
            from robot.libraries.BuiltIn import BuiltIn
            builtin = BuiltIn()

            # Get all library instances
            all_libs = builtin.get_library_instance(all=True)

            for lib_name, lib_instance in all_libs.items():
                # Filter by requested libraries if specified
                if libraries and lib_name not in libraries:
                    continue

                try:
                    # Get keyword names from library
                    if hasattr(lib_instance, "get_keyword_names"):
                        kw_names = lib_instance.get_keyword_names()
                    elif hasattr(lib_instance, "keywords"):
                        kw_names = list(lib_instance.keywords.keys())
                    else:
                        kw_names = [
                            name for name in dir(lib_instance)
                            if not name.startswith("_")
                            and callable(getattr(lib_instance, name))
                        ]

                    for kw_name in kw_names[:limit]:
                        # Get documentation
                        doc = ""
                        args = []

                        if hasattr(lib_instance, "get_keyword_documentation"):
                            doc = lib_instance.get_keyword_documentation(kw_name) or ""

                        if hasattr(lib_instance, "get_keyword_arguments"):
                            args = lib_instance.get_keyword_arguments(kw_name) or []

                        kw_info = KeywordInfo(
                            name=kw_name,
                            library=lib_name,
                            args=[str(a) for a in args],
                            short_doc=doc[:100] if doc else "",
                            full_doc=doc,
                        )
                        keywords.append(kw_info)

                        if len(keywords) >= limit:
                            break

                except Exception as e:
                    logger.debug(f"Error extracting keywords from {lib_name}: {e}")

                if len(keywords) >= limit:
                    break

        except Exception as e:
            logger.debug(f"BuiltIn keyword discovery failed: {e}")

        return keywords

    def find_keywords_for_action(
        self,
        action_description: str,
        context: str = "web",
        limit: int = 5,
    ) -> List[KeywordInfo]:
        """Find relevant keywords for a given action description.

        Uses semantic matching via KeywordMatcher if available.

        Args:
            action_description: Natural language description of action
            context: Automation context (web, mobile, api)
            limit: Maximum number of matches

        Returns:
            List of matching KeywordInfo objects
        """
        if self._keyword_matcher:
            try:
                import asyncio

                # Run async keyword discovery synchronously
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(
                        self._keyword_matcher.discover_keywords(
                            action_description,
                            context=context,
                            current_state={},
                        )
                    )
                finally:
                    loop.close()

                if result.get("success"):
                    keywords = []
                    for match in result.get("matches", [])[:limit]:
                        kw_info = KeywordInfo(
                            name=match.get("keyword_name", ""),
                            library=match.get("library", ""),
                            args=match.get("arguments", []),
                            arg_types=match.get("argument_types", []),
                            short_doc=match.get("documentation", "")[:100],
                        )
                        keywords.append(kw_info)
                    return keywords

            except Exception as e:
                logger.debug(f"Semantic keyword matching failed: {e}")

        # Fallback: return empty list, context will use general keywords
        return []

    def format_keywords_for_context(
        self,
        keywords: List[KeywordInfo],
        include_docs: bool = True,
    ) -> str:
        """Format keywords into a string for AI prompt context.

        Args:
            keywords: List of KeywordInfo objects
            include_docs: Whether to include short documentation

        Returns:
            Formatted string with keyword names and arguments
        """
        if not keywords:
            return ""

        lines = ["Available keywords:"]
        for kw in keywords[:30]:  # Limit for token efficiency
            if include_docs and kw.short_doc:
                # Include args and doc
                args_str = ", ".join(kw.args) if kw.args else ""
                if args_str:
                    lines.append(f"  - {kw.name}    {args_str}")
                    lines.append(f"    # {kw.short_doc[:60]}")
                else:
                    lines.append(f"  - {kw.name}  # {kw.short_doc[:60]}")
            else:
                # Just keyword name
                if kw.args:
                    lines.append(f"  - {kw.name}    {', '.join(kw.args)}")
                else:
                    lines.append(f"  - {kw.name}")

        return "\n".join(lines)

    def format_history_for_context(self, limit: int = 5) -> str:
        """Format execution history into a string for AI prompt context.

        Args:
            limit: Maximum number of recent steps to include

        Returns:
            Formatted string with recent step history
        """
        steps = self.get_execution_history(limit)

        if not steps:
            return ""

        lines = ["Previously executed steps:"]
        for i, step in enumerate(steps, 1):
            status = "PASS" if step["success"] else "FAIL"
            args_str = "    ".join(step["args"]) if step["args"] else ""
            if args_str:
                lines.append(f"  {i}. [{status}] {step['keyword']}    {args_str}")
            else:
                lines.append(f"  {i}. [{status}] {step['keyword']}")

        return "\n".join(lines)

    def get_rich_context(
        self,
        page_state: Dict[str, Any] = None,
        include_keywords: bool = True,
        include_history: bool = True,
        keyword_limit: int = 20,
        history_limit: int = 5,
    ) -> Dict[str, Any]:
        """Get rich context for AI prompt including keywords and history.

        Args:
            page_state: Current page state (url, title, elements)
            include_keywords: Whether to include available keywords
            include_history: Whether to include execution history
            keyword_limit: Max keywords to include
            history_limit: Max history steps to include

        Returns:
            Dictionary with all context information
        """
        context = {}

        # Add page state if provided
        if page_state:
            if "url" in page_state:
                context["url"] = page_state["url"]
            if "title" in page_state:
                context["title"] = page_state["title"]
            if "elements" in page_state:
                context["elements"] = page_state["elements"]

        # Add available keywords with documentation
        if include_keywords:
            keywords = self.get_available_keywords_with_docs(limit=keyword_limit)
            if keywords:
                # Just include keyword names for context
                context["available_keywords"] = [kw.name for kw in keywords]
                # Also store full info for reference
                context["keywords_with_docs"] = self.format_keywords_for_context(
                    keywords, include_docs=True
                )

        # Add execution history
        if include_history:
            history = self.format_history_for_context(history_limit)
            if history:
                context["execution_history"] = history

        return context

    def clear_history(self) -> None:
        """Clear execution history."""
        self._execution_history.clear()

    @property
    def mcp_available(self) -> bool:
        """Whether MCP components are available."""
        return self._mcp_available


# Singleton instance
_mcp_adapter: Optional[MCPServicesAdapter] = None


def get_mcp_adapter() -> MCPServicesAdapter:
    """Get the global MCP Services Adapter instance."""
    global _mcp_adapter
    if _mcp_adapter is None:
        _mcp_adapter = MCPServicesAdapter()
    return _mcp_adapter
