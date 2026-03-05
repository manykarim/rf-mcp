"""FastMCP Memory Adapter - Anti-Corruption Layer.

This adapter registers MCP tools for memory access, translating domain
memory concepts to the FastMCP server's tool format.  All tools are
disabled by default (via DISABLED_TOOL_KWARGS) and enabled selectively
when the memory subsystem is available.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from robotmcp.compat.fastmcp_compat import DISABLED_TOOL_KWARGS

from ..aggregates import MemoryStore
from ..services import EmbeddingService, MemoryHookService, MemoryQueryService
from ..value_objects import MemoryQuery, MemoryType

logger = logging.getLogger(__name__)

# Sentinel returned when the memory subsystem is unavailable.
_UNAVAILABLE_RECALL: Dict[str, Any] = {
    "results": [],
    "count": 0,
    "warning": "Memory service temporarily unavailable",
}


class FastMCPMemoryAdapter:
    """Anti-Corruption Layer adapting memory domain services to MCP tools.

    Responsibilities:
    - Register five memory-related MCP tools on the FastMCP server
    - Translate MCP tool arguments into domain service calls
    - Provide graceful degradation when memory is unavailable
    - Keep infrastructure concerns out of the domain model

    All tools are registered with ``**DISABLED_TOOL_KWARGS`` so they
    are hidden from MCP clients until explicitly enabled via the tool
    profile system.
    """

    def __init__(
        self,
        query_service: MemoryQueryService,
        hook_service: MemoryHookService,
        store: MemoryStore,
        embedding_service: EmbeddingService,
    ) -> None:
        self._query_service = query_service
        self._hook_service = hook_service
        self._store = store
        self._embedding_service = embedding_service

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_tools(self, mcp: Any, *, disabled: bool = True) -> None:
        """Register all memory MCP tools on *mcp*.

        Args:
            mcp: The FastMCP server instance.
            disabled: If True, register with DISABLED_TOOL_KWARGS (hidden).
                      If False, register as visible (when memory is explicitly enabled).
        """
        extra_kwargs = DISABLED_TOOL_KWARGS if disabled else {}
        query_service = self._query_service
        hook_service = self._hook_service
        store = self._store
        embedding_service = self._embedding_service

        # --- recall_step --------------------------------------------------

        @mcp.tool(
            description=(
                "Recall previously successful step sequences for a test scenario"
            ),
            **extra_kwargs,
        )
        async def recall_step(
            scenario: str,
            top_k: int = 5,
        ) -> Dict[str, Any]:
            try:
                results = await query_service.recall_steps(scenario)
                return {
                    "results": [r.to_dict() for r in results],
                    "count": len(results),
                }
            except Exception as exc:
                logger.debug("recall_step failed: %s", exc)
                return dict(_UNAVAILABLE_RECALL)

        # --- recall_fix ---------------------------------------------------

        @mcp.tool(
            description="Recall known fixes for an error message",
            **extra_kwargs,
        )
        async def recall_fix(error_text: str) -> Dict[str, Any]:
            try:
                results = await query_service.recall_for_error(error_text)
                return {
                    "results": [r.to_dict() for r in results],
                    "count": len(results),
                }
            except Exception as exc:
                logger.debug("recall_fix failed: %s", exc)
                return dict(_UNAVAILABLE_RECALL)

        # --- recall_locator -----------------------------------------------

        @mcp.tool(
            description=(
                "Recall structured locator-outcome mappings for an element"
            ),
            **extra_kwargs,
        )
        async def recall_locator(element_description: str) -> Dict[str, Any]:
            try:
                results = await query_service.recall_locators(
                    element_description
                )
                return {
                    "results": [r.to_dict() for r in results],
                    "count": len(results),
                }
            except Exception as exc:
                logger.debug("recall_locator failed: %s", exc)
                return dict(_UNAVAILABLE_RECALL)

        # --- store_knowledge ----------------------------------------------

        @mcp.tool(
            description="Store domain knowledge for future recall",
            **extra_kwargs,
        )
        async def store_knowledge(
            content: str,
            knowledge_type: str = "domain_knowledge",
            tags: List[str] = [],  # noqa: B006 — mutable default required by FastMCP schema
        ) -> Dict[str, Any]:
            if knowledge_type not in ("documentation", "domain_knowledge"):
                return {
                    "stored": False,
                    "reason": (
                        f"Invalid knowledge_type '{knowledge_type}'. "
                        "Must be 'documentation' or 'domain_knowledge'."
                    ),
                }
            try:
                record_id = await hook_service.store_knowledge(
                    content=content,
                    knowledge_type=knowledge_type,
                    tags=tags,
                )
                return {
                    "stored": record_id is not None,
                    "record_id": record_id,
                    "memory_type": knowledge_type,
                }
            except Exception as exc:
                logger.debug("store_knowledge failed: %s", exc)
                return {
                    "stored": False,
                    "reason": f"Storage failed: {exc}",
                }

        # --- get_memory_status --------------------------------------------

        @mcp.tool(
            description=(
                "Get memory subsystem status and collection statistics"
            ),
            **extra_kwargs,
        )
        async def get_memory_status() -> Dict[str, Any]:
            try:
                status = store.to_dict()
                status["backend"] = embedding_service.model_info
                return status
            except Exception as exc:
                logger.debug("get_memory_status failed: %s", exc)
                return {
                    "enabled": False,
                    "error": str(exc),
                }
