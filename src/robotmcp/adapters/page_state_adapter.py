"""Page State Capture Adapter.

Anti-corruption layer bridging recovery domain <-> snapshot/page_source infra.
Implements the recovery.services.PageStateCapture protocol.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class PageStateCaptureAdapter:
    """Adapts existing execution infrastructure to PageStateCapture protocol."""

    def __init__(self, execution_engine: Any):
        self._engine = execution_engine

    async def capture_screenshot(self, session_id: str) -> Optional[str]:
        """Capture screenshot via execution engine."""
        try:
            result = await self._engine.execute_step(
                "Screenshot",
                [],
                session_id,
                detail_level="minimal",
                use_context=True,
            )
            if result.get("success"):
                return result.get("return_value")
        except Exception as e:
            logger.debug("Screenshot capture failed: %s", e)
        return None

    async def capture_page_source(
        self, session_id: str, max_chars: int = 2000
    ) -> Optional[str]:
        """Capture page source via execution engine."""
        try:
            result = await self._engine.execute_step(
                "Get Source",
                [],
                session_id,
                detail_level="minimal",
                use_context=True,
            )
            if result.get("success"):
                source = result.get("return_value", "")
                if isinstance(source, str) and len(source) > max_chars:
                    return source[:max_chars] + "..."
                return source
        except Exception:
            pass
        return None

    async def get_current_url(self, session_id: str) -> Optional[str]:
        """Get current URL via execution engine."""
        try:
            result = await self._engine.execute_step(
                "Get Location",
                [],
                session_id,
                detail_level="minimal",
                use_context=True,
            )
            if result.get("success"):
                return result.get("return_value")
        except Exception:
            pass
        return None

    async def get_page_title(self, session_id: str) -> Optional[str]:
        """Get page title via execution engine."""
        try:
            result = await self._engine.execute_step(
                "Get Title",
                [],
                session_id,
                detail_level="minimal",
                use_context=True,
            )
            if result.get("success"):
                return result.get("return_value")
        except Exception:
            pass
        return None


class EvidenceCollectorImpl:
    """Implements EvidenceCollectorProtocol using PageStateCaptureAdapter."""

    def __init__(self, execution_engine: Any):
        self._page_state = PageStateCaptureAdapter(execution_engine)

    async def collect_evidence(self, session_id: str) -> dict:
        """Collect all available evidence."""
        evidence = {}
        screenshot = await self._page_state.capture_screenshot(session_id)
        if screenshot:
            evidence["screenshot_base64"] = screenshot
        page_source = await self._page_state.capture_page_source(session_id)
        if page_source:
            evidence["page_source_snippet"] = page_source
        url = await self._page_state.get_current_url(session_id)
        if url:
            evidence["current_url"] = url
        title = await self._page_state.get_page_title(session_id)
        if title:
            evidence["page_title"] = title
        return evidence
