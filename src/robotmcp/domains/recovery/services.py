"""Recovery Domain Services.

Domain services coordinate between the RecoveryEngine aggregate,
RecoveryPlan entities, and external infrastructure via protocol-based
anti-corruption layers.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from .value_objects import (
    ErrorClassification, RecoveryAction, RecoveryStrategy, RecoveryTier,
)
from .entities import RecoveryPlan, RecoveryPlanPhase
from .aggregates import RecoveryEngine

logger = logging.getLogger(__name__)


@runtime_checkable
class KeywordRunner(Protocol):
    """Protocol for executing keywords (anti-corruption layer).

    The recovery domain never imports the execution infrastructure
    directly. Instead, callers provide an adapter implementing this
    protocol.
    """
    async def run_keyword(
        self,
        session_id: str,
        keyword: str,
        args: List[str],
        timeout: Optional[str] = None,
    ) -> Any: ...


@runtime_checkable
class PageStateCapture(Protocol):
    """Protocol for capturing page state (anti-corruption layer to snapshot domain).

    Used by EvidenceCollector and Tier2RecoveryService to gather
    diagnostic context without coupling to browser/selenium internals.
    """
    async def capture_screenshot(self, session_id: str) -> Optional[str]: ...
    async def capture_page_source(
        self, session_id: str, max_chars: int = 2000
    ) -> Optional[str]: ...
    async def get_current_url(self, session_id: str) -> Optional[str]: ...
    async def get_page_title(self, session_id: str) -> Optional[str]: ...


@dataclass
class ErrorClassifier:
    """Service that classifies error messages using the RecoveryEngine.

    This is a thin wrapper that delegates to RecoveryEngine.classify().
    It exists as a separate service for composability: callers can
    classify errors without needing the full engine reference.
    """
    engine: RecoveryEngine

    def classify(self, error_message: str) -> ErrorClassification:
        """Classify an error message into an ErrorClassification.

        Args:
            error_message: The error text from a failed keyword.

        Returns:
            The matching ErrorClassification (UNKNOWN if no match).
        """
        return self.engine.classify(error_message)


@dataclass
class Tier1RecoveryService:
    """Executes Tier 1 keyword-specific recovery strategies.

    Tier 1 strategies are simple heuristics: wait, scroll, dismiss
    overlays, handle alerts. They do not require page-level context.
    Actions are executed best-effort: if one fails, remaining actions
    still run.
    """
    keyword_runner: KeywordRunner

    async def execute(
        self,
        session_id: str,
        strategy: RecoveryStrategy,
        plan: RecoveryPlan,
    ) -> List[RecoveryAction]:
        """Execute recovery actions and return all actions that were run.

        Args:
            session_id: Session to execute actions against.
            strategy: The selected Tier 1 strategy.
            plan: The recovery plan to record actions on.

        Returns:
            List of RecoveryActions that were attempted (success or failure).
        """
        executed_actions: List[RecoveryAction] = []
        for action in strategy.actions:
            try:
                await self.keyword_runner.run_keyword(
                    session_id, action.keyword, list(action.args), timeout="5s"
                )
                executed_actions.append(action)
                plan.record_action(action)
            except Exception as e:
                logger.warning(
                    "Tier1 recovery action %s failed: %s", action.keyword, e
                )
                executed_actions.append(action)
                plan.record_action(action)
                # Continue with remaining actions -- best effort
        return executed_actions


@dataclass
class Tier2RecoveryService:
    """Executes Tier 2 context-aware recovery strategies.

    Tier 2 strategies may inspect page state before executing
    recovery actions. They are used on escalation (second attempt)
    or when the error classification implies page-level problems.
    """
    keyword_runner: KeywordRunner
    page_state: Optional[PageStateCapture] = None

    async def execute(
        self,
        session_id: str,
        strategy: RecoveryStrategy,
        plan: RecoveryPlan,
    ) -> List[RecoveryAction]:
        """Execute context-aware recovery actions.

        Captures evidence (current URL, page title) before running
        strategy actions. Evidence is stored on the plan for diagnostics.

        Args:
            session_id: Session to execute actions against.
            strategy: The selected Tier 2 strategy.
            plan: The recovery plan to record actions and evidence on.

        Returns:
            List of RecoveryActions that were attempted.
        """
        executed_actions: List[RecoveryAction] = []

        # Capture evidence first (if page state available)
        if self.page_state:
            try:
                url = await self.page_state.get_current_url(session_id)
                if url:
                    plan.evidence["current_url"] = url
                title = await self.page_state.get_page_title(session_id)
                if title:
                    plan.evidence["page_title"] = title
            except Exception as e:
                logger.warning("Evidence capture failed: %s", e)

        # Execute strategy actions
        for action in strategy.actions:
            try:
                await self.keyword_runner.run_keyword(
                    session_id, action.keyword, list(action.args), timeout="10s"
                )
                executed_actions.append(action)
                plan.record_action(action)
            except Exception as e:
                logger.warning(
                    "Tier2 recovery action %s failed: %s", action.keyword, e
                )
                executed_actions.append(action)
                plan.record_action(action)
        return executed_actions


@dataclass
class EvidenceCollector:
    """Captures diagnostic evidence on failure.

    Collects screenshot, page source snippet, URL, and title.
    Each capture is independent: failure of one does not prevent
    others from being collected.
    """
    page_state: Optional[PageStateCapture] = None

    async def collect(self, session_id: str) -> Dict[str, Any]:
        """Collect all available evidence.

        Args:
            session_id: Session to capture evidence from.

        Returns:
            Dict with available evidence (screenshot_base64,
            page_source_snippet, current_url, page_title).
            Missing evidence keys are omitted, not None.
        """
        evidence: Dict[str, Any] = {}
        if self.page_state is None:
            return evidence
        try:
            screenshot = await self.page_state.capture_screenshot(session_id)
            if screenshot:
                evidence["screenshot_base64"] = screenshot
        except Exception:
            pass
        try:
            page_source = await self.page_state.capture_page_source(
                session_id, max_chars=2000
            )
            if page_source:
                evidence["page_source_snippet"] = page_source
        except Exception:
            pass
        try:
            url = await self.page_state.get_current_url(session_id)
            if url:
                evidence["current_url"] = url
        except Exception:
            pass
        try:
            title = await self.page_state.get_page_title(session_id)
            if title:
                evidence["page_title"] = title
        except Exception:
            pass
        return evidence
