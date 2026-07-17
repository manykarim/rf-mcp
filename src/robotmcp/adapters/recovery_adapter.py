"""Recovery Service Adapter.

Anti-corruption layer bridging batch_execution <-> recovery domain.
Implements the batch_execution.services.RecoveryServiceProtocol by
coordinating ErrorClassifier + RecoveryEngine + Tier1/Tier2 services.
"""
from __future__ import annotations

import contextlib
import logging
import os
import time
from typing import Any, Dict, List, Optional

from robotmcp.domains.recovery import (
    RecoveryEngine, ErrorClassifier,
    Tier1RecoveryService, Tier2RecoveryService,
    RecoveryPlan, RecoveryTier,
    KeywordRunner, PageStateCapture,
)
from robotmcp.domains.batch_execution.entities import RecoveryAttempt

logger = logging.getLogger(__name__)

_BATCH_RETRY_TIMEOUT_ENV = "ROBOTMCP_PLATYNUI_BATCH_RETRY_TIMEOUT"
_BATCH_RETRY_TIMEOUT_DEFAULT = 5.0


def _batch_retry_timeout_cap_seconds() -> float:
    """Capped PlatynUI descriptor-resolution timeout (seconds) for batch
    retries; env-overridable, default 5s. Falls back to the default on a bad
    value."""
    raw = os.environ.get(_BATCH_RETRY_TIMEOUT_ENV, "").strip()
    if not raw:
        return _BATCH_RETRY_TIMEOUT_DEFAULT
    try:
        val = float(raw)
        return val if val > 0 else _BATCH_RETRY_TIMEOUT_DEFAULT
    except ValueError:
        return _BATCH_RETRY_TIMEOUT_DEFAULT


class RecoveryServiceAdapter:
    """Bridges batch_execution <-> recovery domain.

    Implements RecoveryServiceProtocol from batch_execution.services.
    """

    # Opt-in marker: BatchRunner only invokes the desktop retry-safety gate and
    # descriptor-timeout cap when this is truthy on the recovery service, so a
    # Mock/AsyncMock recovery service in tests never triggers them (change:
    # desktop-aware-batch-execution §4/§5).
    supports_desktop_batch_hooks = True

    def __init__(
        self,
        engine: RecoveryEngine,
        keyword_runner: KeywordRunner,
        page_state: Optional[PageStateCapture] = None,
        session_manager: Optional[Any] = None,
    ):
        self._engine = engine
        # Used to resolve the recovery platform (web vs desktop) so a desktop
        # batch gets desktop strategies, never browser ones (change:
        # desktop-aware-batch-execution).
        self._session_manager = session_manager
        self._classifier = ErrorClassifier(engine=engine)
        self._tier1 = Tier1RecoveryService(keyword_runner=keyword_runner)
        self._tier2 = Tier2RecoveryService(
            keyword_runner=keyword_runner, page_state=page_state
        )

    def _resolve_platform(self, session_id: str) -> str:
        """Return "desktop" for a PlatynUI session, else "web" (default)."""
        try:
            if self._session_manager is not None:
                sess = self._session_manager.get_session(session_id)
                if sess is not None and sess.is_desktop_session():
                    return "desktop"
        except Exception:
            pass
        return "web"

    def desktop_retry_blocked(self, session_id: str, error_message: str) -> bool:
        """Desktop retry-safety gate (change: desktop-aware-batch-execution §5).

        On a desktop session a batch step may be retried ONLY when the input
        provably never fired — i.e. the failure classifies as ELEMENT_NOT_FOUND
        (descriptor resolution precedes the pointer/keyboard action). Every other
        desktop failure may have partially acted, so a blind retry is a
        spray-input hazard: return True to block it (record failure immediately).
        Web/api sessions are never blocked. Never raises.
        """
        try:
            if self._resolve_platform(session_id) != "desktop":
                return False
            from robotmcp.domains.recovery.value_objects import ErrorClassification

            classification = self._classifier.classify(error_message)
            return classification != ErrorClassification.ELEMENT_NOT_FOUND
        except Exception:  # pragma: no cover - defensive
            return False

    def _resolve_baremetal(self, session_id: str) -> Optional[Any]:
        """Best-effort resolve the live PlatynUI.BareMetal RF library instance
        (RF execution context is process-global). Returns None on any failure."""
        try:
            from robot.running.context import EXECUTION_CONTEXTS

            ctx = EXECUTION_CONTEXTS.current
            if ctx is None:
                return None
            return ctx.namespace.get_library_instance("PlatynUI.BareMetal")
        except Exception:
            return None

    @contextlib.contextmanager
    def retry_timeout_cap(self, session_id: str):
        """Cap PlatynUI descriptor-resolution timeout during desktop RETRIES
        (change: desktop-aware-batch-execution §4).

        The initial attempt keeps the native ~30s budget; recovery re-attempts
        run with ``BareMetal.query_settings.timeout`` temporarily capped (default
        5s, ``ROBOTMCP_PLATYNUI_BATCH_RETRY_TIMEOUT``) so a bad-descriptor batch
        cannot burn ~30s × retries. No-op — yields unchanged — for non-desktop
        sessions or when the library / ``query_settings`` cannot be resolved, so
        it never crashes a batch. The original timeout is always restored.
        """
        qs = None
        original = None
        try:
            if self._resolve_platform(session_id) == "desktop":
                lib = self._resolve_baremetal(session_id)
                qs = getattr(lib, "query_settings", None) if lib is not None else None
        except Exception:  # pragma: no cover - defensive
            qs = None
        try:
            if qs is not None and hasattr(qs, "timeout"):
                try:
                    original = qs.timeout
                    qs.timeout = _batch_retry_timeout_cap_seconds()
                except Exception:  # pragma: no cover - defensive
                    original = None
            yield
        finally:
            if qs is not None and original is not None:
                try:
                    qs.timeout = original
                except Exception:  # pragma: no cover - defensive
                    pass

    async def attempt_recovery(
        self, session_id: str, keyword: str, args: List[str],
        error_message: str, attempt_number: int,
    ) -> Optional[RecoveryAttempt]:
        """Classify error, select strategy, execute recovery actions."""
        start = time.monotonic()

        # 1. Classify
        classification = self._classifier.classify(error_message)
        logger.debug("Error classified as %s", classification.value)

        # 2. Select strategy (platform-aware: desktop sessions must never get
        #    browser recovery actions — change: desktop-aware-batch-execution).
        platform = self._resolve_platform(session_id)
        strategy = self._engine.select_strategy(
            classification, attempt_number, platform=platform
        )
        if strategy is None:
            logger.debug("No recovery strategy for %s", classification.value)
            return None

        # 3. Execute recovery actions
        plan = RecoveryPlan.create(session_id, keyword, args, error_message)
        plan.set_classification(classification)
        plan.set_strategy(strategy)

        if strategy.tier == RecoveryTier.TIER_1:
            actions = await self._tier1.execute(session_id, strategy, plan)
        else:
            actions = await self._tier2.execute(session_id, strategy, plan)

        plan.finish_execution()
        time_ms = int((time.monotonic() - start) * 1000)

        return RecoveryAttempt(
            attempt_number=attempt_number,
            strategy=strategy.name,
            tier=strategy.tier.value,
            action_description=strategy.description,
            result="ATTEMPTED",  # Actual result determined by retry
            time_ms=time_ms,
        )


class KeywordExecutorAdapter:
    """Adapts the existing execution coordinator to KeywordRunner protocol."""

    def __init__(self, execution_engine: Any):
        self._engine = execution_engine

    async def run_keyword(
        self, session_id: str, keyword: str,
        args: List[str], timeout: Optional[str] = None,
    ) -> Any:
        """Execute a keyword via the existing execution coordinator."""
        result = await self._engine.execute_step(
            keyword, args, session_id,
            detail_level="minimal",
            use_context=True,
        )
        if not result.get("success", False):
            raise RuntimeError(result.get("error", "Keyword execution failed"))
        return result.get("return_value")

    async def execute_keyword(
        self, session_id: str, keyword: str,
        args: List[str], timeout: Optional[str] = None,
        assign_to: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a keyword and return the full result dict."""
        return await self._engine.execute_step(
            keyword, args, session_id,
            detail_level="minimal",
            assign_to=assign_to,
            use_context=True,
        )
