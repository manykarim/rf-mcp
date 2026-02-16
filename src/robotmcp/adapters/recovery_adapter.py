"""Recovery Service Adapter.

Anti-corruption layer bridging batch_execution <-> recovery domain.
Implements the batch_execution.services.RecoveryServiceProtocol by
coordinating ErrorClassifier + RecoveryEngine + Tier1/Tier2 services.
"""
from __future__ import annotations

import logging
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


class RecoveryServiceAdapter:
    """Bridges batch_execution <-> recovery domain.

    Implements RecoveryServiceProtocol from batch_execution.services.
    """

    def __init__(
        self,
        engine: RecoveryEngine,
        keyword_runner: KeywordRunner,
        page_state: Optional[PageStateCapture] = None,
    ):
        self._engine = engine
        self._classifier = ErrorClassifier(engine=engine)
        self._tier1 = Tier1RecoveryService(keyword_runner=keyword_runner)
        self._tier2 = Tier2RecoveryService(
            keyword_runner=keyword_runner, page_state=page_state
        )

    async def attempt_recovery(
        self, session_id: str, keyword: str, args: List[str],
        error_message: str, attempt_number: int,
    ) -> Optional[RecoveryAttempt]:
        """Classify error, select strategy, execute recovery actions."""
        start = time.monotonic()

        # 1. Classify
        classification = self._classifier.classify(error_message)
        logger.debug("Error classified as %s", classification.value)

        # 2. Select strategy
        strategy = self._engine.select_strategy(classification, attempt_number)
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
