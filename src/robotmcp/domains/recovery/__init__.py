"""Recovery Bounded Context (ADR-011).

Protocol-based error classification and automatic recovery strategies
for Robot Framework keyword execution failures. Designed to be
composable with batch execution and potentially single-step execution.
"""
from .value_objects import (
    RecoveryTier, ErrorClassification, ErrorPattern,
    RecoveryAction, RecoveryStrategy,
)
from .entities import RecoveryPlan, RecoveryPlanPhase
from .aggregates import RecoveryEngine
from .services import (
    ErrorClassifier, Tier1RecoveryService, Tier2RecoveryService,
    EvidenceCollector, KeywordRunner, PageStateCapture,
)
from .events import (
    ErrorClassified, RecoveryStrategySelected,
    RecoveryAttempted, RecoverySucceeded,
    RecoveryFailed, EvidenceCollected,
)

__all__ = [
    "RecoveryTier", "ErrorClassification", "ErrorPattern",
    "RecoveryAction", "RecoveryStrategy",
    "RecoveryPlan", "RecoveryPlanPhase",
    "RecoveryEngine",
    "ErrorClassifier", "Tier1RecoveryService", "Tier2RecoveryService",
    "EvidenceCollector", "KeywordRunner", "PageStateCapture",
    "ErrorClassified", "RecoveryStrategySelected",
    "RecoveryAttempted", "RecoverySucceeded",
    "RecoveryFailed", "EvidenceCollected",
]
