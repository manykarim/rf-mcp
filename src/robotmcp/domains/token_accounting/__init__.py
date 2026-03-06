"""Token Accounting Bounded Context (ADR-017).

Provides pluggable token estimation, per-profile reporting, and CI regression detection.
"""

from .value_objects import (
    TokenizerBackend,
    TokenCount,
    TokenBudget,
    BudgetCheckResult,
    TokenReport,
    ProfileTokenSummary,
    TokenRegressionResult,
)
from .entities import TokenMeasurement
from .aggregates import TokenAccountant
from .events import (
    TokenBudgetExceeded,
    TokenRegressionDetected,
    TokenMeasurementRecorded,
)
from .services import (
    TokenEstimator,
    TokenEstimationService,
    TokenReportingService,
    TokenRegressionService,
    get_estimation_service,
)
from .repository import TokenBaselineRepository, JsonFileBaselineRepository

__all__ = [
    "TokenizerBackend",
    "TokenCount",
    "TokenBudget",
    "BudgetCheckResult",
    "TokenReport",
    "ProfileTokenSummary",
    "TokenRegressionResult",
    "TokenMeasurement",
    "TokenAccountant",
    "TokenBudgetExceeded",
    "TokenRegressionDetected",
    "TokenMeasurementRecorded",
    "TokenEstimator",
    "TokenEstimationService",
    "TokenReportingService",
    "TokenRegressionService",
    "get_estimation_service",
    "TokenBaselineRepository",
    "JsonFileBaselineRepository",
]
