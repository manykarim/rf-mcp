"""Recovery Domain Entities.

Entities have identity and mutable state. A RecoveryPlan is
identified by its plan_id and tracks the full lifecycle of a
single recovery attempt.
"""
from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .value_objects import ErrorClassification, RecoveryAction, RecoveryStrategy


class RecoveryPlanPhase(str, enum.Enum):
    """Lifecycle phases of a recovery plan.

    The plan moves strictly forward through these phases:
    CLASSIFY -> STRATEGIZE -> EXECUTE -> EVALUATE -> COMPLETED
    """
    CLASSIFY = "CLASSIFY"
    STRATEGIZE = "STRATEGIZE"
    EXECUTE = "EXECUTE"
    EVALUATE = "EVALUATE"
    COMPLETED = "COMPLETED"

    @property
    def next_phase(self) -> Optional[RecoveryPlanPhase]:
        """Return the next phase in the lifecycle, or None if completed."""
        order = list(RecoveryPlanPhase)
        idx = order.index(self)
        return order[idx + 1] if idx < len(order) - 1 else None


@dataclass
class RecoveryPlan:
    """Lifecycle-tracked recovery attempt entity.

    A RecoveryPlan is created when a keyword execution fails and
    tracks the full recovery lifecycle from error classification
    through strategy execution to final evaluation.

    Invariants:
        - Phase transitions are strictly forward (no going back).
        - Classification can only be set in CLASSIFY phase.
        - Strategy can only be set in STRATEGIZE phase.
        - Actions can only be recorded in EXECUTE phase.
        - Result can only be set in EVALUATE phase.
    """
    plan_id: str
    session_id: str
    keyword: str
    args: List[str]
    error_message: str
    classification: Optional[ErrorClassification] = None
    selected_strategy: Optional[RecoveryStrategy] = None
    actions_executed: List[RecoveryAction] = field(default_factory=list)
    retry_succeeded: Optional[bool] = None
    phase: RecoveryPlanPhase = RecoveryPlanPhase.CLASSIFY
    evidence: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        session_id: str,
        keyword: str,
        args: List[str],
        error_message: str,
    ) -> RecoveryPlan:
        """Factory: create a new plan in CLASSIFY phase."""
        return cls(
            plan_id=f"recovery_{uuid.uuid4().hex[:12]}",
            session_id=session_id,
            keyword=keyword,
            args=list(args),
            error_message=error_message,
        )

    def advance_phase(self) -> None:
        """Advance to the next lifecycle phase.

        Raises:
            ValueError: If already in COMPLETED phase.
        """
        next_p = self.phase.next_phase
        if next_p is None:
            raise ValueError(f"Cannot advance from {self.phase.value}")
        self.phase = next_p

    def set_classification(self, classification: ErrorClassification) -> None:
        """Set error classification (CLASSIFY phase only).

        Raises:
            ValueError: If not in CLASSIFY phase.
        """
        if self.phase != RecoveryPlanPhase.CLASSIFY:
            raise ValueError(f"Cannot classify in phase {self.phase.value}")
        self.classification = classification
        self.advance_phase()

    def set_strategy(self, strategy: RecoveryStrategy) -> None:
        """Set selected recovery strategy (STRATEGIZE phase only).

        Raises:
            ValueError: If not in STRATEGIZE phase.
        """
        if self.phase != RecoveryPlanPhase.STRATEGIZE:
            raise ValueError(f"Cannot set strategy in phase {self.phase.value}")
        self.selected_strategy = strategy
        self.advance_phase()

    def record_action(self, action: RecoveryAction) -> None:
        """Record an executed recovery action (EXECUTE phase only).

        Raises:
            ValueError: If not in EXECUTE phase.
        """
        if self.phase != RecoveryPlanPhase.EXECUTE:
            raise ValueError(f"Cannot record action in phase {self.phase.value}")
        self.actions_executed.append(action)

    def finish_execution(self) -> None:
        """Transition from EXECUTE to EVALUATE phase.

        Raises:
            ValueError: If not in EXECUTE phase.
        """
        if self.phase != RecoveryPlanPhase.EXECUTE:
            raise ValueError(f"Cannot finish execution in phase {self.phase.value}")
        self.advance_phase()

    def set_result(self, succeeded: bool) -> None:
        """Set the final recovery result (EVALUATE phase only).

        Raises:
            ValueError: If not in EVALUATE phase.
        """
        if self.phase != RecoveryPlanPhase.EVALUATE:
            raise ValueError(f"Cannot set result in phase {self.phase.value}")
        self.retry_succeeded = succeeded
        self.advance_phase()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize plan state for MCP responses."""
        d: Dict[str, Any] = {
            "plan_id": self.plan_id,
            "keyword": self.keyword,
            "error_message": self.error_message,
            "phase": self.phase.value,
        }
        if self.classification:
            d["classification"] = self.classification.value
        if self.selected_strategy:
            d["strategy"] = self.selected_strategy.name
        if self.actions_executed:
            d["actions"] = [a.to_dict() for a in self.actions_executed]
        if self.retry_succeeded is not None:
            d["retry_succeeded"] = self.retry_succeeded
        return d
