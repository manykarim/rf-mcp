"""Memory Domain — Value Objects.

Immutable types identified by their attributes, following ADR-001 conventions.
All use @dataclass(frozen=True) with __post_init__ validation.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    ClassVar,
    Dict,
    FrozenSet,
    List,
    Optional,
    Tuple,
)


# ---------------------------------------------------------------------------
# MemoryType
# ---------------------------------------------------------------------------

class MemoryTypeEnum(str, Enum):
    """Enum for memory classification."""

    WORKING_STEPS = "working_steps"
    KEYWORDS = "keywords"
    DOCUMENTATION = "documentation"
    COMMON_ERRORS = "common_errors"
    DOMAIN_KNOWLEDGE = "domain_knowledge"
    LOCATORS = "locators"


@dataclass(frozen=True)
class MemoryType:
    """Classifies the type of knowledge stored in memory."""

    value: str

    WORKING_STEPS: ClassVar[str] = "working_steps"
    KEYWORDS: ClassVar[str] = "keywords"
    DOCUMENTATION: ClassVar[str] = "documentation"
    COMMON_ERRORS: ClassVar[str] = "common_errors"
    DOMAIN_KNOWLEDGE: ClassVar[str] = "domain_knowledge"
    LOCATORS: ClassVar[str] = "locators"

    VALID_TYPES: ClassVar[FrozenSet[str]] = frozenset(
        {
            "working_steps",
            "keywords",
            "documentation",
            "common_errors",
            "domain_knowledge",
            "locators",
        }
    )

    _COLLECTION_PREFIX: ClassVar[str] = "rfmcp_"

    def __post_init__(self) -> None:
        if self.value not in self.VALID_TYPES:
            raise ValueError(
                f"Invalid memory type '{self.value}'. "
                f"Valid: {sorted(self.VALID_TYPES)}"
            )

    # -- Properties ----------------------------------------------------------

    @property
    def collection_name(self) -> str:
        return f"{self._COLLECTION_PREFIX}{self.value}"

    @property
    def is_executable(self) -> bool:
        return self.value in {self.WORKING_STEPS, self.KEYWORDS}

    @property
    def is_reference(self) -> bool:
        return self.value in {self.DOCUMENTATION, self.DOMAIN_KNOWLEDGE}

    # -- Factory methods -----------------------------------------------------

    @classmethod
    def working_steps(cls) -> MemoryType:
        return cls(cls.WORKING_STEPS)

    @classmethod
    def keywords(cls) -> MemoryType:
        return cls(cls.KEYWORDS)

    @classmethod
    def documentation(cls) -> MemoryType:
        return cls(cls.DOCUMENTATION)

    @classmethod
    def common_errors(cls) -> MemoryType:
        return cls(cls.COMMON_ERRORS)

    @classmethod
    def domain_knowledge(cls) -> MemoryType:
        return cls(cls.DOMAIN_KNOWLEDGE)

    @classmethod
    def locators(cls) -> MemoryType:
        return cls(cls.LOCATORS)

    @classmethod
    def from_string(cls, value: str) -> MemoryType:
        return cls(value.lower().strip())

    @classmethod
    def all_types(cls) -> List[MemoryType]:
        return [cls(v) for v in sorted(cls.VALID_TYPES)]


# ---------------------------------------------------------------------------
# EmbeddingVector
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EmbeddingVector:
    """Immutable vector embedding with model provenance."""

    values: Tuple[float, ...]
    model_name: str
    dimensions: int

    SUPPORTED_MODELS: ClassVar[Dict[str, int]] = {
        "potion-base-8M": 256,
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
    }

    def __post_init__(self) -> None:
        if self.dimensions != len(self.values):
            raise ValueError(
                f"dimensions={self.dimensions} != len(values)={len(self.values)}"
            )
        if self.dimensions <= 0:
            raise ValueError("dimensions must be positive")
        for i, v in enumerate(self.values):
            if not math.isfinite(v):
                raise ValueError(f"values[{i}] is not finite: {v}")

    def cosine_similarity(self, other: EmbeddingVector) -> float:
        if self.dimensions != other.dimensions:
            raise ValueError(
                f"Dimension mismatch: {self.dimensions} vs {other.dimensions}"
            )
        dot = sum(a * b for a, b in zip(self.values, other.values))
        norm_a = math.sqrt(sum(a * a for a in self.values))
        norm_b = math.sqrt(sum(b * b for b in other.values))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def to_list(self) -> List[float]:
        return list(self.values)

    @classmethod
    def from_list(cls, values: List[float], model_name: str) -> EmbeddingVector:
        return cls(
            values=tuple(values),
            model_name=model_name,
            dimensions=len(values),
        )


# ---------------------------------------------------------------------------
# SimilarityScore
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SimilarityScore:
    """Bounded similarity score (0.0 to 1.0) with distance metric."""

    value: float
    distance_metric: str = "cosine"

    VALID_METRICS: ClassVar[FrozenSet[str]] = frozenset(
        {"cosine", "euclidean", "dot_product"}
    )

    def __post_init__(self) -> None:
        if not (0.0 <= self.value <= 1.0):
            raise ValueError(f"Score must be 0.0–1.0, got {self.value}")
        if self.distance_metric not in self.VALID_METRICS:
            raise ValueError(
                f"Invalid metric '{self.distance_metric}'. "
                f"Valid: {sorted(self.VALID_METRICS)}"
            )

    @property
    def is_high(self) -> bool:
        return self.value >= 0.8

    @property
    def is_moderate(self) -> bool:
        return 0.5 <= self.value < 0.8

    @property
    def is_low(self) -> bool:
        return self.value < 0.5

    def exceeds(self, threshold: float) -> bool:
        return self.value >= threshold

    @classmethod
    def cosine(cls, value: float) -> SimilarityScore:
        return cls(value=value, distance_metric="cosine")

    @classmethod
    def zero(cls) -> SimilarityScore:
        return cls(value=0.0, distance_metric="cosine")


# ---------------------------------------------------------------------------
# ConfidenceScore
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConfidenceScore:
    """Confidence in an error fix, with action thresholds."""

    value: float

    AUTO_APPLY_THRESHOLD: ClassVar[float] = 0.9
    SUGGEST_THRESHOLD: ClassVar[float] = 0.5

    def __post_init__(self) -> None:
        if not (0.0 <= self.value <= 1.0):
            raise ValueError(f"Confidence must be 0.0–1.0, got {self.value}")

    @property
    def action(self) -> str:
        if self.value >= self.AUTO_APPLY_THRESHOLD:
            return "auto_apply"
        if self.value >= self.SUGGEST_THRESHOLD:
            return "suggest"
        return "deprioritize"

    @property
    def should_auto_apply(self) -> bool:
        return self.value >= self.AUTO_APPLY_THRESHOLD

    @property
    def should_suggest(self) -> bool:
        return self.SUGGEST_THRESHOLD <= self.value < self.AUTO_APPLY_THRESHOLD

    @property
    def is_low(self) -> bool:
        return self.value < self.SUGGEST_THRESHOLD


# ---------------------------------------------------------------------------
# TimeDecayFactor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TimeDecayFactor:
    """Time-based relevance decay with a 0.5 floor.

    Formula: adjusted = similarity * (0.5 + 0.5 * e^(-days_old / half_life))
    At days_old=0: factor=1.0 (full similarity)
    At infinity:   factor=0.5 (old knowledge retains half weight)
    """

    half_life_days: float = 30.0

    def __post_init__(self) -> None:
        if self.half_life_days <= 0:
            raise ValueError("half_life_days must be positive")

    def decay_factor(self, days_old: float) -> float:
        return 0.5 + 0.5 * math.exp(-days_old / self.half_life_days)

    def compute(self, similarity: SimilarityScore, days_old: float) -> SimilarityScore:
        factor = self.decay_factor(max(0.0, days_old))
        adjusted = similarity.value * factor
        return SimilarityScore(
            value=min(1.0, max(0.0, adjusted)),
            distance_metric=similarity.distance_metric,
        )


# ---------------------------------------------------------------------------
# LocatorStrategy
# ---------------------------------------------------------------------------


class LocatorStrategy(str, Enum):
    """Classifies locator addressing strategy."""

    CSS = "css"
    XPATH = "xpath"
    ID = "id"
    TEXT = "text"
    NAME = "name"
    LINK = "link"
    AUTO = "auto"

    @classmethod
    def detect(cls, locator: str) -> LocatorStrategy:
        """Detect strategy from locator prefix."""
        loc = locator.strip()
        if loc.startswith(("css=", "css:")):
            return cls.CSS
        if loc.startswith(("xpath=", "xpath:")) or loc.startswith(("//", "(//")):
            return cls.XPATH
        if loc.startswith(("id=", "id:")):
            return cls.ID
        if loc.startswith(("text=", "text:")):
            return cls.TEXT
        if loc.startswith(("name=", "name:")):
            return cls.NAME
        if loc.startswith(("link=", "link:")):
            return cls.LINK
        return cls.AUTO


# ---------------------------------------------------------------------------
# LocatorOutcome
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LocatorOutcome:
    """Success/failure outcome of a locator interaction."""

    success: bool
    keyword: str
    library: str
    locator: str
    page_url: str = ""
    error_text: str = ""


# ---------------------------------------------------------------------------
# LocatorDescription
# ---------------------------------------------------------------------------


def _extract_description(locator: str, strategy: LocatorStrategy) -> str:
    """Extract human-readable description from a raw locator for embedding."""
    loc = locator.strip()

    # has-text('...') pattern (Browser Library)
    m = re.search(r"has-text\(['\"](.+?)['\"]\)", loc)
    if m:
        return m.group(1)

    # text= prefix
    if strategy == LocatorStrategy.TEXT:
        return re.sub(r"^text[=:]", "", loc).strip().strip("'\"")

    # id= prefix: "id=foo-bar" → "foo bar (id)"
    if strategy == LocatorStrategy.ID:
        val = re.sub(r"^id[=:]", "", loc).strip()
        return val.replace("-", " ").replace("_", " ") + " (id)"

    # name= prefix
    if strategy == LocatorStrategy.NAME:
        val = re.sub(r"^name[=:]", "", loc).strip()
        return val.replace("-", " ").replace("_", " ") + " (name)"

    # link= prefix
    if strategy == LocatorStrategy.LINK:
        return re.sub(r"^link[=:]", "", loc).strip()

    # CSS selectors
    if strategy == LocatorStrategy.CSS:
        val = re.sub(r"^css[=:]", "", loc).strip()
        if val.startswith("#"):
            return val[1:].replace("-", " ").replace("_", " ") + " element"
        if val.startswith("."):
            return val[1:].replace("-", " ").replace("_", " ") + " element"
        cleaned = re.sub(r"[.#\[\]=:>+~]", " ", val)
        return " ".join(cleaned.split())

    # XPath: //tag[@attr='val'] → "val tag (attr)"
    if strategy == LocatorStrategy.XPATH:
        val = re.sub(r"^xpath[=:]", "", loc).strip()
        m = re.search(r"//(\w+)\[@(\w+)=['\"](.+?)['\"]\]", val)
        if m:
            tag, attr, attrval = m.group(1), m.group(2), m.group(3)
            return f"{attrval} {tag} ({attr})"
        m = re.search(r"//(\w+)", val)
        if m:
            return m.group(1) + " element"
        return val.lstrip("/")

    # AUTO / fallback
    cleaned = re.sub(r"^[a-z]+=", "", loc)
    cleaned = cleaned.replace("-", " ").replace("_", " ")
    return cleaned.strip() or loc


@dataclass(frozen=True)
class LocatorDescription:
    """Human-readable description extracted from a raw locator."""

    value: str
    raw_locator: str
    strategy: LocatorStrategy

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("LocatorDescription value must be non-empty")

    @classmethod
    def from_locator(cls, locator: str) -> LocatorDescription:
        """Create from a raw locator string."""
        strategy = LocatorStrategy.detect(locator)
        desc = _extract_description(locator, strategy)
        return cls(value=desc, raw_locator=locator, strategy=strategy)


# ---------------------------------------------------------------------------
# LocatorRecallResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LocatorRecallResult:
    """Structured recall result for locator memory queries."""

    locator: str
    strategy: LocatorStrategy
    keyword: str
    library: str
    outcome: str  # "success" or "failure"
    page_url: str = ""
    description: str = ""
    similarity: float = 0.0
    error_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "locator": self.locator,
            "strategy": self.strategy.value,
            "keyword": self.keyword,
            "library": self.library,
            "outcome": self.outcome,
            "page_url": self.page_url,
            "description": self.description,
            "similarity": round(self.similarity, 4),
        }
        if self.error_text:
            result["error_text"] = self.error_text
        return result


# ---------------------------------------------------------------------------
# MemoryQuery
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MemoryQuery:
    """Search parameters for memory recall."""

    query_text: str
    memory_type: Optional[MemoryType] = None
    top_k: int = 5
    min_similarity: float = 0.3
    apply_time_decay: bool = True
    session_id: Optional[str] = None

    MAX_QUERY_LENGTH: ClassVar[int] = 2000
    MAX_TOP_K: ClassVar[int] = 50

    def __post_init__(self) -> None:
        if not self.query_text or not self.query_text.strip():
            raise ValueError("query_text must be non-empty")
        if len(self.query_text) > self.MAX_QUERY_LENGTH:
            raise ValueError(
                f"query_text exceeds {self.MAX_QUERY_LENGTH} chars"
            )
        if not (1 <= self.top_k <= self.MAX_TOP_K):
            raise ValueError(f"top_k must be 1–{self.MAX_TOP_K}")
        if not (0.0 <= self.min_similarity <= 1.0):
            raise ValueError("min_similarity must be 0.0–1.0")

    @property
    def is_scoped(self) -> bool:
        return self.memory_type is not None

    @property
    def is_session_scoped(self) -> bool:
        return self.session_id is not None

    @property
    def collection_names(self) -> List[str]:
        if self.memory_type is not None:
            return [self.memory_type.collection_name]
        return [mt.collection_name for mt in MemoryType.all_types()]

    @classmethod
    def for_error_fix(
        cls, error_text: str, session_id: Optional[str] = None
    ) -> MemoryQuery:
        return cls(
            query_text=error_text[:cls.MAX_QUERY_LENGTH],
            memory_type=MemoryType.common_errors(),
            top_k=3,
            min_similarity=0.3,
            session_id=session_id,
        )

    @classmethod
    def for_keyword_recall(cls, keyword_hint: str) -> MemoryQuery:
        return cls(
            query_text=keyword_hint[:cls.MAX_QUERY_LENGTH],
            memory_type=MemoryType.keywords(),
            top_k=5,
            min_similarity=0.2,
        )

    @classmethod
    def for_step_recall(cls, scenario_description: str) -> MemoryQuery:
        return cls(
            query_text=scenario_description[:cls.MAX_QUERY_LENGTH],
            memory_type=MemoryType.working_steps(),
            top_k=10,
            min_similarity=0.05,
        )

    @classmethod
    def for_locator_recall(cls, element_description: str) -> MemoryQuery:
        return cls(
            query_text=element_description[:cls.MAX_QUERY_LENGTH],
            memory_type=MemoryType.locators(),
            top_k=5,
            min_similarity=0.2,
        )


# ---------------------------------------------------------------------------
# MemoryEntry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MemoryEntry:
    """Immutable content unit before persistence."""

    content: str
    memory_type: MemoryType
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[EmbeddingVector] = None
    tags: Tuple[str, ...] = ()

    MAX_CONTENT_LENGTH: ClassVar[int] = 50_000

    def __post_init__(self) -> None:
        if not self.content or not self.content.strip():
            raise ValueError("content must be non-empty")
        if len(self.content) > self.MAX_CONTENT_LENGTH:
            raise ValueError(
                f"content exceeds {self.MAX_CONTENT_LENGTH} chars"
            )

    @property
    def has_embedding(self) -> bool:
        return self.embedding is not None

    @property
    def content_preview(self) -> str:
        return self.content[:100]

    @property
    def word_count(self) -> int:
        return len(self.content.split())

    def with_embedding(self, embedding: EmbeddingVector) -> MemoryEntry:
        return MemoryEntry(
            content=self.content,
            memory_type=self.memory_type,
            metadata=self.metadata,
            embedding=embedding,
            tags=self.tags,
        )

    def with_tags(self, *tags: str) -> MemoryEntry:
        return MemoryEntry(
            content=self.content,
            memory_type=self.memory_type,
            metadata=self.metadata,
            embedding=self.embedding,
            tags=tags,
        )


# ---------------------------------------------------------------------------
# RecallResult
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RecallResult:
    """A single result from memory recall with scoring."""

    record_id: str
    content: str
    memory_type: MemoryType
    similarity: SimilarityScore
    adjusted_similarity: SimilarityScore
    age_days: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: Optional[ConfidenceScore] = None
    rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "record_id": self.record_id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "similarity": self.similarity.value,
            "adjusted_similarity": self.adjusted_similarity.value,
            "age_days": round(self.age_days, 1),
            "rank": self.rank,
        }
        if self.metadata:
            result["metadata"] = self.metadata
        if self.confidence is not None:
            result["confidence"] = self.confidence.value
            result["action"] = self.confidence.action
        return result


# ---------------------------------------------------------------------------
# StorageConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StorageConfig:
    """Configuration for the memory storage backend."""

    db_path: str = ""
    embedding_model: str = "potion-base-8M"
    dimension: int = 256
    max_records_per_collection: int = 10_000
    prune_age_days: float = 90.0
    time_decay_half_life: float = 30.0
    enabled: bool = False
    project_id: str = "default"

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")
        if self.max_records_per_collection <= 0:
            raise ValueError("max_records_per_collection must be positive")
        if self.prune_age_days <= 0:
            raise ValueError("prune_age_days must be positive")
        if self.time_decay_half_life <= 0:
            raise ValueError("time_decay_half_life must be positive")

    @classmethod
    def from_env(cls) -> StorageConfig:
        import os

        enabled = os.environ.get("ROBOTMCP_MEMORY_ENABLED", "false").lower() in (
            "true",
            "1",
            "yes",
        )
        db_path = os.environ.get(
            "ROBOTMCP_MEMORY_DB_PATH",
            os.path.expanduser("~/.rf-mcp/memory.db"),
        )
        model = os.environ.get("ROBOTMCP_MEMORY_MODEL", "potion-base-8M")
        dim_map = {"potion-base-8M": 256, "all-MiniLM-L6-v2": 384}
        dimension = dim_map.get(model, 256)
        max_records = int(
            os.environ.get("ROBOTMCP_MEMORY_MAX_RECORDS", "10000")
        )
        prune_days = float(
            os.environ.get("ROBOTMCP_MEMORY_PRUNE_DAYS", "90")
        )
        decay_hl = float(
            os.environ.get("ROBOTMCP_MEMORY_DECAY_HALF_LIFE", "30")
        )
        project_id = os.environ.get("ROBOTMCP_PROJECT_ID", "default")

        return cls(
            db_path=db_path,
            embedding_model=model,
            dimension=dimension,
            max_records_per_collection=max_records,
            prune_age_days=prune_days,
            time_decay_half_life=decay_hl,
            enabled=enabled,
            project_id=project_id,
        )

    @classmethod
    def default(cls) -> StorageConfig:
        import os

        return cls(
            db_path=os.path.expanduser("~/.rf-mcp/memory.db"),
        )
