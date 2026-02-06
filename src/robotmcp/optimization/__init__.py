"""Native self-learning optimization system for rf-mcp.

This package provides performance optimization and self-learning capabilities
for the rf-mcp Robot Framework MCP Server. It uses only Python standard library
features (json, pathlib, statistics) with no external dependencies.

Key Components:
    - PatternStore: Native JSON-based persistent storage for learned patterns
    - PageAnalyzer: Page complexity analysis for optimization decisions
    - CompressionPatternLearner: Learn optimal compression strategies
    - FoldingPatternLearner: Learn optimal list folding thresholds
    - TimeoutPatternLearner: Learn optimal timeout configurations
    - RefUsageLearner: Learn ref usage patterns for predictive preloading
    - PerformanceMetricsCollector: Centralized metrics collection

Example:
    from robotmcp.optimization import PerformanceMetricsCollector

    collector = PerformanceMetricsCollector()

    # Record snapshot generation
    collector.record_snapshot_generation(
        raw_chars=10000,
        aria_chars=1500,
        latency_ms=150,
        page_type="search_results",
        fold_threshold=0.85
    )

    # Get optimization recommendations
    recommendations = collector.get_optimization_recommendations("search_results")

    # Get summary of all metrics
    summary = collector.get_summary()

Storage:
    All learned patterns are stored in ~/.rf-mcp/patterns/ as JSON files,
    organized by namespace (compression, folding, timeouts, refs, metrics,
    instruction_patterns, instruction_records, instruction_sequences).

Instruction Learning Example:
    from robotmcp.optimization import InstructionEffectivenessLearner

    learner = InstructionEffectivenessLearner()

    # Start tracking a session
    tracker = learner.start_session("session-1", "default", "claude-sonnet")
    tracker.record_tool_call("find_keywords", {}, True)
    tracker.record_tool_call("execute_step", {"keyword": "Click"}, True)

    # End session and record effectiveness
    learner.end_session(tracker)

    # Get recommendation for future sessions
    rec = learner.get_recommendation("claude-sonnet", "web_automation")
"""

from .pattern_store import PatternStore
from .page_analyzer import (
    ComplexityLevel,
    PageComplexityProfile,
    PageTypeClassification,
    PageAnalyzer,
)
from .compression_learner import (
    CompressionPattern,
    CompressionResult,
    CompressionPatternLearner,
)
from .timeout_learner import (
    TimeoutPattern,
    TimeoutResult,
    TimeoutPatternLearner,
)
from .folding_learner import (
    FoldingPattern,
    FoldResult,
    FoldingPatternLearner,
)
from .ref_learner import (
    RefUsagePattern,
    SessionRefTracker,
    RefUsageLearner,
)
from .collector import (
    TOKEN_TARGETS,
    LATENCY_TARGETS,
    EFFECTIVENESS_TARGETS,
    MEMORY_TARGETS,
    TokenMetrics,
    LatencyMetrics,
    PerformanceMetricsCollector,
)
from .instruction_learner import (
    InstructionMode,
    ToolCallEvent,
    InstructionEffectivenessRecord,
    LLMBehaviorPattern,
    SuccessfulSequence,
    SessionTracker,
    InstructionEffectivenessLearner,
)
from .instruction_hooks import (
    SessionMetadata,
    InstructionLearningHooks,
    get_hooks,
    track_tool_call,
)

__all__ = [
    # Pattern storage
    "PatternStore",

    # Page analysis
    "ComplexityLevel",
    "PageComplexityProfile",
    "PageTypeClassification",
    "PageAnalyzer",

    # Compression learning
    "CompressionPattern",
    "CompressionResult",
    "CompressionPatternLearner",

    # Timeout learning
    "TimeoutPattern",
    "TimeoutResult",
    "TimeoutPatternLearner",

    # Folding learning
    "FoldingPattern",
    "FoldResult",
    "FoldingPatternLearner",

    # Ref learning
    "RefUsagePattern",
    "SessionRefTracker",
    "RefUsageLearner",

    # Metrics collection
    "TOKEN_TARGETS",
    "LATENCY_TARGETS",
    "EFFECTIVENESS_TARGETS",
    "MEMORY_TARGETS",
    "TokenMetrics",
    "LatencyMetrics",
    "PerformanceMetricsCollector",

    # Instruction learning
    "InstructionMode",
    "ToolCallEvent",
    "InstructionEffectivenessRecord",
    "LLMBehaviorPattern",
    "SuccessfulSequence",
    "SessionTracker",
    "InstructionEffectivenessLearner",

    # Instruction learning hooks
    "SessionMetadata",
    "InstructionLearningHooks",
    "get_hooks",
    "track_tool_call",
]
