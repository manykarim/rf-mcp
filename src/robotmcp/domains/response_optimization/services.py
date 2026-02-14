"""Response Optimization Domain Service."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

from .aggregates import ResponseOptimizationConfig
from .events import ResponseCompressed, SnapshotFolded
from .value_objects import (
    SnapshotCompressionMode,
    TokenEstimate,
    Verbosity,
)

logger = logging.getLogger(__name__)


# Fields that should be omitted when empty/null.
# Mirrors OMIT_WHEN_EMPTY from token_efficient_output.py.
_OMIT_WHEN_EMPTY = frozenset({
    "error", "message", "metadata", "description", "hint",
    "guidance", "warnings", "notes", "details", "extra",
    "context", "trace", "traceback", "stack_trace", "debug_info",
    "assigned_variables", "resolved_arguments", "state_updates",
})


class FieldOptimizer(Protocol):
    """Port wrapping token_efficient_output.TokenEfficientOutput.

    Any object that exposes an ``optimize(response, verbosity)`` method
    satisfies this protocol, including ``TokenEfficientOutput`` itself.
    """

    def optimize(self, response: Dict[str, Any], verbosity: str) -> Dict[str, Any]: ...


class ListFolder(Protocol):
    """Port wrapping domains.snapshot.list_folding.ListFoldingService.

    ``ListFoldingService.fold_list`` accepts ``List[Tuple[str, str]]``
    and returns ``List[FoldedListItem]``.  This protocol abstracts
    just the signature needed by the compressor.
    """

    def fold_list(self, items: List[Tuple[str, str]]) -> Tuple[List[Any], Dict[str, Any]]: ...


@dataclass
class CompressionMetrics:
    """Metrics for a single compression operation."""
    raw_chars: int = 0
    compressed_chars: int = 0
    raw_tokens: int = 0
    compressed_tokens: int = 0
    fields_abbreviated: int = 0
    fields_omitted: int = 0
    strings_truncated: int = 0
    lists_truncated: int = 0

    @property
    def compression_ratio(self) -> float:
        if self.raw_tokens == 0:
            return 0.0
        return 1.0 - (self.compressed_tokens / self.raw_tokens)

    @property
    def tokens_saved(self) -> int:
        return max(0, self.raw_tokens - self.compressed_tokens)


class ResponseCompressor:
    """Orchestrates all response compression for tool outputs.

    Pipeline:
    1. Estimate raw token count
    2. Remove empty/null fields
    3. Apply field abbreviation (if COMPACT)
    4. Apply truncation (if over budget)
    5. Estimate compressed token count
    6. Return compressed response + metrics
    """

    def __init__(
        self,
        event_publisher: Optional[Callable[[object], None]] = None,
    ) -> None:
        self._event_publisher = event_publisher

    def compress_response(
        self,
        raw: Dict[str, Any],
        config: ResponseOptimizationConfig,
    ) -> Tuple[Dict[str, Any], CompressionMetrics]:
        """Compress a single tool response dict."""
        if config.verbosity == Verbosity.VERBOSE:
            estimate = TokenEstimate.from_dict(raw, config.token_budget)
            metrics = CompressionMetrics(
                raw_chars=estimate.char_count,
                compressed_chars=estimate.char_count,
                raw_tokens=estimate.estimated_tokens,
                compressed_tokens=estimate.estimated_tokens,
            )
            return raw, metrics

        # Measure raw
        raw_estimate = TokenEstimate.from_dict(raw, config.token_budget)
        metrics = CompressionMetrics(
            raw_chars=raw_estimate.char_count,
            raw_tokens=raw_estimate.estimated_tokens,
        )

        result = dict(raw)

        # Step 1: Remove empty/null fields
        if config.verbosity in (Verbosity.STANDARD, Verbosity.COMPACT):
            result, omitted = self._remove_empty_fields(result)
            metrics.fields_omitted = omitted

        # Step 2: Apply field abbreviation (COMPACT only)
        if config.verbosity == Verbosity.COMPACT and config.field_abbreviation.enabled:
            result, abbreviated = self._abbreviate_fields(result, config)
            metrics.fields_abbreviated = abbreviated

        # Step 3: Apply truncation
        if config.verbosity == Verbosity.COMPACT:
            result, truncated_strings, truncated_lists = self._apply_truncation(
                result, config
            )
            metrics.strings_truncated = truncated_strings
            metrics.lists_truncated = truncated_lists

        # Measure compressed
        compressed_estimate = TokenEstimate.from_dict(result, config.token_budget)
        metrics.compressed_chars = compressed_estimate.char_count
        metrics.compressed_tokens = compressed_estimate.estimated_tokens

        # Emergency truncation if still over budget
        if not compressed_estimate.within_budget and config.verbosity == Verbosity.COMPACT:
            result = self._emergency_truncate(result, config.token_budget)
            final_estimate = TokenEstimate.from_dict(result, config.token_budget)
            metrics.compressed_chars = final_estimate.char_count
            metrics.compressed_tokens = final_estimate.estimated_tokens

        # Publish event
        self._publish(ResponseCompressed(
            session_id=config.session_id,
            tool_name="unknown",  # caller should set this
            raw_tokens=metrics.raw_tokens,
            compressed_tokens=metrics.compressed_tokens,
            compression_ratio=metrics.compression_ratio,
            verbosity=config.verbosity.value,
        ))

        return result, metrics

    def _remove_empty_fields(
        self, data: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], int]:
        """Remove fields that are empty/null when they are in the omit set."""
        result = {}
        omitted = 0
        for key, value in data.items():
            if key in _OMIT_WHEN_EMPTY and self._is_empty(value):
                omitted += 1
                continue
            if isinstance(value, dict):
                cleaned, sub_omitted = self._remove_empty_fields(value)
                omitted += sub_omitted
                if cleaned or key not in _OMIT_WHEN_EMPTY:
                    result[key] = cleaned
                else:
                    omitted += 1
            else:
                result[key] = value
        return result, omitted

    def _abbreviate_fields(
        self, data: Dict[str, Any], config: ResponseOptimizationConfig,
    ) -> Tuple[Dict[str, Any], int]:
        """Abbreviate field names using the abbreviation map."""
        result = {}
        abbreviated = 0
        for key, value in data.items():
            new_key = config.field_abbreviation.abbreviate(key)
            if new_key != key:
                abbreviated += 1
            if isinstance(value, dict):
                value, sub_abbreviated = self._abbreviate_fields(value, config)
                abbreviated += sub_abbreviated
            result[new_key] = value
        return result, abbreviated

    def _apply_truncation(
        self,
        data: Dict[str, Any],
        config: ResponseOptimizationConfig,
    ) -> Tuple[Dict[str, Any], int, int]:
        """Apply truncation based on policy."""
        result = {}
        truncated_strings = 0
        truncated_lists = 0
        policy = config.truncation

        for key, value in data.items():
            if isinstance(value, str) and len(value) > policy.max_string:
                result[key] = value[:policy.max_string] + "..."
                truncated_strings += 1
            elif isinstance(value, list) and len(value) > policy.max_list_items:
                result[key] = value[:policy.max_list_items]
                truncated_lists += 1
            elif isinstance(value, dict):
                if len(value) > policy.max_dict_items:
                    truncated_value = dict(list(value.items())[:policy.max_dict_items])
                    sub_result, sub_ts, sub_tl = self._apply_truncation(
                        truncated_value, config
                    )
                    result[key] = sub_result
                    truncated_strings += sub_ts
                    truncated_lists += sub_tl + 1
                else:
                    sub_result, sub_ts, sub_tl = self._apply_truncation(value, config)
                    result[key] = sub_result
                    truncated_strings += sub_ts
                    truncated_lists += sub_tl
            else:
                result[key] = value

        return result, truncated_strings, truncated_lists

    def _emergency_truncate(
        self, data: Dict[str, Any], budget: int,
    ) -> Dict[str, Any]:
        """Last-resort truncation to fit within token budget.

        Strategy: Keep essential fields (success/ok, error/err),
        progressively truncate largest remaining fields.
        """
        essential_keys = {"success", "ok", "error", "err", "sid", "session_id"}
        result: Dict[str, Any] = {}

        # Always keep essential fields
        for key in essential_keys:
            if key in data:
                result[key] = data[key]

        # Add remaining fields, checking budget after each
        remaining = {k: v for k, v in data.items() if k not in essential_keys}
        for key, value in remaining.items():
            result[key] = value
            estimate = TokenEstimate.from_dict(result, budget)
            if not estimate.within_budget:
                # Truncate this field's value
                if isinstance(value, str) and len(value) > 100:
                    result[key] = value[:100] + "...[truncated]"
                elif isinstance(value, list) and len(value) > 3:
                    result[key] = value[:3]
                elif isinstance(value, dict) and len(value) > 3:
                    result[key] = dict(list(value.items())[:3])
                else:
                    del result[key]  # Remove entirely if can't truncate

        return result

    @staticmethod
    def _is_empty(value: Any) -> bool:
        """Check if a value is considered empty."""
        if value is None:
            return True
        if isinstance(value, (str, list, dict, tuple, set)) and len(value) == 0:
            return True
        return False

    def _publish(self, event: object) -> None:
        if self._event_publisher:
            try:
                self._event_publisher(event)
            except Exception as e:
                logger.error(f"Failed to publish event: {e}")
