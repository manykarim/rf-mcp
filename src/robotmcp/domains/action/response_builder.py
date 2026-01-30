"""Response Builder Service.

Builds token-optimized responses from action results, applying filtering
and verbosity controls based on ResponseConfig.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, TYPE_CHECKING

from robotmcp.domains.action.value_objects import (
    FilteredResponse,
    ResponseConfig,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# Token estimation constants
# Based on average token-to-character ratios observed in practice
CHARS_PER_TOKEN = 4.0


class ResponseBuilder:
    """Build token-optimized responses.

    The ResponseBuilder applies filtering and verbosity controls to action
    results, producing compact responses that minimize token consumption
    while preserving essential information.

    Typical token reduction:
    - Compact verbosity: 60-70% reduction
    - Standard verbosity: 30-40% reduction
    - Verbose: No reduction (full data)
    """

    def __init__(self, config: ResponseConfig) -> None:
        """Initialize the ResponseBuilder.

        Args:
            config: Response configuration controlling what data to include
        """
        self.config = config

    def build(
        self,
        action: str,
        action_result: Any,
        snapshot: Optional[str] = None,
        error: Optional[str] = None,
        ref: Optional[str] = None,
    ) -> FilteredResponse:
        """Build a filtered response from action results.

        Args:
            action: The action type that was executed
            action_result: The raw result from the action
            snapshot: Optional page snapshot (YAML format)
            error: Optional error message if action failed
            ref: Optional element reference

        Returns:
            A FilteredResponse with token-optimized content
        """
        success = error is None

        # Apply verbosity filtering to result
        filtered_result = self._filter_result(action_result)

        # Handle snapshot based on config
        filtered_snapshot = self._filter_snapshot(snapshot)

        # Build the response data
        response_data: Dict[str, Any] = {
            "success": success,
            "action": action,
        }

        if ref is not None:
            response_data["ref"] = ref

        if filtered_result is not None:
            response_data["result"] = filtered_result

        if error is not None:
            response_data["error"] = self._filter_error(error)

        if filtered_snapshot is not None:
            response_data["snapshot"] = filtered_snapshot

        # Estimate tokens
        token_estimate = self._estimate_tokens(response_data)

        return FilteredResponse(
            success=success,
            action=action,
            ref=ref,
            result=filtered_result,
            error=error if error else None,
            snapshot=filtered_snapshot,
            token_estimate=token_estimate,
        )

    def _filter_result(self, result: Any) -> Any:
        """Apply verbosity filtering to the action result.

        Args:
            result: The raw action result

        Returns:
            Filtered result based on verbosity setting
        """
        if result is None:
            return None

        verbosity = self.config.verbosity

        if verbosity == "compact":
            return self._compact_result(result)
        elif verbosity == "standard":
            return self._standard_result(result)
        else:  # verbose
            return result

    def _compact_result(self, result: Any) -> Any:
        """Create a compact representation of the result.

        Args:
            result: The raw result

        Returns:
            Compact representation
        """
        if result is None:
            return None

        # For strings, truncate if too long
        if isinstance(result, str):
            max_len = 100
            if len(result) > max_len:
                return result[:max_len] + "..."
            return result

        # For dicts, keep only essential keys
        if isinstance(result, dict):
            essential_keys = {
                "success", "status", "message", "error", "value",
                "text", "url", "title", "id", "ref"
            }
            return {
                k: self._compact_result(v)
                for k, v in result.items()
                if k in essential_keys or not isinstance(v, (dict, list))
            }

        # For lists, limit length
        if isinstance(result, list):
            max_items = 3
            if len(result) > max_items:
                compacted = [self._compact_result(item) for item in result[:max_items]]
                compacted.append(f"... and {len(result) - max_items} more")
                return compacted
            return [self._compact_result(item) for item in result]

        # For other types, return as-is
        return result

    def _standard_result(self, result: Any) -> Any:
        """Create a standard representation of the result.

        Args:
            result: The raw result

        Returns:
            Standard representation
        """
        if result is None:
            return None

        # For strings, allow longer content
        if isinstance(result, str):
            max_len = 500
            if len(result) > max_len:
                return result[:max_len] + "..."
            return result

        # For dicts, keep most keys but filter nested structures
        if isinstance(result, dict):
            return {
                k: self._standard_result(v)
                for k, v in result.items()
            }

        # For lists, allow more items
        if isinstance(result, list):
            max_items = 10
            if len(result) > max_items:
                items = [self._standard_result(item) for item in result[:max_items]]
                items.append(f"... and {len(result) - max_items} more")
                return items
            return [self._standard_result(item) for item in result]

        return result

    def _filter_snapshot(self, snapshot: Optional[str]) -> Optional[str]:
        """Filter the snapshot based on config.

        Args:
            snapshot: The raw snapshot string

        Returns:
            Filtered snapshot or None
        """
        if not self.config.include_snapshot:
            return None

        if self.config.snapshot_mode == "none":
            return None

        if snapshot is None:
            return None

        # For incremental mode, the snapshot should already be a diff
        # For full mode, include the full snapshot
        # Apply length limits based on verbosity
        if self.config.verbosity == "compact":
            max_len = 1000
            if len(snapshot) > max_len:
                return snapshot[:max_len] + "\n... (truncated)"
        elif self.config.verbosity == "standard":
            max_len = 5000
            if len(snapshot) > max_len:
                return snapshot[:max_len] + "\n... (truncated)"

        return snapshot

    def _filter_error(self, error: str) -> str:
        """Filter error message based on verbosity.

        Args:
            error: The raw error message

        Returns:
            Filtered error message
        """
        if self.config.verbosity == "compact":
            # Extract just the main error message
            lines = error.split("\n")
            if lines:
                main_error = lines[0]
                if len(main_error) > 200:
                    return main_error[:200] + "..."
                return main_error
            return error[:200] + "..." if len(error) > 200 else error

        elif self.config.verbosity == "standard":
            # Include first few lines of stack trace
            lines = error.split("\n")
            if len(lines) > 5:
                return "\n".join(lines[:5]) + "\n..."
            return error

        return error

    def _estimate_tokens(self, response: Dict[str, Any]) -> int:
        """Estimate the token count for a response.

        Uses a simple character-based estimation. More sophisticated
        estimation could use a tokenizer, but this is faster and
        sufficient for optimization purposes.

        Args:
            response: The response dictionary

        Returns:
            Estimated token count
        """
        # Convert to string representation
        import json
        try:
            response_str = json.dumps(response, default=str)
        except Exception:
            response_str = str(response)

        # Estimate tokens from character count
        char_count = len(response_str)
        token_estimate = int(char_count / CHARS_PER_TOKEN)

        return max(1, token_estimate)

    @classmethod
    def for_action(cls, action_type: str) -> "ResponseBuilder":
        """Create a ResponseBuilder with config appropriate for an action type.

        Args:
            action_type: The type of action

        Returns:
            A ResponseBuilder with appropriate config
        """
        # Navigation actions benefit from full snapshots
        navigation_actions = {
            "navigate", "go_to", "reload", "go_back", "go_forward"
        }

        # Read actions usually don't need snapshots
        read_actions = {
            "get_text", "get_attribute", "get_value", "get_url", "get_title"
        }

        action_lower = action_type.lower().replace(" ", "_")

        if action_lower in navigation_actions:
            config = ResponseConfig(
                include_snapshot=True,
                snapshot_mode="full",
                verbosity="standard",
            )
        elif action_lower in read_actions:
            config = ResponseConfig(
                include_snapshot=False,
                snapshot_mode="none",
                verbosity="compact",
            )
        else:
            # Default for most actions
            config = ResponseConfig.standard()

        return cls(config)


class IncrementalResponseBuilder(ResponseBuilder):
    """Response builder that supports incremental snapshots.

    Extends ResponseBuilder to handle diff-based snapshots that only
    include changes since the last snapshot.
    """

    def __init__(
        self,
        config: ResponseConfig,
        previous_snapshot: Optional[str] = None,
    ) -> None:
        """Initialize the IncrementalResponseBuilder.

        Args:
            config: Response configuration
            previous_snapshot: The previous snapshot for diff calculation
        """
        super().__init__(config)
        self.previous_snapshot = previous_snapshot

    def build_with_diff(
        self,
        action: str,
        action_result: Any,
        current_snapshot: Optional[str] = None,
        error: Optional[str] = None,
        ref: Optional[str] = None,
    ) -> FilteredResponse:
        """Build a response with an incremental snapshot diff.

        Args:
            action: The action type that was executed
            action_result: The raw result from the action
            current_snapshot: The current full snapshot
            error: Optional error message if action failed
            ref: Optional element reference

        Returns:
            A FilteredResponse with incremental snapshot
        """
        if self.config.snapshot_mode == "incremental" and current_snapshot:
            # Calculate diff if we have a previous snapshot
            if self.previous_snapshot:
                diff_snapshot = self._calculate_diff(
                    self.previous_snapshot,
                    current_snapshot,
                )
            else:
                # First snapshot, include full
                diff_snapshot = current_snapshot
        else:
            diff_snapshot = current_snapshot

        return self.build(
            action=action,
            action_result=action_result,
            snapshot=diff_snapshot,
            error=error,
            ref=ref,
        )

    def _calculate_diff(
        self,
        previous: str,
        current: str,
    ) -> str:
        """Calculate a simple diff between snapshots.

        This is a simplified diff that shows added/removed lines.
        A more sophisticated implementation could use a proper
        diffing algorithm for YAML/tree structures.

        Args:
            previous: Previous snapshot
            current: Current snapshot

        Returns:
            Diff representation
        """
        prev_lines = set(previous.strip().split("\n"))
        curr_lines = set(current.strip().split("\n"))

        added = curr_lines - prev_lines
        removed = prev_lines - curr_lines

        if not added and not removed:
            return "(no changes)"

        diff_parts = []

        if added:
            diff_parts.append("# Added:")
            for line in sorted(added)[:20]:  # Limit to 20 lines
                diff_parts.append(f"+ {line}")
            if len(added) > 20:
                diff_parts.append(f"+ ... and {len(added) - 20} more")

        if removed:
            diff_parts.append("# Removed:")
            for line in sorted(removed)[:10]:  # Limit to 10 lines
                diff_parts.append(f"- {line}")
            if len(removed) > 10:
                diff_parts.append(f"- ... and {len(removed) - 10} more")

        return "\n".join(diff_parts)

    def update_previous_snapshot(self, snapshot: str) -> None:
        """Update the previous snapshot for next diff calculation.

        Args:
            snapshot: The new snapshot to use as previous
        """
        self.previous_snapshot = snapshot
