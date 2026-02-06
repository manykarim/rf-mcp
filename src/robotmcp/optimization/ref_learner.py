"""Ref usage pattern learning for predictive preloading.

This module learns element ref usage patterns to enable predictive
preloading and sequence prediction for frequently accessed elements.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
import time

from .pattern_store import PatternStore


@dataclass
class RefUsagePattern:
    """
    Learned ref usage patterns for predictive preloading.

    Attributes:
        page_type: The classified page type
        commonly_used_refs: Most frequently accessed refs
        ref_access_sequence: Common access sequences
        preload_candidates: Refs recommended for preloading
        session_count: Number of sessions this pattern is based on
    """
    page_type: str
    commonly_used_refs: List[str]
    ref_access_sequence: List[str]
    preload_candidates: List[str]
    session_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "page_type": self.page_type,
            "commonly_used_refs": self.commonly_used_refs,
            "ref_access_sequence": self.ref_access_sequence,
            "preload_candidates": self.preload_candidates,
            "session_count": self.session_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RefUsagePattern":
        """Create from dictionary."""
        return cls(
            page_type=data.get("page_type", "unknown"),
            commonly_used_refs=data.get("commonly_used_refs", []),
            ref_access_sequence=data.get("ref_access_sequence", []),
            preload_candidates=data.get("preload_candidates", []),
            session_count=data.get("session_count", 0),
        )


@dataclass
class SessionRefTracker:
    """
    Track ref accesses within a single session.

    Attributes:
        session_id: Unique session identifier
        page_type: The page type for this session
        ref_sequence: Ordered list of refs accessed
        access_times: Timestamps for each access
    """
    session_id: str
    page_type: str
    ref_sequence: List[str] = field(default_factory=list)
    access_times: List[float] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)


class RefUsageLearner:
    """
    Learn ref usage patterns for predictive preloading.

    Tracks which refs are commonly accessed together and in what
    sequence, enabling intelligent preloading of element locators.

    Example:
        learner = RefUsageLearner()
        learner.record_ref_access("search_results", "e1", "session-123")
        learner.record_ref_access("search_results", "e5", "session-123")
        candidates = learner.get_preload_candidates("search_results")
    """

    # Maximum sequences to keep per page type
    MAX_SEQUENCES_PER_TYPE = 100

    # Maximum refs to track in frequency map
    MAX_REFS_PER_TYPE = 1000

    # Session timeout in seconds
    SESSION_TIMEOUT_SECONDS = 3600  # 1 hour

    def __init__(self, pattern_store: Optional[PatternStore] = None):
        """
        Initialize the ref usage learner.

        Args:
            pattern_store: Pattern store for persistence. Creates default if None.
        """
        self.pattern_store = pattern_store or PatternStore()

        # Track ref sequences per page type
        self.ref_sequences: Dict[str, List[List[str]]] = defaultdict(list)

        # Track ref frequencies per page type
        self.ref_frequencies: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Track transition frequencies: page_type -> (from_ref -> to_ref -> count)
        self.ref_transitions: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )

        # Active session trackers
        self._active_sessions: Dict[str, SessionRefTracker] = {}

        # Load persisted patterns
        self._load_persisted_patterns()

    def _load_persisted_patterns(self) -> None:
        """Load previously learned ref patterns."""
        for key in self.pattern_store.list_keys("refs"):
            data = self.pattern_store.retrieve("refs", key)
            if not data:
                continue

            if key.startswith("sequences-"):
                page_type = key.replace("sequences-", "")
                sequences = data.get("sequences", [])
                self.ref_sequences[page_type] = sequences[-self.MAX_SEQUENCES_PER_TYPE:]

                # Rebuild frequencies and transitions from sequences
                for seq in sequences:
                    for i, ref in enumerate(seq):
                        self.ref_frequencies[page_type][ref] += 1
                        if i > 0:
                            self.ref_transitions[page_type][seq[i-1]][ref] += 1

            elif key.startswith("frequencies-"):
                page_type = key.replace("frequencies-", "")
                frequencies = data.get("frequencies", {})
                for ref, count in frequencies.items():
                    self.ref_frequencies[page_type][ref] = count

    def record_ref_access(
        self,
        page_type: str,
        ref: str,
        session_id: str,
    ) -> None:
        """
        Record a ref access event.

        Args:
            page_type: The classified page type
            ref: The ref that was accessed
            session_id: The session identifier
        """
        # Update frequency
        self.ref_frequencies[page_type][ref] += 1

        # Bound frequency map size
        if len(self.ref_frequencies[page_type]) > self.MAX_REFS_PER_TYPE:
            # Remove least frequent refs
            sorted_refs = sorted(
                self.ref_frequencies[page_type].items(),
                key=lambda x: x[1]
            )
            for removed_ref, _ in sorted_refs[:100]:  # Remove bottom 100
                del self.ref_frequencies[page_type][removed_ref]

        # Track in session
        session_key = f"{page_type}:{session_id}"
        current_time = time.time()

        if session_key not in self._active_sessions:
            self._active_sessions[session_key] = SessionRefTracker(
                session_id=session_id,
                page_type=page_type,
            )

        tracker = self._active_sessions[session_key]

        # Check for session timeout
        if current_time - tracker.start_time > self.SESSION_TIMEOUT_SECONDS:
            # End old session and start new one
            self._end_session(session_key)
            self._active_sessions[session_key] = SessionRefTracker(
                session_id=session_id,
                page_type=page_type,
            )
            tracker = self._active_sessions[session_key]

        # Update transitions
        if tracker.ref_sequence:
            previous_ref = tracker.ref_sequence[-1]
            self.ref_transitions[page_type][previous_ref][ref] += 1

        tracker.ref_sequence.append(ref)
        tracker.access_times.append(current_time)

    def _end_session(self, session_key: str) -> None:
        """
        End a session and record its sequence.

        Args:
            session_key: The session key to end
        """
        tracker = self._active_sessions.pop(session_key, None)
        if not tracker or not tracker.ref_sequence:
            return

        # Record the sequence
        self.ref_sequences[tracker.page_type].append(tracker.ref_sequence)

        # Bound sequences
        if len(self.ref_sequences[tracker.page_type]) > self.MAX_SEQUENCES_PER_TYPE:
            self.ref_sequences[tracker.page_type] = \
                self.ref_sequences[tracker.page_type][-self.MAX_SEQUENCES_PER_TYPE:]

    def record_session_sequence(
        self,
        page_type: str,
        ref_sequence: List[str],
    ) -> None:
        """
        Record a complete ref access sequence from a session.

        Useful for batch recording or when session tracking is external.

        Args:
            page_type: The page type
            ref_sequence: List of refs accessed in order
        """
        if not ref_sequence:
            return

        # Update frequencies
        for ref in ref_sequence:
            self.ref_frequencies[page_type][ref] += 1

        # Update transitions
        for i in range(1, len(ref_sequence)):
            self.ref_transitions[page_type][ref_sequence[i-1]][ref_sequence[i]] += 1

        # Store sequence
        self.ref_sequences[page_type].append(ref_sequence)

        # Bound sequences
        if len(self.ref_sequences[page_type]) > self.MAX_SEQUENCES_PER_TYPE:
            self.ref_sequences[page_type] = \
                self.ref_sequences[page_type][-self.MAX_SEQUENCES_PER_TYPE:]

        # Persist periodically
        if len(self.ref_sequences[page_type]) % 10 == 0:
            self.pattern_store.store("refs", f"sequences-{page_type}", {
                "sequences": self.ref_sequences[page_type]
            })

    def end_session(self, page_type: str, session_id: str) -> None:
        """
        Explicitly end a session.

        Args:
            page_type: The page type
            session_id: The session identifier
        """
        session_key = f"{page_type}:{session_id}"
        self._end_session(session_key)

    def get_preload_candidates(self, page_type: str, top_n: int = 10) -> List[str]:
        """
        Get refs to preload based on usage patterns.

        Returns the most frequently accessed refs for a page type.

        Args:
            page_type: The page type
            top_n: Maximum number of refs to return

        Returns:
            List of ref identifiers recommended for preloading
        """
        frequencies = self.ref_frequencies.get(page_type, {})

        if not frequencies:
            return []

        # Sort by frequency descending
        sorted_refs = sorted(
            frequencies.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [ref for ref, _ in sorted_refs[:top_n]]

    def predict_next_refs(
        self,
        page_type: str,
        current_ref: str,
        top_n: int = 3,
    ) -> List[str]:
        """
        Predict next likely refs based on transition patterns.

        Given a current ref, predicts what refs are likely to be
        accessed next based on historical sequences.

        Args:
            page_type: The page type
            current_ref: The most recently accessed ref
            top_n: Maximum number of predictions to return

        Returns:
            List of predicted next refs
        """
        transitions = self.ref_transitions.get(page_type, {})

        if not transitions or current_ref not in transitions:
            # Fall back to most common refs
            return self.get_preload_candidates(page_type, top_n)

        next_refs = transitions[current_ref]

        # Sort by transition count
        sorted_next = sorted(
            next_refs.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [ref for ref, _ in sorted_next[:top_n]]

    def get_common_sequences(
        self,
        page_type: str,
        min_length: int = 2,
        top_n: int = 5,
    ) -> List[List[str]]:
        """
        Get common ref access sequences.

        Args:
            page_type: The page type
            min_length: Minimum sequence length to consider
            top_n: Maximum number of sequences to return

        Returns:
            List of common ref sequences
        """
        sequences = self.ref_sequences.get(page_type, [])

        if not sequences:
            return []

        # Filter by minimum length
        valid_sequences = [s for s in sequences if len(s) >= min_length]

        if not valid_sequences:
            return []

        # Count subsequences of length min_length
        subsequence_counts: Dict[tuple, int] = defaultdict(int)

        for seq in valid_sequences:
            for i in range(len(seq) - min_length + 1):
                subseq = tuple(seq[i:i + min_length])
                subsequence_counts[subseq] += 1

        # Sort by count and return top n
        sorted_subseqs = sorted(
            subsequence_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [list(subseq) for subseq, _ in sorted_subseqs[:top_n]]

    def get_pattern(self, page_type: str) -> Optional[RefUsagePattern]:
        """
        Get the learned pattern for a page type.

        Args:
            page_type: The page type

        Returns:
            RefUsagePattern if data exists, None otherwise
        """
        frequencies = self.ref_frequencies.get(page_type, {})
        sequences = self.ref_sequences.get(page_type, [])

        if not frequencies and not sequences:
            return None

        commonly_used = self.get_preload_candidates(page_type, 20)
        common_sequences = self.get_common_sequences(page_type, min_length=2, top_n=3)
        preload_candidates = self.get_preload_candidates(page_type, 10)

        return RefUsagePattern(
            page_type=page_type,
            commonly_used_refs=commonly_used,
            ref_access_sequence=common_sequences[0] if common_sequences else [],
            preload_candidates=preload_candidates,
            session_count=len(sequences),
        )

    def get_all_patterns(self) -> Dict[str, RefUsagePattern]:
        """
        Get all learned ref usage patterns.

        Returns:
            Dictionary mapping page type to pattern
        """
        all_page_types = set(list(self.ref_frequencies.keys()) +
                           list(self.ref_sequences.keys()))

        patterns = {}
        for page_type in all_page_types:
            pattern = self.get_pattern(page_type)
            if pattern:
                patterns[page_type] = pattern

        return patterns

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about ref learning.

        Returns:
            Dictionary with learning statistics
        """
        stats = {
            "active_sessions": len(self._active_sessions),
            "page_types": {},
        }

        all_page_types = set(list(self.ref_frequencies.keys()) +
                           list(self.ref_sequences.keys()))

        for page_type in all_page_types:
            frequencies = self.ref_frequencies.get(page_type, {})
            sequences = self.ref_sequences.get(page_type, [])

            stats["page_types"][page_type] = {
                "unique_refs": len(frequencies),
                "total_accesses": sum(frequencies.values()),
                "recorded_sequences": len(sequences),
                "top_refs": self.get_preload_candidates(page_type, 5),
            }

        return stats

    def persist_all(self) -> None:
        """Persist all learned patterns to storage."""
        # End all active sessions
        for session_key in list(self._active_sessions.keys()):
            self._end_session(session_key)

        # Persist sequences
        for page_type, sequences in self.ref_sequences.items():
            self.pattern_store.store("refs", f"sequences-{page_type}", {
                "sequences": sequences
            })

        # Persist frequencies
        for page_type, frequencies in self.ref_frequencies.items():
            self.pattern_store.store("refs", f"frequencies-{page_type}", {
                "frequencies": dict(frequencies)
            })

    def reset_learning(self, page_type: Optional[str] = None) -> None:
        """
        Reset learned patterns.

        Args:
            page_type: Specific page type to reset, or None for all
        """
        if page_type:
            self.ref_frequencies.pop(page_type, None)
            self.ref_sequences.pop(page_type, None)
            self.ref_transitions.pop(page_type, None)

            # Remove active sessions for this page type
            keys_to_remove = [
                k for k in self._active_sessions.keys()
                if k.startswith(f"{page_type}:")
            ]
            for key in keys_to_remove:
                self._active_sessions.pop(key, None)

            self.pattern_store.delete("refs", f"sequences-{page_type}")
            self.pattern_store.delete("refs", f"frequencies-{page_type}")
        else:
            self.ref_frequencies.clear()
            self.ref_sequences.clear()
            self.ref_transitions.clear()
            self._active_sessions.clear()

            for key in self.pattern_store.list_keys("refs"):
                self.pattern_store.delete("refs", key)
