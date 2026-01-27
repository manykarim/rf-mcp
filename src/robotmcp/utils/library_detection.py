"""Centralized library detection from scenario text."""

import re
import logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class LibraryDetector:
    """Centralized library detection from scenario text.

    Consolidates detection logic that was previously duplicated in:
    - nlp_processor.py
    - session_models.py
    """

    # Weighted patterns for explicit library detection
    # Format: (pattern, weight) - higher weight = stronger signal
    LIBRARY_PATTERNS: Dict[str, List[Tuple[str, int]]] = {
        'SeleniumLibrary': [
            (r'\b(use|using|with|via|through)\s+(selenium|seleniumlibrary|selenium\s*library)\b', 10),
            (r'\btest\s+automation\s+with\s+selenium\b', 8),
            (r'\bseleniumlibrary\b', 9),
            (r'\bwebdriver\b', 6),
            (r'\bselenium\s+grid\b', 8),
            (r'\bselenium\s+standalone\b', 7),
            (r'\bclassic\s+selenium\b', 7),
            (r'\bselenium\s+automation\b', 7),
        ],
        'Browser': [
            (r'\b(use|using|with|via|through)\s+(browser|browserlibrary|browser\s*library|playwright)\b', 10),
            (r'\bbrowser\s*library\b', 9),
            (r'\bplaywright\b', 9),
            (r'\bmodern\s+web\s+testing\b', 7),
            (r'\bmodern\s+browser\s+automation\b', 8),
            (r'\bcross[- ]browser\s+testing\b', 6),
        ],
        'RequestsLibrary': [
            (r'\b(use|using|with)\s+(requests|requestslibrary|requests\s*library)\b', 10),
            (r'\brequestslibrary\b', 9),
            (r'\brest\s+api\s+testing\b', 7),
            (r'\bhttp\s+requests?\b', 5),
            (r'\bapi\s+automation\b', 6),
        ],
        'AppiumLibrary': [
            (r'\b(use|using|with)\s+(appium|appiumlibrary|appium\s*library)\b', 10),
            (r'\bappiumlibrary\b', 9),
            (r'\bmobile\s+automation\b', 7),
            (r'\bandroid\s+testing\b', 6),
            (r'\bios\s+testing\b', 6),
            (r'\bmobile\s+app\s+testing\b', 7),
        ],
        'DatabaseLibrary': [
            (r'\b(use|using|with)\s+(database|databaselibrary|database\s*library)\b', 10),
            (r'\bdatabaselibrary\b', 9),
            (r'\bsql\s+testing\b', 6),
            (r'\bdatabase\s+validation\b', 6),
        ],
        'SSHLibrary': [
            (r'\b(use|using|with)\s+(ssh|sshlibrary|ssh\s*library)\b', 10),
            (r'\bsshlibrary\b', 9),
            (r'\bremote\s+server\s+commands?\b', 5),
        ],
        'XML': [
            (r'\b(use|using|with)\s+xml\s*library\b', 10),
            (r'\bxml\s+parsing\b', 6),
            (r'\bxml\s+validation\b', 6),
        ],
    }

    # Minimum score required for detection
    DEFAULT_MIN_SCORE = 5

    def __init__(self, min_score: int = None):
        """Initialize LibraryDetector.

        Args:
            min_score: Minimum score required for library detection
        """
        self.min_score = min_score or self.DEFAULT_MIN_SCORE
        self._compiled_patterns: Dict[str, List[Tuple[re.Pattern, int]]] = {}
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns for performance."""
        for lib, patterns in self.LIBRARY_PATTERNS.items():
            self._compiled_patterns[lib] = [
                (re.compile(pattern, re.IGNORECASE), weight)
                for pattern, weight in patterns
            ]

    def detect(self, text: str, min_score: int = None) -> Optional[str]:
        """Detect explicit library preference from text.

        Args:
            text: Scenario text to analyze
            min_score: Override minimum score threshold

        Returns:
            Library name if detected with sufficient confidence, None otherwise
        """
        if not text:
            return None

        scores = self.get_scores(text)

        if not scores:
            return None

        threshold = min_score or self.min_score
        best_lib, best_score = max(scores.items(), key=lambda x: x[1])

        if best_score >= threshold:
            logger.debug(f"Detected library preference: {best_lib} (score: {best_score})")
            return best_lib

        return None

    def get_scores(self, text: str) -> Dict[str, int]:
        """Get detection scores for all libraries.

        Args:
            text: Scenario text to analyze

        Returns:
            Dictionary of library names to scores
        """
        if not text:
            return {}

        text_lower = text.lower()
        scores: Dict[str, int] = defaultdict(int)

        for lib, patterns in self._compiled_patterns.items():
            for pattern, weight in patterns:
                matches = len(pattern.findall(text_lower))
                if matches > 0:
                    scores[lib] += matches * weight

        return dict(scores)

    def detect_all(self, text: str, min_score: int = None) -> List[Tuple[str, int]]:
        """Detect all libraries above threshold, sorted by score.

        Args:
            text: Scenario text to analyze
            min_score: Override minimum score threshold

        Returns:
            List of (library_name, score) tuples, sorted by score descending
        """
        scores = self.get_scores(text)
        threshold = min_score or self.min_score

        detected = [(lib, score) for lib, score in scores.items() if score >= threshold]
        return sorted(detected, key=lambda x: -x[1])

    def get_conflicting_detections(self, text: str) -> Dict[str, List[str]]:
        """Detect if conflicting libraries are mentioned.

        Args:
            text: Scenario text to analyze

        Returns:
            Dictionary of conflict groups to detected libraries
        """
        CONFLICT_GROUPS = {
            'web_automation': ['Browser', 'SeleniumLibrary'],
        }

        scores = self.get_scores(text)
        conflicts = {}

        for group_name, group_libs in CONFLICT_GROUPS.items():
            detected = [lib for lib in group_libs if scores.get(lib, 0) > 0]
            if len(detected) > 1:
                conflicts[group_name] = detected

        return conflicts


# Module-level singleton for convenience
_detector: Optional[LibraryDetector] = None


def get_library_detector() -> LibraryDetector:
    """Get the global LibraryDetector instance."""
    global _detector
    if _detector is None:
        _detector = LibraryDetector()
    return _detector


def detect_library_preference(text: str, min_score: int = 5) -> Optional[str]:
    """Convenience function to detect library preference.

    Args:
        text: Scenario text to analyze
        min_score: Minimum score threshold

    Returns:
        Library name if detected, None otherwise
    """
    return get_library_detector().detect(text, min_score)
