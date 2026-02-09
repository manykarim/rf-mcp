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
            (r'\bselenium\b', 6),  # Standalone mention — catches "Selenium test", "run Selenium", etc.
            (r'\bwebdriver\b', 6),
            (r'\bselenium\s+grid\b', 8),
            (r'\bselenium\s+standalone\b', 7),
            (r'\bclassic\s+selenium\b', 7),
            (r'\bselenium\s+automation\b', 7),
            (r'\b(chromedriver|geckodriver|edgedriver|safaridriver)\b', 7),
            (r'\bopen\s+browser\b', 6),
            (r'\b(create\s+webdriver|get\s+webelement)\b', 8),
            (r'\b(selenium\s+(2|3|4)|selenium2library)\b', 8),
            (r'\b(implicit|explicit)\s+wait\b', 6),
            (r'\b(input\s+text|click\s+element|page\s+should\s+contain)\b', 6),
            (r'\b(desired\s+capabilities|driver\s+capabilities)\b', 7),
        ],
        'Browser': [
            (r'\b(use|using|with|via|through)\s+(browser|browserlibrary|browser\s*library|playwright)\b', 10),
            (r'\bbrowser\s*library\b', 9),
            (r'\bplaywright\b', 9),
            (r'\bmodern\s+web\s+testing\b', 7),
            (r'\bmodern\s+browser\s+automation\b', 8),
            (r'\bcross[- ]browser\s+testing\b', 6),
            (r'\bchromium\b', 7),
            (r'\bwebkit\b', 7),
            (r'\bnew\s+(browser|page|context)\b', 8),
            (r'\bfill\s+(text|secret)\b', 7),
            (r'\b(rfbrowser|robotframework-browser)\b', 9),
            (r'\b(headless\s+browser|headless\s+chromium)\b', 6),
            (r'\b(shadow\s+dom|web\s+components?)\b', 6),
            (r'\b(SPA|single\s+page\s+app(lication)?)\b', 5),
            (r'\b(e2e|end.to.end)\s+(test|automat)', 5),
        ],
        'RequestsLibrary': [
            (r'\b(use|using|with)\s+(requests|requestslibrary|requests\s*library)\b', 10),
            (r'\brequestslibrary\b', 9),
            (r'\brest\s+api\s+testing\b', 7),
            (r'\bhttp\s+requests?\b', 5),
            (r'\bapi\s+automation\b', 6),
            (r'\b(create\s+session|get\s+on\s+session|post\s+on\s+session)\b', 8),
            (r'\b(status\s+should\s+be|request\s+should\s+be)\b', 7),
            (r'\b(GET|POST|PUT|DELETE|PATCH)\s+(request|on\s+session)\b', 7),
            (r'\b(webservice|web\s+service)\b', 5),
            (r'\bmicroservice\b', 5),
            (r'\b(bearer\s+token|JWT|OAuth2?)\b', 5),
            (r'\b(swagger|openapi)\b', 5),
            (r'\b(graphql|gRPC|SOAP)\b', 5),
            (r'\b(webhook|callback\s+url)\b', 5),
            (r'\bstatus\s+code\b', 5),
        ],
        'AppiumLibrary': [
            (r'\b(use|using|with)\s+(appium|appiumlibrary|appium\s*library)\b', 10),
            (r'\bappiumlibrary\b', 9),
            (r'\bmobile\s+automation\b', 7),
            (r'\bandroid\s+testing\b', 6),
            (r'\bios\s+testing\b', 6),
            (r'\bmobile\s+app\s+testing\b', 7),
            (r'\b(open\s+application|close\s+application)\b', 8),
            (r'\b(tap|swipe|long\s+press|double\s+tap|flick)\b', 6),
            (r'\b(emulator|simulator)\b', 5),
            (r'\b(native\s+app|hybrid\s+app|webview)\b', 6),
            (r'\b(APK|IPA|bundle\s+id|package\s+name)\b', 6),
            (r'\b(device\s+farm|BrowserStack|Sauce\s+Labs)\b', 5),
            (r'\b(UIAutomator2?|XCUITest|Espresso)\b', 7),
            (r'\b(iphone|ipad|tablet|smartphone)\b', 5),
        ],
        'DatabaseLibrary': [
            (r'\b(use|using|with)\s+(database|databaselibrary|database\s*library)\b', 10),
            (r'\bdatabaselibrary\b', 9),
            (r'\bsql\s+testing\b', 6),
            (r'\bdatabase\s+validation\b', 6),
            (r'\b(connect\s+to\s+database|execute\s+sql|call\s+stored\s+procedure)\b', 8),
            (r'\b(row\s+count|check\s+if\s+exists)\b', 7),
            (r'\b(postgres(ql)?|mysql|mariadb|sqlite)\b', 5),
            (r'\b(oracle|sql\s+server|mssql|mongodb)\b', 5),
            (r'\b(connection\s+string|DSN|ODBC)\b', 5),
            (r'\bstored\s+procedure\b', 6),
            (r'\b(CRUD|schema\s+migration)\b', 5),
            (r'\b(SELECT|INSERT|UPDATE|DELETE)\s+(FROM|INTO|SET)\b', 5),
        ],
        'SSHLibrary': [
            (r'\b(use|using|with)\s+(ssh|sshlibrary|ssh\s*library)\b', 10),
            (r'\bsshlibrary\b', 9),
            (r'\bremote\s+server\s+commands?\b', 5),
            (r'\b(open\s+connection|login\s+with\s+public\s+key)\b', 7),
            (r'\b(execute\s+command|start\s+command)\b', 6),
            (r'\b(get\s+file|put\s+file|get\s+directory|put\s+directory)\b', 6),
            (r'\b(sftp|scp)\b', 6),
            (r'\b(remote\s+(server|execution|machine))\b', 5),
            (r'\b(linux|unix)\s+(server|machine|system)\b', 5),
        ],
        'XML': [
            (r'\b(use|using|with)\s+xml\s*library\b', 10),
            (r'\bxml\s+parsing\b', 6),
            (r'\bxml\s+validation\b', 6),
            (r'\b(parse\s+xml|save\s+xml|log\s+element)\b', 7),
            (r'\b(get\s+element\s+text|get\s+element\s+attribute)\b', 6),
            (r'\b(xslt|dtd|xsd)\b', 6),
            (r'\b(namespace|element\s+tree|lxml)\b', 5),
            (r'\bxml\s+(file|document|response|config)\b', 5),
            (r'\bxpath\s+(expression|query|selector)\b', 6),
        ],
    }

    # Minimum score required for detection
    DEFAULT_MIN_SCORE = 5

    # Negation patterns that indicate the user does NOT want a library
    NEGATION_PATTERNS = [
        re.compile(r'\b(not|don\'t|do\s+not|without|stop|avoid)\s+(?:using\s+)?', re.IGNORECASE),
        re.compile(r'\b(instead\s+of|migrate\s+from|replace|replacing|move\s+away\s+from)\s+', re.IGNORECASE),
    ]

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

        # Check for negation context
        for lib_name, score in list(scores.items()):
            if score > 0:
                for neg_pattern in self.NEGATION_PATTERNS:
                    for pattern_str, weight in self.LIBRARY_PATTERNS.get(lib_name, []):
                        combined = neg_pattern.pattern + r'.*?' + pattern_str
                        if re.search(combined, text, re.IGNORECASE):
                            scores[lib_name] = max(0, scores[lib_name] - weight * 2)
                            break

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
