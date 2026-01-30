"""Page complexity analysis for optimization decisions.

This module analyzes web page characteristics to determine optimal
optimization strategies without external dependencies.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional


class ComplexityLevel(Enum):
    """Page complexity classification levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class PageComplexityProfile:
    """
    Profile of page complexity for optimization decisions.

    Analyzes various page characteristics to determine the optimal
    approach for ARIA snapshots, list folding, and timeout configuration.

    Attributes:
        element_count: Total number of DOM elements
        form_count: Number of form elements
        list_item_count: Number of list items (li, tr, option, etc.)
        depth: Maximum nesting depth of the DOM tree
        has_shadow_dom: Whether the page uses Shadow DOM
        has_iframes: Whether the page contains iframes
        interactive_element_count: Number of interactive elements
        text_content_length: Total text content length in characters
        image_count: Number of image elements
    """
    element_count: int
    form_count: int
    list_item_count: int
    depth: int
    has_shadow_dom: bool
    has_iframes: bool
    interactive_element_count: int = 0
    text_content_length: int = 0
    image_count: int = 0

    @property
    def complexity_level(self) -> ComplexityLevel:
        """
        Determine complexity level: low, medium, or high.

        Uses a weighted scoring system based on page characteristics.
        Higher scores indicate more complex pages that benefit from
        aggressive optimization.

        Returns:
            ComplexityLevel enum value
        """
        score = (
            (self.element_count > 500) * 2 +
            (self.element_count > 1000) * 1 +
            (self.list_item_count > 50) * 1 +
            (self.list_item_count > 100) * 1 +
            (self.depth > 10) * 1 +
            (self.depth > 15) * 1 +
            self.has_shadow_dom * 2 +
            self.has_iframes * 1 +
            (self.form_count > 3) * 1 +
            (self.interactive_element_count > 100) * 1
        )

        if score >= 5:
            return ComplexityLevel.HIGH
        elif score >= 2:
            return ComplexityLevel.MEDIUM
        return ComplexityLevel.LOW

    @property
    def complexity_score(self) -> int:
        """
        Get the raw complexity score for fine-grained decisions.

        Returns:
            Integer score (0-12 typical range)
        """
        return (
            (self.element_count > 500) * 2 +
            (self.element_count > 1000) * 1 +
            (self.list_item_count > 50) * 1 +
            (self.list_item_count > 100) * 1 +
            (self.depth > 10) * 1 +
            (self.depth > 15) * 1 +
            self.has_shadow_dom * 2 +
            self.has_iframes * 1 +
            (self.form_count > 3) * 1 +
            (self.interactive_element_count > 100) * 1
        )

    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """
        Get optimization recommendations based on page complexity.

        Returns a dictionary of recommended optimization settings
        tailored to the page's characteristics.

        Returns:
            Dictionary with optimization recommendations
        """
        complexity = self.complexity_level

        return {
            "use_aria_snapshot": True,  # Always use ARIA snapshots
            "enable_list_folding": self.list_item_count > 20,
            "enable_incremental_diff": complexity != ComplexityLevel.LOW,
            "aggressive_filtering": complexity == ComplexityLevel.HIGH,
            "fold_threshold": self._calculate_optimal_fold_threshold(),
            "recommended_action_timeout_ms": self._calculate_action_timeout(),
            "recommended_navigation_timeout_ms": self._calculate_navigation_timeout(),
            "enable_ref_preloading": self.interactive_element_count > 50,
            "max_snapshot_depth": self._calculate_max_depth(),
            "complexity_level": complexity.value,
            "complexity_score": self.complexity_score,
        }

    def _calculate_optimal_fold_threshold(self) -> float:
        """
        Calculate optimal SimHash threshold based on page characteristics.

        Lower thresholds mean more aggressive folding (more items grouped).
        Higher thresholds mean more conservative folding (fewer items grouped).

        Returns:
            Float threshold value between 0.75 and 0.95
        """
        base_threshold = 0.85

        if self.list_item_count > 200:
            # Very large lists: aggressive folding
            return max(0.75, base_threshold - 0.10)
        elif self.list_item_count > 100:
            # Large lists: moderately aggressive
            return max(0.78, base_threshold - 0.07)
        elif self.list_item_count > 50:
            # Medium lists: slightly aggressive
            return base_threshold - 0.03
        elif self.list_item_count < 10:
            # Small lists: conservative (less folding)
            return min(0.95, base_threshold + 0.08)
        elif self.list_item_count < 20:
            # Smaller lists: somewhat conservative
            return min(0.92, base_threshold + 0.05)

        return base_threshold

    def _calculate_action_timeout(self) -> int:
        """
        Calculate recommended action timeout based on complexity.

        Returns:
            Timeout in milliseconds
        """
        base_timeout = 5000  # 5 seconds

        complexity = self.complexity_level
        if complexity == ComplexityLevel.HIGH:
            return base_timeout + 3000  # 8 seconds for complex pages
        elif complexity == ComplexityLevel.MEDIUM:
            return base_timeout + 1500  # 6.5 seconds for medium pages

        return base_timeout

    def _calculate_navigation_timeout(self) -> int:
        """
        Calculate recommended navigation timeout based on complexity.

        Returns:
            Timeout in milliseconds
        """
        base_timeout = 30000  # 30 seconds

        complexity = self.complexity_level
        if complexity == ComplexityLevel.HIGH:
            return base_timeout + 30000  # 60 seconds for complex pages
        elif complexity == ComplexityLevel.MEDIUM:
            return base_timeout + 15000  # 45 seconds for medium pages

        return base_timeout

    def _calculate_max_depth(self) -> Optional[int]:
        """
        Calculate maximum recommended snapshot depth.

        Returns:
            Maximum depth or None for unlimited
        """
        if self.depth > 20:
            return 15  # Limit depth for very deep pages
        elif self.depth > 15:
            return 12

        return None  # No limit for normal pages


@dataclass
class PageTypeClassification:
    """
    Classification of page type for pattern matching.

    Attributes:
        page_type: Identified page type string
        confidence: Confidence score (0.0 to 1.0)
        indicators: List of indicators that led to classification
    """
    page_type: str
    confidence: float
    indicators: List[str] = field(default_factory=list)


class PageAnalyzer:
    """
    Analyzes pages to determine type and complexity.

    This analyzer uses heuristics based on page structure to classify
    pages and determine optimal optimization strategies.
    """

    # Page type indicators
    PAGE_TYPE_INDICATORS = {
        "search_results": [
            "search", "results", "query", "filter", "sort",
            "pagination", "page", "showing"
        ],
        "product_listing": [
            "product", "item", "price", "cart", "add to cart",
            "buy", "shop", "catalog"
        ],
        "form_page": [
            "form", "input", "submit", "login", "register",
            "sign up", "contact", "email"
        ],
        "article": [
            "article", "post", "blog", "content", "read",
            "author", "published", "comments"
        ],
        "dashboard": [
            "dashboard", "analytics", "metrics", "chart",
            "graph", "statistics", "overview"
        ],
        "data_table": [
            "table", "row", "column", "data", "export",
            "sort", "filter", "records"
        ],
    }

    def __init__(self):
        """Initialize the page analyzer."""
        self._type_cache: Dict[str, PageTypeClassification] = {}

    def analyze_complexity(
        self,
        element_count: int,
        form_count: int,
        list_item_count: int,
        depth: int,
        has_shadow_dom: bool = False,
        has_iframes: bool = False,
        interactive_element_count: int = 0,
        text_content_length: int = 0,
        image_count: int = 0,
    ) -> PageComplexityProfile:
        """
        Create a PageComplexityProfile from raw metrics.

        Args:
            element_count: Total number of DOM elements
            form_count: Number of form elements
            list_item_count: Number of list items
            depth: Maximum nesting depth
            has_shadow_dom: Whether Shadow DOM is present
            has_iframes: Whether iframes are present
            interactive_element_count: Number of interactive elements
            text_content_length: Total text length
            image_count: Number of images

        Returns:
            PageComplexityProfile instance
        """
        return PageComplexityProfile(
            element_count=element_count,
            form_count=form_count,
            list_item_count=list_item_count,
            depth=depth,
            has_shadow_dom=has_shadow_dom,
            has_iframes=has_iframes,
            interactive_element_count=interactive_element_count,
            text_content_length=text_content_length,
            image_count=image_count,
        )

    def classify_page_type(
        self,
        url: str,
        title: str = "",
        text_content: str = "",
        element_types: Optional[Dict[str, int]] = None,
    ) -> PageTypeClassification:
        """
        Classify the type of page based on URL and content.

        Args:
            url: Page URL
            title: Page title
            text_content: Sample of page text content
            element_types: Dictionary of element tag names to counts

        Returns:
            PageTypeClassification with type and confidence
        """
        # Check cache
        cache_key = url.lower()
        if cache_key in self._type_cache:
            return self._type_cache[cache_key]

        scores: Dict[str, float] = {}
        indicators_found: Dict[str, List[str]] = {}

        combined_text = f"{url} {title} {text_content}".lower()

        # Score each page type based on indicator presence
        for page_type, indicators in self.PAGE_TYPE_INDICATORS.items():
            score = 0.0
            found = []

            for indicator in indicators:
                if indicator in combined_text:
                    score += 1.0
                    found.append(indicator)

            # Bonus for URL patterns
            if page_type in url.lower():
                score += 2.0
                found.append(f"url:{page_type}")

            scores[page_type] = score
            indicators_found[page_type] = found

        # Additional heuristics based on element types
        if element_types:
            if element_types.get("table", 0) > 2:
                scores["data_table"] = scores.get("data_table", 0) + 2.0
                indicators_found.setdefault("data_table", []).append("multiple_tables")

            if element_types.get("form", 0) > 0:
                scores["form_page"] = scores.get("form_page", 0) + 1.5
                indicators_found.setdefault("form_page", []).append("has_forms")

            if element_types.get("article", 0) > 0:
                scores["article"] = scores.get("article", 0) + 2.0
                indicators_found.setdefault("article", []).append("article_element")

        # Find best match
        if not scores or max(scores.values()) == 0:
            result = PageTypeClassification(
                page_type="generic",
                confidence=0.5,
                indicators=["no_indicators_matched"]
            )
        else:
            best_type = max(scores.keys(), key=lambda k: scores[k])
            max_score = scores[best_type]
            # Normalize confidence (5+ indicators = high confidence)
            confidence = min(1.0, max_score / 5.0)

            result = PageTypeClassification(
                page_type=best_type,
                confidence=confidence,
                indicators=indicators_found.get(best_type, [])
            )

        # Cache result
        self._type_cache[cache_key] = result
        return result

    def get_optimization_profile(
        self,
        complexity: PageComplexityProfile,
        page_type: PageTypeClassification,
    ) -> Dict[str, Any]:
        """
        Get a combined optimization profile from complexity and type.

        Args:
            complexity: Page complexity profile
            page_type: Page type classification

        Returns:
            Dictionary with all optimization recommendations
        """
        recommendations = complexity.get_optimization_recommendations()

        # Adjust based on page type
        if page_type.page_type == "data_table":
            recommendations["enable_list_folding"] = True
            recommendations["fold_threshold"] = min(
                recommendations["fold_threshold"],
                0.80  # More aggressive for data tables
            )

        elif page_type.page_type == "form_page":
            recommendations["enable_list_folding"] = False
            recommendations["enable_ref_preloading"] = True

        elif page_type.page_type == "search_results":
            recommendations["enable_list_folding"] = True
            recommendations["enable_incremental_diff"] = True

        recommendations["page_type"] = page_type.page_type
        recommendations["page_type_confidence"] = page_type.confidence

        return recommendations
