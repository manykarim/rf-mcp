"""Locator Normalizer Adapter.

Wraps the existing LocatorConverter for the Intent domain.
"""
from __future__ import annotations

import logging
from typing import Optional

from ..value_objects import IntentTarget, LocatorStrategy, NormalizedLocator

logger = logging.getLogger(__name__)


class LocatorNormalizerAdapter:
    """Adapts locator normalization for the Intent domain.

    Implements the LocatorNormalizer protocol from services.py.
    Handles library-specific locator translation including:
    - Bare text -> library-appropriate prefix
    - CSS shorthand (#id, .class) -> explicit prefix for SL
    - XPath auto-detection (//) -> explicit prefix for SL
    - Cross-library prefix translation (text= <-> link:)
    """

    # CSS indicators that distinguish CSS from bare text
    _CSS_INDICATORS = frozenset({'#', '.', '[', '>', '+', '~', ':'})
    _XPATH_INDICATORS = ('//', '..')

    # Known prefixes per library family
    _BROWSER_PREFIXES = frozenset({
        'css=', 'xpath=', 'text=', 'id=', 'data-testid=',
    })
    _SELENIUM_PREFIXES = frozenset({
        'css=', 'css:', 'xpath=', 'xpath:', 'id=', 'id:',
        'name=', 'name:', 'link=', 'link:', 'partial link=',
        'partial link:', 'class=', 'class:', 'tag=', 'tag:',
        'dom=', 'dom:',
    })

    def normalize(
        self, target: IntentTarget, target_library: str,
    ) -> NormalizedLocator:
        """Normalize a locator for the target library."""
        original = target.locator
        locator = original
        strategy = "pass_through"
        was_transformed = False

        # If locator already has a valid prefix for target library, pass through
        if target.has_prefix:
            locator = self._translate_prefix(original, target_library)
            was_transformed = (locator != original)
            strategy = "prefix_translation" if was_transformed else "pass_through"
        elif target.has_explicit_strategy and target.strategy != LocatorStrategy.AUTO:
            locator = self._apply_strategy(original, target.strategy, target_library)
            was_transformed = (locator != original)
            strategy = f"explicit_{target.strategy.value}"
        elif self._looks_like_xpath(original):
            locator = self._normalize_xpath(original, target_library)
            was_transformed = (locator != original)
            strategy = "xpath_auto_detect"
        elif self._is_url(original):
            # URLs pass through unchanged (checked AFTER xpath to avoid
            # treating //button as a protocol-relative URL)
            strategy = "url_pass_through"
        elif self._looks_like_css(original):
            locator = self._normalize_css(original, target_library)
            was_transformed = (locator != original)
            strategy = "css_auto_detect"
        else:
            # Bare text
            locator = self._normalize_bare_text(original, target_library)
            was_transformed = (locator != original)
            strategy = "bare_text"

        return NormalizedLocator(
            value=locator,
            source_locator=original,
            target_library=target_library,
            strategy_applied=strategy,
            was_transformed=was_transformed,
        )

    def _is_url(self, locator: str) -> bool:
        return locator.startswith(('http://', 'https://', '/'))

    def _looks_like_xpath(self, locator: str) -> bool:
        return locator.startswith('//') or locator.startswith('..')

    def _looks_like_css(self, locator: str) -> bool:
        if not locator:
            return False
        first_char = locator[0]
        return first_char in self._CSS_INDICATORS or locator.startswith('[')

    def _translate_prefix(self, locator: str, target_library: str) -> str:
        """Translate locator prefix between libraries."""
        if target_library == "Browser":
            # SL -> Browser translations
            if locator.startswith('link:'):
                return f'text={locator[5:]}'
            if locator.startswith('link='):
                return f'text={locator[5:]}'
            if locator.startswith('partial link:'):
                return f'text={locator[13:]}'
            if locator.startswith('partial link='):
                return f'text={locator[13:]}'
            if locator.startswith('id:'):
                return f'id={locator[3:]}'
            if locator.startswith('css:'):
                return f'css={locator[4:]}'
            if locator.startswith('xpath:'):
                return f'xpath={locator[6:]}'
        elif target_library == "SeleniumLibrary":
            # Browser -> SL translations
            if locator.startswith('text='):
                text = locator[5:]
                if len(text) < 50:
                    return f'link:{text}'
                return f'xpath=//*[contains(text(),"{text}")]'
        elif target_library == "AppiumLibrary":
            if locator.startswith('text='):
                return f'accessibility_id={locator[5:]}'
        return locator

    def _apply_strategy(
        self, locator: str, strategy: LocatorStrategy, target_library: str,
    ) -> str:
        """Apply explicit strategy to bare locator."""
        prefix_map = {
            LocatorStrategy.CSS: "css=",
            LocatorStrategy.XPATH: "xpath=",
            LocatorStrategy.TEXT: "text=" if target_library == "Browser" else "link:",
            LocatorStrategy.ID: "id=",
            LocatorStrategy.NAME: "name=",
            LocatorStrategy.LINK: "text=" if target_library == "Browser" else "link:",
            LocatorStrategy.PARTIAL_LINK: "text=" if target_library == "Browser" else "partial link:",
            LocatorStrategy.ACCESSIBILITY_ID: "accessibility_id=",
        }
        prefix = prefix_map.get(strategy, "")
        return f"{prefix}{locator}" if prefix else locator

    def _normalize_xpath(self, locator: str, target_library: str) -> str:
        """Normalize XPath expressions."""
        if target_library in ("SeleniumLibrary", "AppiumLibrary"):
            if not locator.startswith('xpath=') and not locator.startswith('xpath:'):
                return f'xpath={locator}'
        return locator

    def _normalize_css(self, locator: str, target_library: str) -> str:
        """Normalize CSS selectors."""
        if target_library in ("SeleniumLibrary", "AppiumLibrary"):
            # #id shorthand -> id= for SL
            if locator.startswith('#') and ' ' not in locator and '>' not in locator:
                return f'id={locator[1:]}'
            # .class or complex CSS -> css= prefix
            if not locator.startswith('css=') and not locator.startswith('css:'):
                return f'css={locator}'
        return locator

    def _normalize_bare_text(self, locator: str, target_library: str) -> str:
        """Normalize bare text to library-appropriate locator."""
        if target_library == "Browser":
            return f'text={locator}'
        elif target_library == "SeleniumLibrary":
            if len(locator) < 50:
                return f'link:{locator}'
            return f'xpath=//*[contains(text(),"{locator}")]'
        elif target_library == "AppiumLibrary":
            return f'accessibility_id={locator}'
        return locator
