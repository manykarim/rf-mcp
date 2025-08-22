"""Locator conversion service between different browser and mobile libraries."""

import logging
import re
import json
from typing import Optional, Dict, List, Tuple

from robotmcp.models.config_models import ExecutionConfig
from robotmcp.models.session_models import PlatformType

logger = logging.getLogger(__name__)


class LocatorConverter:
    """Converts locators between different browser and mobile automation libraries."""
    
    # Mobile locator strategies and their patterns
    MOBILE_STRATEGIES = {
        'accessibility_id': r'^accessibility_id=(.+)$',
        'android_uiautomator': r'^android=(.+)$',
        'android_viewtag': r'^android_viewtag=(.+)$',
        'android_datamatcher': r'^android_datamatcher=(.+)$',
        'ios_predicate': r'^-ios predicate string:(.+)$',
        'ios_class_chain': r'^-ios class chain:(.+)$',
        'image': r'^-image:(.+)$',
        'id': r'^id=(.+)$',
        'class': r'^class=(.+)$',
        'xpath': r'^xpath=(.+)$|^//.+$'
    }
    
    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()
        self.platform_type = PlatformType.WEB  # Default platform
    
    def add_explicit_strategy_prefix(self, locator: str, for_test_suite: bool = False, target_library: str = "Browser") -> str:
        """
        Add explicit strategy prefix to locators for better Robot Framework compatibility.
        
        This helps avoid Robot Framework escaping issues with # characters and makes
        the intent clearer to AI agents parsing the generated test suites.
        
        Args:
            locator: The locator string to add prefix to
            for_test_suite: If True, add prefix for test suite generation;
                          If False, keep original format for execution
            target_library: Target library ("Browser" or "SeleniumLibrary")
            
        Returns:
            str: Locator with or without explicit strategy prefix based on context
        """
        if not self.config.ADD_EXPLICIT_SELECTOR_STRATEGIES:
            return locator
        
        # For test suite generation, always add prefix to avoid RF escaping
        if for_test_suite:
            # Already has a strategy prefix
            if self._has_explicit_strategy(locator):
                return locator
            
            # Determine and add appropriate strategy prefix based on target library
            strategy = self._detect_selector_strategy(locator)
            if strategy:
                return self._format_strategy_prefix(strategy, locator, target_library)
            return locator
        
        # For execution, keep original format (Browser Library handles prefixes internally)
        # Only add prefix if it helps avoid ambiguity, but this might cause argument parsing issues
        return locator
    
    def _detect_selector_strategy(self, locator: str) -> Optional[str]:
        """
        Detect the appropriate selector strategy for a locator.
        
        Args:
            locator: Locator to analyze
            
        Returns:
            str: Strategy name (css, xpath, text) or None if no prefix needed
        """
        # XPath selectors
        if locator.startswith('//') or locator.startswith('..'):
            return "xpath"
        
        # CSS selectors (most common cases)
        css_indicators = [
            locator.startswith('#'),      # ID selector
            locator.startswith('.'),      # Class selector
            locator.startswith('['),      # Attribute selector
            ' > ' in locator,             # Child combinator
            ' + ' in locator,             # Adjacent sibling
            ' ~ ' in locator,             # General sibling
            ':' in locator,               # Pseudo-selectors
        ]
        
        if any(css_indicators):
            return "css"
        
        # Simple tag selectors
        if locator.lower() in ['input', 'button', 'a', 'div', 'span', 'form', 'textarea', 
                              'select', 'option', 'label', 'img', 'table', 'tr', 'td']:
            return "css"
        
        # Likely text content (contains spaces or common button text)
        text_indicators = [
            ' ' in locator,                                    # Contains spaces
            locator in ['Login', 'Submit', 'Submit Form',     # Common button texts
                       'Click here', 'Sign Up', 'Register',
                       'Cancel', 'OK', 'Save', 'Delete', 'Edit'],
            len(locator) > 20,                                # Long strings likely text
        ]
        
        if any(text_indicators):
            return "text"
        
        # Default to CSS for Browser Library compatibility
        return "css"
    
    def _format_strategy_prefix(self, strategy: str, locator: str, target_library: str) -> str:
        """
        Format strategy prefix according to target library conventions.
        
        Args:
            strategy: Strategy type (css, xpath, text, etc.)
            locator: Original locator
            target_library: Target library ("Browser" or "SeleniumLibrary")
            
        Returns:
            str: Formatted locator with appropriate prefix
        """
        if target_library == "SeleniumLibrary":
            # SeleniumLibrary prefers strategy:value syntax
            if strategy == "text":
                # SeleniumLibrary doesn't have text strategy, convert to link or xpath
                if ' ' in locator and len(locator) < 50:  # Likely button/link text
                    return f"link:{locator}"
                else:
                    # Use xpath for complex text matching
                    escaped_text = locator.replace('"', '\"')
                    return f'xpath://*[contains(text(),"{escaped_text}")]'
            else:
                return f"{strategy}:{locator}"
        else:
            # Browser Library uses strategy=value syntax
            return f"{strategy}={locator}"
    
    def _has_explicit_strategy(self, locator: str) -> bool:
        """Check if locator already has an explicit strategy prefix."""
        # Browser Library strategies (strategy=value)
        browser_strategies = ['id=', 'css=', 'xpath=', 'text=']
        
        # SeleniumLibrary strategies (strategy:value or strategy=value)
        selenium_strategies = ['id:', 'name:', 'identifier:', 'class:', 'tag:', 
                             'xpath:', 'css:', 'dom:', 'link:', 'partial link:', 
                             'data:', 'jquery:', 'default:']
        
        all_strategies = browser_strategies + selenium_strategies + \
                        [s.replace(':', '=') for s in selenium_strategies if ':' in s]
        
        return any(locator.startswith(strategy) for strategy in all_strategies)

    def convert_locator_for_library(self, locator: str, target_library: str) -> str:
        """
        Convert locator format between different libraries.
        
        SeleniumLibrary supports:
        - id, name, identifier (default strategies)
        - class, tag, xpath, css, dom, link, partial link, data, jquery (explicit)
        - Implicit xpath detection (starts with //)
        
        Browser Library supports:
        - CSS (default)
        - xpath (auto-detected if starts with // or ..)
        - text (finds by text content)
        - id (CSS shorthand)
        - Cascaded selectors with >>
        - Shadow DOM piercing
        
        Args:
            locator: The locator string to convert
            target_library: Target library ("Browser" or "SeleniumLibrary")
            
        Returns:
            str: Converted locator string
        """
        if not locator:
            return locator
        
        if not self.config.ENABLE_LOCATOR_CONVERSION:
            return locator
            
        # NOTE: Locator conversion disabled for Browser Library to preserve strategy prefixes
        # Browser Library supports strategy prefixes (css=, id=, xpath=, etc.) and we should preserve them
        # Only convert for SeleniumLibrary if needed
        if target_library == "SeleniumLibrary":
            return self._convert_to_selenium_library(locator)
        # For Browser Library, preserve original locator to avoid stripping strategy prefixes
        elif target_library == "Browser":
            return locator  # No conversion - preserve strategy prefixes
        
        return locator
    
    def _convert_to_browser_library(self, locator: str) -> str:
        """Convert locator to Browser Library format."""
        # Handle explicit strategy syntax (strategy=value or strategy:value)
        if "=" in locator and self._is_explicit_strategy(locator):
            strategy, value = locator.split("=", 1) if "=" in locator else locator.split(":", 1)
            strategy = strategy.lower().strip()
            
            if strategy == "id":
                # id=element -> id=element (Browser supports both #element and id=element)
                return f"id={value}"
            elif strategy == "css":
                # css=selector -> selector (CSS is default in Browser)
                return value
            elif strategy == "xpath":
                # xpath=//path -> //path (Browser auto-detects xpath)
                return value
            elif strategy == "name":
                # name=attr -> [name="attr"]
                return f'[name="{value}"]'
            elif strategy == "class":
                # class=classname -> .classname
                return f".{value}"
            elif strategy == "tag":
                # tag=div -> div
                return value
            elif strategy == "link" or strategy == "partial link":
                # link=text -> text=text (Browser uses text selectors)
                return f'text={value}'
            elif strategy == "data":
                # data=value -> [data-testid="value"] - common convention
                return f'[data-testid="{value}"]'
            elif strategy == "jquery":
                # jquery selectors -> CSS equivalent (best effort)
                if self.config.CONVERT_JQUERY_SELECTORS:
                    return self._convert_jquery_to_css(value)
                else:
                    logger.warning(f"jQuery locator '{locator}' cannot be converted - jQuery conversion disabled")
                    return locator
            elif strategy == "dom":
                # dom expressions can't be directly converted
                logger.warning(f"DOM locator '{locator}' cannot be converted to Browser Library format")
                return locator
            elif strategy == "identifier":
                # identifier -> try id first, fallback to name
                return f'id={value}, [name="{value}"]'
        
        # Handle implicit xpath (starts with // or ..)
        elif locator.startswith("//") or locator.startswith(".."):
            # XPath is auto-detected in Browser Library
            return locator
        
        # Handle CSS shortcuts that might need conversion
        elif locator.startswith("#") or locator.startswith(".") or locator.startswith("["):
            # Already in CSS format, keep as-is
            return locator
        
        # Handle cascaded selectors
        elif self.config.CONVERT_CASCADED_SELECTORS and ">>" in locator:
            # Already in Browser Library cascaded format
            return locator
        
        return locator
    
    def _convert_to_selenium_library(self, locator: str) -> str:
        """Convert locator to SeleniumLibrary format."""
        # Handle text selectors
        if locator.startswith("text="):
            # text=content -> xpath=//*[contains(text(),'content')]
            text_content = locator[5:]
            return f'xpath=//*[contains(text(),"{text_content}")]'
        
        # Handle CSS shortcuts
        elif locator.startswith("#"):
            # #id -> id=id
            return f"id={locator[1:]}"
        elif locator.startswith("."):
            # .class -> css=.class
            return f"css={locator}"
        elif locator.startswith("["):
            # [attr] -> css=[attr]
            return f"css={locator}"
        
        # Handle cascaded selectors (Browser Library specific)
        elif ">>" in locator:
            # Convert >> to descendant CSS selector
            parts = locator.split(">>")
            css_selector = " ".join(part.strip() for part in parts)
            return f"css={css_selector}"
        
        # Handle implicit xpath
        elif locator.startswith("//") or locator.startswith(".."):
            # xpath=//path (explicit for SeleniumLibrary)
            return f"xpath={locator}"
        
        # Handle id= format (Browser Library)
        elif locator.startswith("id="):
            # Keep id= format as SeleniumLibrary supports it
            return locator
        
        # Plain CSS selector - make it explicit for SeleniumLibrary
        elif self._is_css_selector(locator):
            return f"css={locator}"
        
        return locator
    
    def _convert_jquery_to_css(self, jquery_selector: str) -> str:
        """
        Convert jQuery selectors to CSS equivalents (best effort).
        
        Args:
            jquery_selector: jQuery selector string
            
        Returns:
            str: CSS selector equivalent
        """
        if not self.config.CONVERT_JQUERY_SELECTORS:
            return jquery_selector
        
        css_selector = jquery_selector
        
        # Convert jQuery pseudo-selectors to CSS equivalents
        conversions = {
            ':first': ':first-child',
            ':last': ':last-child',
            ':eq(': ':nth-child(',
            ':even': ':nth-child(even)',
            ':odd': ':nth-child(odd)',
            ':gt(': ':nth-child(n+',  # Approximate
            ':lt(': ':nth-child(-n+',  # Approximate
            ':visible': ':not([hidden])',  # Approximation
            ':hidden': '[hidden]',  # Approximation
            ':checked': ':checked',  # Same
            ':selected': ':checked',  # Close enough for most cases
            ':disabled': ':disabled',  # Same
            ':enabled': ':not(:disabled)',  # CSS equivalent
            ':focus': ':focus'  # Same
        }
        
        for jquery_pseudo, css_pseudo in conversions.items():
            css_selector = css_selector.replace(jquery_pseudo, css_pseudo)
        
        # Handle :contains() pseudo-selector (jQuery specific)
        contains_pattern = r':contains\(["\']([^"\']+)["\']\)'
        css_selector = re.sub(contains_pattern, r'[title*="\1"], [alt*="\1"]', css_selector)
        
        # Handle :has() pseudo-selector (limited browser support)
        has_pattern = r':has\(([^)]+)\)'
        css_selector = re.sub(has_pattern, r':has(\1)', css_selector)
        
        if css_selector != jquery_selector:
            logger.debug(f"Converted jQuery selector '{jquery_selector}' to CSS '{css_selector}'")
        
        return css_selector
    
    def _is_explicit_strategy(self, locator: str) -> bool:
        """
        Check if locator uses explicit strategy syntax.
        
        Args:
            locator: Locator string to check
            
        Returns:
            bool: True if uses explicit strategy (strategy=value)
        """
        if "=" not in locator and ":" not in locator:
            return False
        
        # Check for known strategies before the = or :
        known_strategies = [
            'id', 'name', 'class', 'tag', 'xpath', 'css', 'dom', 
            'link', 'partial link', 'data', 'jquery', 'identifier', 'text'
        ]
        
        separator = "=" if "=" in locator else ":"
        potential_strategy = locator.split(separator, 1)[0].lower().strip()
        
        return potential_strategy in known_strategies
    
    def _is_css_selector(self, locator: str) -> bool:
        """
        Check if locator appears to be a CSS selector.
        
        Args:
            locator: Locator string to check
            
        Returns:
            bool: True if appears to be CSS selector
        """
        # Simple heuristics for CSS selectors
        css_indicators = [
            locator.startswith("#"),  # ID selector
            locator.startswith("."),  # Class selector
            locator.startswith("["),  # Attribute selector
            " > " in locator,         # Child combinator
            " + " in locator,         # Adjacent sibling
            " ~ " in locator,         # General sibling
            ":nth-child(" in locator, # Pseudo-selector
            ":first-child" in locator,
            ":last-child" in locator,
            ":hover" in locator,
            ":focus" in locator
        ]
        
        return any(css_indicators) and not locator.startswith("//")
    
    def get_conversion_stats(self) -> Dict[str, int]:
        """
        Get statistics about locator conversions performed.
        
        Returns:
            dict: Conversion statistics
        """
        # In a real implementation, this would track conversion counts
        return {
            "total_conversions": 0,
            "to_browser_library": 0,
            "to_selenium_library": 0,
            "jquery_conversions": 0,
            "failed_conversions": 0
        }
    
    def validate_locator(self, locator: str, library_type: str) -> Dict[str, bool]:
        """
        Validate locator syntax for a specific library.
        
        Args:
            locator: Locator string to validate
            library_type: Library to validate for ("Browser" or "SeleniumLibrary")
            
        Returns:
            dict: Validation results
        """
        validation = {
            "valid": True,
            "warnings": []
        }
        
        if not locator:
            validation["valid"] = False
            validation["warnings"].append("Empty locator")
            return validation
        
        if library_type == "Browser":
            # Check for SeleniumLibrary-specific syntax that might not work
            if locator.startswith("dom="):
                validation["warnings"].append("DOM locators not supported in Browser Library")
            elif "identifier=" in locator:
                validation["warnings"].append("identifier= strategy not directly supported in Browser Library")
        
        elif library_type == "SeleniumLibrary":
            # Check for Browser Library-specific syntax
            if ">>" in locator:
                validation["warnings"].append("Cascaded selectors (>>) may need conversion for SeleniumLibrary")
            elif locator.startswith("text="):
                validation["warnings"].append("text= selectors need conversion for SeleniumLibrary")
        
        return validation
    
    def set_platform_type(self, platform: PlatformType) -> None:
        """Set the platform type for locator conversion."""
        self.platform_type = platform
    
    def is_mobile_locator(self, locator: str) -> bool:
        """Check if a locator is mobile-specific."""
        for pattern in self.MOBILE_STRATEGIES.values():
            if re.match(pattern, locator):
                # Check if it's a mobile-only strategy
                mobile_only = ['accessibility_id', 'android_uiautomator', 'android_viewtag',
                              'android_datamatcher', 'ios_predicate', 'ios_class_chain', 'image']
                for strategy in mobile_only:
                    if re.match(self.MOBILE_STRATEGIES[strategy], locator):
                        return True
        return False
    
    def validate_mobile_locator(self, locator: str, platform: str = None) -> Tuple[bool, List[str]]:
        """
        Validate a mobile locator.
        
        Args:
            locator: Locator string to validate
            platform: 'Android' or 'iOS' (optional)
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check if it matches any mobile strategy
        matched = False
        for strategy, pattern in self.MOBILE_STRATEGIES.items():
            if re.match(pattern, locator):
                matched = True
                
                # Platform-specific validation
                if platform == 'Android' and strategy in ['ios_predicate', 'ios_class_chain']:
                    errors.append(f"iOS-only locator strategy '{strategy}' used for Android")
                elif platform == 'iOS' and strategy in ['android_uiautomator', 'android_viewtag', 'android_datamatcher']:
                    errors.append(f"Android-only locator strategy '{strategy}' used for iOS")
                    
                # Validate specific strategies
                if strategy == 'android_uiautomator':
                    if not self._validate_uiautomator_syntax(locator):
                        errors.append("Invalid UiAutomator syntax")
                elif strategy == 'android_datamatcher':
                    if not self._validate_datamatcher_syntax(locator):
                        errors.append("Invalid data matcher JSON syntax")
                elif strategy == 'ios_predicate':
                    if not self._validate_predicate_syntax(locator):
                        errors.append("Invalid iOS predicate syntax")
                        
                break
                
        if not matched:
            errors.append(f"Locator does not match any known mobile strategy: {locator}")
            
        return len(errors) == 0, errors
    
    def _validate_uiautomator_syntax(self, locator: str) -> bool:
        """Validate Android UiAutomator syntax."""
        match = re.match(r'^android=(.+)$', locator)
        if match:
            selector = match.group(1)
            # Basic validation - check for new UiSelector()
            return 'new UiSelector()' in selector
        return False
    
    def _validate_datamatcher_syntax(self, locator: str) -> bool:
        """Validate Android data matcher JSON syntax."""
        match = re.match(r'^android_datamatcher=(.+)$', locator)
        if match:
            json_str = match.group(1)
            try:
                json.loads(json_str)
                return True
            except json.JSONDecodeError:
                return False
        return False
    
    def _validate_predicate_syntax(self, locator: str) -> bool:
        """Validate iOS predicate syntax."""
        match = re.match(r'^-ios predicate string:(.+)$', locator)
        if match:
            predicate = match.group(1)
            # Basic validation - check for common operators
            return any(op in predicate for op in ['==', '!=', 'CONTAINS', 'BEGINSWITH', 'ENDSWITH', 'MATCHES'])
        return False
    
    def convert_mobile_locator(self, locator: str, from_platform: str, to_platform: str) -> str:
        """
        Convert mobile locator between platforms if possible.
        
        Args:
            locator: Locator to convert
            from_platform: Source platform ('Android' or 'iOS')
            to_platform: Target platform ('Android' or 'iOS')
            
        Returns:
            Converted locator or original if conversion not possible
        """
        if from_platform == to_platform:
            return locator
            
        # Try to convert common strategies
        if locator.startswith('id='):
            # ID works on both platforms
            return locator
        elif locator.startswith('class='):
            # Class names are different between platforms
            class_name = locator[6:]
            if from_platform == 'Android' and to_platform == 'iOS':
                # Convert Android class to iOS
                conversions = {
                    'android.widget.Button': 'XCUIElementTypeButton',
                    'android.widget.TextView': 'XCUIElementTypeStaticText',
                    'android.widget.EditText': 'XCUIElementTypeTextField',
                    'android.widget.ImageView': 'XCUIElementTypeImage'
                }
                for android_class, ios_class in conversions.items():
                    if android_class in class_name:
                        return f'class={class_name.replace(android_class, ios_class)}'
            elif from_platform == 'iOS' and to_platform == 'Android':
                # Convert iOS class to Android
                conversions = {
                    'XCUIElementTypeButton': 'android.widget.Button',
                    'XCUIElementTypeStaticText': 'android.widget.TextView',
                    'XCUIElementTypeTextField': 'android.widget.EditText',
                    'XCUIElementTypeImage': 'android.widget.ImageView'
                }
                for ios_class, android_class in conversions.items():
                    if ios_class in class_name:
                        return f'class={class_name.replace(ios_class, android_class)}'
        elif locator.startswith('accessibility_id='):
            # Accessibility ID works on both platforms
            return locator
            
        # Platform-specific strategies cannot be converted
        logger.warning(f"Cannot convert platform-specific locator from {from_platform} to {to_platform}: {locator}")
        return locator
    
    def get_supported_strategies(self, library_type: str) -> List[str]:
        """
        Get list of supported locator strategies for a library.
        
        Args:
            library_type: Library type ("Browser", "SeleniumLibrary", or "AppiumLibrary")
            
        Returns:
            list: List of supported strategy names
        """
        if library_type == "Browser":
            return [
                "css", "xpath", "text", "id", "cascaded", "shadow"
            ]
        elif library_type == "SeleniumLibrary":
            return [
                "id", "name", "identifier", "class", "tag", "xpath", 
                "css", "dom", "link", "partial link", "data", "jquery"
            ]
        elif library_type == "AppiumLibrary":
            return [
                "accessibility_id", "id", "class", "xpath",
                "android_uiautomator", "android_viewtag", "android_datamatcher",
                "ios_predicate", "ios_class_chain", "image"
            ]
        else:
            return []