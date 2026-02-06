"""Robot Framework Context Bridge.

Provides integration with Robot Framework execution context:
- Variable access and resolution
- Library discovery
- Keyword execution
- Logging integration
- MCP services integration for rich context
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from robot.api import logger as rf_logger

# Import Robot Framework components
try:
    from robot.libraries.BuiltIn import BuiltIn

    ROBOT_AVAILABLE = True
except ImportError:
    BuiltIn = None
    ROBOT_AVAILABLE = False

# Import MCP adapter for rich context
from robotmcp.lib.mcp_adapter import get_mcp_adapter

logger = logging.getLogger(__name__)


class RFContextBridge:
    """Bridge between AILibrary and Robot Framework execution context."""

    def __init__(self):
        """Initialize the RF context bridge."""
        self._builtin: Optional[BuiltIn] = None
        self._keyword_cache: Dict[str, List[str]] = {}
        self._mcp_adapter = get_mcp_adapter()

    @property
    def builtin(self) -> BuiltIn:
        """Get BuiltIn library instance (lazy initialization)."""
        if self._builtin is None:
            if not ROBOT_AVAILABLE:
                raise RuntimeError(
                    "Robot Framework is not available. "
                    "AILibrary must be used within a Robot Framework test."
                )
            self._builtin = BuiltIn()
        return self._builtin

    def get_variables(self) -> Dict[str, Any]:
        """Get all variables in current scope.

        Returns:
            Dictionary of all RF variables in current scope.
        """
        try:
            return self.builtin.get_variables()
        except Exception as e:
            logger.warning(f"Failed to get variables: {e}")
            return {}

    def get_variable(self, name: str, default: Any = None) -> Any:
        """Get a specific variable value.

        Args:
            name: Variable name (with or without ${} syntax)
            default: Default value if variable not found

        Returns:
            Variable value or default
        """
        try:
            # Normalize variable name
            if not name.startswith("${"):
                name = f"${{{name}}}"
            return self.builtin.get_variable_value(name, default)
        except Exception as e:
            logger.debug(f"Variable {name} not found: {e}")
            return default

    def set_variable(self, name: str, value: Any, scope: str = "test") -> None:
        """Set a variable in the specified scope.

        Args:
            name: Variable name
            value: Variable value
            scope: Variable scope ('test', 'suite', 'global')
        """
        # Normalize variable name
        if not name.startswith("${"):
            name = f"${{{name}}}"

        if scope == "test":
            self.builtin.set_test_variable(name, value)
        elif scope == "suite":
            self.builtin.set_suite_variable(name, value)
        elif scope == "global":
            self.builtin.set_global_variable(name, value)
        else:
            raise ValueError(f"Invalid scope: {scope}. Use 'test', 'suite', or 'global'")

    def resolve_variables(self, text: str) -> str:
        """Resolve RF variables in text.

        Args:
            text: Text containing RF variable references

        Returns:
            Text with variables resolved
        """
        try:
            return self.builtin.replace_variables(text)
        except Exception as e:
            logger.debug(f"Variable resolution failed: {e}")
            return text

    def get_available_keywords(self) -> List[Dict[str, Any]]:
        """Get all keywords from imported libraries.

        Uses MCP adapter for rich keyword context when available,
        falls back to basic BuiltIn discovery otherwise.

        Returns:
            List of keyword info dictionaries with name, library, args, and doc.
        """
        keywords = []

        # Try MCP adapter first for rich context (includes args, types, descriptions)
        try:
            if self._mcp_adapter:
                mcp_keywords = self._mcp_adapter.get_available_keywords_with_docs(limit=100)
                if mcp_keywords:
                    keywords = [
                        {
                            "name": kw.name,
                            "library": kw.library,
                            "args": kw.args,
                            "documentation": kw.short_doc,
                        }
                        for kw in mcp_keywords
                    ]
                    self._keyword_cache = {kw["name"].lower(): kw for kw in keywords}
                    return keywords
        except Exception as e:
            logger.debug(f"MCP keyword discovery failed, using fallback: {e}")

        # Fallback to basic BuiltIn discovery
        try:
            # Get all imported libraries
            libraries = self.builtin.get_library_instance(all=True)

            for lib_name, lib_instance in libraries.items():
                lib_keywords = self._get_library_keywords(lib_name, lib_instance)
                keywords.extend(lib_keywords)

            # Cache for performance
            self._keyword_cache = {kw["name"].lower(): kw for kw in keywords}

        except Exception as e:
            logger.warning(f"Failed to discover keywords: {e}")

        return keywords

    def _get_library_keywords(
        self, lib_name: str, lib_instance: Any
    ) -> List[Dict[str, Any]]:
        """Extract keywords from a library instance.

        Args:
            lib_name: Library name
            lib_instance: Library instance

        Returns:
            List of keyword dictionaries
        """
        keywords = []

        try:
            # Try to get keywords from the library
            if hasattr(lib_instance, "get_keyword_names"):
                keyword_names = lib_instance.get_keyword_names()
            elif hasattr(lib_instance, "keywords"):
                keyword_names = list(lib_instance.keywords.keys())
            else:
                # Fallback: inspect public methods
                keyword_names = [
                    name
                    for name in dir(lib_instance)
                    if not name.startswith("_") and callable(getattr(lib_instance, name))
                ]

            for kw_name in keyword_names:
                doc = self._get_keyword_doc(lib_instance, kw_name)
                keywords.append(
                    {
                        "name": kw_name,
                        "library": lib_name,
                        "documentation": doc,
                    }
                )

        except Exception as e:
            logger.debug(f"Failed to extract keywords from {lib_name}: {e}")

        return keywords

    def _get_keyword_doc(self, lib_instance: Any, keyword_name: str) -> str:
        """Get documentation for a keyword.

        Args:
            lib_instance: Library instance
            keyword_name: Keyword name

        Returns:
            Keyword documentation string
        """
        try:
            if hasattr(lib_instance, "get_keyword_documentation"):
                return lib_instance.get_keyword_documentation(keyword_name)
            else:
                method = getattr(lib_instance, keyword_name, None)
                if method and hasattr(method, "__doc__"):
                    return method.__doc__ or ""
        except Exception:
            pass
        return ""

    def run_keyword(self, keyword: str, *args, **kwargs) -> Any:
        """Execute a Robot Framework keyword.

        Args:
            keyword: Keyword name
            *args: Positional arguments
            **kwargs: Named arguments

        Returns:
            Keyword return value
        """
        try:
            # Convert kwargs to RF format (name=value pairs as additional args)
            all_args = list(args)
            for key, value in kwargs.items():
                all_args.append(f"{key}={value}")

            return self.builtin.run_keyword(keyword, *all_args)
        except Exception as e:
            logger.error(f"Keyword execution failed: {keyword} - {e}")
            raise

    def run_keyword_and_return_status(
        self, keyword: str, *args, **kwargs
    ) -> Tuple[bool, Any, Optional[str]]:
        """Execute a keyword and return status information.

        Args:
            keyword: Keyword name
            *args: Positional arguments
            **kwargs: Named arguments

        Returns:
            Tuple of (success, result, error_message)
        """
        try:
            result = self.run_keyword(keyword, *args, **kwargs)
            return (True, result, None)
        except Exception as e:
            return (False, None, str(e))

    def log(self, message: str, level: str = "INFO") -> None:
        """Log a message to Robot Framework log.

        Args:
            message: Message to log
            level: Log level (DEBUG, INFO, WARN, ERROR)
        """
        level = level.upper()
        if level == "DEBUG":
            rf_logger.debug(message)
        elif level == "INFO":
            rf_logger.info(message)
        elif level == "WARN":
            rf_logger.warn(message)
        elif level == "ERROR":
            rf_logger.error(message)
        else:
            rf_logger.info(message)

    def get_library_instance(self, name: str) -> Optional[Any]:
        """Get a library instance by name.

        Args:
            name: Library name

        Returns:
            Library instance or None
        """
        try:
            return self.builtin.get_library_instance(name)
        except Exception as e:
            logger.debug(f"Failed to get library {name}: {e}")
            return None

    def import_library(self, name: str, *args, **kwargs) -> None:
        """Import a library dynamically.

        Args:
            name: Library name or path
            *args: Library arguments
            **kwargs: Library keyword arguments
        """
        try:
            self.builtin.import_library(name, *args, **kwargs)
        except Exception as e:
            logger.error(f"Failed to import library {name}: {e}")
            raise

    def get_current_test_name(self) -> Optional[str]:
        """Get the name of the currently executing test.

        Returns:
            Test name or None
        """
        try:
            return self.builtin.get_variable_value("${TEST_NAME}")
        except Exception:
            return None

    def get_current_suite_name(self) -> Optional[str]:
        """Get the name of the currently executing suite.

        Returns:
            Suite name or None
        """
        try:
            return self.builtin.get_variable_value("${SUITE_NAME}")
        except Exception:
            return None

    def find_keyword(self, keyword_name: str) -> Optional[Dict[str, Any]]:
        """Find keyword info by name.

        Args:
            keyword_name: Keyword name to find

        Returns:
            Keyword info dictionary or None
        """
        if not self._keyword_cache:
            self.get_available_keywords()

        return self._keyword_cache.get(keyword_name.lower())

    def record_step(
        self,
        keyword: str,
        args: List[str],
        success: bool,
        result: Any = None,
        error: Optional[str] = None,
    ) -> None:
        """Record an executed step for history tracking.

        Args:
            keyword: Keyword name that was executed
            args: Arguments passed to the keyword
            success: Whether execution succeeded
            result: Result of the execution (if any)
            error: Error message (if failed)
        """
        if self._mcp_adapter:
            self._mcp_adapter.record_step(keyword, args, success, result, error)

    def get_execution_history(self, limit: int = 5) -> str:
        """Get formatted execution history for AI context.

        Args:
            limit: Maximum number of steps to include

        Returns:
            Formatted string with recent execution history
        """
        if self._mcp_adapter:
            return self._mcp_adapter.format_history_for_context(limit)
        return ""

    def get_rich_context(
        self,
        include_keywords: bool = True,
        include_history: bool = True,
        keyword_limit: int = 30,
        history_limit: int = 5,
    ) -> Dict[str, Any]:
        """Get rich context for AI prompts including page state, keywords, and history.

        Args:
            include_keywords: Whether to include available keywords
            include_history: Whether to include execution history
            keyword_limit: Max keywords to include
            history_limit: Max history steps to include

        Returns:
            Dictionary with all context information for AI
        """
        # Get page state
        context = self.get_page_state()

        # Add rich context from MCP adapter
        if self._mcp_adapter:
            mcp_context = self._mcp_adapter.get_rich_context(
                page_state=context,
                include_keywords=include_keywords,
                include_history=include_history,
                keyword_limit=keyword_limit,
                history_limit=history_limit,
            )
            context.update(mcp_context)

        return context

    def get_page_state(self) -> Dict[str, Any]:
        """Get current page state for web automation context.

        Returns:
            Dictionary with page state information including element selectors
        """
        state = {}

        # Try Browser Library
        try:
            browser = self.get_library_instance("Browser")
            if browser:
                state["library"] = "Browser"
                try:
                    state["url"] = self.run_keyword("Get Url")
                except Exception:
                    pass
                try:
                    state["title"] = self.run_keyword("Get Title")
                except Exception:
                    pass
                # Get page element context for better AI prompts
                try:
                    elements = self._get_browser_elements()
                    state["elements"] = elements
                    if elements:
                        rf_logger.info(f"Page elements found:\n{elements[:500]}...")
                    else:
                        rf_logger.info("No page elements found")
                except Exception as e:
                    logger.debug(f"Could not get page elements: {e}")
                    rf_logger.warn(f"Could not get page elements: {e}")
                return state
        except Exception:
            pass

        # Try SeleniumLibrary
        try:
            selenium = self.get_library_instance("SeleniumLibrary")
            if selenium:
                state["library"] = "SeleniumLibrary"
                try:
                    state["url"] = self.run_keyword("Get Location")
                except Exception:
                    pass
                try:
                    state["title"] = self.run_keyword("Get Title")
                except Exception:
                    pass
                return state
        except Exception:
            pass

        return state

    def _get_browser_elements(self) -> str:
        """Get key interactive elements from the page for AI context.

        Uses the existing PageSourceService.extract_page_context() for comprehensive
        element extraction when available, with fallback to direct HTML parsing.

        Returns:
            String description of page elements with their selectors
        """
        try:
            # Get page source
            html = self.run_keyword("Get Page Source")
            if not html:
                rf_logger.debug("No HTML from Get Page Source")
                return ""

            rf_logger.debug(f"Got HTML source, length: {len(html)}")

            # Try to use the existing PageSourceService for comprehensive parsing
            try:
                from robotmcp.components.execution.page_source_service import PageSourceService
                import asyncio

                service = PageSourceService()
                # Run async method synchronously
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    context = loop.run_until_complete(service.extract_page_context(html))
                finally:
                    loop.close()

                if context:
                    formatted = self._format_context_for_ai(context)
                    rf_logger.debug(f"PageSourceService returned: {formatted[:200] if formatted else 'empty'}")
                    if formatted:
                        return formatted
            except ImportError:
                rf_logger.debug("PageSourceService not available, using fallback parsing")
            except Exception as e:
                rf_logger.debug(f"PageSourceService failed: {e}, using fallback")

            # Fallback: basic HTML parsing for essential elements
            result = self._parse_elements_basic(html)
            rf_logger.debug(f"Fallback parser returned: {result[:200] if result else 'empty'}")
            return result

        except Exception as e:
            rf_logger.debug(f"Error extracting elements: {e}")
            return ""

    def _format_context_for_ai(self, context: Dict[str, Any]) -> str:
        """Format extracted page context into a string for AI prompts.

        Uses category-based budgets to ensure all element types are represented,
        even on complex pages with many elements. Deduplicates similar elements
        to reduce noise while preserving useful selectors.

        Args:
            context: Dictionary from PageSourceService.extract_page_context()

        Returns:
            Formatted string with element selectors
        """
        # Category-based budgets ensure each element type gets representation
        # even on complex pages with 100+ elements
        BUDGET = {
            "links": 12,        # Navigation is critical (cart, checkout, menu)
            "inputs": 12,       # Form inputs for data entry
            "buttons": 12,      # Action buttons
            "text_elements": 14 # Product info, labels, messages
        }
        # Total max elements: 12+12+12+14 = 50

        elements = []
        seen_selectors = set()  # Deduplicate identical selectors

        def add_element(desc: str, selector: str) -> bool:
            """Add element if not duplicate. Returns True if added."""
            if selector in seen_selectors:
                return False
            seen_selectors.add(selector)
            elements.append(f"  - {desc}: {selector}")
            return True

        # === LINKS (Critical for navigation: cart, checkout, menu items) ===
        link_count = 0
        for link in context.get("links", []):
            if link_count >= BUDGET["links"]:
                break

            link_text = link.get("text", "")
            link_id = link.get("id", "")
            data_test = link.get("data_test", "")
            link_class = link.get("class", "")

            # Build selector with priority: data-test > id > class > text
            if data_test:
                selector = f"[data-test='{data_test}']"
            elif link_id:
                selector = f"#{link_id}"
            elif link_class:
                first_class = link_class.split()[0] if link_class else ""
                if first_class:
                    selector = f".{first_class}"
                else:
                    continue
            elif link_text:
                selector = f"text={link_text}"
            else:
                continue

            # Build description
            if link_text:
                desc = f"link '{link_text}'"
            elif data_test:
                readable = data_test.replace('-', ' ').replace('_', ' ')
                desc = f"link ({readable})"
            elif link_class:
                readable = link_class.split()[0].replace('_', ' ').replace('-', ' ')
                desc = f"link ({readable})"
            else:
                desc = "link"

            if add_element(desc, selector):
                link_count += 1

        # === INPUTS (Form fields for data entry) ===
        input_count = 0

        # Process form inputs first
        for form in context.get("forms", []):
            if input_count >= BUDGET["inputs"]:
                break
            for inp in form.get("inputs", []):
                if input_count >= BUDGET["inputs"]:
                    break

                inp_type = inp.get("type", "text")
                inp_id = inp.get("id", "")
                inp_name = inp.get("name", "")
                placeholder = inp.get("placeholder", "")
                data_test = inp.get("data_test", "")

                if data_test:
                    selector = f"[data-test='{data_test}']"
                elif inp_id:
                    selector = f"#{inp_id}"
                elif inp_name:
                    selector = f"[name='{inp_name}']"
                else:
                    continue

                desc = f"input[type={inp_type}]"
                if placeholder:
                    desc += f" '{placeholder}'"

                if add_element(desc, selector):
                    input_count += 1

        # Process standalone inputs
        for inp in context.get("inputs", []):
            if input_count >= BUDGET["inputs"]:
                break

            inp_type = inp.get("type", "text")
            inp_id = inp.get("id", "")
            inp_name = inp.get("name", "")
            placeholder = inp.get("placeholder", "")
            data_test = inp.get("data_test", "")

            if data_test:
                selector = f"[data-test='{data_test}']"
            elif inp_id:
                selector = f"#{inp_id}"
            elif inp_name:
                selector = f"[name='{inp_name}']"
            else:
                continue

            desc = f"input[type={inp_type}]"
            if placeholder:
                desc += f" '{placeholder}'"

            if add_element(desc, selector):
                input_count += 1

        # === BUTTONS (Action triggers) ===
        button_count = 0
        for btn in context.get("buttons", []):
            if button_count >= BUDGET["buttons"]:
                break

            btn_text = btn.get("text", "")
            btn_id = btn.get("id", "")
            data_test = btn.get("data_test", "")

            if data_test:
                selector = f"[data-test='{data_test}']"
            elif btn_id:
                selector = f"#{btn_id}"
            elif btn_text:
                selector = f"text={btn_text}"
            else:
                continue

            if add_element(f"button '{btn_text}'", selector):
                button_count += 1

        # === TEXT ELEMENTS (Product info, prices, labels, messages) ===
        text_count = 0
        for text_elem in context.get("text_elements", []):
            if text_count >= BUDGET["text_elements"]:
                break

            text_content = text_elem.get("text", "")
            css_class = text_elem.get("class", "")
            data_test = text_elem.get("data_test", "")
            elem_type = text_elem.get("type", "text")

            if data_test:
                selector = f"[data-test='{data_test}']"
            elif css_class:
                selector = f".{css_class}"
            else:
                continue

            display_text = text_content[:40] + "..." if len(text_content) > 40 else text_content
            if add_element(f"{elem_type} '{display_text}'", selector):
                text_count += 1

        if elements:
            return "Page elements:\n" + "\n".join(elements)
        return ""

    def _parse_elements_basic(self, html: str) -> str:
        """Basic HTML parsing fallback when PageSourceService is not available.

        Extracts key interactive elements and text content with their selectors.
        For elements that appear multiple times (like product names), provides
        indexed selectors using >> nth=N notation.

        Args:
            html: Raw HTML source

        Returns:
            String with basic element selectors and text content
        """
        import re

        elements = []

        # Extract input elements with data-test (highest priority for SauceDemo-like sites)
        data_test_pattern = r'<input[^>]*data-test=["\']([^"\']*)["\'][^>]*>'
        for match in re.finditer(data_test_pattern, html, re.IGNORECASE):
            full_match = match.group(0)
            data_test = match.group(1)
            type_match = re.search(r'type=["\']([^"\']*)["\']', full_match, re.IGNORECASE)
            input_type = type_match.group(1) if type_match else "text"
            selector = f"[data-test='{data_test}']"
            elements.append(f"  - input[type={input_type}]: {selector}")

        # Extract input elements with IDs
        id_pattern = r'<input[^>]*id=["\']([^"\']*)["\'][^>]*>'
        for match in re.finditer(id_pattern, html, re.IGNORECASE):
            full_match = match.group(0)
            elem_id = match.group(1)
            # Skip if already found via data-test
            if f"#{elem_id}" in str(elements):
                continue
            type_match = re.search(r'type=["\']([^"\']*)["\']', full_match, re.IGNORECASE)
            input_type = type_match.group(1) if type_match else "text"
            selector = f"#{elem_id}"
            elements.append(f"  - input[type={input_type}]: {selector}")

        # Extract buttons - scan for data-test, id in any order
        button_pattern = r'<(?:button|input[^>]*type=["\']submit["\'])[^>]*>([^<]*)?'
        for match in re.finditer(button_pattern, html, re.IGNORECASE):
            full_tag = match.group(0)
            btn_text = (match.group(1) or "").strip()

            # Look for data-test first (highest priority)
            data_test_match = re.search(r'data-test=["\']([^"\']*)["\']', full_tag, re.IGNORECASE)
            id_match = re.search(r'\bid=["\']([^"\']*)["\']', full_tag, re.IGNORECASE)

            if data_test_match:
                selector = f"[data-test='{data_test_match.group(1)}']"
            elif id_match:
                selector = f"#{id_match.group(1)}"
            else:
                continue

            elements.append(f"  - button '{btn_text}': {selector}")

        # Extract anchor/link elements - scan for data-test, id, or class in any order
        anchor_pattern = r'<a[^>]*>([^<]*)?'
        for match in re.finditer(anchor_pattern, html, re.IGNORECASE):
            full_tag = match.group(0)
            link_text = (match.group(1) or "").strip()

            # Look for data-test first (highest priority)
            data_test_match = re.search(r'data-test=["\']([^"\']*)["\']', full_tag, re.IGNORECASE)
            id_match = re.search(r'\bid=["\']([^"\']*)["\']', full_tag, re.IGNORECASE)
            class_match = re.search(r'\bclass=["\']([^"\']*)["\']', full_tag, re.IGNORECASE)

            if data_test_match:
                selector = f"[data-test='{data_test_match.group(1)}']"
            elif id_match:
                selector = f"#{id_match.group(1)}"
            elif class_match:
                first_class = class_match.group(1).split()[0]
                if first_class:
                    selector = f".{first_class}"
                else:
                    continue
            else:
                continue

            desc = f"link '{link_text}'" if link_text else "link"
            elements.append(f"  - {desc}: {selector}")

        # Extract text elements (headings, divs, spans) with data-test attributes
        # These often contain confirmation messages, error messages, etc.
        text_element_pattern = r'<(h[1-6]|div|span|p)[^>]*data-test=["\']([^"\']*)["\'][^>]*>([^<]*)?'
        for match in re.finditer(text_element_pattern, html, re.IGNORECASE):
            tag = match.group(1)
            data_test = match.group(2)
            text_content = (match.group(3) or "").strip()
            selector = f"[data-test='{data_test}']"
            desc = f"text '{text_content[:30]}'" if text_content else f"{tag}"
            elements.append(f"  - {desc}: {selector}")

        # ============================================================
        # ENHANCED: Extract elements with common e-commerce patterns
        # Extract ACTUAL TEXT CONTENT and provide INDEXED selectors
        # ============================================================

        # Pattern definitions: (class_pattern, element_type_description, ordinal_prefix)
        text_class_patterns = [
            # SauceDemo specific patterns
            (r"inventory_item_name", "product name"),
            (r"inventory_item_price", "product price"),
            (r"inventory_item_desc", "product description"),
            (r"cart_quantity", "cart quantity"),
            (r"cart_item_label", "cart item"),
            # Generic e-commerce patterns
            (r"product-name", "product name"),
            (r"product-price", "product price"),
            (r"product-title", "product title"),
            (r"item-name", "item name"),
            (r"item-price", "item price"),
            (r"price", "price"),
            # Shopping cart patterns
            (r"cart-item", "cart item"),
            (r"checkout-item", "checkout item"),
        ]

        # Track elements by class for indexed output
        for class_pattern, elem_type in text_class_patterns:
            # Match elements containing this class with their text content
            # Pattern matches: <div class="...inventory_item_name...">TEXT</div>
            # Also matches nested content like <div class="x"><a>TEXT</a></div>
            pattern = rf'<(?:div|span|p|a)[^>]*class=["\'][^"\']*{class_pattern}[^"\']*["\'][^>]*>(.*?)</(?:div|span|p|a)>'
            matches = list(re.finditer(pattern, html, re.IGNORECASE | re.DOTALL))

            if matches:
                # Ordinal labels for readability
                ordinals = ["first", "second", "third", "fourth", "fifth", "sixth"]

                for idx, match in enumerate(matches[:6]):  # Limit to 6 items
                    raw_content = match.group(1)
                    # Clean up the text content (remove nested tags, excess whitespace)
                    text_content = re.sub(r'<[^>]+>', '', raw_content).strip()
                    text_content = ' '.join(text_content.split())  # Normalize whitespace

                    if text_content:
                        # Use ordinal for first few, then numeric index
                        if idx < len(ordinals):
                            ordinal = ordinals[idx]
                        else:
                            ordinal = f"item {idx + 1}"

                        # Truncate long text
                        display_text = text_content[:35] + "..." if len(text_content) > 35 else text_content

                        # Provide selector with nth index
                        selector = f".{class_pattern} >> nth={idx}"
                        elements.append(f"  - {ordinal} {elem_type} '{display_text}': {selector}")

        # Also extract elements by data-test attribute with text content
        data_test_text_pattern = r'<(?:div|span|p|a|button)[^>]*data-test=["\']([^"\']+)["\'][^>]*>(.*?)</(?:div|span|p|a|button)>'
        seen_data_tests = set()
        for match in re.finditer(data_test_text_pattern, html, re.IGNORECASE | re.DOTALL):
            data_test = match.group(1)
            if data_test in seen_data_tests:
                continue
            seen_data_tests.add(data_test)

            raw_content = match.group(2)
            text_content = re.sub(r'<[^>]+>', '', raw_content).strip()
            text_content = ' '.join(text_content.split())

            if text_content and len(text_content) < 100:  # Skip very long content
                display_text = text_content[:35] + "..." if len(text_content) > 35 else text_content
                selector = f"[data-test='{data_test}']"

                # Determine element type from data-test name
                if "name" in data_test.lower():
                    elem_type = "name"
                elif "price" in data_test.lower():
                    elem_type = "price"
                elif "desc" in data_test.lower():
                    elem_type = "description"
                elif "title" in data_test.lower():
                    elem_type = "title"
                elif "error" in data_test.lower():
                    elem_type = "error"
                elif "message" in data_test.lower():
                    elem_type = "message"
                else:
                    elem_type = "text"

                elements.append(f"  - {elem_type} '{display_text}': {selector}")

        if elements:
            return "Page elements:\n" + "\n".join(elements[:40])
        return ""
