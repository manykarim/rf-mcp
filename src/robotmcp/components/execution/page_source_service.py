"""Page source management and filtering service for web and mobile."""

import logging
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Union

from robotmcp.config import library_registry
from robotmcp.models.session_models import ExecutionSession, PlatformType
from robotmcp.models.config_models import ExecutionConfig
from robotmcp.plugins import get_library_plugin_manager

logger = logging.getLogger(__name__)

# Import BeautifulSoup for DOM filtering
try:
    from bs4 import BeautifulSoup, Comment
    BS4_AVAILABLE = True
except ImportError:
    BeautifulSoup = None
    Comment = None
    BS4_AVAILABLE = False


class PageSourceService:
    """Manages page source retrieval, filtering, and context extraction for web and mobile."""

    # Library-to-keyword mappings — verified against libdoc catalogs.
    # Browser Library (Playwright), SeleniumLibrary (Selenium), AppiumLibrary (Appium).
    _SOURCE_KEYWORDS: Dict[str, Optional[str]] = {
        "Browser": "Get Page Source",
        "SeleniumLibrary": "Get Source",
        "AppiumLibrary": "Get Source",
        "PlatynUI": None,
        "PlatynUI.BareMetal": None,
    }
    _URL_KEYWORDS: Dict[str, Optional[str]] = {
        "Browser": "Get Url",
        "SeleniumLibrary": "Get Location",
        "AppiumLibrary": "Get Window Url",
        "PlatynUI": None,
        "PlatynUI.BareMetal": None,
    }
    _TITLE_KEYWORDS: Dict[str, Optional[str]] = {
        "Browser": "Get Title",
        "SeleniumLibrary": "Get Title",
        "AppiumLibrary": "Get Window Title",
        "PlatynUI": None,
        "PlatynUI.BareMetal": None,
    }

    @staticmethod
    def _keyword_candidates(
        mapping: Dict[str, str],
        imported: list,
    ) -> tuple:
        """Return an ordered tuple of keyword candidates based on imported libraries.

        Picks only keywords whose library is actually imported, avoiding
        ERROR-level log noise from attempting keywords that don't exist in
        the session.  Falls back to the full candidate list when no library
        match is found (backwards compatibility).
        """
        candidates: list = []
        seen: set = set()
        for lib in imported:
            kw = mapping.get(lib)
            if kw and kw not in seen:
                candidates.append(kw)
                seen.add(kw)
        if candidates:
            return tuple(candidates)
        # Fallback: return all unique non-None keywords from the mapping
        return tuple(dict.fromkeys(v for v in mapping.values() if v))

    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()

    def _get_page_source_via_rf_context(self, session: ExecutionSession) -> str:
        """Attempt to fetch page source through the Robot Framework native context.

        IMPORTANT: This is a blocking call that enters _suppress_stdout().
        Callers from async code MUST use asyncio.to_thread() to avoid:
        1. Blocking the asyncio event loop
        2. Racing with concurrent _suppress_stdout() usage (fd 1 redirect)
        """
        page_source = ""
        try:
            from robotmcp.components.execution.rf_native_context_manager import (
                get_rf_native_context_manager,
            )

            rf_mgr = get_rf_native_context_manager()
            # Select library-appropriate keywords to avoid log noise
            imported = getattr(session, "imported_libraries", []) or []
            candidates = self._keyword_candidates(self._SOURCE_KEYWORDS, imported)
            logger.info(
                "Page source retrieval for session %s: imported=%s, candidates=%s",
                session.session_id, imported, candidates,
            )

            for kw in candidates:
                res = rf_mgr.execute_keyword_with_context(
                    session_id=session.session_id,
                    keyword_name=kw,
                    arguments=[],
                    assign_to=None,
                    session_variables=dict(session.variables),
                )
                if res and res.get("success"):
                    out = res.get("output") or res.get("result")
                    if isinstance(out, str) and out:
                        page_source = out
                        session.browser_state.page_source = out
                        break
                    else:
                        logger.warning(
                            "Page source keyword '%s' succeeded but returned "
                            "empty/non-string output: type=%s, output=%r, result=%r",
                            kw, type(res.get("result")).__name__,
                            str(res.get("output", ""))[:100],
                            str(res.get("result", ""))[:100],
                        )
                else:
                    logger.warning(
                        "Page source keyword '%s' failed for session %s: %s",
                        kw, session.session_id,
                        res.get("error", "unknown") if res else "no response",
                    )

            # Update URL and title where possible
            for url_kw in self._keyword_candidates(self._URL_KEYWORDS, imported):
                try:
                    res = rf_mgr.execute_keyword_with_context(
                        session_id=session.session_id,
                        keyword_name=url_kw,
                        arguments=[],
                    )
                    if res and res.get("success") and res.get("output"):
                        session.browser_state.current_url = res.get("output")
                        break
                except Exception:
                    pass

            for title_kw in self._keyword_candidates(self._TITLE_KEYWORDS, imported):
                try:
                    res = rf_mgr.execute_keyword_with_context(
                        session_id=session.session_id,
                        keyword_name=title_kw,
                        arguments=[],
                    )
                    if res and res.get("success") and res.get("output"):
                        session.browser_state.page_title = res.get("output")
                        break
                except Exception:
                    pass
        except Exception as exc:
            logger.warning("RF-context page source retrieval failed: %s", exc)
            page_source = session.browser_state.page_source or ""

        return page_source

    async def _get_page_source_from_plugin(
        self,
        session: ExecutionSession,
        *,
        browser_library_manager: Any,
        full_source: bool,
        filtered: bool,
        filtering_level: str,
        include_reduced_dom: bool,
    ) -> Optional[Dict[str, Any]]:
        """Attempt to retrieve page source via a plugin-provided state provider."""
        try:
            # Ensure plugin registry is initialized before lookup
            library_registry.get_all_libraries()
            plugin_manager = get_library_plugin_manager()
            active_library = (
                session.get_web_automation_library()
                or session.get_active_library()
            )
            if not active_library:
                return None

            provider = plugin_manager.get_state_provider(active_library)
            if not provider:
                return None

            return await provider.get_page_source(
                session,
                full_source=full_source,
                filtered=filtered,
                filtering_level=filtering_level,
                include_reduced_dom=include_reduced_dom,
                service=self,
                browser_library_manager=browser_library_manager,
            )
        except Exception as exc:
            logger.warning(
                "Plugin page source provider failed for %s: %s",
                getattr(session, "session_id", "unknown"),
                exc,
            )
            return None

    def filter_page_source(self, html: str, filtering_level: str = "standard") -> str:
        """
        Filter HTML page source to keep only automation-relevant content.
        
        Removes scripts, styles, metadata, and other elements that are not useful
        for web automation, making the DOM tree cleaner and more focused.
        
        Args:
            html: Raw HTML source code
            filtering_level: Filtering intensity ('minimal', 'standard', 'aggressive')
            
        Returns:
            str: Filtered HTML with only automation-relevant elements
        """
        if not html or not BS4_AVAILABLE:
            return html
            
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Define filtering rules based on level
            if filtering_level == "minimal":
                elements_to_remove = ['script', 'style']
                attributes_to_remove = ['onclick', 'onload', 'onmouseover']
                remove_comments = True
                simplify_head = False
                
            elif filtering_level == "aggressive":
                elements_to_remove = [
                    'script', 'style', 'svg', 'noscript', 'meta', 'link', 
                    'video', 'audio', 'embed', 'object', 'canvas'
                ]
                attributes_to_remove = [
                    'style', 'onclick', 'onload', 'onmouseover', 'onmouseout',
                    'onfocus', 'onblur', 'onchange', 'onsubmit', 'ondblclick',
                    'onkeydown', 'onkeyup', 'onkeypress'
                ]
                remove_comments = True
                simplify_head = True
                
            else:  # standard (default)
                elements_to_remove = [
                    'script', 'style', 'noscript', 'svg', 'meta', 'link', 
                    'embed', 'object'
                ]
                attributes_to_remove = [
                    'style', 'onclick', 'onload', 'onmouseover', 'onmouseout',
                    'onfocus', 'onblur', 'onchange', 'onsubmit'
                ]
                remove_comments = True
                simplify_head = True
            
            # Remove unwanted elements completely
            for tag_name in elements_to_remove:
                for element in soup.find_all(tag_name):
                    element.decompose()
            
            # Remove HTML comments
            if remove_comments:
                for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                    comment.extract()
            
            # Simplify head section - keep only title
            if simplify_head:
                head = soup.find('head')
                if head:
                    title = head.find('title')
                    head.clear()
                    if title:
                        head.append(title)
            
            # For "standard" filtering, keep non-visible and hidden elements
            # This allows automation tools to interact with elements that might be shown/hidden dynamically
            # Only remove elements that are truly non-interactive (this logic moved to aggressive filtering)
            
            # Note: Non-visible elements are kept in standard filtering as they may become visible
            # through user interactions or JavaScript, and automation tools need to detect them
            
            # For aggressive filtering, also remove hidden/non-visible elements
            if filtering_level == "aggressive":
                elements_to_check = soup.find_all()
                for element in elements_to_check[:]:  # Create a copy to safely modify during iteration
                    should_remove = False
                    
                    # Check for hidden attribute
                    if element.get('hidden') is not None:
                        should_remove = True
                    
                    # Check for display:none or visibility:hidden in style attribute
                    style_attr = element.get('style', '')
                    if style_attr:
                        style_lower = style_attr.lower()
                        if ('display:none' in style_lower.replace(' ', '') or 
                            'display: none' in style_lower or
                            'visibility:hidden' in style_lower.replace(' ', '') or 
                            'visibility: hidden' in style_lower):
                            should_remove = True
                    
                    # Check for common CSS classes that indicate hidden elements
                    class_attr = element.get('class', [])
                    if isinstance(class_attr, list):
                        class_names = ' '.join(class_attr).lower()
                    else:
                        class_names = str(class_attr).lower()
                    
                    if any(hidden_class in class_names for hidden_class in 
                          ['hidden', 'invisible', 'd-none', 'hide', 'sr-only', 'visually-hidden']):
                        should_remove = True
                    
                    # Remove if determined to be hidden
                    if should_remove:
                        element.decompose()
            
            # Remove unwanted attributes from remaining elements
            for element in soup.find_all():
                attrs_to_remove = []
                
                # Check each attribute
                for attr_name in element.attrs.keys():
                    if attr_name.lower() in attributes_to_remove:
                        attrs_to_remove.append(attr_name)
                
                # Remove unwanted attributes
                for attr in attrs_to_remove:
                    del element.attrs[attr]
            
            # Return cleaned HTML
            return str(soup)
            
        except Exception as e:
            logger.error(f"Error filtering page source: {e}")
            return html

    async def get_page_source(
        self, 
        session: ExecutionSession, 
        browser_library_manager: Any,  # BrowserLibraryManager
        full_source: bool = False, 
        filtered: bool = False, 
        filtering_level: str = "standard",
        include_reduced_dom: bool = True,
    ) -> Dict[str, Any]:
        """
        Get page source for a session (web or mobile).
        
        Args:
            session: ExecutionSession to get page source from
            browser_library_manager: BrowserLibraryManager instance for source retrieval
            full_source: If True, returns complete page source. If False, returns preview.
            filtered: If True, returns filtered page source with only automation-relevant content.
            filtering_level: Filtering intensity when filtered=True ('minimal', 'standard', 'aggressive').
            include_reduced_dom: When True, attempts to capture Browser Library reduced DOM (aria snapshot).
        """
        try:
            plugin_result = await self._get_page_source_from_plugin(
                session,
                browser_library_manager=browser_library_manager,
                full_source=full_source,
                filtered=filtered,
                filtering_level=filtering_level,
                include_reduced_dom=include_reduced_dom,
            )
            if plugin_result is not None:
                return plugin_result

            # Mobile sessions fall back to mobile source path
            try:
                web_pref = session.get_web_automation_library()
            except Exception:
                web_pref = None

            if session.is_mobile_session() and web_pref not in ("Browser", "SeleniumLibrary"):
                # Only use mobile source path when not in a web automation session
                return await self._get_mobile_source(
                    session, full_source, filtered, filtering_level
                )

            # RF-context fallback when no plugin handles the library.
            # Run in a worker thread to avoid blocking the event loop and
            # to prevent _suppress_stdout() fd-redirect races.
            import asyncio as _asyncio
            page_source = await _asyncio.to_thread(
                self._get_page_source_via_rf_context, session
            )

            if not page_source:
                return {
                    "success": False,
                    "error": "No page source available for this session"
                }

            # Apply filtering if requested
            if filtered:
                filtered_source = self.filter_page_source(page_source, filtering_level)
                result = {
                    "success": True,
                    "session_id": session.session_id,
                    "page_source_length": len(page_source),
                    "filtered_page_source_length": len(filtered_source),
                    "current_url": session.browser_state.current_url,
                    "page_title": session.browser_state.page_title,
                    "context": await self.extract_page_context(page_source),
                    "filtering_applied": True,
                    "filtering_level": filtering_level
                }
                
                if full_source:
                    result["page_source"] = filtered_source
                else:
                    # Return preview of filtered source
                    preview_size = self.config.PAGE_SOURCE_PREVIEW_SIZE
                    if len(filtered_source) > preview_size:
                        result["page_source_preview"] = (
                            filtered_source[:preview_size] + 
                            "...\n[Truncated - use full_source=True for complete filtered source]"
                        )
                    else:
                        result["page_source_preview"] = filtered_source
            else:
                result = {
                    "success": True,
                    "session_id": session.session_id,
                    "page_source_length": len(page_source),
                    "current_url": session.browser_state.current_url,
                    "page_title": session.browser_state.page_title,
                    "context": await self.extract_page_context(page_source),
                    "filtering_applied": False
                }
                
                if full_source:
                    result["page_source"] = page_source
                else:
                    # Return preview of raw source
                    preview_size = self.config.PAGE_SOURCE_PREVIEW_SIZE
                    if len(page_source) > preview_size:
                        result["page_source_preview"] = (
                            page_source[:preview_size] + 
                            "...\n[Truncated - use full_source=True for complete source]"
                        )
                    else:
                        result["page_source_preview"] = page_source
                    # Add generic 'source' field for compatibility with older tests
                    result["source"] = result["page_source_preview"]
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting page source for session {session.session_id}: {e}")
            return {
                "success": False,
                "error": f"Failed to get page source: {str(e)}"
            }

    async def _capture_browser_aria_snapshot(
        self,
        session: ExecutionSession,
        browser_library_manager: Any,
        selector: str = "css=html",
    ) -> Dict[str, Any]:
        """Attempt to capture Browser Library reduced DOM (aria snapshot)."""
        info: Dict[str, Any] = {
            "success": False,
            "selector": selector,
            "format": "yaml",
            "library": "Browser",
        }

        try:
            _, library_type = browser_library_manager.get_active_browser_library(session)
        except Exception as detection_error:
            info["error"] = f"library_detection_failed: {detection_error}"
            logger.debug(
                "Browser library detection failed for session %s: %s",
                session.session_id,
                detection_error,
            )
            return info

        if library_type != "browser":
            info["error"] = "browser_library_not_active"
            return info

        try:
            from robotmcp.components.execution.rf_native_context_manager import (
                get_rf_native_context_manager,
            )

            rf_mgr = get_rf_native_context_manager()
        except Exception as ctx_error:
            info["error"] = f"rf_context_unavailable: {ctx_error}"
            logger.debug(
                "RF native context unavailable when requesting aria snapshot for session %s: %s",
                session.session_id,
                ctx_error,
            )
            return info

        # Rely on Browser/Playwright to traverse frames and shadow-dom when capturing the root element.
        arguments = [selector, "return_type=yaml"]

        try:
            res = rf_mgr.execute_keyword_with_context(
                session_id=session.session_id,
                keyword_name="Get Aria Snapshot",
                arguments=arguments,
                assign_to=None,
                session_variables=dict(session.variables),
            )
        except Exception as exec_error:
            info["error"] = f"keyword_execution_failed: {exec_error}"
            logger.debug(
                "Executing Browser.Get Aria Snapshot failed for session %s: %s",
                session.session_id,
                exec_error,
            )
            return info

        if not res:
            info["error"] = "no_response"
            return info

        if res.get("success"):
            snapshot = res.get("output") or res.get("result")
            if snapshot:
                info["success"] = True
                info["content"] = snapshot
                return info
            info["error"] = "empty_snapshot"
            return info

        info["error"] = res.get("error") or res.get("message") or "keyword_failed"
        return info

    async def _get_page_source_unified_async(self, session: ExecutionSession, browser_library_manager: Any) -> Optional[str]:
        """
        Get page source using the keyword discovery system for consistency.
        
        Args:
            session: ExecutionSession to get source from
            browser_library_manager: BrowserLibraryManager instance
            
        Returns:
            str: Page source HTML or None if not available
        """
        try:
            from robotmcp.components.execution.rf_native_context_manager import (
                get_rf_native_context_manager,
            )

            mgr = get_rf_native_context_manager()
            imported = getattr(session, "imported_libraries", []) or []

            for kw in self._keyword_candidates(self._SOURCE_KEYWORDS, imported):
                res = mgr.execute_keyword_with_context(
                    session_id=session.session_id,
                    keyword_name=kw,
                    arguments=[],
                    assign_to=None,
                    session_variables=dict(session.variables),
                )
                if res and res.get("success"):
                    out = res.get("output") or res.get("result")
                    if isinstance(out, str) and out:
                        # Cache for later
                        session.browser_state.page_source = out
                        return out
        except Exception as rf_fallback_err:
            logger.debug(f"RF-context page source retrieval not available: {rf_fallback_err}")

        return None

    async def _get_current_url(self, session: ExecutionSession, browser_library_manager: Any) -> Optional[str]:
        """Get current URL using RF context in the session."""
        try:
            from robotmcp.components.execution.rf_native_context_manager import (
                get_rf_native_context_manager,
            )

            mgr = get_rf_native_context_manager()
            imported = getattr(session, "imported_libraries", []) or []
            for kw in self._keyword_candidates(self._URL_KEYWORDS, imported):
                res = mgr.execute_keyword_with_context(
                    session_id=session.session_id,
                    keyword_name=kw,
                    arguments=[],
                )
                if res and res.get("success") and res.get("output"):
                    return res["output"]
        except Exception as e:
            logger.debug(f"Could not get current URL: {e}")
        return None
    
    async def _get_page_title(self, session: ExecutionSession, browser_library_manager: Any) -> Optional[str]:
        """Get page title using RF context in the session."""
        try:
            from robotmcp.components.execution.rf_native_context_manager import (
                get_rf_native_context_manager,
            )

            mgr = get_rf_native_context_manager()
            imported = getattr(session, "imported_libraries", []) or []
            for kw in self._keyword_candidates(self._TITLE_KEYWORDS, imported):
                res = mgr.execute_keyword_with_context(
                    session_id=session.session_id,
                    keyword_name=kw,
                    arguments=[],
                )
                if res and res.get("success") and res.get("output"):
                    return res["output"]
        except Exception as e:
            logger.debug(f"Could not get page title: {e}")
        return None

    async def extract_page_context(self, html: str) -> Dict[str, Any]:
        """
        Extract useful context information from page source.
        
        Args:
            html: Raw HTML source
            
        Returns:
            dict: Context information including forms, buttons, inputs, etc.
        """
        if not html or not BS4_AVAILABLE:
            return {}
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            context = {
                "forms": [],
                "buttons": [],
                "inputs": [],
                "links": [],
                "images": [],
                "page_title": "",
                "headings": []
            }
            
            # Extract page title
            title_tag = soup.find('title')
            if title_tag:
                context["page_title"] = title_tag.get_text().strip()
            
            # Extract forms
            for form in soup.find_all('form'):
                form_info = {
                    "action": form.get('action', ''),
                    "method": form.get('method', 'GET').upper(),
                    "inputs": []
                }
                
                # Get form inputs
                for input_elem in form.find_all(['input', 'textarea', 'select']):
                    input_info = {
                        "type": input_elem.get('type', 'text'),
                        "name": input_elem.get('name', ''),
                        "id": input_elem.get('id', ''),
                        "placeholder": input_elem.get('placeholder', ''),
                        "required": input_elem.get('required') is not None
                    }
                    form_info["inputs"].append(input_info)
                
                context["forms"].append(form_info)
            
            # Extract buttons
            for button in soup.find_all(['button', 'input']):
                if button.name == 'input' and button.get('type') not in ['button', 'submit', 'reset']:
                    continue
                
                button_info = {
                    "text": button.get_text().strip() or button.get('value', ''),
                    "type": button.get('type', 'button'),
                    "id": button.get('id', ''),
                    "class": button.get('class', [])
                }
                context["buttons"].append(button_info)
            
            # Extract standalone inputs
            for input_elem in soup.find_all('input'):
                if input_elem.find_parent('form'):
                    continue  # Skip inputs already captured in forms
                
                input_info = {
                    "type": input_elem.get('type', 'text'),
                    "name": input_elem.get('name', ''),
                    "id": input_elem.get('id', ''),
                    "placeholder": input_elem.get('placeholder', ''),
                    "value": input_elem.get('value', '')
                }
                context["inputs"].append(input_info)
            
            # Extract links (limit to first 20 to avoid too much data)
            for link in soup.find_all('a', href=True)[:20]:
                link_info = {
                    "href": link.get('href', ''),
                    "text": link.get_text().strip(),
                    "title": link.get('title', '')
                }
                context["links"].append(link_info)
            
            # Extract headings
            for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                heading_info = {
                    "level": heading.name,
                    "text": heading.get_text().strip(),
                    "id": heading.get('id', '')
                }
                context["headings"].append(heading_info)
            
            # Extract images (limit to first 10)
            for img in soup.find_all('img')[:10]:
                img_info = {
                    "src": img.get('src', ''),
                    "alt": img.get('alt', ''),
                    "title": img.get('title', '')
                }
                context["images"].append(img_info)
            
            return context
            
        except Exception as e:
            logger.error(f"Error extracting page context: {e}")
            return {}

    def get_filtered_source_stats(self, original_html: str, filtered_html: str) -> Dict[str, Any]:
        """
        Get statistics about filtering operations.
        
        Args:
            original_html: Original HTML before filtering
            filtered_html: HTML after filtering
            
        Returns:
            dict: Statistics about the filtering operation
        """
        if not BS4_AVAILABLE:
            return {}
        
        try:
            original_soup = BeautifulSoup(original_html, 'html.parser')
            filtered_soup = BeautifulSoup(filtered_html, 'html.parser')
            
            original_elements = original_soup.find_all()
            filtered_elements = filtered_soup.find_all()
            
            stats = {
                "original_size": len(original_html),
                "filtered_size": len(filtered_html),
                "size_reduction_percent": round(((len(original_html) - len(filtered_html)) / len(original_html)) * 100, 2) if original_html else 0,
                "original_element_count": len(original_elements),
                "filtered_element_count": len(filtered_elements),
                "elements_removed": len(original_elements) - len(filtered_elements)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating filtering stats: {e}")
            return {}

    def validate_filtering_level(self, filtering_level: str) -> bool:
        """
        Validate that the filtering level is supported.
        
        Args:
            filtering_level: Filtering level to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        return filtering_level in ["minimal", "standard", "aggressive"]

    def get_supported_filtering_levels(self) -> List[str]:
        """
        Get list of supported filtering levels.
        
        Returns:
            list: List of supported filtering level names
        """
        return ["minimal", "standard", "aggressive"]
    
    async def _get_mobile_source(
        self,
        session: ExecutionSession,
        full_source: bool = False,
        filtered: bool = False,
        filtering_level: str = "standard"
    ) -> Dict[str, Any]:
        """
        Get source from mobile app using AppiumLibrary.
        
        Args:
            session: Mobile session
            full_source: Return complete source
            filtered: Apply filtering
            filtering_level: Filtering intensity
            
        Returns:
            Mobile app source and metadata
        """
        try:
            # Check if AppiumLibrary is loaded
            if 'AppiumLibrary' not in session.imported_libraries:
                return {
                    "success": False,
                    "error": "AppiumLibrary not loaded in session"
                }
            
            # Use RF context manager to execute AppiumLibrary keyword in-session
            try:
                from robotmcp.components.execution.rf_native_context_manager import (
                    get_rf_native_context_manager,
                )
                rf_mgr = get_rf_native_context_manager()
                res = rf_mgr.execute_keyword_with_context(
                    session_id=session.session_id,
                    keyword_name="Get Source",
                    arguments=[],
                    assign_to=None,
                    session_variables=dict(session.variables),
                )
                if res and res.get("success"):
                    if res.get("output"):
                        source = res["output"]
                    else:
                        source = res.get("result")
                else:
                    error_msg = res.get("error", "Unknown error") if res else "No result"
                    logger.debug(f"RF context call failed for Get Source: {error_msg}")
                    source = await self._get_mobile_source_direct(session)
                    if not source:
                        return {
                            "success": False,
                            "error": f"Failed to get mobile source: {error_msg}"
                        }
            except Exception as rf_err:
                logger.debug(f"RF context execution failed for Get Source: {rf_err}")
                source = await self._get_mobile_source_direct(session)
                if not source:
                    return {
                        "success": False,
                        "error": f"No active mobile app session: {str(rf_err)}"
                    }
            
            if not source:
                return {
                    "success": False,
                    "error": "No source available from mobile app"
                }
            
            # Parse mobile source based on platform
            platform = session.mobile_config.platform_name if session.mobile_config else None
            
            if platform == 'Android':
                parsed_source = self._parse_android_source(source, filtered)
            elif platform == 'iOS':
                parsed_source = self._parse_ios_source(source, filtered)
            else:
                parsed_source = {"raw_source": source}
            
            result = {
                "success": True,
                "session_id": session.session_id,
                "platform": platform,
                "source_type": "mobile_app",
                "source_length": len(source) if isinstance(source, str) else 0,
                "current_context": session.current_context or "NATIVE_APP"
            }
            
            if session.mobile_config:
                result["app_info"] = {
                    "package": session.mobile_config.app_package,
                    "activity": session.mobile_config.app_activity,
                    "device": session.mobile_config.device_name
                }
            
            if full_source:
                result["source"] = source
                result["parsed_structure"] = parsed_source
            else:
                # Return preview
                preview_size = self.config.PAGE_SOURCE_PREVIEW_SIZE
                if isinstance(source, str) and len(source) > preview_size:
                    result["source_preview"] = source[:preview_size] + "...[Truncated]"
                else:
                    result["source_preview"] = source
                result["parsed_structure_summary"] = self._get_structure_summary(parsed_source)
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting mobile source: {e}")
            return {
                "success": False,
                "error": f"Failed to get mobile source: {str(e)}"
            }
    
    def _parse_android_source(self, xml_source: str, filtered: bool = False) -> Dict[str, Any]:
        """Parse Android UI XML hierarchy."""
        try:
            root = ET.fromstring(xml_source)
            
            def parse_element(elem, depth=0):
                # Skip certain elements if filtering
                if filtered and elem.tag in ['hierarchy', 'android.view.ViewGroup']:
                    if not elem.get('resource-id') and not elem.get('content-desc'):
                        # Skip container elements without IDs
                        children = []
                        for child in elem:
                            children.extend(parse_element(child, depth))
                        return children
                
                element_info = {
                    'tag': elem.tag.split('.')[-1] if '.' in elem.tag else elem.tag,
                    'class': elem.get('class', ''),
                    'resource_id': elem.get('resource-id', ''),
                    'text': elem.get('text', ''),
                    'content_desc': elem.get('content-desc', ''),
                    'enabled': elem.get('enabled') == 'true',
                    'clickable': elem.get('clickable') == 'true',
                    'bounds': elem.get('bounds', ''),
                    'depth': depth
                }
                
                # Only include non-empty fields
                element_info = {k: v for k, v in element_info.items() if v or v is False}
                
                # Parse children
                children = []
                for child in elem:
                    children.extend(parse_element(child, depth + 1))
                
                if children:
                    element_info['children'] = children
                    
                return [element_info]
            
            parsed = parse_element(root)
            return {"elements": parsed, "type": "android_xml"}
            
        except Exception as e:
            logger.error(f"Error parsing Android source: {e}")
            return {"error": str(e), "raw": xml_source[:500] if len(xml_source) > 500 else xml_source}
    
    def _parse_ios_source(self, source: Union[str, dict], filtered: bool = False) -> Dict[str, Any]:
        """Parse iOS XCUITest hierarchy."""
        try:
            if isinstance(source, dict):
                # Already parsed as dict
                return {"elements": [source], "type": "ios_dict"}
            elif isinstance(source, str):
                # Try to parse as XML
                root = ET.fromstring(source)
                
                def parse_element(elem, depth=0):
                    element_info = {
                        'type': elem.get('type', ''),
                        'name': elem.get('name', ''),
                        'label': elem.get('label', ''),
                        'value': elem.get('value', ''),
                        'enabled': elem.get('enabled') == 'true',
                        'visible': elem.get('visible') == 'true',
                        'x': elem.get('x', ''),
                        'y': elem.get('y', ''),
                        'width': elem.get('width', ''),
                        'height': elem.get('height', ''),
                        'depth': depth
                    }
                    
                    # Only include non-empty fields
                    element_info = {k: v for k, v in element_info.items() if v or v is False}
                    
                    # Parse children
                    children = []
                    for child in elem:
                        children.append(parse_element(child, depth + 1))
                    
                    if children:
                        element_info['children'] = children
                        
                    return element_info
                
                parsed = parse_element(root)
                return {"elements": [parsed], "type": "ios_xml"}
            else:
                return {"raw": str(source), "type": "unknown"}
                
        except Exception as e:
            logger.error(f"Error parsing iOS source: {e}")
            return {"error": str(e), "raw": str(source)[:500] if len(str(source)) > 500 else str(source)}
    
    def _get_structure_summary(self, parsed_source: Dict[str, Any]) -> Dict[str, Any]:
        """Get summary of parsed mobile structure."""
        if 'error' in parsed_source:
            return parsed_source
            
        elements = parsed_source.get('elements', [])
        if not elements:
            return {"element_count": 0}
            
        # Count different element types
        def count_elements(elem_list):
            counts = {}
            total = 0
            for elem in elem_list:
                if isinstance(elem, dict):
                    elem_type = elem.get('tag') or elem.get('type') or elem.get('class', 'unknown')
                    counts[elem_type] = counts.get(elem_type, 0) + 1
                    total += 1
                    
                    if 'children' in elem:
                        child_counts, child_total = count_elements(elem['children'])
                        for k, v in child_counts.items():
                            counts[k] = counts.get(k, 0) + v
                        total += child_total
                        
            return counts, total
        
        type_counts, total_count = count_elements(elements)
        
        return {
            "total_elements": total_count,
            "element_types": type_counts,
            "structure_type": parsed_source.get('type', 'unknown')
        }
    
    async def _get_mobile_source_direct(self, session: ExecutionSession) -> Optional[str]:
        """
        Fallback method to get mobile source via direct library method call.
        
        This method attempts to get the AppiumLibrary instance directly and call
        the get_source method without going through Robot Framework's BuiltIn.run_keyword
        which requires an execution context.
        
        Args:
            session: ExecutionSession containing mobile configuration
            
        Returns:
            str: Mobile app source XML or None if failed
        """
        try:
            # Try to get AppiumLibrary instance from the library manager
            from robotmcp.core.library_manager import LibraryManager
            
            library_manager = LibraryManager()
            
            # Check if AppiumLibrary is available in the loaded libraries
            appium_lib = None
            
            # First, try to get it from the library manager's loaded libraries
            if hasattr(library_manager, '_libraries'):
                for lib_name, lib_instance in library_manager._libraries.items():
                    if lib_name == "AppiumLibrary" or "appium" in lib_name.lower():
                        appium_lib = lib_instance
                        break
            
            # If not found, try importing and getting instance directly
            if not appium_lib:
                try:
                    import importlib
                    appium_module = importlib.import_module('AppiumLibrary')
                    # This might not work as libraries usually need proper initialization
                    logger.debug("AppiumLibrary module imported but instance not available")
                    return None
                except ImportError:
                    logger.debug("AppiumLibrary not installed")
                    return None
            
            # Call get_source method directly on the library instance
            if appium_lib and hasattr(appium_lib, 'get_source'):
                source = appium_lib.get_source()
                logger.debug(f"Got mobile source directly from AppiumLibrary: {len(source) if source else 0} chars")
                return source
            else:
                logger.debug("AppiumLibrary instance found but get_source method not available")
                return None

        except Exception as e:
            logger.debug(f"Direct mobile source retrieval failed: {e}")
            return None
