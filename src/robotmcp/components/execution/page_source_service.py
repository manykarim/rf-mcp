"""Page source management and filtering service for web and mobile."""

import logging
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Union

from robotmcp.models.session_models import ExecutionSession, PlatformType
from robotmcp.models.config_models import ExecutionConfig

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
    
    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()
    
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
        filtering_level: str = "standard"
    ) -> Dict[str, Any]:
        """
        Get page source for a session (web or mobile).
        
        Args:
            session: ExecutionSession to get page source from
            browser_library_manager: BrowserLibraryManager instance for source retrieval
            full_source: If True, returns complete page source. If False, returns preview.
            filtered: If True, returns filtered page source with only automation-relevant content.
            filtering_level: Filtering intensity when filtered=True ('minimal', 'standard', 'aggressive').
        """
        try:
            # Check if this is a mobile session
            if session.is_mobile_session():
                return await self._get_mobile_source(session, full_source, filtered, filtering_level)
            
            # Try to get fresh page source using browser library (web)
            page_source = ""
            try:
                page_source = await self._get_page_source_unified_async(session, browser_library_manager)
                if page_source:
                    session.browser_state.page_source = page_source
                    # Also update URL and title if we can extract them
                    url = await self._get_current_url(session, browser_library_manager)
                    if url and url != "about:blank":
                        session.browser_state.current_url = url
                    title = await self._get_page_title(session, browser_library_manager)
                    if title and title != "Generic Page":
                        session.browser_state.page_title = title
                else:
                    page_source = session.browser_state.page_source or ""
            except Exception as e:
                logger.debug(f"Could not get fresh page source: {e}")
                page_source = session.browser_state.page_source or ""
            
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
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting page source for session {session.session_id}: {e}")
            return {
                "success": False,
                "error": f"Failed to get page source: {str(e)}"
            }

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
            from robotmcp.core.dynamic_keyword_orchestrator import get_keyword_discovery
            
            # Use the same keyword discovery system that handles Browser Library execution
            keyword_discovery = get_keyword_discovery()
            
            # Determine which library to use
            library, library_type = browser_library_manager.get_active_browser_library(session)
            
            if library_type == "browser":
                # Use Browser Library via keyword discovery
                result = await keyword_discovery.execute_keyword("Get Page Source", [], session.variables, session_id=session.session_id)
                if result and result.get("success") and result.get("output"):
                    return result["output"]
                    
            elif library_type == "selenium":
                # Use SeleniumLibrary via keyword discovery  
                result = await keyword_discovery.execute_keyword("Get Source", [], session.variables, session_id=session.session_id)
                if result and result.get("success") and result.get("output"):
                    return result["output"]
                    
            else:
                logger.debug("No active browser library available for page source")
                return None
                
        except Exception as e:
            logger.error(f"Error getting page source via keyword discovery: {e}")
            
            # Fallback to direct library calls if keyword discovery fails
            try:
                library, library_type = browser_library_manager.get_active_browser_library(session)
                
                if library_type == "browser" and library:
                    # Browser Library - direct call as fallback
                    return library.get_page_source()
                    
                elif library_type == "selenium" and library:
                    # SeleniumLibrary - direct call as fallback
                    return library.get_source()
                    
            except Exception as fallback_error:
                logger.error(f"Fallback page source retrieval also failed: {fallback_error}")
                
            return None
    
    async def _get_current_url(self, session: ExecutionSession, browser_library_manager: Any) -> Optional[str]:
        """Get current URL using keyword discovery."""
        try:
            from robotmcp.core.dynamic_keyword_orchestrator import get_keyword_discovery
            
            keyword_discovery = get_keyword_discovery()
            library, library_type = browser_library_manager.get_active_browser_library(session)
            
            if library_type == "browser":
                result = await keyword_discovery.execute_keyword("Get Url", [], session.variables, session_id=session.session_id)
                if result and result.get("success") and result.get("output"):
                    return result["output"]
            elif library_type == "selenium":
                result = await keyword_discovery.execute_keyword("Get Location", [], session.variables, session_id=session.session_id)
                if result and result.get("success") and result.get("output"):
                    return result["output"]
                    
        except Exception as e:
            logger.debug(f"Could not get current URL: {e}")
            
        return None
    
    async def _get_page_title(self, session: ExecutionSession, browser_library_manager: Any) -> Optional[str]:
        """Get page title using keyword discovery."""
        try:
            from robotmcp.core.dynamic_keyword_orchestrator import get_keyword_discovery
            
            keyword_discovery = get_keyword_discovery()
            library, library_type = browser_library_manager.get_active_browser_library(session)
            
            if library_type == "browser":
                result = await keyword_discovery.execute_keyword("Get Title", [], session.variables, session_id=session.session_id)
                if result and result.get("success") and result.get("output"):
                    return result["output"]
            elif library_type == "selenium":
                result = await keyword_discovery.execute_keyword("Get Title", [], session.variables, session_id=session.session_id)  
                if result and result.get("success") and result.get("output"):
                    return result["output"]
                    
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
            
            # FIXED: Use keyword discovery system instead of BuiltIn.run_keyword
            # This avoids the "Cannot access execution context" error
            try:
                from robotmcp.core.dynamic_keyword_orchestrator import get_keyword_discovery
                
                keyword_discovery = get_keyword_discovery()
                result = await keyword_discovery.execute_keyword("Get Source", [], session.variables, session_id=session.session_id)
                
                if result and result.get("success") and result.get("output"):
                    source = result["output"]
                elif result and result.get("success") and result.get("result"):
                    source = result["result"]
                else:
                    error_msg = result.get('error', 'Unknown error') if result else 'No result from keyword discovery'
                    logger.debug(f"Keyword discovery failed for Get Source: {error_msg}")
                    # Fall back to direct library method call
                    source = await self._get_mobile_source_direct(session)
                    if not source:
                        return {
                            "success": False,
                            "error": f"Failed to get mobile source: {error_msg}"
                        }
                        
            except Exception as kd_error:
                logger.debug(f"Keyword discovery failed for Get Source: {kd_error}")
                # Fall back to direct library method call
                source = await self._get_mobile_source_direct(session)
                if not source:
                    return {
                        "success": False,
                        "error": f"No active mobile app session: {str(kd_error)}"
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