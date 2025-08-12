"""Keyword execution service."""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from robotmcp.models.session_models import ExecutionSession
from robotmcp.models.execution_models import ExecutionStep
from robotmcp.models.config_models import ExecutionConfig
from robotmcp.utils.argument_processor import ArgumentProcessor
from robotmcp.utils.rf_native_type_converter import RobotFrameworkNativeConverter
from robotmcp.core.dynamic_keyword_orchestrator import get_keyword_discovery

logger = logging.getLogger(__name__)

# Import Robot Framework components
try:
    from robot.libraries.BuiltIn import BuiltIn
    ROBOT_AVAILABLE = True
except ImportError:
    BuiltIn = None
    ROBOT_AVAILABLE = False


class KeywordExecutor:
    """Handles keyword execution with proper library routing and error handling."""
    
    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()
        self.keyword_discovery = get_keyword_discovery()
        self.argument_processor = ArgumentProcessor()
        self.rf_converter = RobotFrameworkNativeConverter()
    
    async def execute_keyword(
        self, 
        session: ExecutionSession,
        keyword: str,
        arguments: List[str],
        browser_library_manager: Any,  # BrowserLibraryManager
        detail_level: str = "minimal",
        library_prefix: str = None
    ) -> Dict[str, Any]:
        """
        Execute a single Robot Framework keyword step with optional library prefix.
        
        Args:
            session: ExecutionSession to run in
            keyword: Robot Framework keyword name (supports Library.Keyword syntax)
            arguments: List of arguments for the keyword
            browser_library_manager: BrowserLibraryManager instance
            detail_level: Level of detail in response ('minimal', 'standard', 'full')
            library_prefix: Optional explicit library name to override session search order
            
        Returns:
            Execution result with status, output, and state
        """
        try:
            # Create execution step
            step = ExecutionStep(
                step_id=str(uuid.uuid4()),
                keyword=keyword,
                arguments=arguments,
                start_time=datetime.now()
            )
            
            # Update session activity
            session.update_activity()
            
            # Mark step as running
            step.status = "running"
            
            logger.info(f"Executing keyword: {keyword} with args: {arguments}")
            
            # Execute the keyword with library prefix support
            result = await self._execute_keyword_internal(session, step, browser_library_manager, library_prefix)
            
            # Update step status
            step.end_time = datetime.now()
            step.result = result.get("output")
            
            if result["success"]:
                step.mark_success(result.get("output"))
                # Only append successful steps to the session for suite generation
                session.add_step(step)
                logger.debug(f"Added successful step to session: {keyword}")
            else:
                step.mark_failure(result.get("error"))
                logger.debug(f"Failed step not added to session: {keyword} - {result.get('error')}")
            
            # Update session variables if any were set
            if "variables" in result:
                session.variables.update(result["variables"])
            
            # Build response based on detail level
            response = await self._build_response_by_detail_level(
                detail_level, result, step, keyword, arguments, session
            )
            return response
            
        except Exception as e:
            logger.error(f"Error executing step {keyword}: {e}")
            
            # Create a failed step for error reporting
            step = ExecutionStep(
                step_id=str(uuid.uuid4()),
                keyword=keyword,
                arguments=arguments,
                start_time=datetime.now(),
                end_time=datetime.now()
            )
            step.mark_failure(str(e))
            
            return {
                "success": False,
                "error": str(e),
                "step_id": step.step_id,
                "keyword": keyword,
                "arguments": arguments,
                "status": "fail",
                "execution_time": step.execution_time,
                "session_variables": dict(session.variables)
            }

    async def _execute_keyword_internal(
        self, 
        session: ExecutionSession, 
        step: ExecutionStep,
        browser_library_manager: Any,
        library_prefix: str = None
    ) -> Dict[str, Any]:
        """Execute a specific keyword with error handling and library prefix support."""
        try:
            keyword_name = step.keyword
            args = step.arguments
            
            # Only detect and set active library if session doesn't have one already
            current_active = session.get_active_library()
            if not current_active or current_active == "auto":
                detected_library = browser_library_manager.detect_library_from_keyword(keyword_name, args)
                if detected_library in ["browser", "selenium"]:
                    browser_library_manager.set_active_library(session, detected_library)
            
            # Handle special built-in keywords first
            if keyword_name.lower() in ["set variable", "log", "should be equal"]:
                return await self._execute_builtin_keyword(session, keyword_name, args)
            
            # If library prefix is specified, use direct execution
            if library_prefix:
                return await self._execute_with_library_prefix(session, keyword_name, args, library_prefix)
            
            # Get active browser library and execute
            library, library_type = browser_library_manager.get_active_browser_library(session)
            
            if library_type == "browser":
                return await self._execute_browser_keyword(session, keyword_name, args, library)
            elif library_type == "selenium":
                return await self._execute_selenium_keyword(session, keyword_name, args, library)
            else:
                # Try built-in execution as fallback
                return await self._execute_builtin_keyword(session, keyword_name, args)
                
        except Exception as e:
            logger.error(f"Error in keyword execution: {e}")
            return {
                "success": False,
                "error": f"Execution failed: {str(e)}",
                "output": "",
                "variables": {},
                "state_updates": {}
            }
    
    async def _execute_with_library_prefix(
        self,
        session: ExecutionSession,
        keyword: str,
        args: List[str],
        library_prefix: str
    ) -> Dict[str, Any]:
        """Execute keyword with explicit library prefix."""
        try:
            # Use keyword discovery with library prefix
            result = await self.keyword_discovery.execute_keyword(
                keyword_name=keyword,
                args=args,
                session_variables=session.variables,
                active_library=None,  # Don't use active library when prefix is explicit
                session_id=session.session_id,
                library_prefix=library_prefix
            )
            
            # Update session state if successful
            if result.get("success"):
                # Extract any state updates based on the library and keyword
                if library_prefix.lower() == "browser":
                    state_updates = self._extract_browser_state_updates(keyword, args, result.get("output"))
                    self._apply_state_updates(session, state_updates)
                
                # Add library prefix information to result
                result["library_prefix_used"] = library_prefix
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing {library_prefix}.{keyword}: {e}")
            return {
                "success": False,
                "error": f"Library prefix execution failed: {str(e)}",
                "output": "",
                "variables": {},
                "state_updates": {},
                "library_prefix_used": library_prefix
            }

    async def _execute_browser_keyword(
        self, 
        session: ExecutionSession, 
        keyword: str, 
        args: List[str], 
        library: Any
    ) -> Dict[str, Any]:
        """Execute a Browser Library keyword using the dynamic execution handler."""
        try:
            # Use the keyword discovery's execute_keyword method with Browser Library filter
            result = await self.keyword_discovery.execute_keyword(
                keyword_name=keyword,
                args=args,
                session_variables=session.variables,
                active_library="Browser",
                session_id=session.session_id
            )
            
            # Update session browser state based on keyword if successful
            if result.get("success"):
                state_updates = self._extract_browser_state_updates(keyword, args, result.get("output"))
                self._apply_state_updates(session, state_updates)
                result["state_updates"] = state_updates
            else:
                # Add Browser Library-specific error guidance for failed keywords
                result["browser_guidance"] = self._get_browser_error_guidance(keyword, args, result.get("error", ""))
            
            return result
                
        except Exception as e:
            logger.error(f"Error executing Browser Library keyword {keyword}: {e}")
            error_msg = f"Browser keyword execution failed: {str(e)}"
            
            # Include locator guidance for Browser Library errors
            guidance = self._get_browser_error_guidance(keyword, args, str(e))
            
            return {
                "success": False,
                "error": error_msg,
                "output": "",
                "variables": {},
                "state_updates": {},
                "browser_guidance": guidance
            }

    async def _execute_selenium_keyword(
        self, 
        session: ExecutionSession, 
        keyword: str, 
        args: List[str], 
        library: Any
    ) -> Dict[str, Any]:
        """Execute a SeleniumLibrary keyword using the dynamic execution handler."""
        try:
            # Use the keyword discovery's execute_keyword method with SeleniumLibrary filter
            result = await self.keyword_discovery.execute_keyword(
                keyword_name=keyword,
                args=args,
                session_variables=session.variables,
                active_library="SeleniumLibrary",
                session_id=session.session_id
            )
            
            # Update session browser state based on keyword if successful
            if result.get("success"):
                state_updates = self._extract_selenium_state_updates(keyword, args, result.get("output"))
                self._apply_state_updates(session, state_updates)
                result["state_updates"] = state_updates
            else:
                # Add SeleniumLibrary-specific error guidance for failed keywords
                result["selenium_guidance"] = self._get_selenium_error_guidance(keyword, args, result.get("error", ""))
            
            return result
                
        except Exception as e:
            logger.error(f"Error executing SeleniumLibrary keyword {keyword}: {e}")
            error_msg = f"Selenium keyword execution failed: {str(e)}"
            
            # Include locator guidance for SeleniumLibrary errors
            guidance = self._get_selenium_error_guidance(keyword, args, str(e))
            
            return {
                "success": False,
                "error": error_msg,
                "output": "",
                "variables": {},
                "state_updates": {},
                "selenium_guidance": guidance
            }

    async def _execute_builtin_keyword(
        self, 
        session: ExecutionSession, 
        keyword: str, 
        args: List[str]
    ) -> Dict[str, Any]:
        """Execute a built-in Robot Framework keyword."""
        try:
            if not ROBOT_AVAILABLE:
                return {
                    "success": False,
                    "error": "Robot Framework not available for built-in keywords",
                    "output": "",
                    "variables": {},
                    "state_updates": {}
                }
            
            builtin = BuiltIn()
            keyword_lower = keyword.lower()
            
            # Handle common built-in keywords
            if keyword_lower == "set variable":
                if args:
                    var_value = args[0]
                    return {
                        "success": True,
                        "output": var_value,
                        "variables": {"${VARIABLE}": var_value},
                        "state_updates": {}
                    }
            
            elif keyword_lower == "log":
                message = args[0] if args else ""
                logger.info(f"Robot Log: {message}")
                return {
                    "success": True,
                    "output": message,
                    "variables": {},
                    "state_updates": {}
                }
            
            elif keyword_lower == "should be equal":
                if len(args) >= 2:
                    if args[0] == args[1]:
                        return {
                            "success": True,
                            "output": f"'{args[0]}' == '{args[1]}'",
                            "variables": {},
                            "state_updates": {}
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"'{args[0]}' != '{args[1]}'",
                            "output": "",
                            "variables": {},
                            "state_updates": {}
                        }
            
            # Try to execute using BuiltIn library
            try:
                result = builtin.run_keyword(keyword, *args)
                return {
                    "success": True,
                    "output": str(result) if result is not None else "OK",
                    "variables": {},
                    "state_updates": {}
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Built-in keyword execution failed: {str(e)}",
                    "output": "",
                    "variables": {},
                    "state_updates": {}
                }
                
        except Exception as e:
            logger.error(f"Error executing built-in keyword {keyword}: {e}")
            return {
                "success": False,
                "error": f"Built-in keyword execution failed: {str(e)}",
                "output": "",
                "variables": {},
                "state_updates": {}
            }

    def _extract_browser_state_updates(self, keyword: str, args: List[str], result: Any) -> Dict[str, Any]:
        """Extract state updates from Browser Library keyword execution."""
        state_updates = {}
        keyword_lower = keyword.lower()
        
        # Extract state changes based on keyword
        if "new browser" in keyword_lower:
            browser_type = args[0] if args else "chromium"
            state_updates["current_browser"] = {"type": browser_type}
        elif "new context" in keyword_lower:
            state_updates["current_context"] = {"id": str(result) if result else "context"}
        elif "new page" in keyword_lower:
            url = args[0] if args else ""
            state_updates["current_page"] = {"id": str(result) if result else "page", "url": url}
        elif "go to" in keyword_lower:
            url = args[0] if args else ""
            state_updates["current_page"] = {"url": url}
        
        return state_updates

    def _extract_selenium_state_updates(self, keyword: str, args: List[str], result: Any) -> Dict[str, Any]:
        """Extract state updates from SeleniumLibrary keyword execution."""
        state_updates = {}
        keyword_lower = keyword.lower()
        
        # Extract state changes based on keyword
        if "open browser" in keyword_lower:
            state_updates["current_browser"] = {"type": args[1] if len(args) > 1 else "firefox"}
        elif "go to" in keyword_lower:
            state_updates["current_page"] = {"url": args[0] if args else ""}
        
        return state_updates

    def _apply_state_updates(self, session: ExecutionSession, state_updates: Dict[str, Any]) -> None:
        """Apply state updates to session browser state."""
        if not state_updates:
            return
        
        browser_state = session.browser_state
        
        for key, value in state_updates.items():
            if key == "current_browser":
                if isinstance(value, dict):
                    browser_state.browser_type = value.get("type")
            elif key == "current_context":
                if isinstance(value, dict):
                    browser_state.context_id = value.get("id")
            elif key == "current_page":
                if isinstance(value, dict):
                    browser_state.current_url = value.get("url")
                    browser_state.page_id = value.get("id")

    async def _build_response_by_detail_level(
        self, 
        detail_level: str, 
        result: Dict[str, Any], 
        step: ExecutionStep,
        keyword: str,
        arguments: List[str],
        session: ExecutionSession
    ) -> Dict[str, Any]:
        """Build execution response based on requested detail level."""
        base_response = {
            "success": result["success"],
            "step_id": step.step_id,
            "keyword": keyword,
            "arguments": arguments,
            "status": step.status,
            "execution_time": step.execution_time
        }
        
        if not result["success"]:
            base_response["error"] = result.get("error", "Unknown error")
        
        if detail_level == "minimal":
            base_response["output"] = result.get("output", "")
            
        elif detail_level == "standard":
            base_response.update({
                "output": result.get("output", ""),
                "session_variables": dict(session.variables),
                "active_library": session.get_active_library()
            })
            
        elif detail_level == "full":
            base_response.update({
                "output": result.get("output", ""),
                "session_variables": dict(session.variables),
                "state_updates": result.get("state_updates", {}),
                "active_library": session.get_active_library(),
                "browser_state": {
                    "browser_type": session.browser_state.browser_type,
                    "current_url": session.browser_state.current_url,
                    "context_id": session.browser_state.context_id,
                    "page_id": session.browser_state.page_id
                },
                "step_count": session.step_count,
                "duration": session.duration
            })
        
        return base_response

    def get_supported_detail_levels(self) -> List[str]:
        """Get list of supported detail levels."""
        return ["minimal", "standard", "full"]

    def validate_detail_level(self, detail_level: str) -> bool:
        """Validate that the detail level is supported."""
        return detail_level in self.get_supported_detail_levels()
    
    def _get_selenium_error_guidance(self, keyword: str, args: List[str], error_message: str) -> Dict[str, Any]:
        """Generate SeleniumLibrary-specific error guidance for agents."""
        # Get base locator guidance
        guidance = self.rf_converter.get_selenium_locator_guidance(error_message, keyword)
        
        # Add keyword-specific guidance
        keyword_lower = keyword.lower()
        
        if any(term in keyword_lower for term in ["click", "input", "select", "clear", "wait"]):
            # Element interaction keywords
            guidance["keyword_specific_tips"] = [
                f"'{keyword}' requires a valid element locator as the first argument",
                "Common locator patterns: 'id:elementId', 'name:fieldName', 'css:.className'",
                "Ensure the element is visible and interactable before interaction"
            ]
            
            # Analyze the locator argument if provided
            if args:
                locator = args[0]
                if not any(strategy in locator for strategy in [":", "="]):
                    guidance["locator_analysis"] = {
                        "provided_locator": locator,
                        "issue": "Locator appears to be missing strategy prefix",
                        "suggestions": [
                            f"Try 'id:{locator}' if it's an ID",
                            f"Try 'name:{locator}' if it's a name attribute", 
                            f"Try 'css:{locator}' if it's a CSS selector",
                            f"Try 'xpath://*[@id=\"{locator}\"]' for XPath"
                        ]
                    }
                elif "=" in locator and ":" not in locator:
                    guidance["locator_analysis"] = {
                        "provided_locator": locator,
                        "issue": "Contains '=' but no strategy prefix - may be parsed as named argument",
                        "correct_format": f"name:{locator}" if locator.startswith("name=") else f"Use appropriate strategy prefix",
                        "note": "SeleniumLibrary requires 'strategy:value' format, not 'strategy=value'"
                    }
        
        elif "open" in keyword_lower or "browser" in keyword_lower:
            guidance["keyword_specific_tips"] = [
                f"'{keyword}' manages browser/session state",
                "Ensure proper browser initialization before element interactions",
                "Check browser driver compatibility and installation"
            ]
        
        return guidance
    
    def _get_browser_error_guidance(self, keyword: str, args: List[str], error_message: str) -> Dict[str, Any]:
        """Generate Browser Library-specific error guidance for agents."""
        # Get base locator guidance
        guidance = self.rf_converter.get_browser_locator_guidance(error_message, keyword)
        
        # Add keyword-specific guidance
        keyword_lower = keyword.lower()
        
        if any(term in keyword_lower for term in ["click", "fill", "select", "check", "type", "press", "hover"]):
            # Element interaction keywords
            guidance["keyword_specific_tips"] = [
                f"'{keyword}' requires a valid element selector",
                "Browser Library uses CSS selectors by default (no prefix needed)",
                "Common patterns: '.class', '#id', 'button', 'input[type=\"submit\"]'",
                "For complex elements, use cascaded selectors: 'div.container >> .button'"
            ]
            
            # Analyze the selector argument if provided
            if args:
                selector = args[0]
                guidance.update(self._analyze_browser_selector(selector))
        
        elif any(term in keyword_lower for term in ["new browser", "new page", "new context", "go to"]):
            guidance["keyword_specific_tips"] = [
                f"'{keyword}' manages browser/page state",
                "Ensure proper browser initialization sequence",
                "Check browser installation and dependencies",
                "Verify URL accessibility for navigation keywords"
            ]
        
        elif "wait" in keyword_lower:
            guidance["keyword_specific_tips"] = [
                f"'{keyword}' handles dynamic content and timing",
                "Adjust timeout values for slow-loading elements",
                "Use appropriate wait conditions (visible, hidden, enabled, etc.)",
                "Consider page load states for complete readiness"
            ]
        
        return guidance
    
    def _analyze_browser_selector(self, selector: str) -> Dict[str, Any]:
        """Analyze a Browser Library selector and provide specific guidance."""
        analysis = {}
        
        # Detect selector patterns and provide guidance (order matters - check >>> before >>)
        if ">>>" in selector:
            analysis["iframe_selector_detected"] = {
                "type": "iFrame piercing selector",
                "explanation": "Using >>> to access elements inside frames",
                "tip": "Left side selects frame, right side selects element inside frame"
            }
        
        elif selector.startswith("#") and not selector.startswith("\\#"):
            analysis["selector_warning"] = {
                "issue": "ID selector may need escaping in Robot Framework",
                "provided_selector": selector,
                "recommended": f"\\{selector}",
                "explanation": "# is a comment character in Robot Framework, use \\# for ID selectors"
            }
        
        elif ">>" in selector:
            analysis["cascaded_selector_detected"] = {
                "type": "Cascaded selector (good practice)",
                "explanation": "Using >> to chain multiple selector strategies",
                "tip": "Each part of the chain is relative to the previous match"
            }
        
        elif selector.startswith('"') and selector.endswith('"'):
            analysis["text_selector_detected"] = {
                "type": "Text selector (implicit)",
                "explanation": "Quoted strings are treated as text selectors",
                "equivalent_explicit": f"text={selector}",
                "tip": "Use for exact text matching"
            }
        
        elif selector.startswith("//") or selector.startswith(".."):
            analysis["xpath_selector_detected"] = {
                "type": "XPath selector (implicit)",
                "explanation": "Selectors starting with // or .. are treated as XPath",
                "equivalent_explicit": f"xpath={selector}",
                "tip": "XPath provides powerful element traversal capabilities"
            }
        
        elif "=" in selector and any(selector.startswith(prefix) for prefix in ["css=", "xpath=", "text=", "id="]):
            strategy = selector.split("=", 1)[0]
            analysis["explicit_strategy_detected"] = {
                "type": f"Explicit {strategy} selector",
                "explanation": f"Using explicit {strategy} strategy",
                "tip": f"Good practice to be explicit with selector strategies"
            }
        
        else:
            analysis["implicit_css_detected"] = {
                "type": "CSS selector (implicit default)",
                "explanation": "Plain selectors are treated as CSS by default",
                "equivalent_explicit": f"css={selector}",
                "tip": "Browser Library defaults to CSS selectors"
            }
        
        return analysis
    
