"""Shared utility for detecting which library a keyword belongs to."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def detect_library_from_keyword(keyword: str, keyword_discovery=None) -> Optional[str]:
    """
    Detect which library a keyword belongs to using dynamic discovery and fallback patterns.
    
    This consolidates the library detection logic that was duplicated across
    execution_engine.py and test_builder.py.
    
    Args:
        keyword: The keyword name to detect library for
        keyword_discovery: Optional dynamic keyword discovery instance
        
    Returns:
        str: Library name (e.g., "Browser", "SeleniumLibrary", "BuiltIn") or None
    """
    if not keyword:
        return None
        
    # First try dynamic keyword discovery if available
    if keyword_discovery and hasattr(keyword_discovery, 'find_keyword'):
        try:
            keyword_info = keyword_discovery.find_keyword(keyword)
            if keyword_info and keyword_info.library:
                logger.debug(f"Dynamic detection: '{keyword}' -> {keyword_info.library}")
                return keyword_info.library
        except Exception as e:
            logger.debug(f"Dynamic keyword detection failed for '{keyword}': {e}")
    
    # Fallback to pattern-based detection
    keyword_lower = keyword.lower().strip()
    
    # Check for specific SeleniumLibrary keywords (excluding conflicting ones like 'go to')
    # These patterns should only match SeleniumLibrary-specific keywords that don't exist in Browser Library
    if any(kw in keyword_lower for kw in [
        'open browser', 'input text', 'click button', 'select from list', 
        'wait until element is visible', 'page should contain', 'element should be visible',
        'capture page screenshot', 'maximize browser window', 'set window size'
    ]):
        logger.debug(f"Pattern detection: '{keyword}' -> SeleniumLibrary")
        return "SeleniumLibrary"
    
    # Browser Library keywords (prioritized for modern web testing)
    elif any(kw in keyword_lower for kw in [
        'new browser', 'new context', 'new page', 'close context', 'close page', 'go to',
        'get viewport size', 'set viewport size', 'wait for elements state', 'get element count',
        'get element', 'get elements', 'fill text', 'fill', 'get text', 'get property',
        'select options by', 'check checkbox', 'get page source', 'click'
    ]):
        logger.debug(f"Pattern detection: '{keyword}' -> Browser")
        return "Browser"
    
    # RequestsLibrary keywords
    elif any(kw in keyword_lower for kw in [
        'get request', 'post request', 'put request', 'delete request', 'patch request',
        'head request', 'options request', 'response should', 'create session',
        'get on session', 'post on session'
    ]):
        logger.debug(f"Pattern detection: '{keyword}' -> RequestsLibrary")
        return "RequestsLibrary"
    
    # DatabaseLibrary keywords
    elif any(kw in keyword_lower for kw in [
        'connect to database', 'disconnect from database', 'execute sql', 'query',
        'check if exists in database', 'check if not exists in database'
    ]):
        logger.debug(f"Pattern detection: '{keyword}' -> DatabaseLibrary")
        return "DatabaseLibrary"
    
    # String manipulation keywords
    elif any(kw in keyword_lower for kw in [
        'convert to upper case', 'convert to lower case', 'split string',
        'get substring', 'replace string', 'strip string'
    ]):
        logger.debug(f"Pattern detection: '{keyword}' -> String")
        return "String"
    
    # Collections keywords
    elif any(kw in keyword_lower for kw in [
        'append to list', 'get from list', 'create list', 'create dictionary',
        'get from dictionary', 'set to dictionary', 'remove from list'
    ]) or ('create' in keyword_lower and 'list' in keyword_lower):
        logger.debug(f"Pattern detection: '{keyword}' -> Collections")
        return "Collections"
    
    # Operating System keywords
    elif any(kw in keyword_lower for kw in [
        'copy file', 'create directory', 'file should exist', 'directory should exist',
        'create file', 'remove file', 'remove directory', 'move file'
    ]):
        logger.debug(f"Pattern detection: '{keyword}' -> OperatingSystem")
        return "OperatingSystem"
    
    # Process keywords
    elif any(kw in keyword_lower for kw in [
        'start process', 'run process', 'terminate process', 'wait for process',
        'process should be running', 'process should be stopped'
    ]):
        logger.debug(f"Pattern detection: '{keyword}' -> Process")
        return "Process"
    
    # BuiltIn keywords (note: BuiltIn is automatically available)
    elif any(kw in keyword_lower for kw in [
        'log', 'set variable', 'should be equal', 'should contain', 'should not contain',
        'convert to string', 'convert to integer', 'convert to number', 'catenate',
        'create list', 'length should be', 'run keyword', 'run keyword if'
    ]):
        logger.debug(f"Pattern detection: '{keyword}' -> BuiltIn")
        return "BuiltIn"
    
    # PRIORITY 1: AppiumLibrary keywords (mobile testing) - COMPREHENSIVE LIST
    # Checked BEFORE Browser Library to prevent mis-detection
    elif any(kw in keyword_lower for kw in [
        # Application lifecycle
        'open application', 'close application', 'launch application', 'quit application',
        'remove application', 'reset application', 'background application', 'activate application',
        
        # Source and page operations (CRITICAL: includes 'get source')
        'get source', 'get page source', 'capture page screenshot', 'save screenshot',
        
        # Element interaction
        'tap', 'click element', 'press keycode', 'long press', 'swipe', 'scroll',
        'input text', 'input password', 'clear text', 'set text', 'input text into element',
        
        # Element queries
        'get text', 'get element attribute', 'get element location', 'get element size',
        'element should be visible', 'element should not be visible', 'element should be enabled',
        
        # Waits and conditions
        'wait until element is visible', 'wait until page contains element',
        'wait until element is enabled', 'wait until page does not contain element',
        
        # Context and session
        'get contexts', 'get current context', 'switch to context',
        
        # Device operations
        'shake', 'lock', 'unlock', 'get orientation', 'set orientation',
        'get network connection status', 'set network connection status', 'get device',
        
        # Advanced mobile features
        'execute script', 'install app', 'remove app', 'get app strings', 'end test'
    ]):
        logger.debug(f"Pattern detection: '{keyword}' -> AppiumLibrary")
        return "AppiumLibrary"
    
    logger.debug(f"No library detected for keyword: '{keyword}'")
    return None


def detect_library_type_from_keyword(keyword: str, arguments: list = None) -> str:
    """
    Detect library type for execution engine purposes.
    
    This replaces the execution_engine specific detection logic.
    
    Args:
        keyword: The keyword name
        arguments: Optional keyword arguments (for context)
        
    Returns:
        str: "browser", "selenium", or "auto"
    """
    # Import here to avoid circular imports
    from robotmcp.core.dynamic_keyword_orchestrator import get_keyword_discovery
    
    try:
        # Use dynamic keyword discovery for accurate library detection
        keyword_discovery = get_keyword_discovery()
        library = detect_library_from_keyword(keyword, keyword_discovery)
    except Exception as e:
        logger.debug(f"Failed to get dynamic keyword discovery: {e}, using pattern-based detection")
        library = detect_library_from_keyword(keyword)
    
    if library == "Browser":
        return "browser"
    elif library == "SeleniumLibrary":
        return "selenium"
    else:
        return "auto"  # Let the system auto-detect for other libraries