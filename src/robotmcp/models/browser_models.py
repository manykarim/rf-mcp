"""Browser-related data models."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BrowserState:
    """Represents Browser Library and SeleniumLibrary state."""
    # Common browser state
    browser_type: Optional[str] = None
    current_url: Optional[str] = None
    page_title: Optional[str] = None
    viewport: Dict[str, int] = field(default_factory=lambda: {"width": 1280, "height": 720})
    page_source: Optional[str] = None
    cookies: List[Dict[str, Any]] = field(default_factory=list)
    local_storage: Dict[str, str] = field(default_factory=dict)
    
    # Browser Library specific state
    browser_id: Optional[str] = None
    context_id: Optional[str] = None
    page_id: Optional[str] = None
    
    # SeleniumLibrary specific state
    driver_instance: Optional[Any] = None
    selenium_session_id: Optional[str] = None
    
    # Active library indicator ("browser" or "selenium" or None)
    active_library: Optional[str] = None
    session_storage: Dict[str, str] = field(default_factory=dict)
    page_elements: List[Dict[str, Any]] = field(default_factory=list)
    
    def is_browser_library_active(self) -> bool:
        """Check if Browser Library is the active library."""
        return self.active_library == "browser"
    
    def is_selenium_library_active(self) -> bool:
        """Check if SeleniumLibrary is the active library."""
        return self.active_library == "selenium"
    
    def has_browser_session(self) -> bool:
        """Check if there's an active browser session."""
        return (self.browser_id is not None or 
                self.driver_instance is not None)
    
    def has_page_loaded(self) -> bool:
        """Check if a page is currently loaded."""
        return (self.page_id is not None or 
                self.current_url is not None)
    
    def reset(self) -> None:
        """Reset browser state to initial values."""
        self.browser_type = None
        self.current_url = None
        self.page_title = None
        self.page_source = None
        self.cookies.clear()
        self.local_storage.clear()
        self.browser_id = None
        self.context_id = None
        self.page_id = None
        self.driver_instance = None
        self.selenium_session_id = None
        self.active_library = None
        self.session_storage.clear()
        self.page_elements.clear()