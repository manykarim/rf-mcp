"""State Manager for tracking and capturing application state during test execution."""

import json
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import re

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class DOMElement:
    """Represents a DOM element with its properties."""
    tag: str
    id: Optional[str] = None
    class_name: Optional[str] = None
    text: Optional[str] = None
    attributes: Dict[str, str] = field(default_factory=dict)
    xpath: Optional[str] = None
    css_selector: Optional[str] = None
    visible: bool = True
    clickable: bool = False
    children: List['DOMElement'] = field(default_factory=list)

@dataclass
class PageState:
    """Represents the current state of a web page."""
    url: str
    title: str
    elements: List[DOMElement] = field(default_factory=list)
    forms: List[Dict[str, Any]] = field(default_factory=list)
    links: List[Dict[str, str]] = field(default_factory=list)
    images: List[Dict[str, str]] = field(default_factory=list)
    scripts: List[str] = field(default_factory=list)
    cookies: Dict[str, str] = field(default_factory=dict)
    local_storage: Dict[str, str] = field(default_factory=dict)
    page_source: Optional[str] = None

@dataclass
class APIState:
    """Represents the state of API interactions."""
    last_request: Optional[Dict[str, Any]] = None
    last_response: Optional[Dict[str, Any]] = None
    request_history: List[Dict[str, Any]] = field(default_factory=list)
    base_url: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    cookies: Dict[str, str] = field(default_factory=dict)

@dataclass
class DatabaseState:
    """Represents the state of database interactions."""
    connection_string: Optional[str] = None
    last_query: Optional[str] = None
    last_result: Optional[List[Dict[str, Any]]] = None
    query_history: List[Dict[str, Any]] = field(default_factory=list)
    active_transaction: bool = False

@dataclass
class ApplicationState:
    """Complete application state snapshot."""
    timestamp: datetime
    session_id: str
    page_state: Optional[PageState] = None
    api_state: Optional[APIState] = None
    database_state: Optional[DatabaseState] = None
    variables: Dict[str, Any] = field(default_factory=dict)
    screenshots: List[str] = field(default_factory=list)

class StateManager:
    """Manages application state tracking across different contexts."""
    
    def __init__(self):
        self.state_history: Dict[str, List[ApplicationState]] = {}
        self.current_states: Dict[str, ApplicationState] = {}
        
    async def get_state(
        self,
        state_type: str = "all",
        elements_of_interest: List[str] = None,
        session_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Retrieve current application state.
        
        Args:
            state_type: Type of state to retrieve (dom, api, database, all)
            elements_of_interest: Specific elements to focus on
            session_id: Session identifier
            
        Returns:
            Current application state
        """
        try:
            if elements_of_interest is None:
                elements_of_interest = []
            
            # Get current state for session
            current_state = self.current_states.get(session_id)
            
            if not current_state:
                # Create initial state
                current_state = ApplicationState(
                    timestamp=datetime.now(),
                    session_id=session_id
                )
                self.current_states[session_id] = current_state
            
            result = {
                "success": True,
                "session_id": session_id,
                "timestamp": current_state.timestamp.isoformat(),
                "state_type": state_type
            }
            
            if state_type in ["dom", "all"]:
                dom_state = await self._get_dom_state(session_id, elements_of_interest)
                result["dom"] = dom_state
                
            if state_type in ["api", "all"]:
                api_state = await self._get_api_state(session_id)
                result["api"] = api_state
                
            if state_type in ["database", "all"]:
                db_state = await self._get_database_state(session_id)
                result["database"] = db_state
            
            # Always include variables
            result["variables"] = dict(current_state.variables)
            
            # Update state history
            await self._update_state_history(session_id, current_state)
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting application state: {e}")
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id
            }

    async def _get_dom_state(self, session_id: str, elements_of_interest: List[str]) -> Dict[str, Any]:
        """Get DOM state for web applications."""
        try:
            # In a real implementation, this would interface with Selenium/Browser library
            # For now, we'll simulate DOM state
            
            # Simulate getting page info from browser
            mock_page_state = await self._simulate_page_state(session_id)
            
            # Filter elements if specific ones are requested
            if elements_of_interest:
                filtered_elements = []
                for element in mock_page_state.elements:
                    if self._element_matches_interest(element, elements_of_interest):
                        filtered_elements.append(element)
                mock_page_state.elements = filtered_elements
            
            return {
                "url": mock_page_state.url,
                "title": mock_page_state.title,
                "elements": [
                    {
                        "tag": elem.tag,
                        "id": elem.id,
                        "class": elem.class_name,
                        "text": elem.text,
                        "attributes": elem.attributes,
                        "xpath": elem.xpath,
                        "css_selector": elem.css_selector,
                        "visible": elem.visible,
                        "clickable": elem.clickable
                    } for elem in mock_page_state.elements
                ],
                "forms": mock_page_state.forms,
                "links": mock_page_state.links,
                "element_count": len(mock_page_state.elements),
                "interactive_elements": len([e for e in mock_page_state.elements if e.clickable])
            }
            
        except Exception as e:
            logger.error(f"Error getting DOM state: {e}")
            return {"error": str(e)}

    async def _get_api_state(self, session_id: str) -> Dict[str, Any]:
        """Get API interaction state."""
        try:
            # In a real implementation, this would interface with RequestsLibrary
            # For now, we'll simulate API state
            
            mock_api_state = APIState(
                base_url="https://api.example.com",
                headers={"Content-Type": "application/json", "User-Agent": "RobotFramework"},
                last_request={
                    "method": "GET",
                    "url": "/users",
                    "timestamp": datetime.now().isoformat()
                },
                last_response={
                    "status_code": 200,
                    "headers": {"Content-Type": "application/json"},
                    "body": {"users": []},
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            return {
                "base_url": mock_api_state.base_url,
                "headers": mock_api_state.headers,
                "cookies": mock_api_state.cookies,
                "last_request": mock_api_state.last_request,
                "last_response": mock_api_state.last_response,
                "request_count": len(mock_api_state.request_history)
            }
            
        except Exception as e:
            logger.error(f"Error getting API state: {e}")
            return {"error": str(e)}

    async def _get_database_state(self, session_id: str) -> Dict[str, Any]:
        """Get database interaction state."""
        try:
            # In a real implementation, this would interface with DatabaseLibrary
            # For now, we'll simulate database state
            
            mock_db_state = DatabaseState(
                connection_string="sqlite:///test.db",
                last_query="SELECT * FROM users",
                last_result=[{"id": 1, "name": "test_user"}],
                active_transaction=False
            )
            
            return {
                "connection_string": mock_db_state.connection_string,
                "last_query": mock_db_state.last_query,
                "last_result": mock_db_state.last_result,
                "query_count": len(mock_db_state.query_history),
                "active_transaction": mock_db_state.active_transaction
            }
            
        except Exception as e:
            logger.error(f"Error getting database state: {e}")
            return {"error": str(e)}

    async def _simulate_page_state(self, session_id: str) -> PageState:
        """Simulate getting current page state."""
        # This would be replaced with actual Selenium/Browser library calls
        
        # Get current variables to determine state
        current_state = self.current_states.get(session_id)
        current_url = "about:blank"
        
        if current_state and "current_url" in current_state.variables:
            current_url = current_state.variables["current_url"]
        
        # Simulate different pages based on URL
        if "login" in current_url.lower():
            return self._create_login_page_state(current_url)
        elif "dashboard" in current_url.lower():
            return self._create_dashboard_page_state(current_url)
        elif "example.com" in current_url:
            return self._create_example_page_state(current_url)
        else:
            return self._create_generic_page_state(current_url)

    def _create_login_page_state(self, url: str) -> PageState:
        """Create a simulated login page state."""
        elements = [
            DOMElement(
                tag="input",
                id="username",
                attributes={"type": "text", "name": "username", "placeholder": "Username"},
                xpath="//input[@id='username']",
                css_selector="input#username",
                visible=True,
                clickable=True
            ),
            DOMElement(
                tag="input", 
                id="password",
                attributes={"type": "password", "name": "password", "placeholder": "Password"},
                xpath="//input[@id='password']",
                css_selector="input#password",
                visible=True,
                clickable=True
            ),
            DOMElement(
                tag="button",
                id="login-btn",
                text="Login",
                attributes={"type": "submit", "class": "btn btn-primary"},
                xpath="//button[@id='login-btn']",
                css_selector="button#login-btn",
                visible=True,
                clickable=True
            ),
            DOMElement(
                tag="a",
                text="Forgot Password?",
                attributes={"href": "/forgot-password"},
                xpath="//a[text()='Forgot Password?']",
                css_selector="a[href='/forgot-password']",
                visible=True,
                clickable=True
            )
        ]
        
        forms = [
            {
                "id": "login-form",
                "action": "/login",
                "method": "POST",
                "fields": ["username", "password"]
            }
        ]
        
        return PageState(
            url=url,
            title="Login - Example App",
            elements=elements,
            forms=forms,
            links=[{"text": "Forgot Password?", "href": "/forgot-password"}]
        )

    def _create_dashboard_page_state(self, url: str) -> PageState:
        """Create a simulated dashboard page state."""
        elements = [
            DOMElement(
                tag="h1",
                text="Dashboard",
                class_name="page-title",
                xpath="//h1[@class='page-title']",
                css_selector="h1.page-title",
                visible=True
            ),
            DOMElement(
                tag="button",
                id="new-item-btn",
                text="Create New Item",
                class_name="btn btn-success",
                xpath="//button[@id='new-item-btn']",
                css_selector="button#new-item-btn",
                visible=True,
                clickable=True
            ),
            DOMElement(
                tag="table",
                id="items-table",
                class_name="table table-striped",
                xpath="//table[@id='items-table']",
                css_selector="table#items-table",
                visible=True
            ),
            DOMElement(
                tag="a",
                text="Logout",
                class_name="logout-link",
                attributes={"href": "/logout"},
                xpath="//a[@class='logout-link']",
                css_selector="a.logout-link",
                visible=True,
                clickable=True
            )
        ]
        
        return PageState(
            url=url,
            title="Dashboard - Example App",
            elements=elements,
            links=[{"text": "Logout", "href": "/logout"}]
        )

    def _create_example_page_state(self, url: str) -> PageState:
        """Create a simulated example.com page state."""
        elements = [
            DOMElement(
                tag="h1",
                text="Example Domain",
                xpath="//h1",
                css_selector="h1",
                visible=True
            ),
            DOMElement(
                tag="p",
                text="This domain is for use in illustrative examples in documents.",
                xpath="//p[1]",
                css_selector="p:first-child",
                visible=True
            ),
            DOMElement(
                tag="a",
                text="More information...",
                attributes={"href": "https://www.iana.org/domains/example"},
                xpath="//a",
                css_selector="a",
                visible=True,
                clickable=True
            )
        ]
        
        return PageState(
            url=url,
            title="Example Domain",
            elements=elements,
            links=[{"text": "More information...", "href": "https://www.iana.org/domains/example"}]
        )

    def _create_generic_page_state(self, url: str) -> PageState:
        """Create a generic page state."""
        elements = [
            DOMElement(
                tag="html",
                xpath="//html",
                css_selector="html",
                visible=True
            ),
            DOMElement(
                tag="body",
                xpath="//body",
                css_selector="body",
                visible=True
            )
        ]
        
        return PageState(
            url=url,
            title="Generic Page",
            elements=elements
        )

    def _element_matches_interest(self, element: DOMElement, interests: List[str]) -> bool:
        """Check if an element matches any of the interests."""
        for interest in interests:
            interest_lower = interest.lower()
            
            # Check ID
            if element.id and interest_lower in element.id.lower():
                return True
            
            # Check class name
            if element.class_name and interest_lower in element.class_name.lower():
                return True
            
            # Check text content
            if element.text and interest_lower in element.text.lower():
                return True
            
            # Check tag name
            if interest_lower in element.tag.lower():
                return True
            
            # Check attributes
            for attr_value in element.attributes.values():
                if isinstance(attr_value, str) and interest_lower in attr_value.lower():
                    return True
        
        return False

    async def _update_state_history(self, session_id: str, state: ApplicationState) -> None:
        """Update state history for a session."""
        if session_id not in self.state_history:
            self.state_history[session_id] = []
        
        # Keep only last 50 state snapshots
        self.state_history[session_id].append(state)
        if len(self.state_history[session_id]) > 50:
            self.state_history[session_id] = self.state_history[session_id][-50:]

    async def update_variables(self, session_id: str, variables: Dict[str, Any]) -> None:
        """Update session variables."""
        if session_id not in self.current_states:
            self.current_states[session_id] = ApplicationState(
                timestamp=datetime.now(),
                session_id=session_id
            )
        
        self.current_states[session_id].variables.update(variables)
        self.current_states[session_id].timestamp = datetime.now()

    async def get_state_history(self, session_id: str, limit: int = 10) -> Dict[str, Any]:
        """Get state history for a session."""
        try:
            history = self.state_history.get(session_id, [])
            
            # Get the most recent states
            recent_history = history[-limit:] if len(history) > limit else history
            
            return {
                "success": True,
                "session_id": session_id,
                "total_states": len(history),
                "returned_states": len(recent_history),
                "history": [
                    {
                        "timestamp": state.timestamp.isoformat(),
                        "has_page_state": state.page_state is not None,
                        "has_api_state": state.api_state is not None,
                        "has_database_state": state.database_state is not None,
                        "variable_count": len(state.variables)
                    } for state in recent_history
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting state history: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def clear_session_state(self, session_id: str) -> Dict[str, Any]:
        """Clear state for a session."""
        try:
            if session_id in self.current_states:
                del self.current_states[session_id]
            
            if session_id in self.state_history:
                del self.state_history[session_id]
            
            return {
                "success": True,
                "message": f"State cleared for session '{session_id}'"
            }
            
        except Exception as e:
            logger.error(f"Error clearing session state: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def compare_states(self, session_id: str, state1_index: int, state2_index: int) -> Dict[str, Any]:
        """Compare two states from history."""
        try:
            history = self.state_history.get(session_id, [])
            
            if state1_index >= len(history) or state2_index >= len(history):
                return {
                    "success": False,
                    "error": "Invalid state indices"
                }
            
            state1 = history[state1_index]
            state2 = history[state2_index]
            
            # Compare basic properties
            differences = {
                "timestamp_diff": (state2.timestamp - state1.timestamp).total_seconds(),
                "variable_changes": self._compare_dicts(state1.variables, state2.variables),
                "state_types_changed": []
            }
            
            # Add state type specific differences
            if state1.page_state and state2.page_state:
                if state1.page_state.url != state2.page_state.url:
                    differences["state_types_changed"].append("page_url")
                if len(state1.page_state.elements) != len(state2.page_state.elements):
                    differences["state_types_changed"].append("page_elements")
            
            return {
                "success": True,
                "session_id": session_id,
                "state1_timestamp": state1.timestamp.isoformat(),
                "state2_timestamp": state2.timestamp.isoformat(),
                "differences": differences
            }
            
        except Exception as e:
            logger.error(f"Error comparing states: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _compare_dicts(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two dictionaries and return differences."""
        added = {k: v for k, v in dict2.items() if k not in dict1}
        removed = {k: v for k, v in dict1.items() if k not in dict2}
        changed = {k: {"old": dict1[k], "new": dict2[k]} 
                  for k in dict1.keys() & dict2.keys() 
                  if dict1[k] != dict2[k]}
        
        return {
            "added": added,
            "removed": removed,
            "changed": changed
        }