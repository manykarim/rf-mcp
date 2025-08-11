"""Session-related data models."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .execution_models import ExecutionStep
from .browser_models import BrowserState


@dataclass
class ExecutionSession:
    """Manages execution state for a test session."""
    session_id: str
    suite: Optional[Any] = None
    steps: List[ExecutionStep] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    imported_libraries: List[str] = field(default_factory=list)
    current_browser: Optional[str] = None
    browser_state: BrowserState = field(default_factory=BrowserState)
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    
    def add_step(self, step: ExecutionStep) -> None:
        """Add a successful step to the session."""
        if step.is_successful:
            self.steps.append(step)
            self.last_activity = datetime.now()
    
    def update_activity(self) -> None:
        """Update the last activity timestamp."""
        self.last_activity = datetime.now()
    
    def is_browser_session(self) -> bool:
        """Check if this session has browser automation capabilities."""
        return (self.browser_state.has_browser_session() or 
                'Browser' in self.imported_libraries or 
                'SeleniumLibrary' in self.imported_libraries)
    
    def get_active_library(self) -> Optional[str]:
        """Get the currently active browser automation library."""
        return self.browser_state.active_library
    
    def set_variable(self, name: str, value: Any) -> None:
        """Set a session variable."""
        self.variables[name] = value
        self.update_activity()
    
    def get_variable(self, name: str, default: Any = None) -> Any:
        """Get a session variable."""
        return self.variables.get(name, default)
    
    def import_library(self, library_name: str) -> None:
        """
        Mark a library as imported in this session.
        
        Enforces exclusion rules - Browser Library and SeleniumLibrary cannot
        coexist in the same session.
        
        Args:
            library_name: Name of the library to import
            
        Raises:
            ValueError: If trying to import a conflicting library
        """
        if library_name not in self.imported_libraries:
            # Enforce web automation library exclusion
            web_automation_libs = ['Browser', 'SeleniumLibrary']
            
            if library_name in web_automation_libs:
                # Check if another web automation library is already imported
                existing_web_libs = [lib for lib in self.imported_libraries if lib in web_automation_libs]
                
                if existing_web_libs and library_name not in existing_web_libs:
                    existing_lib = existing_web_libs[0]
                    raise ValueError(
                        f"Cannot import '{library_name}' - session already has '{existing_lib}'. "
                        f"Browser Library and SeleniumLibrary are mutually exclusive per session."
                    )
            
            self.imported_libraries.append(library_name)
            self.update_activity()
    
    def get_web_automation_library(self) -> Optional[str]:
        """Get the web automation library imported in this session."""
        web_automation_libs = ['Browser', 'SeleniumLibrary']
        for lib in self.imported_libraries:
            if lib in web_automation_libs:
                return lib
        return None
    
    def get_successful_steps(self) -> List[ExecutionStep]:
        """Get all successfully executed steps."""
        return [step for step in self.steps if step.is_successful]
    
    def get_failed_steps(self) -> List[ExecutionStep]:
        """Get all failed steps (Note: failed steps are not added to self.steps)."""
        # This would need to be tracked separately if needed
        return []
    
    @property
    def step_count(self) -> int:
        """Get the total number of successful steps."""
        return len(self.steps)
    
    @property
    def duration(self) -> float:
        """Calculate session duration in seconds."""
        return (self.last_activity - self.created_at).total_seconds()
    
    def cleanup(self) -> None:
        """Clean up session resources."""
        self.browser_state.reset()
        self.steps.clear()
        # Keep variables and imported_libraries for potential reuse