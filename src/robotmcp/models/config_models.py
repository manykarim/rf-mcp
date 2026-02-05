"""Configuration data models."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from robotmcp.domains.timeout import TimeoutPolicy


@dataclass
class ExecutionConfig:
    """Centralized configuration for execution engine."""

    # Timeout settings (legacy - prefer TimeoutPolicy for new code)
    DEFAULT_TIMEOUT: int = 10000  # milliseconds (legacy, use timeout_policy)
    SESSION_CLEANUP_TIMEOUT: int = 1800  # seconds (30 minutes)

    # New DDD timeout settings (matching Playwright MCP dual timeout strategy)
    # ACTION_TIMEOUT: 5s for clicks, typing (fast failure detection)
    # NAVIGATION_TIMEOUT: 60s for page loads (allow time for network)
    ACTION_TIMEOUT: int = 5000  # milliseconds - element actions (click, fill, etc.)
    NAVIGATION_TIMEOUT: int = 60000  # milliseconds - page navigation
    ASSERTION_TIMEOUT: int = 10000  # milliseconds - assertion retries
    READ_TIMEOUT: int = 2000  # milliseconds - read operations
    PRE_VALIDATION_TIMEOUT: int = 500  # milliseconds - fast pre-validation checks
    
    # Page source settings
    DEFAULT_FILTERING_LEVEL: str = "standard"
    MAX_PAGE_SOURCE_SIZE: int = 1000000  # characters
    PAGE_SOURCE_PREVIEW_SIZE: int = 2000  # characters
    
    # Browser settings
    DEFAULT_BROWSER_TYPE: str = "chromium"
    DEFAULT_HEADLESS: bool = True
    DEFAULT_VIEWPORT_WIDTH: int = 1280
    DEFAULT_VIEWPORT_HEIGHT: int = 720
    
    # Library preferences
    PREFERRED_WEB_LIBRARY: str = "Browser"  # Browser or SeleniumLibrary
    AUTO_LIBRARY_SELECTION: bool = True
    
    # Execution settings
    CAPTURE_PAGE_SOURCE_ON_DOM_CHANGE: bool = True
    CAPTURE_PAGE_SOURCE_ON_ERROR: bool = True
    MAX_EXECUTION_TIME: int = 300  # seconds
    
    # Filtering settings
    REMOVE_SCRIPTS_IN_STANDARD: bool = True
    REMOVE_STYLES_IN_STANDARD: bool = True
    KEEP_HIDDEN_ELEMENTS_IN_STANDARD: bool = True
    
    # Locator conversion settings (disabled during execution; handle in suite generation)
    ENABLE_LOCATOR_CONVERSION: bool = False
    CONVERT_JQUERY_SELECTORS: bool = True
    CONVERT_CASCADED_SELECTORS: bool = True
    ADD_EXPLICIT_SELECTOR_STRATEGIES: bool = True  # Add css=, xpath=, text= prefixes
    
    # Error handling
    FAIL_FAST_ON_ENUM_ERRORS: bool = True
    PROVIDE_ENUM_SUGGESTIONS: bool = True
    LOG_FAILED_CONVERSIONS: bool = True
    
    @classmethod
    def from_dict(cls, config: Dict) -> 'ExecutionConfig':
        """Create configuration from dictionary."""
        instance = cls()
        for key, value in config.items():
            if hasattr(instance, key):
                setattr(instance, key, value)
        return instance
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_')
        }
    
    def update(self, **kwargs) -> None:
        """Update configuration values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration key: {key}")
    
    def validate(self) -> List[str]:
        """Validate configuration values and return any errors."""
        errors = []
        
        if self.DEFAULT_TIMEOUT <= 0:
            errors.append("DEFAULT_TIMEOUT must be positive")
        
        if self.MAX_PAGE_SOURCE_SIZE <= 0:
            errors.append("MAX_PAGE_SOURCE_SIZE must be positive")
        
        if self.DEFAULT_FILTERING_LEVEL not in ["minimal", "standard", "aggressive"]:
            errors.append("DEFAULT_FILTERING_LEVEL must be 'minimal', 'standard', or 'aggressive'")
        
        if self.PREFERRED_WEB_LIBRARY not in ["Browser", "SeleniumLibrary"]:
            errors.append("PREFERRED_WEB_LIBRARY must be 'Browser' or 'SeleniumLibrary'")

        # Validate DDD timeout settings
        if self.ACTION_TIMEOUT <= 0:
            errors.append("ACTION_TIMEOUT must be positive")

        if self.NAVIGATION_TIMEOUT <= 0:
            errors.append("NAVIGATION_TIMEOUT must be positive")

        if self.ACTION_TIMEOUT > 30000:
            errors.append("ACTION_TIMEOUT should not exceed 30 seconds")

        if self.NAVIGATION_TIMEOUT > 300000:
            errors.append("NAVIGATION_TIMEOUT should not exceed 5 minutes")

        return errors

    def create_timeout_policy(self, session_id: str) -> "TimeoutPolicy":
        """Create a TimeoutPolicy from this configuration.

        This method integrates the legacy ExecutionConfig with the new
        DDD TimeoutPolicy aggregate from the timeout domain.

        Args:
            session_id: The session ID for the policy.

        Returns:
            A TimeoutPolicy configured with values from this config.
        """
        from robotmcp.domains.timeout import TimeoutPolicy, Milliseconds

        policy = TimeoutPolicy.create_default(session_id)

        # Apply config overrides if they differ from defaults
        if self.ACTION_TIMEOUT != 5000:
            policy = policy.with_action_timeout(Milliseconds(self.ACTION_TIMEOUT))

        if self.NAVIGATION_TIMEOUT != 60000:
            policy = policy.with_navigation_timeout(Milliseconds(self.NAVIGATION_TIMEOUT))

        return policy

    def get_timeout_for_action_type(self, action_type: str) -> int:
        """Get the appropriate timeout for an action type.

        Uses the dual timeout strategy:
        - Navigation actions: NAVIGATION_TIMEOUT (60s default)
        - Element actions: ACTION_TIMEOUT (5s default)
        - Read actions: READ_TIMEOUT (2s default)

        Args:
            action_type: The type of action (e.g., 'click', 'navigate', 'get_text')

        Returns:
            Timeout in milliseconds.
        """
        navigation_actions = {'navigate', 'reload', 'go_back', 'go_forward', 'wait_for_navigation'}
        read_actions = {'get_text', 'get_attribute', 'get_state', 'get_value', 'snapshot', 'screenshot'}

        action_lower = action_type.lower()

        if action_lower in navigation_actions:
            return self.NAVIGATION_TIMEOUT
        elif action_lower in read_actions:
            return self.READ_TIMEOUT
        else:
            return self.ACTION_TIMEOUT
