"""Configuration data models."""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ExecutionConfig:
    """Centralized configuration for execution engine."""
    
    # Timeout settings
    DEFAULT_TIMEOUT: int = 10000  # milliseconds
    SESSION_CLEANUP_TIMEOUT: int = 1800  # seconds (30 minutes)
    
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
        
        return errors
