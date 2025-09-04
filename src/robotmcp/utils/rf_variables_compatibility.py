"""Robot Framework Variables compatibility wrapper."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class CompatibleVariables:
    """Variables wrapper that provides backward compatibility for set_global method."""
    
    def __init__(self, original_variables):
        """Initialize with original Variables instance."""
        self._original = original_variables
        
    def __getattr__(self, name: str) -> Any:
        """Delegate all attribute access to original Variables."""
        return getattr(self._original, name)
        
    def __setitem__(self, key: str, value: Any) -> None:
        """Delegate item assignment to original Variables."""
        self._original[key] = value
        
    def __getitem__(self, key: str) -> Any:
        """Delegate item access to original Variables."""
        return self._original[key]
        
    def __contains__(self, key: str) -> bool:
        """Delegate contains check to original Variables."""
        return key in self._original
        
    def set_global(self, name: str, value: Any) -> None:
        """Provide set_global method for BuiltIn compatibility.
        
        This method provides compatibility for older RF versions that expected
        set_global method on Variables class. Maps to __setitem__.
        """
        # Ensure name has variable syntax
        if not name.startswith('${'):
            name = f"${{{name}}}"
        if not name.endswith('}'):
            name = f"{name}" + "}"
            
        logger.debug(f"Variables.set_global({name}, {value}) -> __setitem__")
        self._original[name] = value

    # Added for RF 6/7 compatibility: BuiltIn uses variables.set_test/set_suite
    def _normalize_var_name(self, name: str) -> str:
        if not name.startswith('${'):
            name = f"${{{name}}}"
        if not name.endswith('}'):
            name = f"{name}" + "}"
        return name

    def set_test(self, name: str, value: Any) -> None:
        """Set test-scoped variable (compatibility shim)."""
        name = self._normalize_var_name(name)
        logger.debug(f"Variables.set_test({name}, {value}) -> __setitem__")
        self._original[name] = value

    def set_suite(self, name: str, value: Any) -> None:
        """Set suite-scoped variable (compatibility shim)."""
        name = self._normalize_var_name(name)
        logger.debug(f"Variables.set_suite({name}, {value}) -> __setitem__")
        self._original[name] = value
        
    def get_variable(self, name: str, default: Any = None) -> Any:
        """Get variable value with RF-compatible name handling."""
        # Ensure name has variable syntax
        if not name.startswith('${'):
            name = f"${{{name}}}"
        if not name.endswith('}'):
            name = f"{name}" + "}"
            
        try:
            return self._original[name]
        except KeyError:
            if default is not None:
                return default
            raise
            
    def replace_variables(self, text: str) -> str:
        """Replace variables in text using RF's replace_string method."""
        return self._original.replace_string(text)


class CompatibleNamespace:
    """Namespace wrapper that provides BuiltIn-compatible interface."""
    
    def __init__(self, original_namespace, compatible_variables):
        """Initialize with original namespace and compatible variables."""
        self._original = original_namespace
        self.variables = compatible_variables
        
    def __getattr__(self, name: str) -> Any:
        """Delegate all other attribute access to original namespace."""
        return getattr(self._original, name)


def create_compatible_variables(original_variables):
    """Create Variables instance compatible with BuiltIn expectations."""
    return CompatibleVariables(original_variables)


def create_compatible_namespace(original_namespace, compatible_variables):
    """Create Namespace instance compatible with BuiltIn expectations."""
    return CompatibleNamespace(original_namespace, compatible_variables)
