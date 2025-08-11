"""Argument processing and type conversion utilities."""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from robotmcp.models.library_models import KeywordInfo, ParsedArguments, ArgumentInfo
from robotmcp.utils.rf_libdoc_integration import get_rf_doc_storage

logger = logging.getLogger(__name__)


class ArgumentProcessor:
    """Handles argument parsing, type conversion, and LibDoc integration."""
    
    def parse_arguments(self, args: List[str]) -> ParsedArguments:
        """Parse a list of arguments into positional and named arguments."""
        parsed = ParsedArguments()
        
        # Common Robot Framework locator strategy prefixes that should remain as positional arguments
        locator_strategies = {
            'id=', 'name=', 'class=', 'tag=', 'css=', 'xpath=', 'text=', 'link=', 
            'partial_link=', 'link_text=', 'partial_link_text=', 'dom=', 'jquery=',
            'id:', 'name:', 'class:', 'tag:', 'css:', 'xpath:', 'text:', 'link:',
            'partial_link:', 'link_text:', 'partial_link_text:', 'dom:', 'jquery:'
        }
        
        for arg in args:
            if '=' in arg:
                # Check if this is a locator strategy prefix (should remain positional)
                is_locator_strategy = any(arg.lower().startswith(prefix) for prefix in locator_strategies)
                
                if is_locator_strategy:
                    # Treat as positional argument
                    if not parsed.named:
                        parsed.positional.append(arg)
                    else:
                        # After named arguments, treat as named with empty key
                        parsed.named[arg] = ""
                else:
                    # Named argument
                    key, value = arg.split('=', 1)
                    parsed.named[key] = value
            else:
                # Positional argument (unless we already have named args)
                if not parsed.named:
                    parsed.positional.append(arg)
                else:
                    # After named arguments, treat as named with empty key
                    parsed.named[arg] = ""
        
        return parsed
    
    def convert_browser_arguments(self, keyword_name: str, args: List[str], library_name: str = None) -> Dict[str, Any]:
        """
        Convert arguments for Browser Library keywords using LibDoc type information.
        
        Args:
            keyword_name: Name of the keyword
            args: List of arguments
            library_name: Optional library name for context
            
        Returns:
            Dict with converted arguments and metadata about LibDoc usage
        """
        converted_kwargs = {}
        libdoc_type_info = {}  # Track which arguments have LibDoc type info
        
        # Get LibDoc argument information
        libdoc_args = self.get_libdoc_argument_info(keyword_name, library_name)
        
        # Common Robot Framework locator strategy prefixes that should remain as positional arguments
        locator_strategies = {
            'id=', 'name=', 'class=', 'tag=', 'css=', 'xpath=', 'text=', 'link=', 
            'partial_link=', 'link_text=', 'partial_link_text=', 'dom=', 'jquery=',
            'id:', 'name:', 'class:', 'tag:', 'css:', 'xpath:', 'text:', 'link:',
            'partial_link:', 'link_text:', 'partial_link_text:', 'dom:', 'jquery:'
        }
        
        for i, arg in enumerate(args):
            if '=' in arg:
                # Check if this is a locator strategy prefix (should remain positional)
                is_locator_strategy = any(arg.lower().startswith(prefix) for prefix in locator_strategies)
                
                if is_locator_strategy:
                    # Treat as positional argument
                    param_type = 'str'  # default
                    has_libdoc_info = False
                    if i < len(libdoc_args):
                        param_type = self.detect_argument_type(libdoc_args[i].type_hint)
                        has_libdoc_info = True
                    
                    arg_key = f'arg_{i}'
                    libdoc_type_info[arg_key] = has_libdoc_info
                    
                    # Only apply smart detection if LibDoc didn't provide type info
                    if param_type == 'str' and not has_libdoc_info:
                        param_type = self._smart_detect_argument_type(arg_key, arg, library_name)
                    
                    if param_type and param_type != 'str':
                        converted_kwargs[arg_key] = self.convert_string_value(arg, param_type)
                    else:
                        converted_kwargs[arg_key] = arg
                else:
                    # Named argument
                    key, value = arg.split('=', 1)
                    
                    # Find matching LibDoc argument info
                    param_type = 'str'  # default
                    has_libdoc_info = False
                    for arg_info in libdoc_args:
                        if arg_info.name == key:
                            param_type = self.detect_argument_type(arg_info.type_hint)
                            has_libdoc_info = True
                            break
                    
                    libdoc_type_info[key] = has_libdoc_info
                    
                    # If LibDoc didn't provide type info, try smart detection for common patterns
                    if param_type == 'str' and not has_libdoc_info:
                        param_type = self._smart_detect_argument_type(key, value, library_name)
                    
                    # Convert value to appropriate type
                    if param_type and param_type != 'str':
                        converted_kwargs[key] = self.convert_string_value(value, param_type)
                    else:
                        converted_kwargs[key] = value
            else:
                # Positional argument
                param_type = 'str'  # default
                has_libdoc_info = False
                if i < len(libdoc_args):
                    param_type = self.detect_argument_type(libdoc_args[i].type_hint)
                    has_libdoc_info = True
                
                arg_key = f'arg_{i}'
                libdoc_type_info[arg_key] = has_libdoc_info
                
                # Only apply smart detection if LibDoc didn't provide type info
                if param_type == 'str' and not has_libdoc_info:
                    param_type = self._smart_detect_argument_type(arg_key, arg, library_name)
                
                if param_type and param_type != 'str':
                    converted_kwargs[arg_key] = self.convert_string_value(arg, param_type)
                else:
                    converted_kwargs[arg_key] = arg
        
        # Add metadata about LibDoc usage
        converted_kwargs['_libdoc_type_info'] = libdoc_type_info
        
        return converted_kwargs
    
    def convert_string_value(self, value: str, target_type: str):
        """Convert string value to target type."""
        
        if target_type == 'bool':
            return value.lower().strip() in ['true', '1', 'yes', 'on']
        
        elif target_type == 'int':
            try:
                return int(value)
            except ValueError:
                return value
        
        elif target_type == 'float':
            try:
                return float(value)
            except ValueError:
                return value
        
        elif target_type == 'browser_enum':
            # Handle SupportedBrowsers enum conversion using built-in enum access
            try:
                from Browser.utils.data_types import SupportedBrowsers
                # Use Python's built-in enum string access (case-sensitive)
                return SupportedBrowsers[value.lower()]
            except (ImportError, KeyError):
                # If can't import the enum or value not found, fallback to chromium
                try:
                    from Browser.utils.data_types import SupportedBrowsers
                    return SupportedBrowsers.chromium
                except ImportError:
                    return value
        
        elif target_type.endswith('_enum'):
            # Handle all Browser Library enum conversions using Python's built-in enum access
            return self._convert_browser_enum(target_type, value)
        
        elif target_type in ['dict', 'list', 'tuple']:
            try:
                # First try ast.literal_eval for safe evaluation of Python literals
                import ast
                result = ast.literal_eval(value)
                
                # Convert to target type if needed
                if target_type == 'tuple' and isinstance(result, (list, dict)):
                    return tuple(result) if isinstance(result, list) else tuple(result.items())
                elif target_type == 'list' and isinstance(result, (tuple, dict)):
                    return list(result) if isinstance(result, tuple) else list(result.items())
                elif target_type == 'dict' and isinstance(result, (list, tuple)):
                    # Try to convert list/tuple of pairs to dict
                    return dict(result) if len(result) > 0 and all(len(item) == 2 for item in result if isinstance(item, (list, tuple))) else result
                
                return result
                
            except (ValueError, SyntaxError):
                # If ast.literal_eval fails, try JSON parsing
                try:
                    result = json.loads(value)
                    
                    # Convert to target type if needed
                    if target_type == 'tuple':
                        return tuple(result) if isinstance(result, list) else result
                    elif target_type == 'list':
                        return list(result) if isinstance(result, dict) else result
                    
                    return result
                    
                except json.JSONDecodeError:
                    # If both fail, return original string
                    return value
        
        # For enum, timedelta, etc., return as string for now
        return value
    
    def parse_argument_signature(self, signature: str) -> List[ArgumentInfo]:
        """Parse Robot Framework argument signature to extract type information."""
        
        if not signature or not signature.strip():
            return []
        
        # Split by comma, but respect nested brackets/parentheses
        args = self.split_signature_args(signature)
        parsed_args = []
        
        for arg in args:
            arg = arg.strip()
            if not arg:
                continue
                
            # Parse individual argument
            arg_info = self.parse_single_argument(arg)
            if arg_info:
                parsed_args.append(arg_info)
        
        return parsed_args
    
    def split_signature_args(self, signature: str) -> List[str]:
        """Split signature by comma, respecting nested structures."""
        args = []
        current_arg = ""
        bracket_depth = 0
        paren_depth = 0
        
        for char in signature:
            if char in '[{':
                bracket_depth += 1
            elif char in ']}':
                bracket_depth -= 1
            elif char == '(':
                paren_depth += 1
            elif char == ')':
                paren_depth -= 1
            elif char == ',' and bracket_depth == 0 and paren_depth == 0:
                args.append(current_arg.strip())
                current_arg = ""
                continue
            
            current_arg += char
        
        if current_arg.strip():
            args.append(current_arg.strip())
        
        return args
    
    def parse_single_argument(self, arg: str) -> Optional[ArgumentInfo]:
        """Parse a single argument specification."""
        # Pattern: name: type = default
        # Examples: 
        # - timeout: float = 30.0
        # - *args: List[str]
        # - **kwargs: Dict[str, Any]
        # - viewport: ViewportDimensions = None
        
        # Handle varargs and kwargs
        is_varargs = arg.startswith('*') and not arg.startswith('**')
        is_kwargs = arg.startswith('**')
        
        if is_varargs:
            arg = arg[1:]  # Remove *
        elif is_kwargs:
            arg = arg[2:]  # Remove **
        
        # Split by colon for type hints
        if ':' in arg:
            name_part, type_part = arg.split(':', 1)
            name = name_part.strip()
            
            # Check for default value
            default_value = None
            if '=' in type_part:
                type_hint, default_part = type_part.split('=', 1)
                type_hint = type_hint.strip()
                default_value = default_part.strip()
            else:
                type_hint = type_part.strip()
        else:
            # No type hint, might have default value
            if '=' in arg:
                name, default_part = arg.split('=', 1)
                name = name.strip()
                default_value = default_part.strip()
                type_hint = 'str'  # Default type
            else:
                name = arg.strip()
                type_hint = 'str'  # Default type
                default_value = None
        
        if name:
            return ArgumentInfo(
                name=name,
                type_hint=type_hint,
                default_value=default_value,
                is_varargs=is_varargs,
                is_kwargs=is_kwargs
            )
        
        return None
    
    def detect_argument_type(self, type_hint: str) -> str:
        """Detect the basic type from a type hint."""
        if not type_hint:
            return 'str'
        
        type_lower = type_hint.lower()
        
        # Dictionary types
        if any(dict_type in type_lower for dict_type in ['dict', 'mapping', 'viewportdimensions']):
            return 'dict'
        
        # List/sequence types  
        if any(list_type in type_lower for list_type in ['list', 'sequence', 'tuple']):
            return 'list' if 'list' in type_lower else 'tuple'
        
        # Boolean types
        if 'bool' in type_lower:
            return 'bool'
        
        # Numeric types
        if 'int' in type_lower:
            return 'int'
        if 'float' in type_lower:
            return 'float'
        
        # Browser Library specific enum types that need special handling
        if any(browser_enum in type_lower for browser_enum in ['supportedbrowsers', 'browserlibrary']):
            return 'browser_enum'
        if 'selectattribute' in type_lower:
            return 'select_attribute_enum'
        if 'mousebutton' in type_lower:
            return 'mouse_button_enum'
        if 'elementstate' in type_lower:
            return 'element_state_enum'
        if 'pageloadstates' in type_lower:
            return 'page_load_states_enum'
        if 'dialogaction' in type_lower:
            return 'dialog_action_enum'
        if 'requestmethod' in type_lower:
            return 'request_method_enum'
        if 'scrollbehavior' in type_lower:
            return 'scroll_behavior_enum'
        
        # Enum types (treat as string for now)
        if 'enum' in type_lower:
            return 'str'
            
        # Time-related (treat as string for now)
        if any(time_type in type_lower for time_type in ['timedelta', 'duration']):
            return 'str'
        
        # Default to string
        return 'str'
    
    def _convert_browser_enum(self, target_type: str, value: str):
        """Convert string to Browser Library enum using Python's built-in enum access."""
        
        # Map target types to enum classes and their string formats
        enum_mapping = {
            'select_attribute_enum': ('SelectAttribute', str.lower),
            'mouse_button_enum': ('MouseButton', str.lower), 
            'element_state_enum': ('ElementState', str.lower),
            'page_load_states_enum': ('PageLoadStates', str.lower),
            'dialog_action_enum': ('DialogAction', str.lower),
            'request_method_enum': ('RequestMethod', str.upper),
            'scroll_behavior_enum': ('ScrollBehavior', str.lower),
        }
        
        if target_type not in enum_mapping:
            return value
            
        enum_name, formatter = enum_mapping[target_type]
        formatted_value = formatter(value)
        
        try:
            # Dynamically import the enum class
            from Browser.utils.data_types import (
                SelectAttribute, MouseButton, ElementState, PageLoadStates,
                DialogAction, RequestMethod, ScrollBehavior
            )
            
            # Get the enum class
            enum_class = locals()[enum_name]
            
            # Use Python's built-in enum string access
            try:
                return enum_class[formatted_value]
            except KeyError:
                # Invalid enum value - fail loudly with helpful error message
                valid_values = [e.name for e in enum_class]
                raise ValueError(
                    f"Invalid {enum_name} value: '{value}' (formatted as '{formatted_value}'). "
                    f"Valid options are: {valid_values}"
                )
                
        except ImportError:
            # If can't import Browser library, return original string
            return value
    
    def _smart_detect_argument_type(self, arg_name: str, arg_value: str, library_name: str = None) -> str:
        """Smart detection for argument types when LibDoc doesn't provide enough info."""
        
        # Common boolean parameters across libraries
        boolean_params = {
            # Browser Library
            'force', 'headless', 'devtools', 'chromiumSandbox', 'handleSIGHUP', 
            'handleSIGINT', 'handleSIGTERM', 'noWaitAfter',
            # SeleniumLibrary
            'implicit_wait', 'page_load_timeout', 'desired_capabilities',
            # RequestsLibrary  
            'verify', 'allow_redirects',
            # General
            'debug', 'verbose', 'enabled', 'disabled', 'visible', 'hidden'
        }
        
        # Common integer/numeric parameters
        numeric_params = {
            # Browser Library
            'timeout', 'width', 'height', 'clickCount', 'delay',
            # SeleniumLibrary
            'index', 'size', 'position',
            # RequestsLibrary
            'timeout', 'max_retries',
            # General
            'count', 'limit', 'retry', 'attempts'
        }
        
        arg_name_lower = arg_name.lower()
        
        # Check if it's a known boolean parameter
        if arg_name_lower in boolean_params:
            return 'bool'
        
        # Check if it's a known numeric parameter
        if arg_name_lower in numeric_params:
            return 'int'
        
        # Smart pattern detection based on value
        if arg_value.lower() in ['true', 'false']:
            return 'bool'
        
        if arg_value.isdigit():
            return 'int'
        
        if arg_value.startswith('{') and arg_value.endswith('}'):
            return 'dict'
        
        if arg_value.startswith('[') and arg_value.endswith(']'):
            return 'list'
        
        # Default to string
        return 'str'
    
    def get_libdoc_argument_info(self, keyword_name: str, library_name: str = None) -> List[ArgumentInfo]:
        """Get argument information from LibDoc for a keyword."""
        
        # Get LibDoc storage
        rf_storage = get_rf_doc_storage()
        
        if not rf_storage.is_available():
            return []
        
        try:
            # Try to find keyword in LibDoc
            keyword_info = rf_storage.find_keyword(keyword_name)
            if keyword_info:
                # Check if library matches if specified
                if library_name and keyword_info.library.lower() != library_name.lower():
                    return []
                
                # Parse argument information - try arg_types first, then fall back to args
                if keyword_info.arg_types:
                    # arg_types is a list of strings, convert to signature format
                    signature_parts = []
                    for i, arg_type in enumerate(keyword_info.arg_types):
                        if i < len(keyword_info.args):
                            arg_name = keyword_info.args[i]
                            signature_parts.append(f"{arg_name}: {arg_type}")
                        else:
                            signature_parts.append(f"arg_{i}: {arg_type}")
                    
                    signature = ", ".join(signature_parts)
                    return self.parse_argument_signature(signature)
                elif keyword_info.args:
                    # If no arg_types, try to parse from args (which may contain type info)
                    # Format: ['browser: SupportedBrowsers = chromium', 'headless: bool = True', ...]
                    return self.parse_argument_signature(", ".join(keyword_info.args))
        
        except Exception as e:
            logger.debug(f"LibDoc argument info lookup failed for '{keyword_name}': {e}")
        
        return []