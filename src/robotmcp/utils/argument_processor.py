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
        
        for arg in args:
            if '=' in arg:
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
            Dict with converted arguments
        """
        converted_kwargs = {}
        
        # Get LibDoc argument information
        libdoc_args = self.get_libdoc_argument_info(keyword_name, library_name)
        
        for i, arg in enumerate(args):
            if '=' in arg:
                key, value = arg.split('=', 1)
                
                # Find matching LibDoc argument info
                param_type = 'str'  # default
                for arg_info in libdoc_args:
                    if arg_info.name == key:
                        param_type = self.detect_argument_type(arg_info.type_hint)
                        break
                
                # Convert value to appropriate type
                if param_type and param_type != 'str':
                    converted_kwargs[key] = self.convert_string_value(value, param_type)
                else:
                    converted_kwargs[key] = value
            else:
                # Positional argument
                param_type = 'str'  # default
                if i < len(libdoc_args):
                    param_type = self.detect_argument_type(libdoc_args[i].type_hint)
                
                if param_type and param_type != 'str':
                    converted_kwargs[f'arg_{i}'] = self.convert_string_value(arg, param_type)
                else:
                    converted_kwargs[f'arg_{i}'] = arg
        
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
        
        # Enum types (treat as string for now)
        if 'enum' in type_lower:
            return 'str'
            
        # Time-related (treat as string for now)
        if any(time_type in type_lower for time_type in ['timedelta', 'duration']):
            return 'str'
        
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
            if library_name:
                lib_info = rf_storage.get_library_info(library_name)
                if lib_info:
                    keyword_info = lib_info.get_keyword_info(keyword_name)
                    if keyword_info and hasattr(keyword_info, 'arg_types'):
                        return self.parse_argument_signature(keyword_info.arg_types)
            
            # Fallback: search across all libraries
            all_libs = rf_storage.get_all_library_names()
            for lib_name in all_libs:
                lib_info = rf_storage.get_library_info(lib_name)
                if lib_info:
                    keyword_info = lib_info.get_keyword_info(keyword_name)
                    if keyword_info and hasattr(keyword_info, 'arg_types'):
                        return self.parse_argument_signature(keyword_info.arg_types)
        
        except Exception as e:
            logger.debug(f"LibDoc argument info lookup failed for '{keyword_name}': {e}")
        
        return []