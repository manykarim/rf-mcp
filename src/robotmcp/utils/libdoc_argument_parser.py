"""LibDoc-based argument parser for Robot Framework keywords."""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from robotmcp.models.library_models import ParsedArguments
from robotmcp.utils.rf_libdoc_integration import get_rf_doc_storage

logger = logging.getLogger(__name__)


class LibDocArgumentParser:
    """Parser that uses Robot Framework LibDoc to correctly parse keyword arguments."""
    
    def __init__(self):
        self.rf_storage = get_rf_doc_storage()
    
    def parse_arguments_with_libdoc(
        self, 
        keyword_name: str, 
        args: List[str], 
        library_name: Optional[str] = None
    ) -> ParsedArguments:
        """
        Parse arguments using LibDoc information for accurate parameter mapping.
        
        Args:
            keyword_name: Name of the keyword
            args: List of argument strings from user
            library_name: Optional library name for disambiguation
            
        Returns:
            ParsedArguments with correct positional/named split
        """
        # Get keyword info from LibDoc
        keyword_info = self._get_keyword_info(keyword_name, library_name)
        
        if not keyword_info or not keyword_info.args:
            # Fallback to simple parsing if no LibDoc info
            logger.debug(f"No LibDoc info for {keyword_name}, using fallback parsing")
            return self._fallback_parse(args)
        
        # Parse the keyword signature to understand parameter structure
        params = self._parse_keyword_signature(keyword_info.args)
        
        # Map user arguments to parameters
        return self._map_arguments_to_parameters(args, params)
    
    def _get_keyword_info(self, keyword_name: str, library_name: Optional[str] = None):
        """Get keyword information from LibDoc storage."""
        if not self.rf_storage.is_available():
            return None
            
        try:
            # Refresh library if specified
            if library_name:
                self.rf_storage.refresh_library(library_name)
            
            # Find keyword
            keyword_info = self.rf_storage.find_keyword(keyword_name)
            
            # Check library matches if specified
            if keyword_info and library_name:
                if keyword_info.library.lower() != library_name.lower():
                    return None
                    
            return keyword_info
        except Exception as e:
            logger.debug(f"Failed to get LibDoc info for {keyword_name}: {e}")
            return None
    
    def _parse_keyword_signature(self, signature_args: List[str]) -> List[Dict[str, Any]]:
        """
        Parse Robot Framework keyword signature into parameter information.
        
        Args:
            signature_args: List of argument strings like ['selector: str', 'txt: str', 'force: bool = False']
            
        Returns:
            List of parameter dictionaries with name, type, required, default info
        """
        params = []
        
        for i, arg_str in enumerate(signature_args):
            param = {
                'index': i,
                'name': f'arg_{i}',  # fallback name
                'type': 'str',       # fallback type
                'required': True,    # fallback required
                'default': None      # fallback default
            }
            
            # Parse "name: type = default" format
            if ':' in arg_str:
                # Split name and type part
                name_part, type_and_default = arg_str.split(':', 1)
                param['name'] = name_part.strip()
                
                # Parse type and default
                if '=' in type_and_default:
                    # Has default value
                    type_part, default_part = type_and_default.split('=', 1)
                    param['type'] = type_part.strip()
                    param['default'] = default_part.strip()
                    param['required'] = False
                else:
                    # No default value
                    param['type'] = type_and_default.strip()
                    param['required'] = True
            else:
                # Simple format, just use as name
                if '=' in arg_str:
                    # Has default
                    name, default = arg_str.split('=', 1)
                    param['name'] = name.strip()
                    param['default'] = default.strip()
                    param['required'] = False
                else:
                    # Required parameter
                    param['name'] = arg_str.strip()
                    param['required'] = True
            
            params.append(param)
        
        return params
    
    def _map_arguments_to_parameters(self, args: List[str], params: List[Dict[str, Any]]) -> ParsedArguments:
        """
        Map user-provided arguments to keyword parameters based on signature.
        
        Args:
            args: User-provided argument strings
            params: Parsed parameter information from signature
            
        Returns:
            ParsedArguments with correct positional/named mapping and type conversion
        """
        parsed = ParsedArguments()
        
        # Track which parameters have been filled
        param_filled = [False] * len(params)
        arg_index = 0
        
        for arg in args:
            if self._is_named_argument(arg, params):
                # Parse as named argument
                key, value = arg.split('=', 1)
                key = key.strip()
                
                # Find parameter by name
                param_index = self._find_parameter_by_name(key, params)
                if param_index is not None:
                    # Apply type conversion based on LibDoc type info
                    param_type = params[param_index]['type']
                    converted_value = self._convert_value_by_type(value, param_type)
                    parsed.named[key] = converted_value
                    param_filled[param_index] = True
                else:
                    # Unknown parameter, add to named without conversion
                    parsed.named[key] = value
            else:
                # Treat as positional argument
                if arg_index < len(params) and not param_filled[arg_index]:
                    # Apply type conversion for positional args too
                    param_type = params[arg_index]['type']
                    converted_value = self._convert_value_by_type(arg, param_type)
                    parsed.positional.append(converted_value)
                    param_filled[arg_index] = True
                    arg_index += 1
                else:
                    # Extra positional arguments - no type info available
                    parsed.positional.append(arg)
        
        return parsed
    
    def _convert_value_by_type(self, value: str, param_type: str) -> Any:
        """
        Convert string value to appropriate Python type based on LibDoc type information.
        
        Args:
            value: String value from user
            param_type: Type information from LibDoc (e.g., "dict[str, str] | None", "list[str]", "bool")
            
        Returns:
            Converted value of appropriate type
        """
        # Skip conversion for non-string values
        if not isinstance(value, str):
            return value
        
        # Handle None/null values
        if value.lower() in ['none', 'null']:
            return None
        
        # Extract base type from complex type annotations
        base_type = self._extract_base_type(param_type)
        
        if base_type == 'bool':
            return value.lower().strip() in ['true', '1', 'yes', 'on']
        
        elif base_type == 'int':
            try:
                return int(value)
            except ValueError:
                return value
        
        elif base_type == 'float':
            try:
                return float(value)
            except ValueError:
                return value
        
        elif base_type == 'dict':
            try:
                import ast
                result = ast.literal_eval(value)
                return result if isinstance(result, dict) else value
            except (ValueError, SyntaxError):
                return value
        
        elif base_type == 'list':
            try:
                import ast
                result = ast.literal_eval(value)
                return result if isinstance(result, list) else value
            except (ValueError, SyntaxError):
                return value
        
        # For other types (str, custom classes, enums), keep as string
        return value
    
    def _extract_base_type(self, param_type: str) -> str:
        """
        Extract the base type from complex type annotations.
        
        Examples:
            "dict[str, str] | None" → "dict"
            "list[str]" → "list" 
            "bool" → "bool"
            "ViewportDimensions | None" → "dict" (TypedDict treated as dict)
        """
        if not param_type:
            return "str"
        
        # Remove union with None (| None)
        clean_type = param_type.split('|')[0].strip()
        
        # Extract base type from generics
        if '[' in clean_type:
            base_type = clean_type.split('[')[0]
        else:
            base_type = clean_type
        
        # Map known types
        base_type_lower = base_type.lower()
        if base_type_lower in ['dict', 'dictionary']:
            return 'dict'
        elif base_type_lower in ['list', 'array']:
            return 'list'
        elif base_type_lower in ['bool', 'boolean']:
            return 'bool'
        elif base_type_lower in ['int', 'integer']:
            return 'int'
        elif base_type_lower in ['float', 'double']:
            return 'float'
        
        # Special handling for Browser Library TypedDict types
        # These are TypedDicts that should accept Python dict objects
        browser_typed_dicts = [
            'ViewportDimensions', 'GeoLocation', 'HttpCredentials', 
            'RecordHar', 'RecordVideo', 'Proxy', 'ClientCertificate'
        ]
        
        if base_type in browser_typed_dicts:
            return 'dict'
        
        # For other custom types, enums, etc., keep as string
        return 'str'
    
    def _is_named_argument(self, arg: str, params: List[Dict[str, Any]]) -> bool:
        """
        Determine if an argument string is a named argument based on the parameter signature.
        
        Args:
            arg: Argument string to check
            params: Parameter signature information
            
        Returns:
            True if this should be parsed as named argument
        """
        if '=' not in arg:
            return False
        
        # Split and check the key part
        key_part = arg.split('=', 1)[0].strip()
        
        # Must be a valid identifier
        if not key_part.isidentifier():
            return False
        
        # Check if it matches a parameter name
        param_names = [p['name'] for p in params]
        if key_part in param_names:
            return True
        
        # If we don't recognize the parameter name, but it looks like a valid
        # named argument format, treat it as named
        return True
    
    def _find_parameter_by_name(self, name: str, params: List[Dict[str, Any]]) -> Optional[int]:
        """Find parameter index by name."""
        for i, param in enumerate(params):
            if param['name'] == name:
                return i
        return None
    
    def _fallback_parse(self, args: List[str]) -> ParsedArguments:
        """
        Simple fallback parsing when LibDoc is not available.
        Treats everything as positional unless it's clearly a named argument.
        """
        parsed = ParsedArguments()
        
        for arg in args:
            if self._is_simple_named_argument(arg):
                key, value = arg.split('=', 1)
                parsed.named[key.strip()] = value
            else:
                parsed.positional.append(arg)
        
        return parsed
    
    def _is_simple_named_argument(self, arg: str) -> bool:
        """Simple check for named arguments when no LibDoc info available."""
        if '=' not in arg:
            return False
        
        key_part = arg.split('=', 1)[0].strip()
        
        # Must be valid identifier
        if not key_part.isidentifier():
            return False
        
        # Must not contain complex CSS selector patterns
        if any(char in arg for char in ['[', ']', '#', ':', '(', ')', '\'', '"']):
            return False
        
        return True