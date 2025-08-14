"""Argument processing and type conversion utilities."""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from robotmcp.models.library_models import KeywordInfo, ParsedArguments, ArgumentInfo
from robotmcp.utils.rf_libdoc_integration import get_rf_doc_storage
from robotmcp.utils.rf_native_type_converter import RobotFrameworkNativeConverter

logger = logging.getLogger(__name__)


class ArgumentProcessor:
    """Handles argument parsing, type conversion, and LibDoc integration."""
    
    def __init__(self):
        self.rf_native_converter = RobotFrameworkNativeConverter()
    
    def parse_arguments_for_keyword(self, keyword_name: str, args: List[str], library_name: str = None) -> ParsedArguments:
        """
        Parse arguments for a specific keyword using Robot Framework's native type conversion.
        
        This is the most accurate method as it uses Robot Framework's own argument
        resolution and type conversion systems.
        """
        return self.rf_native_converter.parse_and_convert_arguments(keyword_name, args, library_name)
    
    def parse_arguments(self, args: List[str]) -> ParsedArguments:
        """
        Simple fallback argument parsing when LibDoc information is not available.
        
        DEPRECATED: Use parse_arguments_for_keyword() instead for better accuracy.
        """
        parsed = ParsedArguments()
        
        for arg in args:
            if '=' in arg and self._looks_like_named_argument(arg):
                # Parse as named argument
                key, value = arg.split('=', 1)
                parsed.named[key.strip()] = value
            else:
                # Treat as positional argument
                parsed.positional.append(arg)
        
        return parsed
    
    def _looks_like_named_argument(self, arg: str) -> bool:
        """Simple check if an argument looks like a named argument."""
        if '=' not in arg:
            return False
        
        key_part = arg.split('=', 1)[0].strip()
        
        # Must be a valid Python identifier (no spaces, special chars, etc.)
        return key_part.isidentifier()
    
    
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