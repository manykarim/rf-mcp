"""Robot Framework native type conversion integration."""

import logging
from typing import Any, Dict, List, Optional

from robotmcp.models.library_models import ParsedArguments
from robotmcp.utils.rf_libdoc_integration import get_rf_doc_storage

logger = logging.getLogger(__name__)

# Import Robot Framework native type conversion
try:
    from robot.running.arguments.typeinfo import TypeInfo
    from robot.running.arguments.typeconverters import TypeConverter
    from robot.running.arguments import ArgumentSpec
    from robot.running.arguments.argumentresolver import ArgumentResolver
    RF_NATIVE_CONVERSION_AVAILABLE = True
except ImportError:
    RF_NATIVE_CONVERSION_AVAILABLE = False
    logger.warning("Robot Framework native type conversion not available")


class RobotFrameworkNativeConverter:
    """Uses Robot Framework's native type conversion system for maximum accuracy."""
    
    def __init__(self):
        self.rf_storage = get_rf_doc_storage()
    
    def parse_and_convert_arguments(
        self, 
        keyword_name: str, 
        args: List[str], 
        library_name: Optional[str] = None
    ) -> ParsedArguments:
        """
        Parse and convert arguments using Robot Framework's native systems.
        
        This is the most accurate approach as it uses the exact same logic
        that Robot Framework uses internally for keyword execution.
        
        Args:
            keyword_name: Name of the keyword
            args: List of argument strings from user
            library_name: Optional library name for disambiguation
            
        Returns:
            ParsedArguments with correctly converted types
        """
        if not RF_NATIVE_CONVERSION_AVAILABLE:
            # Fallback to simple parsing
            return self._fallback_parse(args)
        
        # Get keyword info from LibDoc
        keyword_info = self._get_keyword_info(keyword_name, library_name)
        
        if not keyword_info or not keyword_info.args:
            # No signature info available, use fallback
            logger.debug(f"No LibDoc signature for {keyword_name}, using fallback")
            return self._fallback_parse(args)
        
        try:
            # Create Robot Framework ArgumentSpec from LibDoc signature
            spec = self._create_argument_spec(keyword_info.args)
            
            # Pre-parse named arguments from the args list
            positional_args, named_args = self._split_args_into_positional_and_named(args)
            
            # Use Robot Framework's ArgumentResolver
            resolver = ArgumentResolver(spec)
            resolved_positional, resolved_named = resolver.resolve(positional_args, named_args)
            
            # Apply type conversion using Robot Framework's native converters
            converted_positional = self._convert_positional_args(resolved_positional, keyword_info.args)
            
            # Handle different formats that RF ArgumentResolver might return
            if isinstance(resolved_named, dict):
                converted_named = self._convert_named_args(resolved_named, keyword_info.args)
            else:
                # If it's not a dict, convert to dict first
                named_dict = dict(resolved_named) if resolved_named else {}
                converted_named = self._convert_named_args(named_dict, keyword_info.args)
            
            # Build result
            result = ParsedArguments()
            result.positional = converted_positional
            result.named = converted_named
            
            return result
            
        except Exception as e:
            logger.debug(f"Robot Framework native parsing failed for {keyword_name}: {e}")
            # Fallback to simple parsing
            return self._fallback_parse(args)
    
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
    
    def _create_argument_spec(self, signature_args: List[str]) -> ArgumentSpec:
        """
        Create Robot Framework ArgumentSpec from LibDoc signature.
        
        Args:
            signature_args: List like ['selector: str', 'txt: str', 'force: bool = False']
            
        Returns:
            ArgumentSpec that Robot Framework can use
        """
        positional_or_named = []
        defaults = {}
        
        for arg_str in signature_args:
            if ':' in arg_str:
                # Parse "name: type = default" format
                name_part, type_and_default = arg_str.split(':', 1)
                name = name_part.strip()
                
                if '=' in type_and_default:
                    # Has default value
                    type_part, default_part = type_and_default.split('=', 1)
                    default_value = default_part.strip()
                    
                    # Convert default to appropriate Python type
                    if default_value.lower() == 'none':
                        defaults[name] = None
                    elif default_value.lower() in ['true', 'false']:
                        defaults[name] = default_value.lower() == 'true'
                    elif default_value.isdigit():
                        defaults[name] = int(default_value)
                    else:
                        # Keep as string, Robot Framework will handle it
                        defaults[name] = default_value
                
                positional_or_named.append(name)
            elif '=' in arg_str:
                # Simple format with default
                name, default = arg_str.split('=', 1)
                name = name.strip()
                positional_or_named.append(name)
                defaults[name] = default.strip()
            else:
                # Required parameter
                name = arg_str.strip()
                if name not in ['*', '**']:  # Skip varargs markers
                    positional_or_named.append(name)
        
        return ArgumentSpec(
            positional_or_named=positional_or_named,
            defaults=defaults
        )
    
    def _split_args_into_positional_and_named(self, args: List[str]) -> tuple[List[str], Dict[str, str]]:
        """
        Split user arguments into positional and named arguments.
        
        This uses simple heuristics since we'll let Robot Framework handle
        the complex argument resolution.
        """
        positional = []
        named = {}
        
        for arg in args:
            if '=' in arg and self._looks_like_named_arg(arg):
                key, value = arg.split('=', 1)
                named[key.strip()] = value
            else:
                positional.append(arg)
        
        return positional, named
    
    def _looks_like_named_arg(self, arg: str) -> bool:
        """Simple check if an argument looks like a named argument."""
        if '=' not in arg:
            return False
        
        key_part = arg.split('=', 1)[0].strip()
        
        # Must be valid Python identifier (no spaces, special chars, etc.)
        return key_part.isidentifier()
    
    def _convert_positional_args(self, args: List[str], signature_args: List[str]) -> List[Any]:
        """Convert positional arguments using Robot Framework's type converters."""
        converted = []
        
        for i, arg in enumerate(args):
            if i < len(signature_args):
                # Get type information from signature
                type_info = self._parse_type_from_signature(signature_args[i])
                if type_info:
                    converted_value = self._convert_with_rf_converter(arg, type_info)
                    converted.append(converted_value)
                else:
                    # No type info, keep as string
                    converted.append(arg)
            else:
                # Extra args, keep as string
                converted.append(arg)
        
        return converted
    
    def _convert_named_args(self, args: Dict[str, str], signature_args: List[str]) -> Dict[str, Any]:
        """Convert named arguments using Robot Framework's type converters."""
        converted = {}
        
        # Build parameter name to type mapping
        param_types = {}
        for arg_str in signature_args:
            if ':' in arg_str:
                name_part, type_part = arg_str.split(':', 1)
                name = name_part.strip()
                # Extract just the type part (before =)
                if '=' in type_part:
                    type_str = type_part.split('=', 1)[0].strip()
                else:
                    type_str = type_part.strip()
                param_types[name] = type_str
        
        # Convert each named argument
        for key, value in args.items():
            if key in param_types:
                type_str = param_types[key]
                type_info = self._parse_type_string(type_str)
                if type_info:
                    converted_value = self._convert_with_rf_converter(value, type_info)
                    converted[key] = converted_value
                else:
                    converted[key] = value
            else:
                # Unknown parameter, keep as string
                converted[key] = value
        
        return converted
    
    def _parse_type_from_signature(self, arg_str: str) -> Optional['TypeInfo']:
        """Parse type information from a single argument signature."""
        if ':' not in arg_str:
            return None
        
        name_part, type_and_default = arg_str.split(':', 1)
        
        # Extract type part (before =)
        if '=' in type_and_default:
            type_str = type_and_default.split('=', 1)[0].strip()
        else:
            type_str = type_and_default.strip()
        
        return self._parse_type_string(type_str)
    
    def _parse_type_string(self, type_str: str) -> Optional['TypeInfo']:
        """Parse a type string into Robot Framework TypeInfo."""
        try:
            # Handle Union types by extracting the primary type (first non-None)
            if '|' in type_str:
                # Handle Union types like "ViewportDimensions | None"
                union_types = [t.strip() for t in type_str.split('|')]
                primary_type = None
                for t in union_types:
                    if t.lower() != 'none':
                        primary_type = t
                        break
                
                if primary_type:
                    # Try to get TypeInfo for the primary type
                    return self._parse_single_type(primary_type)
                else:
                    # All types were None, default to str
                    return TypeInfo.from_string('str')
            
            return self._parse_single_type(type_str)
        except Exception as e:
            logger.debug(f"Failed to parse type string '{type_str}': {e}")
            return None
    
    def _parse_single_type(self, type_str: str) -> Optional['TypeInfo']:
        """Parse a single type string, handling custom Browser Library types."""
        # First try Robot Framework's native parsing
        type_info = TypeInfo.from_string(type_str)
        if type_info and type_info.type is not None:
            return type_info
        
        # For Browser Library TypedDict types, treat as dict
        browser_typed_dicts = [
            'ViewportDimensions', 'GeoLocation', 'HttpCredentials', 
            'RecordHar', 'RecordVideo', 'Proxy', 'ClientCertificate'
        ]
        if type_str in browser_typed_dicts:
            return TypeInfo.from_string('dict')
        
        # Try to import and use Browser Library enum types
        browser_enum_types = {
            'SupportedBrowsers': 'SupportedBrowsers',
            'SelectAttribute': 'SelectAttribute', 
            'MouseButton': 'MouseButton',
            'ElementState': 'ElementState',
            'PageLoadStates': 'PageLoadStates',
            'DialogAction': 'DialogAction',
            'RequestMethod': 'RequestMethod',
            'ScrollBehavior': 'ScrollBehavior',
            'ColorScheme': 'ColorScheme',
            'ForcedColors': 'ForcedColors',
            'ReduceMotion': 'ReduceMotion',
        }
        
        if type_str in browser_enum_types:
            try:
                # Import the actual enum class
                enum_class = self._import_browser_enum(browser_enum_types[type_str])
                if enum_class:
                    return TypeInfo.from_type(enum_class)
            except Exception as e:
                logger.debug(f"Failed to import Browser enum {type_str}: {e}")
        
        # Fallback to None
        return None
    
    def _import_browser_enum(self, enum_name: str):
        """Import Browser Library enum class by name."""
        try:
            if enum_name == 'SupportedBrowsers':
                from Browser.utils.data_types import SupportedBrowsers
                return SupportedBrowsers
            elif enum_name == 'SelectAttribute':
                from Browser.utils.data_types import SelectAttribute
                return SelectAttribute
            elif enum_name == 'MouseButton':
                from Browser.utils.data_types import MouseButton
                return MouseButton
            elif enum_name == 'ElementState':
                from Browser.utils.data_types import ElementState
                return ElementState
            elif enum_name == 'PageLoadStates':
                from Browser.utils.data_types import PageLoadStates
                return PageLoadStates
            elif enum_name == 'DialogAction':
                from Browser.utils.data_types import DialogAction
                return DialogAction
            elif enum_name == 'RequestMethod':
                from Browser.utils.data_types import RequestMethod
                return RequestMethod
            elif enum_name == 'ScrollBehavior':
                from Browser.utils.data_types import ScrollBehavior
                return ScrollBehavior
            elif enum_name == 'ColorScheme':
                from Browser.utils.data_types import ColorScheme
                return ColorScheme
            elif enum_name == 'ForcedColors':
                from Browser.utils.data_types import ForcedColors
                return ForcedColors
            elif enum_name == 'ReduceMotion':
                from Browser.utils.data_types import ReduceMotion
                return ReduceMotion
        except ImportError:
            pass
        return None
    
    def _convert_with_rf_converter(self, value: str, type_info: 'TypeInfo') -> Any:
        """Convert a value using Robot Framework's native type converter."""
        try:
            converter = TypeConverter.converter_for(type_info)
            return converter.convert(value, None)
        except Exception as e:
            logger.debug(f"Type conversion failed for '{value}' to {type_info.type}: {e}")
            # Return original value if conversion fails
            return value
    
    
    def _fallback_parse(self, args: List[str]) -> ParsedArguments:
        """Simple fallback parsing when Robot Framework native systems aren't available."""
        parsed = ParsedArguments()
        
        for arg in args:
            if '=' in arg and self._looks_like_named_arg(arg):
                # Parse as named argument
                key, value = arg.split('=', 1)
                parsed.named[key.strip()] = value
            else:
                # Treat as positional argument
                parsed.positional.append(arg)
        
        return parsed