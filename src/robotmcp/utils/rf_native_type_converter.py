"""Robot Framework native type conversion integration."""

import logging
from typing import Any, Dict, List, Optional

from robotmcp.models.library_models import ParsedArguments
from robotmcp.utils.rf_libdoc_integration import get_rf_doc_storage

logger = logging.getLogger(__name__)

# Note: Hardcoded REQUESTS_LIBRARY_SIGNATURES cache removed
# Now using Robot Framework native APIs with hybrid extraction approach

# SeleniumLibrary locator strategies for error guidance
SELENIUM_LOCATOR_STRATEGIES = {
    "id": "Element id (e.g., 'id:example')",
    "name": "name attribute (e.g., 'name:example')",
    "identifier": "Either id or name (e.g., 'identifier:example')",
    "class": "Element class (e.g., 'class:example')",
    "tag": "Tag name (e.g., 'tag:div')",
    "xpath": "XPath expression (e.g., 'xpath://div[@id=\"example\"]')",
    "css": "CSS selector (e.g., 'css:div#example')",
    "dom": "DOM expression (e.g., 'dom:document.images[5]')",
    "link": "Exact text a link has (e.g., 'link:The example')",
    "partial link": "Partial link text (e.g., 'partial link:he ex')",
    "sizzle": "Sizzle selector deprecated (e.g., 'sizzle:div.example')",
    "data": "Element data-* attribute (e.g., 'data:id:my_id')",
    "jquery": "jQuery expression (e.g., 'jquery:div.example')",
    "default": "Keyword specific default behavior (e.g., 'default:example')"
}

# Browser Library (Playwright) locator strategies for error guidance
BROWSER_LOCATOR_STRATEGIES = {
    "css": "CSS selector (default strategy) - e.g., 'css=.class > #login_btn' or just '.class > #login_btn'",
    "xpath": "XPath expression - e.g., 'xpath=//input[@id=\"login_btn\"]' or '//input[@id=\"login_btn\"]'",
    "text": "Browser text engine (exact/partial/regex) - e.g., 'text=Login' or \"Login\"",
    "id": "Element ID attribute - e.g., 'id=login_btn'",
    "css:light": "CSS without shadow DOM piercing - e.g., 'css:light=article div'",
    "text:light": "Text without shadow DOM piercing - e.g., 'text:light=Login'",
    "data-testid": "data-testid attribute - e.g., 'data-testid=submit-button'",
    "data-test-id": "data-test-id attribute - e.g., 'data-test-id=submit-button'",
    "data-test": "data-test attribute - e.g., 'data-test=submit-button'",
    "id:light": "ID without shadow DOM piercing - e.g., 'id:light=login_btn'"
}

# Browser Library selector format patterns
BROWSER_SELECTOR_PATTERNS = {
    "explicit": "strategy=value (e.g., 'css=.button', 'xpath=//button')",
    "implicit_css": "Plain selectors default to CSS (e.g., '.button' becomes 'css=.button')",
    "implicit_xpath": "Selectors starting with // or .. become XPath (e.g., '//button')",
    "implicit_text": "Quoted selectors become text (e.g., '\"Login\"' becomes 'text=Login')",
    "cascaded": "Multiple strategies with >> separator (e.g., 'text=Hello >> ../.. >> .select_button')",
    "iframe_piercing": "Frame piercing with >>> (e.g., 'id=iframe >>> id=btn')",
    "element_reference": "Element reference with element= (e.g., '${ref} >> .child')"
}

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
    
    def _extract_requests_library_signature(self, keyword_name: str, library_name: str = None) -> Optional[List[str]]:
        """
        Extract keyword signature using Robot Framework native APIs.
        
        Uses hybrid approach:
        1. Native RF ArgumentSpec for non-decorated keywords
        2. Closure inspection for decorated keywords  
        3. No hardcoded cache dependency
        """
        return self._extract_keyword_signature_hybrid(keyword_name, library_name)

    def _extract_keyword_signature_hybrid(self, keyword_name: str, library_name: str = None) -> Optional[List[str]]:
        """
        Hybrid signature extraction using RF native APIs with closure fallback.
        
        This is the general solution that works with any Robot Framework library:
        1. Load library using RF native TestLibrary
        2. Check if keyword uses native ArgumentSpec or is decorated
        3. Extract using appropriate method
        """
        if not library_name:
            return None
            
        try:
            # Load library using Robot Framework's native TestLibrary
            from robot.running.testlibraries import TestLibrary
            
            lib = TestLibrary.from_name(library_name)
            
            # Find the keyword in the library
            keyword_obj = None
            for kw in lib.keywords:
                if kw.name == keyword_name:
                    keyword_obj = kw
                    break
            
            if not keyword_obj:
                logger.debug(f"Keyword '{keyword_name}' not found in library '{library_name}'")
                return None
            
            if not hasattr(keyword_obj, 'args'):
                logger.debug(f"Keyword '{keyword_name}' has no args attribute")
                return None
                
            args_spec = keyword_obj.args
            
            # Check if this is a decorated keyword (shows generic *args, **kwargs)
            if self._is_decorated_keyword(args_spec):
                logger.debug(f"Detected decorated keyword '{keyword_name}' - using closure inspection")
                return self._extract_from_closure(keyword_obj, keyword_name, library_name)
            else:
                logger.debug(f"Detected native keyword '{keyword_name}' - using ArgumentSpec")
                return self._extract_from_argumentspec(args_spec)
                
        except Exception as e:
            logger.debug(f"Error in hybrid extraction for {keyword_name}: {e}")
            return None

    def _is_decorated_keyword(self, args_spec) -> bool:
        """
        Detect if keyword is decorated by checking ArgumentSpec pattern.
        
        Decorated keywords show:
        - var_positional = 'args' 
        - var_named = 'kwargs'
        - positional_or_named = () (empty)
        
        This indicates the original signature is masked by decoration.
        """
        try:
            return (
                args_spec.var_positional == 'args' and
                args_spec.var_named == 'kwargs' and 
                len(args_spec.positional_or_named) == 0
            )
        except Exception:
            return False

    def _extract_from_argumentspec(self, args_spec) -> List[str]:
        """
        Extract signature from Robot Framework's native ArgumentSpec.
        
        This works for non-decorated keywords and provides full signature details
        including positional args, named args, defaults, *args, and **kwargs.
        """
        signature = []
        
        try:
            # Required positional args (no defaults)
            required_positional = []
            for name in args_spec.positional:
                if name not in args_spec.defaults:
                    required_positional.append(name)
            
            # Args with defaults (can be positional or named)
            optional_args = []
            for name in args_spec.positional_or_named:
                if name in args_spec.defaults:
                    default = args_spec.defaults[name]
                    optional_args.append(f"{name}={default}")
                elif name not in required_positional:  # Not already added as required
                    optional_args.append(name)
            
            # Combine in order: required positional, optional, *args, **kwargs
            signature.extend(required_positional)
            signature.extend(optional_args)
            
            # Add *args if present
            if args_spec.var_positional:
                signature.append(f"*{args_spec.var_positional}")
            
            # Add **kwargs if present
            if args_spec.var_named:
                signature.append(f"**{args_spec.var_named}")
            
            logger.debug(f"Native ArgumentSpec extraction result: {signature}")
            return signature
            
        except Exception as e:
            logger.debug(f"Error extracting from ArgumentSpec: {e}")
            return []

    def _extract_from_closure(self, keyword_obj, keyword_name: str, library_name: str) -> Optional[List[str]]:
        """
        Extract signature from decorated keyword using closure inspection.
        
        This is only used when ArgumentSpec shows generic *args, **kwargs
        due to decoration masking the real signature.
        """
        try:
            # Check different possible attributes for the handler function
            handler = None
            if hasattr(keyword_obj, 'method'):
                handler = keyword_obj.method
            elif hasattr(keyword_obj, '_handler'):
                handler = keyword_obj._handler
            elif hasattr(keyword_obj, 'handler'):
                handler = keyword_obj.handler
            
            if not handler:
                return None
            
            # For RequestsLibrary, use existing closure inspection logic
            if library_name == "RequestsLibrary" and hasattr(handler, '__closure__') and handler.__closure__:
                return self._extract_requestslibrary_from_closure(handler, keyword_name)
            
            # For other libraries, could extend closure inspection here
            # but most libraries don't use decorators that mask signatures
            
        except Exception as e:
            logger.debug(f"Error in closure extraction: {e}")
            
        return None

    def _extract_requestslibrary_from_closure(self, handler, keyword_name: str) -> Optional[List[str]]:
        """
        Extract RequestsLibrary signature from closure using existing logic.
        
        This maintains the current working closure inspection for RequestsLibrary
        decorated methods while making it part of the hybrid system.
        """
        try:
            import inspect
            
            # Map keyword names to actual method names  
            keyword_to_method = {
                "POST On Session": "post_on_session",
                "POST": "session_less_post",
                "GET On Session": "get_on_session", 
                "GET": "session_less_get",
                "PUT On Session": "put_on_session",
                "PUT": "session_less_put",
                "PATCH On Session": "patch_on_session",
                "PATCH": "session_less_patch",
                "DELETE On Session": "delete_on_session",
                "DELETE": "session_less_delete",
            }
            
            method_name = keyword_to_method.get(keyword_name)
            if not method_name:
                return None
            
            # Extract from closure cells
            for cell in handler.__closure__:
                try:
                    content = cell.cell_contents
                    if (callable(content) and 
                        hasattr(content, '__name__') and 
                        content.__name__ == method_name):
                        
                        original_sig = inspect.signature(content)
                        # Convert to signature args format
                        signature_args = []
                        for name, param in original_sig.parameters.items():
                            if name != 'self':
                                if param.kind == inspect.Parameter.VAR_KEYWORD:
                                    signature_args.append(f"**{name}")
                                elif param.kind == inspect.Parameter.VAR_POSITIONAL:
                                    signature_args.append(f"*{name}")
                                elif param.default is param.empty:
                                    signature_args.append(name)
                                else:
                                    signature_args.append(f"{name}={param.default}")
                        
                        logger.debug(f"Closure extraction for {keyword_name}: {signature_args}")
                        return signature_args
                        
                except Exception as cell_error:
                    continue
                    
        except Exception as e:
            logger.debug(f"RequestsLibrary closure extraction error: {e}")
            
        return None
    
    def _is_dictionary_literal(self, arg_str: str) -> bool:
        """Check if the argument string contains a dictionary literal that needs conversion."""
        if '=' not in arg_str:
            return False
        
        param_name, param_value = arg_str.split('=', 1)
        param_value = param_value.strip()
        
        # Check for dictionary-like patterns
        dict_patterns = [
            param_value == '{}',  # Empty dict
            param_value.startswith('{') and param_value.endswith('}'),  # Dict literal
            param_value.startswith('[') and param_value.endswith(']'),  # List literal
        ]
        
        return any(dict_patterns)
    
    def _looks_like_named_parameter(self, arg_str: str) -> bool:
        """Check if argument looks like a named parameter (param=value)."""
        if '=' not in arg_str:
            return False
        
        param_name = arg_str.split('=', 1)[0].strip()
        
        # Named parameters should have valid Python identifier names
        # Avoid splitting on '=' inside strings, URLs, etc.
        return (param_name.isidentifier() and 
                not arg_str.startswith('http') and 
                not arg_str.count('=') > 10)  # Avoid complex expressions
    
    def _process_object_preserving_args(self, keyword_name: str, args: List[Any]) -> List[Any]:
        """
        Proper solution: Handle ObjectPreservingArgument objects correctly for RF execution.
        
        Robot Framework's ArgumentResolver expects to receive arguments in a format that
        allows proper separation of positional and named arguments. For named parameters
        with object values, we need to return both the parameter name and the object value
        in a way that RF can process.
        """
        from robotmcp.components.variables.variable_resolver import ObjectPreservingArgument
        
        processed_args = []
        
        for arg in args:
            if isinstance(arg, ObjectPreservingArgument):
                # CORRECT FIX: Return a special format that preserves the object
                # This will be handled by the argument resolution system properly
                processed_args.append((arg.param_name, arg.value))
            elif isinstance(arg, tuple) and len(arg) == 2:
                # Handle named parameter tuples (like ('json', {'test': 'value'}))
                param_name, param_value = arg
                if isinstance(param_name, str) and param_name.isidentifier():
                    # Preserve the tuple as-is for named parameter handling
                    processed_args.append(arg)
                else:
                    # Not a valid named parameter tuple, convert to string
                    processed_args.append(str(arg))
            else:
                processed_args.append(str(arg) if not isinstance(arg, str) else arg)
        
        return processed_args
    
    def parse_and_convert_arguments(
        self, 
        keyword_name: str, 
        args: List[Any], 
        library_name: Optional[str] = None,
        session_variables: Optional[Dict[str, Any]] = None
    ) -> ParsedArguments:
        """
        Parse and convert arguments using Robot Framework's native ArgumentResolver.
        
        This uses RF's native argument resolution system which handles:
        - Positional vs named argument separation
        - Type conversion based on method signatures
        - Decorator-aware keyword resolution
        - ObjectPreservingArgument objects for object type preservation
        
        Args:
            keyword_name: Name of the keyword
            args: List of arguments from user (may include ObjectPreservingArgument objects)
            library_name: Optional library name for disambiguation
            
        Returns:
            ParsedArguments with correctly converted types
        """
        # Handle ObjectPreservingArgument objects from variable resolution
        processed_args = self._process_object_preserving_args(keyword_name, args)
        return self._parse_with_rf_native_resolver(keyword_name, processed_args, library_name, session_variables)
    
    def _parse_with_rf_native_resolver(self, keyword_name: str, args: List[Any], library_name: Optional[str] = None, session_variables: Optional[Dict[str, Any]] = None) -> ParsedArguments:
        """Use Robot Framework's native argument resolution - GENERAL SOLUTION."""
        if not RF_NATIVE_CONVERSION_AVAILABLE:
            logger.debug(f"RF native parsing not available, using fallback for {keyword_name}")
            return self._fallback_parse(args)
        
        # GENERAL SOLUTION: Use RF's native Variables and ArgumentResolver
        try:
            from robot.variables import Variables
            from robot.running.arguments import ArgumentSpec
            from robot.running.arguments.argumentresolver import ArgumentResolver
            
            # Step 1: Create Variables instance and resolve all variables in args
            variables = Variables()
            # Add session variables to enable ${variable} resolution
            if session_variables:
                for var_name, var_value in session_variables.items():
                    # Store variables in RF format (with ${} syntax)
                    if not var_name.startswith('${'):
                        var_name = f'${{{var_name}}}'
                    variables[var_name] = var_value
                    logger.debug(f"Added session variable: {var_name} = {var_value} (type: {type(var_value)})")
                logger.debug(f"Session variables loaded: {len(session_variables)} variables")
            else:
                logger.debug(f"No session variables provided for {keyword_name}")
            
            # Use Robot Framework's native argument resolution approach
            # Instead of custom logic, let RF handle variable resolution and argument parsing
            
            # For named arguments like "json=${body}", we need to parse them properly
            # RF's approach is to separate the name and value, then resolve variables
            positional_args = []
            named_args = {}
            
            logger.debug(f"RF_NATIVE_CONVERTER: Processing args: {args}")
            logger.debug(f"RF_NATIVE_CONVERTER: Session variables: {list(session_variables.keys()) if session_variables else 'None'}")
            
            # PHASE 1: Get keyword argument specification to validate named arguments
            keyword_spec = self._get_keyword_argument_spec(keyword_name, library_name)
            valid_param_names = set()
            has_kwargs = False
            
            if keyword_spec:
                # Get all valid parameter names from the keyword specification
                if hasattr(keyword_spec, 'positional'):
                    valid_param_names.update(keyword_spec.positional)
                if hasattr(keyword_spec, 'named_only') and keyword_spec.named_only:
                    valid_param_names.update(keyword_spec.named_only)
                if hasattr(keyword_spec, 'positional_or_named') and keyword_spec.positional_or_named:
                    valid_param_names.update(keyword_spec.positional_or_named)
                
                # CRITICAL: Check if keyword accepts **kwargs
                has_kwargs = hasattr(keyword_spec, 'var_named') and keyword_spec.var_named is not None
                
                logger.debug(f"ArgumentSpec analysis for {keyword_name}: valid_params={valid_param_names}, has_kwargs={has_kwargs} (var_named='{getattr(keyword_spec, 'var_named', None)}')")
            
            for arg in args:
                if isinstance(arg, tuple) and len(arg) == 2:
                    # Handle ObjectPreservingArgument tuples (already resolved)
                    param_name, param_value = arg
                    named_args[param_name] = param_value
                elif isinstance(arg, str) and '=' in arg:
                    # ENHANCED FIX: Handle **kwargs properly
                    parts = arg.split('=', 1)
                    if len(parts) == 2 and parts[0].strip().isidentifier():
                        param_name = parts[0].strip()
                        param_value_str = parts[1].strip()
                        
                        # Check if this parameter name is valid for this keyword
                        if valid_param_names and param_name in valid_param_names:
                            # Valid explicit parameter - treat as named argument
                            try:
                                resolved_value = variables.replace_scalar(param_value_str)
                                named_args[param_name] = resolved_value
                                logger.debug(f"RF explicit named argument: {param_name}={resolved_value} (type: {type(resolved_value)})")
                            except Exception as var_error:
                                logger.debug(f"Variable resolution failed for '{param_value_str}': {var_error}")
                                named_args[param_name] = param_value_str
                        elif has_kwargs:
                            # **KWARGS FIX**: Keyword accepts **kwargs and this is not an explicit parameter
                            # Treat as **kwargs argument (named argument)
                            try:
                                resolved_value = variables.replace_scalar(param_value_str)
                                named_args[param_name] = resolved_value
                                logger.debug(f"RF **kwargs argument: {param_name}={resolved_value} (type: {type(resolved_value)})")
                            except Exception as var_error:
                                logger.debug(f"Variable resolution failed for '{param_value_str}': {var_error}")
                                named_args[param_name] = param_value_str
                        else:
                            # No **kwargs support and parameter name not explicit - treat as positional
                            logger.debug(f"'{param_name}' not in valid parameters {valid_param_names} for {keyword_name}, no **kwargs, treating '{arg}' as positional")
                            try:
                                resolved_arg = variables.replace_scalar(arg)
                                positional_args.append(resolved_arg)
                                logger.debug(f"RF positional resolution: '{arg}' -> {resolved_arg} (type: {type(resolved_arg)})")
                            except Exception as var_error:
                                logger.debug(f"Variable resolution failed for '{arg}': {var_error}")
                                positional_args.append(arg)
                    else:
                        # Not a valid named parameter format, treat as positional
                        try:
                            resolved_arg = variables.replace_scalar(arg)
                            positional_args.append(resolved_arg)
                            logger.debug(f"RF positional resolution: '{arg}' -> {resolved_arg} (type: {type(resolved_arg)})")
                        except Exception as var_error:
                            logger.debug(f"Variable resolution failed for '{arg}': {var_error}")
                            positional_args.append(arg)
                else:
                    # Positional argument
                    if isinstance(arg, str) and '${' in arg:
                        try:
                            resolved_arg = variables.replace_scalar(arg)
                            positional_args.append(resolved_arg)
                            logger.debug(f"RF positional resolution: '{arg}' -> {resolved_arg} (type: {type(resolved_arg)})")
                        except Exception as var_error:
                            logger.debug(f"Variable resolution failed for '{arg}': {var_error}")
                            positional_args.append(arg)
                    else:
                        positional_args.append(arg)
            
            # Create result using Robot Framework's native approach
            result = ParsedArguments()
            result.positional = positional_args
            result.named = named_args
            
            logger.debug(f"RF NATIVE SOLUTION - {keyword_name}: positional={positional_args}, named={named_args}")
            return result
            
        except Exception as e:
            logger.debug(f"RF native resolution failed for {keyword_name}: {e}")
            # Fallback to original logic
            pass
        
        try:
            # Use Robot Framework's native LibDoc to get keyword signature
            from robot.libdoc import LibraryDocumentation
            
            if library_name:
                lib_doc = LibraryDocumentation(library_name)
                keyword_doc = None
                
                # Find the specific keyword
                for kw_doc in lib_doc.keywords:
                    if kw_doc.name == keyword_name:
                        keyword_doc = kw_doc
                        break
                
                if keyword_doc and hasattr(keyword_doc, 'args') and keyword_doc.args:
                    # Check if signature is decorator-masked (e.g., *args, **kwargs)
                    arg_strings = [str(arg) for arg in keyword_doc.args]
                    is_decorator_masked = (
                        len(arg_strings) <= 2 and 
                        any('*args' in arg or '**kwargs' in arg for arg in arg_strings)
                    )
                    
                    if not is_decorator_masked:
                        # Use RF's native ArgumentSpec and ArgumentResolver for clean signatures
                        from robot.running.arguments import ArgumentSpec
                        from robot.running.arguments.argumentresolver import ArgumentResolver
                        
                        spec = ArgumentSpec(arg_strings)
                        resolver = ArgumentResolver(spec)
                        
                        # Convert our tuple format to RF-compatible format
                        rf_args = []
                        object_params = {}
                        
                        for arg in args:
                            if isinstance(arg, tuple) and len(arg) == 2:
                                # This is an ObjectPreservingArgument tuple (param_name, value)
                                param_name, param_value = arg
                                object_params[param_name] = param_value
                                # Add to RF args in named parameter format
                                rf_args.append(f"{param_name}=__OBJECT_PLACEHOLDER__")
                            else:
                                # Handle named parameters that might need type conversion
                                arg_str = str(arg)
                                if '=' in arg_str and self._is_dictionary_literal(arg_str):
                                    # Convert dictionary literals like "headers={}" to actual dicts
                                    param_name, param_value_str = arg_str.split('=', 1)
                                    try:
                                        # Safely evaluate the dictionary literal
                                        import ast
                                        param_value = ast.literal_eval(param_value_str)
                                        object_params[param_name] = param_value
                                        rf_args.append(f"{param_name}=__OBJECT_PLACEHOLDER__")
                                    except (ValueError, SyntaxError):
                                        # If it's not a valid literal, keep as string
                                        rf_args.append(arg_str)
                                else:
                                    rf_args.append(arg_str)
                        
                        # Resolve arguments using RF native logic
                        positional, named = resolver.resolve(rf_args)
                        
                        # Replace object placeholders with actual objects
                        for i, (key, value) in enumerate(named):
                            if value == "__OBJECT_PLACEHOLDER__" and key in object_params:
                                named[i] = (key, object_params[key])
                        
                        logger.debug(f"RF native resolution for {keyword_name}: pos={len(positional)}, named={len(named)}")
                        
                        result = ParsedArguments()
                        result.positional = positional
                        result.named = dict(named) if named else {}
                        
                        return result
                    else:
                        logger.debug(f"Decorator-masked signature detected for {keyword_name}, using custom extraction")
                        # Fall through to signature extraction logic
        except Exception as e:
            logger.debug(f"RF native parsing failed for {keyword_name}: {e}")
        
        # Fallback to simple parsing
        return self._fallback_parse(args)
    
    # REMOVED: Complex signature awareness logic - replaced by general RF native solution above
    
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
            signature_args: List like ['selector: str', 'attribute: SelectAttribute', '*values']
            
        Returns:
            ArgumentSpec that Robot Framework can use
        """
        positional_or_named = []
        defaults = {}
        var_positional = None
        var_named = None
        
        for arg_str in signature_args:
            if ':' in arg_str:
                # Parse "name: type = default" format
                name_part, type_and_default = arg_str.split(':', 1)
                name = name_part.strip()
                
                # Handle variadic arguments
                if name.startswith('**'):
                    var_named = name[2:]  # Remove **
                    continue
                elif name.startswith('*'):
                    var_positional = name[1:]  # Remove *
                    continue
                
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
                
                # Handle variadic arguments
                if name.startswith('**'):
                    var_named = name[2:]
                    continue
                elif name.startswith('*'):
                    var_positional = name[1:]
                    continue
                    
                positional_or_named.append(name)
                defaults[name] = default.strip()
            else:
                # Required parameter or variadic argument
                name = arg_str.strip()
                
                if name.startswith('**'):
                    var_named = name[2:]  # Remove **
                elif name.startswith('*'):
                    var_positional = name[1:]  # Remove *
                elif name not in ['*', '**']:  # Only add regular parameters
                    positional_or_named.append(name)
        
        return ArgumentSpec(
            positional_or_named=positional_or_named,
            defaults=defaults,
            var_positional=var_positional,
            var_named=var_named
        )
    
    def _detect_argument_ordering_violation(self, args: List[str], signature_args: List[str] = None) -> Dict[str, Any]:
        """
        Detect Robot Framework argument ordering violations and provide helpful error messages.
        
        Returns dict with 'violation', 'message', 'suggestion', and 'fix_examples' keys.
        """
        found_named_at = None
        
        for i, arg in enumerate(args):
            if '=' in arg and self._looks_like_named_arg(arg, self._get_valid_param_names(signature_args)):
                if found_named_at is None:
                    found_named_at = i
            elif found_named_at is not None:
                # Found positional after named - violation!
                return {
                    'violation': True,
                    'message': f"Positional argument '{arg}' found after named argument at position {found_named_at}",
                    'rf_error': f"Keyword got positional argument after named arguments",
                    'suggestion': "Robot Framework requires all arguments after a named argument to also be named",
                    'user_pattern': args,
                    'fix_examples': self._generate_fix_examples(args, signature_args)
                }
        
        return {'violation': False}
    
    def _get_valid_param_names(self, signature_args: List[str] = None) -> set:
        """Extract valid parameter names from signature args."""
        if not signature_args:
            return set()
        
        valid_params = set()
        for sig_arg in signature_args:
            if '=' in sig_arg:
                param_name = sig_arg.split('=', 1)[0].strip()
            else:
                param_name = sig_arg.strip()
            valid_params.add(param_name)
        
        return valid_params
    
    def _generate_fix_examples(self, args: List[str], signature_args: List[str] = None) -> List[Dict[str, Any]]:
        """Generate example fixes for invalid argument patterns."""
        examples = []
        
        # Option 1: Convert all to positional (remove named syntax)
        positional_fix = []
        for arg in args:
            if '=' in arg and self._looks_like_named_arg(arg, self._get_valid_param_names(signature_args)):
                value = arg.split('=', 1)[1]
                positional_fix.append(value)
            else:
                positional_fix.append(arg)
        
        examples.append({
            'name': 'All positional arguments',
            'args': positional_fix,
            'description': 'Remove parameter names, keep only values in correct order'
        })
        
        # Option 2: Move named arguments to end
        positional_args = []
        named_args = []
        
        for arg in args:
            if '=' in arg and self._looks_like_named_arg(arg, self._get_valid_param_names(signature_args)):
                named_args.append(arg)
            else:
                positional_args.append(arg)
        
        reordered = positional_args + named_args
        examples.append({
            'name': 'Reorder arguments (named at end)',
            'args': reordered,
            'description': 'Move all named arguments to the end'
        })
        
        return examples

    def _split_args_into_positional_and_named(self, args: List[str], signature_args: List[str] = None) -> tuple[List[str], Dict[str, str]]:
        """
        Split user arguments into positional and named arguments.
        
        Uses LibDoc signature information to accurately distinguish between
        locator strings (like "name=firstname") and actual named arguments.
        
        Raises helpful error messages for argument ordering violations.
        """
        # Check for argument ordering violations and auto-fix if needed
        violation_check = self._detect_argument_ordering_violation(args, signature_args)
        if violation_check.get('violation'):
            # Auto-fix: Use the reordered arguments (move named args to end)
            fix_examples = violation_check['fix_examples']
            if fix_examples:
                fixed_args = fix_examples[1]['args'] if len(fix_examples) > 1 else fix_examples[0]['args']  # Prefer reordering
                logger.info(f"AUTO-FIX: Detected invalid argument pattern for {signature_args}. "
                           f"Automatically fixed: {args} → {fixed_args}")
                args = fixed_args
            else:
                # If no fix examples available, still raise the error
                error_msg = f"Invalid argument pattern: {violation_check['rf_error']}\n\n"
                error_msg += f"Problem: {violation_check['message']}\n"
                error_msg += f"Suggestion: {violation_check['suggestion']}"
                raise ValueError(error_msg)
        
        positional = []
        named = {}
        
        # Build list of valid parameter names from signature
        valid_param_names = set()
        if signature_args:
            for arg_str in signature_args:
                if ':' in arg_str:
                    param_name = arg_str.split(':', 1)[0].strip()
                    if param_name.startswith('*'):
                        param_name = param_name[1:]  # Remove * for varargs
                    if param_name.startswith('*'):
                        param_name = param_name[1:]  # Remove ** for kwargs
                    if param_name and not param_name.startswith('*'):
                        valid_param_names.add(param_name)
        
        for arg in args:
            if '=' in arg and self._looks_like_named_arg(arg, valid_param_names):
                key, value = arg.split('=', 1)
                named[key.strip()] = value
            else:
                positional.append(arg)
        
        return positional, named
    
    def _looks_like_named_arg(self, arg: str, valid_param_names: set = None) -> bool:
        """
        Check if an argument looks like a named argument.
        
        Uses valid parameter names from LibDoc signature to distinguish between
        actual named parameters and locator strings containing '=' characters.
        """
        if '=' not in arg:
            return False
        
        key_part = arg.split('=', 1)[0].strip()
        
        # Must be valid Python identifier
        if not key_part.isidentifier():
            return False
        
        # If we have valid parameter names from signature, only treat as named arg
        # if the key matches an actual parameter name
        if valid_param_names:
            return key_part in valid_param_names
        
        # Fallback: treat as named arg if it's a valid identifier
        return True
    
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
    
    
    def _fallback_parse(self, args: List[str], signature_args: List[str] = None) -> ParsedArguments:
        """Simple fallback parsing when Robot Framework native systems aren't available.
        
        Args:
            args: List of argument strings from user
            signature_args: Optional keyword signature arguments for parameter validation
            
        Returns:
            ParsedArguments with proper positional/named argument separation
        """
        parsed = ParsedArguments()
        
        # Build list of valid parameter names from signature
        valid_param_names = set()
        if signature_args:
            for arg_str in signature_args:
                if ':' in arg_str:
                    param_name = arg_str.split(':', 1)[0].strip()
                    if param_name.startswith('*'):
                        param_name = param_name[1:]  # Remove * for varargs
                    if param_name.startswith('*'):
                        param_name = param_name[1:]  # Remove ** for kwargs
                    if param_name and not param_name.startswith('*'):
                        valid_param_names.add(param_name)
                else:
                    # Simple parameter name without type info
                    param_name = arg_str.strip()
                    if param_name and not param_name.startswith('*'):
                        valid_param_names.add(param_name)
        
        for arg in args:
            if '=' in arg and self._looks_like_named_arg(arg, valid_param_names):
                # Parse as named argument
                key, value = arg.split('=', 1)
                parsed.named[key.strip()] = value
            else:
                # Treat as positional argument
                parsed.positional.append(arg)
        
        return parsed
    
    def get_selenium_locator_guidance(self, error_message: str = None, keyword_name: str = None) -> Dict[str, Any]:
        """
        Provide SeleniumLibrary locator strategy guidance for agents.
        
        Args:
            error_message: Optional error message to analyze
            keyword_name: Optional keyword name that failed
            
        Returns:
            Dict with locator strategies and guidance
        """
        guidance = {
            "locator_strategies": SELENIUM_LOCATOR_STRATEGIES,
            "common_examples": {
                "By ID": "id:my-button",
                "By Name": "name:firstname", 
                "By CSS": "css:#submit-btn",
                "By XPath": "xpath://input[@type='submit']",
                "By Class": "class:button-primary",
                "By Link Text": "link:Click Here"
            },
            "tips": [
                "For form elements, 'name:fieldname' is often most reliable",
                "CSS selectors use 'css:' prefix, not just the selector",
                "XPath expressions must start with 'xpath:' prefix",
                "Use 'identifier:' to match either id or name attributes",
                "For buttons/links, try 'link:' for exact text matching"
            ]
        }
        
        # Add specific guidance based on error analysis
        if error_message and keyword_name:
            if "element not found" in error_message.lower():
                guidance["element_not_found_suggestions"] = [
                    "Verify the element exists on the current page",
                    "Try different locator strategies (id, name, css, xpath)",
                    "Check if element is in an iframe or shadow DOM",
                    "Ensure page has fully loaded before locating element",
                    "Use browser developer tools to inspect element attributes"
                ]
            
            if "timeout" in error_message.lower():
                guidance["timeout_suggestions"] = [
                    "Increase wait time for dynamic content",
                    "Use explicit waits (Wait Until Element Is Visible)",
                    "Check if element loads asynchronously",
                    "Verify locator strategy is correct"
                ]
        
        return guidance
    
    def get_browser_locator_guidance(self, error_message: str = None, keyword_name: str = None) -> Dict[str, Any]:
        """
        Provide Browser Library (Playwright) locator strategy guidance for agents.
        
        Args:
            error_message: Optional error message to analyze
            keyword_name: Optional keyword name that failed
            
        Returns:
            Dict with Browser Library locator strategies and guidance
        """
        guidance = {
            "locator_strategies": BROWSER_LOCATOR_STRATEGIES,
            "selector_patterns": BROWSER_SELECTOR_PATTERNS,
            "common_examples": {
                "CSS (default)": ".button-primary",
                "CSS explicit": "css=.button-primary", 
                "CSS with ID": "\\#submit-btn",  # Note: # needs escaping in Robot Framework
                "XPath": "//input[@type='submit']",
                "XPath implicit": "//button[contains(text(), 'Login')]",
                "Text exact": "text=Login",
                "Text implicit": "\"Login\"",
                "Text regex": "text=/^Log(in|out)$/i",
                "ID": "id=submit-button",
                "Cascaded": "text=Hello >> ../.. >> .select_button",
                "iFrame piercing": "id=myframe >>> .inner-button"
            },
            "selector_format_rules": {
                "Default strategy": "CSS - plain selectors are treated as CSS",
                "Explicit format": "strategy=value (spaces around = are ignored)",
                "XPath detection": "Selectors starting with // or .. become XPath automatically",
                "Text detection": "Quoted selectors (\"text\" or 'text') become text selectors",
                "Cascading": "Use >> to chain selectors (css=div >> text=Login >> .button)",
                "iFrame access": "Use >>> to pierce iFrames (id=frame >>> id=element)",
                "Element refs": "Use element=${ref} >> .child for element references"
            },
            "strict_mode_info": {
                "description": "Browser Library uses strict mode by default",
                "strict_true": "Keyword fails if selector finds multiple elements",
                "strict_false": "Keyword succeeds even with multiple matches (uses first)",
                "how_to_change": "Use 'Set Strict Mode' keyword or library import parameter"
            },
            "shadow_dom_support": {
                "automatic_piercing": "CSS and text engines automatically pierce open shadow roots",
                "light_engines": "Use css:light= or text:light= to disable shadow DOM piercing",
                "closed_shadow_roots": "Closed shadow roots cannot be accessed"
            },
            "tips": [
                "Browser Library uses CSS selectors by default (no prefix needed)",
                "Use \\# instead of # for ID selectors (Robot Framework escaping)",
                "XPath: Start with // or .. for automatic detection",
                "Text: Use quotes for exact text matching or regex patterns",
                "Cascaded selectors: Chain with >> for complex element paths",
                "iFrames: Use >>> to access elements inside frames",
                "Shadow DOM: CSS pierces automatically, use :light for light DOM only",
                "Strict mode: Controls behavior when multiple elements match"
            ]
        }
        
        # Add specific guidance based on error analysis
        if error_message and keyword_name:
            guidance.update(self._analyze_browser_error(error_message, keyword_name))
        
        return guidance
    
    def _analyze_browser_error(self, error_message: str, keyword_name: str) -> Dict[str, Any]:
        """Analyze Browser Library specific errors and provide targeted guidance."""
        analysis = {}
        error_lower = error_message.lower()
        
        if "strict mode violation" in error_lower or "multiple elements" in error_lower:
            analysis["strict_mode_violation"] = {
                "issue": "Selector matches multiple elements but strict mode is enabled",
                "solutions": [
                    "Make selector more specific to match only one element",
                    "Use 'Set Strict Mode    False' to allow multiple matches",
                    "Add more specific CSS selectors or attributes",
                    "Use nth-child() or other CSS pseudo-selectors for specific elements"
                ],
                "examples": [
                    "Instead of '.button' use '.button.primary' or '.button:nth-child(1)'",
                    "Instead of 'div' use 'div.container > div.content'",
                    "Add unique attributes like '[data-testid=\"submit-btn\"]'"
                ]
            }
        
        if "element not found" in error_lower or "waiting for selector" in error_lower:
            analysis["element_not_found_suggestions"] = [
                "Verify element exists on current page",
                "Check if element loads asynchronously (use Wait For Elements State)",
                "Try different selector strategies (CSS, XPath, text, ID)",
                "Check if element is inside an iFrame (use >>> syntax)",
                "Verify element is not in closed shadow DOM",
                "Use browser developer tools to inspect element",
                "Check if element appears after user interaction"
            ]
        
        if "timeout" in error_lower:
            analysis["timeout_suggestions"] = [
                "Increase timeout with explicit waits",
                "Use 'Wait For Elements State' before interaction",
                "Check if element loads dynamically",
                "Verify selector syntax is correct",
                "Use 'Wait For Load State' to ensure page is ready"
            ]
        
        if "shadow" in error_lower or "shadow root" in error_lower:
            analysis["shadow_dom_guidance"] = {
                "issue": "Element may be in shadow DOM",
                "solutions": [
                    "Use regular CSS (automatic shadow piercing): 'css=.my-element'",
                    "Use text selectors (automatic shadow piercing): 'text=Button Text'", 
                    "Avoid css:light= for shadow DOM elements",
                    "Check if shadow root is closed (not accessible)"
                ],
                "note": "Browser Library automatically pierces open shadow roots with CSS and text engines"
            }
        
        if "iframe" in error_lower or "frame" in error_lower:
            analysis["iframe_guidance"] = {
                "issue": "Element may be inside an iFrame",
                "solutions": [
                    "Use frame piercing syntax: 'id=myframe >>> .inner-element'",
                    "First select the frame, then the element inside",
                    "Use 'Set Selector Prefix' for multiple operations in same frame"
                ],
                "examples": [
                    "Click    id=login-frame >>> input[name='username']",
                    "Set Selector Prefix    id=content-frame\nClick    .submit-button"
                ]
            }
        
        return analysis
    
    def _get_keyword_argument_spec(self, keyword_name: str, library_name: Optional[str] = None):
        """Get Robot Framework ArgumentSpec for a keyword to validate named arguments.
        
        ENHANCED: Now properly handles **kwargs using DynamicArgumentParser.
        
        This is the general solution using RF's native APIs to get accurate parameter information
        including proper **kwargs detection for keywords like AppiumLibrary.Open Application.
        
        Args:
            keyword_name: Name of the keyword
            library_name: Optional library name for disambiguation
            
        Returns:
            ArgumentSpec object with proper var_named (**kwargs) support, None otherwise
        """
        if not RF_NATIVE_CONVERSION_AVAILABLE:
            logger.debug(f"RF native APIs not available, cannot get ArgumentSpec for {keyword_name}")
            return None
        
        try:
            # Method 1: LibDoc + DynamicArgumentParser for **kwargs support (primary method)
            # NOTE: TestLibrary approach skipped due to init parameter requirement issues
            try:
                from robot.libdoc import LibraryDocumentation
                from robot.running.arguments.argumentparser import DynamicArgumentParser
                
                if library_name:
                    libdoc = LibraryDocumentation(library_name)
                    
                    # Find the keyword in LibDoc
                    for kw in libdoc.keywords:
                        if kw.name == keyword_name:
                            if kw.args:
                                # Convert libdoc args to list of strings
                                arg_strings = [str(arg) for arg in kw.args]
                                logger.debug(f"LibDoc args for {keyword_name}: {arg_strings}")
                                
                                # Use DynamicArgumentParser to properly parse **kwargs
                                parser = DynamicArgumentParser()
                                try:
                                    spec = parser.parse(arg_strings, keyword_name)
                                    logger.debug(f"DynamicArgumentParser result for {keyword_name}: var_named='{spec.var_named}', positional={spec.positional_or_named}")
                                    return spec
                                except Exception as parse_error:
                                    logger.debug(f"DynamicArgumentParser failed for {keyword_name}: {parse_error}")
                            break
                        
            except Exception as libdoc_error:
                logger.debug(f"LibDoc approach failed for {keyword_name}: {libdoc_error}")
                
        except Exception as e:
            logger.debug(f"Failed to get ArgumentSpec for {keyword_name} from {library_name}: {e}")
        
        return None
    
    def get_appium_locator_guidance(self, error_message: str = None, keyword_name: str = None) -> Dict[str, Any]:
        """
        Provide AppiumLibrary locator strategy guidance for agents.
        
        Args:
            error_message: Optional error message to analyze
            keyword_name: Optional keyword name that failed
            
        Returns:
            Dict with AppiumLibrary locator strategies and guidance
        """
        guidance = {
            "locator_strategies": {
                "id": "Element ID - e.g., 'id=my_element' or just 'my_element' (default behavior)",
                "xpath": "XPath expression - e.g., '//*[@type=\"android.widget.EditText\"]'", 
                "identifier": "Matches by @id attribute - e.g., 'identifier=my_element'",
                "accessibility_id": "Accessibility options utilize - e.g., 'accessibility_id=button3'",
                "class": "Matches by class - e.g., 'class=UIAPickerWheel'",
                "name": "Matches by @name attribute - e.g., 'name=my_element' (Selendroid only)",
                "android": "Android UI Automator - e.g., 'android=UiSelector().description(\"Apps\")'",
                "ios": "iOS UI Automation - e.g., 'ios=.buttons().withName(\"Apps\")'",
                "predicate": "iOS Predicate - e.g., 'predicate=name==\"login\"'",
                "chain": "iOS Class Chain - e.g., 'chain=XCUIElementTypeWindow[1]/*'",
                "css": "CSS selector in webview - e.g., 'css=.green_button'"
            },
            "common_examples": {
                "By ID (default)": "my_element",
                "By ID explicit": "id=my_element", 
                "By XPath": "//*[@type='android.widget.EditText']",
                "By XPath explicit": "xpath=//*[@text='Login']",
                "By Accessibility ID": "accessibility_id=submit-button",
                "By Class": "class=android.widget.Button",
                "By Android UiAutomator": "android=UiSelector().description('Login')",
                "By iOS Predicate": "predicate=name=='login_button'",
                "By iOS Class Chain": "chain=XCUIElementTypeWindow[1]/XCUIElementTypeButton[2]",
                "WebView CSS": "css=.login-form .submit-btn"
            },
            "default_behavior": {
                "plain_text": "Plain text locators (e.g., 'my_element') are treated as ID lookups",
                "key_attributes": "By default, locators match against key attributes (id for all elements)",
                "xpath_detection": "XPath expressions should start with // or use explicit 'xpath=' prefix",
                "strategy_prefix": "Use 'strategy=value' format for explicit strategy selection"
            },
            "platform_specific": {
                "android": {
                    "ui_automator": "Use 'android=UiSelector()...' for complex Android element queries",
                    "examples": [
                        "android=UiSelector().className('android.widget.Button').text('Login')",
                        "android=UiSelector().resourceId('com.app:id/submit').enabled(true)",
                        "android=UiSelector().description('Search').clickable(true)"
                    ]
                },
                "ios": {
                    "predicate": "Use 'predicate=' for iOS NSPredicate queries",
                    "class_chain": "Use 'chain=' for iOS class chain queries",
                    "examples": [
                        "predicate=name BEGINSWITH 'login' AND visible == 1",
                        "predicate=type == 'XCUIElementTypeButton' AND name == 'Submit'", 
                        "chain=XCUIElementTypeWindow[1]/XCUIElementTypeButton[@name='Login']"
                    ]
                }
            },
            "webelement_support": {
                "description": "AppiumLibrary v1.4+ supports WebElement objects",
                "usage": [
                    "Get elements with: Get WebElements or Get WebElement",
                    "Use directly: Click Element ${element}",
                    "List access: Click Element @{elements}[2]"
                ],
                "example": """
*** Test Cases ***
Use WebElement
    @{elements}    Get WebElements    class=android.widget.Button
    Click Element    @{elements}[0]
                """
            },
            "tips": [
                "Plain locators (e.g., 'login_btn') are treated as ID lookups by default",
                "XPath expressions should start with // for automatic detection",
                "Use accessibility_id for accessible elements (recommended for cross-platform)",
                "Android UiAutomator provides powerful element selection capabilities",
                "iOS predicates offer flexible element matching with NSPredicate syntax",
                "WebView elements can use CSS selectors with 'css=' prefix",
                "Always verify element visibility and state before interaction"
            ]
        }
        
        # Add specific guidance based on error analysis
        if error_message and keyword_name:
            guidance.update(self._analyze_appium_error(error_message, keyword_name))
        
        return guidance
    
    def _analyze_appium_error(self, error_message: str, keyword_name: str) -> Dict[str, Any]:
        """Analyze AppiumLibrary specific errors and provide targeted guidance."""
        analysis = {}
        error_lower = error_message.lower()
        
        if "element not found" in error_lower or "no such element" in error_lower:
            analysis["element_not_found_suggestions"] = [
                "Verify the element exists on the current screen",
                "Try different locator strategies (id, xpath, accessibility_id, class)",
                "Check if element appears after app interaction or loading", 
                "Use explicit waits (Wait Until Element Is Visible)",
                "Verify app context is correct (native vs webview)",
                "Check if element is scrollable into view",
                "Use Appium Inspector to examine element attributes"
            ]
        
        if "timeout" in error_lower or "wait" in error_lower:
            analysis["timeout_suggestions"] = [
                "Increase implicit wait time for dynamic content",
                "Use explicit waits (Wait Until Element Is Visible/Enabled)",
                "Check if element loads asynchronously after user actions",
                "Verify locator strategy matches element attributes",
                "Consider element loading time in mobile networks",
                "Use Wait Until Page Contains Element for page-level waits"
            ]
        
        if "context" in error_lower or "webview" in error_lower:
            analysis["context_guidance"] = {
                "issue": "May need to switch between native and webview contexts",
                "solutions": [
                    "Use 'Get Contexts' to list available contexts",
                    "Switch to webview: 'Switch To Context    WEBVIEW_1'", 
                    "Switch to native: 'Switch To Context    NATIVE_APP'",
                    "Use CSS selectors only in webview context",
                    "Use native locators (id, xpath) in native context"
                ],
                "example": """
*** Test Cases ***
Handle WebView
    @{contexts}    Get Contexts
    Switch To Context    WEBVIEW_1
    Click Element    css=.login-button
    Switch To Context    NATIVE_APP
                """
            }
        
        if "session" in error_lower or "driver" in error_lower:
            analysis["session_guidance"] = {
                "issue": "Mobile session or driver may not be properly initialized",
                "solutions": [
                    "Ensure Open Application was called with correct capabilities",
                    "Check device connection and availability",
                    "Verify Appium server is running and accessible",
                    "Review device capabilities (platformName, deviceName, app path)",
                    "Check if app installation is required"
                ]
            }
        
        if "stale" in error_lower or "reference" in error_lower:
            analysis["stale_element_guidance"] = {
                "issue": "Element reference has become stale (element no longer attached to DOM)",
                "solutions": [
                    "Re-find the element before interaction",
                    "Avoid storing element references for long periods",
                    "Use locator strings instead of WebElement objects when possible",
                    "Refresh page or screen if element structure changed"
                ]
            }
        
        if "permission" in error_lower or "security" in error_lower:
            analysis["permission_guidance"] = {
                "issue": "App permissions or security restrictions may be blocking interaction",
                "solutions": [
                    "Grant required app permissions before testing",
                    "Handle permission dialogs with explicit waits and clicks",
                    "Check if device security settings block automation",
                    "Verify app is properly signed for testing"
                ]
            }
        
        return analysis
    
    def _get_keyword_argument_spec(self, keyword_name: str, library_name: Optional[str] = None):
        """Get Robot Framework ArgumentSpec for a keyword to validate named arguments.
        
        ENHANCED: Now properly handles **kwargs using DynamicArgumentParser.
        
        This is the general solution using RF's native APIs to get accurate parameter information
        including proper **kwargs detection for keywords like AppiumLibrary.Open Application.
        
        Args:
            keyword_name: Name of the keyword
            library_name: Optional library name for disambiguation
            
        Returns:
            ArgumentSpec object with proper var_named (**kwargs) support, None otherwise
        """
        if not RF_NATIVE_CONVERSION_AVAILABLE:
            logger.debug(f"RF native APIs not available, cannot get ArgumentSpec for {keyword_name}")
            return None
        
        try:
            # Method 1: LibDoc + DynamicArgumentParser for **kwargs support (primary method)
            # NOTE: TestLibrary approach skipped due to init parameter requirement issues
            try:
                from robot.libdoc import LibraryDocumentation
                from robot.running.arguments.argumentparser import DynamicArgumentParser
                
                if library_name:
                    libdoc = LibraryDocumentation(library_name)
                    
                    # Find the keyword in LibDoc
                    for kw in libdoc.keywords:
                        if kw.name == keyword_name:
                            if kw.args:
                                # Convert libdoc args to list of strings
                                arg_strings = [str(arg) for arg in kw.args]
                                logger.debug(f"LibDoc args for {keyword_name}: {arg_strings}")
                                
                                # Use DynamicArgumentParser to properly parse **kwargs
                                parser = DynamicArgumentParser()
                                try:
                                    spec = parser.parse(arg_strings, keyword_name)
                                    logger.debug(f"DynamicArgumentParser result for {keyword_name}: var_named='{spec.var_named}', positional={spec.positional_or_named}")
                                    return spec
                                except Exception as parse_error:
                                    logger.debug(f"DynamicArgumentParser failed for {keyword_name}: {parse_error}")
                            break
                        
            except Exception as libdoc_error:
                logger.debug(f"LibDoc approach failed for {keyword_name}: {libdoc_error}")
                
        except Exception as e:
            logger.debug(f"Failed to get ArgumentSpec for {keyword_name} from {library_name}: {e}")
        
        return None