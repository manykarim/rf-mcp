"""Robot Framework-compatible variable resolution engine."""

import os
import re
import tempfile
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class VariableResolutionError(Exception):
    """Exception raised when variable resolution fails."""
    
    def __init__(self, variable_name: str, available_vars: List[str] = None, message: str = None):
        if available_vars is None:
            available_vars = []
        
        if message:
            super().__init__(message)
        else:
            suggestions = self._suggest_similar_variables(variable_name, available_vars)
            suggestion_text = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
            available_text = f" Available variables: {', '.join(available_vars[:5])}" if available_vars else ""
            super().__init__(f"Variable '${{{variable_name}}}' not found.{available_text}{suggestion_text}")
    
    def _suggest_similar_variables(self, target: str, available: List[str], max_suggestions: int = 3) -> List[str]:
        """Suggest similar variable names using simple string similarity."""
        if not available:
            return []
        
        # Simple similarity scoring based on common characters
        def similarity_score(a: str, b: str) -> float:
            a_lower, b_lower = a.lower(), b.lower()
            if a_lower == b_lower:
                return 1.0
            if a_lower in b_lower or b_lower in a_lower:
                return 0.8
            
            # Count common characters
            common = sum(1 for char in a_lower if char in b_lower)
            return common / max(len(a), len(b))
        
        # Score all variables and return top matches
        scored = [(var, similarity_score(target, var)) for var in available]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Return suggestions with score > 0.3
        return [var for var, score in scored[:max_suggestions] if score > 0.3]


class CircularReferenceError(VariableResolutionError):
    """Exception raised when circular variable references are detected."""
    
    def __init__(self, variable_chain: List[str]):
        chain_text = " → ".join(f"${{{var}}}" for var in variable_chain)
        super().__init__(
            variable_chain[0], 
            message=f"Circular variable reference detected: {chain_text}"
        )


class VariableResolver:
    """
    Robot Framework-compatible variable resolution engine.
    
    Provides comprehensive variable substitution that matches Robot Framework's
    behavior, including built-in variables, nested variables, and proper error handling.
    """
    
    # Regex patterns for different variable types
    VARIABLE_PATTERN = re.compile(r'\$\{([^}]+)\}')  # For normal variable resolution
    VARIABLE_SYNTAX_PATTERN = re.compile(r'\$\{([^}]*)\}')  # For syntax validation (allows empty)
    # Enhanced pattern to match variables with indexing in text: ${var}[index1][index2]...
    VARIABLE_WITH_INDEX_PATTERN = re.compile(r'\$\{[^}]+\}(?:\[[^\]]+\])*')
    LIST_ACCESS_PATTERN = re.compile(r'^([^[\]]+)\[([^\]]+)\]$')
    DICT_ACCESS_PATTERN = re.compile(r'^([^[\]]+)\[([^\]]+)\]$')  # Same as list for now
    
    def __init__(self):
        """Initialize the variable resolver with built-in variables."""
        self.builtin_variables = self._init_builtin_variables()
        self._resolution_stack = []  # For circular reference detection
    
    def resolve_arguments(self, 
                         arguments: List[str], 
                         session_variables: Dict[str, Any],
                         include_builtins: bool = True) -> List[str]:
        """
        Resolve variables in keyword arguments.
        
        Args:
            arguments: List of keyword arguments that may contain variables
            session_variables: Session-specific variables
            include_builtins: Whether to include Robot Framework built-in variables
            
        Returns:
            List of arguments with variables resolved to their actual values
            
        Raises:
            VariableResolutionError: If a variable cannot be resolved
            CircularReferenceError: If circular references are detected
        """
        if not arguments:
            return []
        
        try:
            # Clear resolution stack for each new resolution request
            self._resolution_stack.clear()
            
            # Combine all available variables
            all_variables = {}
            if include_builtins:
                all_variables.update(self.builtin_variables)
            all_variables.update(session_variables)
            
            resolved_args = []
            for arg in arguments:
                if isinstance(arg, str):
                    resolved_arg = self.resolve_single_argument(arg, session_variables, include_builtins)
                    resolved_args.append(resolved_arg)
                else:
                    # Non-string arguments pass through unchanged
                    resolved_args.append(arg)
            
            logger.debug(f"Resolved arguments: {arguments} → {resolved_args}")
            return resolved_args
            
        except Exception as e:
            logger.error(f"Error resolving arguments {arguments}: {e}")
            raise
        finally:
            # Always clear the resolution stack
            self._resolution_stack.clear()
    
    def resolve_single_argument(self, 
                               arg: str, 
                               variables: Dict[str, Any],
                               include_builtins: bool = True) -> Any:
        """
        Resolve variables in a single argument.
        
        Args:
            arg: Argument string that may contain variables
            variables: Available variables for resolution
            include_builtins: Whether to include Robot Framework built-in variables
            
        Returns:
            Resolved argument value (may be string, int, bool, list, etc.)
            
        Raises:
            VariableResolutionError: If a variable cannot be resolved
            CircularReferenceError: If circular references are detected
        """
        if not isinstance(arg, str):
            return arg
        
        # Combine variables with built-ins if requested
        all_variables = {}
        if include_builtins:
            all_variables.update(self.builtin_variables)
        all_variables.update(variables)
        
        # Check if the entire argument is a single variable
        if self._is_single_variable(arg):
            # Check for method calls or attribute access first
            if self._has_method_call_or_attribute_access(arg):
                return self._resolve_enhanced_variable_expression(arg, all_variables)
            # Handle both ${var} and ${var}[index] patterns
            elif '[' in arg and arg.endswith(']'):
                # Handle multiple index access like ${var}[index1][index2]
                return self._resolve_complex_indexed_access(arg, all_variables)
            else:
                # Simple variable: ${var}
                var_name = arg[2:-1]  # Remove ${ and }
                return self._resolve_variable_reference(var_name, all_variables)
        
        # Perform text substitution for variables within text
        return self._substitute_variables_in_text(arg, all_variables)
    
    def _is_single_variable(self, text: str) -> bool:
        """Check if text is exactly one variable (e.g., '${var}' or '${var}[index1][index2]')."""
        if not text.startswith('${'):
            return False
        
        # Find the matching closing brace for the variable
        brace_count = 0
        var_end = -1
        
        for i, char in enumerate(text):
            if char == '{' and i > 0 and text[i-1] == '$':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    var_end = i
                    break
        
        if var_end == -1:
            return False
        
        # Check if the rest is either nothing or valid index access pattern(s)
        remaining = text[var_end + 1:]
        if not remaining:
            return True  # Pure variable like ${var}
        
        # Check if it's one or more index access patterns like [0] or [0][key]
        return self._is_valid_index_chain(remaining)
    
    def _is_valid_index_chain(self, text: str) -> bool:
        """Check if text is a valid chain of index accesses like [0][key][item]."""
        if not text:
            return True
        
        if not text.startswith('['):
            return False
        
        pos = 0
        while pos < len(text):
            if text[pos] != '[':
                return False
            
            # Find the matching closing bracket
            bracket_count = 1
            pos += 1
            start_pos = pos
            
            while pos < len(text) and bracket_count > 0:
                if text[pos] == '[':
                    bracket_count += 1
                elif text[pos] == ']':
                    bracket_count -= 1
                pos += 1
            
            if bracket_count != 0:
                return False  # Unmatched brackets
            
            # Check that there's content between the brackets
            if pos - 1 <= start_pos:
                return False  # Empty brackets []
        
        return True
    
    def _substitute_variables_in_text(self, text: str, variables: Dict[str, Any]) -> str:
        """
        Substitute all variables in text, maintaining string context.
        
        This handles cases like "Hello ${name}!" or "Value: ${num}" 
        and also indexed variables like "User: ${user}[name]"
        """
        def replace_variable_with_index(match):
            full_var_expr = match.group(0)  # e.g., "${user}[name]" or "${list}[0][item]"
            
            try:
                # Use the complex indexed access resolver for full expressions
                if self._is_single_variable(full_var_expr):
                    if '[' in full_var_expr:
                        value = self._resolve_complex_indexed_access(full_var_expr, variables)
                    else:
                        # Simple variable
                        var_name = full_var_expr[2:-1]  # Remove ${ and }
                        value = self._resolve_variable_reference(var_name, variables)
                else:
                    # Fallback to simple variable resolution
                    var_name = full_var_expr[2:full_var_expr.find('}')]
                    value = self._resolve_variable_reference(var_name, variables)
                
                # Convert to string for text substitution
                return str(value)
            except CircularReferenceError:
                # Let circular reference errors pass through unchanged
                raise
            except VariableResolutionError as e:
                # Re-raise with better context
                raise VariableResolutionError(
                    full_var_expr, 
                    list(variables.keys()),
                    f"Variable '{full_var_expr}' not found in text: '{text}'"
                )
        
        # First, handle variables with indexing (${var}[index]...)
        # This needs to be done before simple variables to avoid partial matches
        return self.VARIABLE_WITH_INDEX_PATTERN.sub(replace_variable_with_index, text)
    
    def _resolve_variable_reference(self, var_name: str, variables: Dict[str, Any]) -> Any:
        """
        Resolve a single variable reference with support for complex patterns.
        
        Supports:
        - Simple variables: var_name
        - List access: list_var[0] 
        - Dict access: dict_var[key]
        - Nested variables: outer_${inner}
        """
        # Check for circular references
        if var_name in self._resolution_stack:
            raise CircularReferenceError(self._resolution_stack + [var_name])
        
        try:
            self._resolution_stack.append(var_name)
            
            # Handle nested variables first (e.g., outer_${inner})
            if '${' in var_name:
                resolved_var_name = self._substitute_variables_in_text(var_name, variables)
                return self._resolve_simple_or_indexed_variable(resolved_var_name, variables)
            
            # Handle simple variables - but check if value contains more variables
            normalized_name = self._normalize_variable_name(var_name)
            
            if normalized_name in variables:
                value = variables[normalized_name]
                logger.debug(f"Resolved variable {normalized_name} = {value}")
                
                # If the value is a string that contains variables, resolve them too
                if isinstance(value, str) and '${' in value:
                    return self._substitute_variables_in_text(value, variables)
                
                return value
            
            # Variable not found
            available_vars = [name.replace('${', '').replace('}', '') 
                             for name in variables.keys() 
                             if name.startswith('${')]
            raise VariableResolutionError(var_name, available_vars)
            
        finally:
            # Always remove from stack
            if var_name in self._resolution_stack:
                self._resolution_stack.remove(var_name)
    
    def _resolve_simple_or_indexed_variable(self, var_name: str, variables: Dict[str, Any]) -> Any:
        """Resolve simple variable access (no indexing handled here)."""
        
        # Simple variable access only
        normalized_name = self._normalize_variable_name(var_name)
        
        if normalized_name in variables:
            value = variables[normalized_name]
            logger.debug(f"Resolved variable {normalized_name} = {value}")
            return value
        
        # Variable not found
        available_vars = [name.replace('${', '').replace('}', '') 
                         for name in variables.keys() 
                         if name.startswith('${')]
        raise VariableResolutionError(var_name, available_vars)
    
    def _resolve_complex_indexed_access(self, arg: str, variables: Dict[str, Any]) -> Any:
        """
        Resolve complex indexed access like ${var}[index1][index2][index3].
        
        Args:
            arg: Full argument like '${TEST_DATA}[${CURRENT_ROW}][name]' or '${${prefix}_var}[0]'
            variables: Available variables
            
        Returns:
            Final resolved value after all index accesses
        """
        # Parse: ${var}[index1][index2]... -> var_name and [index1, index2, ...]
        
        if not arg.startswith('${'):
            raise VariableResolutionError("", message=f"Invalid variable syntax: {arg}")
        
        # For nested variables like ${${prefix}_var}[0], we need to find the matching closing brace
        # This is more complex than just finding the first }
        
        brace_count = 0
        var_end = -1
        
        for i, char in enumerate(arg):
            if char == '{' and i > 0 and arg[i-1] == '$':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    var_end = i
                    break
        
        if var_end == -1:
            raise VariableResolutionError("", message=f"Unclosed variable brace: {arg}")
        
        var_name = arg[2:var_end]  # Extract variable name without ${}
        
        # Extract all index expressions: [index1][index2]...
        remaining = arg[var_end + 1:]
        
        if not remaining:
            # No indexing, just resolve the variable
            return self._resolve_variable_reference(var_name, variables)
        
        # Parse multiple index expressions
        indices = []
        current_pos = 0
        
        while current_pos < len(remaining):
            if remaining[current_pos] != '[':
                raise VariableResolutionError("", message=f"Expected '[' at position {current_pos} in: {remaining}")
            
            # Find the matching closing bracket
            bracket_count = 0
            start_pos = current_pos + 1
            end_pos = start_pos
            
            for i in range(start_pos, len(remaining)):
                if remaining[i] == '[':
                    bracket_count += 1
                elif remaining[i] == ']':
                    if bracket_count == 0:
                        end_pos = i
                        break
                    else:
                        bracket_count -= 1
            else:
                raise VariableResolutionError("", message=f"Unmatched '[' in: {remaining}")
            
            # Extract the index expression
            index_expr = remaining[start_pos:end_pos]
            indices.append(index_expr)
            current_pos = end_pos + 1
        
        # Start with the base variable and apply each index in sequence
        current_value = self._resolve_variable_reference(var_name, variables)
        
        for i, index_expr in enumerate(indices):
            try:
                current_value = self._apply_single_index(current_value, index_expr, variables, f"{var_name}[{'...'.join(indices[:i+1])}]")
            except Exception as e:
                raise VariableResolutionError(
                    f"{var_name}[{'...'.join(indices[:i+1])}]",
                    message=f"Failed to apply index '{index_expr}' to {var_name}: {e}"
                )
        
        return current_value
    
    def _apply_single_index(self, base_value: Any, index_expr: str, variables: Dict[str, Any], full_path: str) -> Any:
        """Apply a single index to a value."""
        # Resolve the index expression (it might contain variables)
        if '${' in index_expr:
            resolved_index = self._substitute_variables_in_text(index_expr, variables)
        else:
            resolved_index = index_expr
        
        # Perform the access
        try:
            if isinstance(base_value, (list, tuple)):
                # List/tuple access - index should be integer
                index = int(resolved_index)
                return base_value[index]
            elif isinstance(base_value, dict):
                # Dictionary access - try key as-is, then as string
                if resolved_index in base_value:
                    return base_value[resolved_index]
                # Try converting index to int for numeric keys
                try:
                    numeric_index = int(resolved_index)
                    if numeric_index in base_value:
                        return base_value[numeric_index]
                except ValueError:
                    pass
                # Key not found
                raise KeyError(resolved_index)
            else:
                raise TypeError(f"Value is not indexable (type: {type(base_value)})")
                
        except (IndexError, KeyError, TypeError, ValueError) as e:
            raise VariableResolutionError(
                full_path,
                message=f"Failed to access index '{resolved_index}': {e}"
            )
    
    def _resolve_indexed_access(self, base_var: str, index_expr: str, variables: Dict[str, Any]) -> Any:
        """Resolve indexed access like list[0] or dict[key]."""
        
        # Get the base variable
        base_value = self._resolve_variable_reference(base_var, variables)
        
        # Resolve the index expression (it might contain variables too)
        if '${' in index_expr:
            resolved_index = self._substitute_variables_in_text(index_expr, variables)
        else:
            resolved_index = index_expr
        
        # Perform the access
        try:
            if isinstance(base_value, (list, tuple)):
                # List/tuple access - index should be integer
                index = int(resolved_index)
                return base_value[index]
            elif isinstance(base_value, dict):
                # Dictionary access - try key as-is, then as string
                if resolved_index in base_value:
                    return base_value[resolved_index]
                # Try converting index to int for numeric keys
                try:
                    numeric_index = int(resolved_index)
                    if numeric_index in base_value:
                        return base_value[numeric_index]
                except ValueError:
                    pass
                # Key not found
                raise KeyError(resolved_index)
            else:
                raise TypeError(f"Variable '{base_var}' is not indexable (type: {type(base_value)})")
                
        except (IndexError, KeyError, TypeError, ValueError) as e:
            raise VariableResolutionError(
                f"{base_var}[{index_expr}]",
                message=f"Failed to access index '{resolved_index}' in variable '{base_var}': {e}"
            )
    
    def _normalize_variable_name(self, name: str) -> str:
        """Normalize variable name to Robot Framework format (${name})."""
        if not name.startswith('${') or not name.endswith('}'):
            return f"${{{name}}}"
        return name
    
    def _init_builtin_variables(self) -> Dict[str, Any]:
        """
        Initialize Robot Framework built-in variables.
        
        Returns dictionary of standard RF built-in variables that are always available.
        """
        current_dir = Path.cwd()
        
        builtin_vars = {
            # Directory and path variables
            "${CURDIR}": str(current_dir),
            "${EXECDIR}": str(current_dir),  # In real RF, this would be different
            "${TEMPDIR}": tempfile.gettempdir(),
            
            # Special characters and constants
            "${SPACE}": " ",
            "${EMPTY}": "",
            "${TRUE}": True,
            "${FALSE}": False,
            "${NULL}": None,
            "${NONE}": None,  # Alternative to NULL
            
            # Path separators
            "${/}": os.sep,
            "${:}": os.pathsep,
            
            # Escape characters  
            "${\\n}": "\n",
            "${\\r}": "\r",
            "${\\t}": "\t",
            "${\\\\}": "\\",
            
            # Numeric constants
            "${0}": 0,
            "${1}": 1,
            "${-1}": -1,
        }
        
        logger.debug(f"Initialized {len(builtin_vars)} built-in variables")
        return builtin_vars
    
    def get_available_variables(self, session_variables: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get all available variables (built-in + session).
        
        Args:
            session_variables: Session-specific variables
            
        Returns:
            Combined dictionary of all available variables
        """
        all_vars = self.builtin_variables.copy()
        if session_variables:
            all_vars.update(session_variables)
        return all_vars
    
    def validate_variable_syntax(self, text: str) -> Tuple[bool, List[str]]:
        """
        Validate variable syntax in text.
        
        Args:
            text: Text to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Find all variable patterns (using syntax pattern that allows empty variables)
        variables = self.VARIABLE_SYNTAX_PATTERN.findall(text)
        
        for var in variables:
            # Check for basic syntax issues
            if not var.strip():
                errors.append("Empty variable name: ${}")
            elif var.startswith('{') or var.endswith('}'):
                errors.append(f"Invalid nested braces in variable: ${{{var}}}")
            elif '$$' in var:
                errors.append(f"Invalid double dollar sign in variable: ${{{var}}}")
        
        # Check for unmatched braces (more sophisticated check)
        # Count only ${...} patterns, not standalone braces
        syntax_matches = self.VARIABLE_SYNTAX_PATTERN.findall(text)
        dollar_brace_count = text.count('${')
        
        # If we have ${...} patterns but different counts, there might be malformed variables
        if dollar_brace_count != len(syntax_matches):
            unmatched_variables = []
            # Find potential unmatched ${...} patterns
            import re
            potential_vars = re.findall(r'\$\{[^}]*', text)  # Finds ${... without closing }
            for var in potential_vars:
                if not var.endswith('}'):
                    unmatched_variables.append(var)
            
            if unmatched_variables:
                errors.append(f"Unclosed variable braces: {', '.join(unmatched_variables)}")
        
        return len(errors) == 0, errors
    
    def preview_resolution(self, text: str, variables: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preview variable resolution without actually performing it.
        
        Useful for debugging and showing users what variables will be resolved.
        
        Args:
            text: Text containing variables to preview
            variables: Available variables
            
        Returns:
            Dictionary with resolution preview information
        """
        preview = {
            "original": text,
            "variables_found": [],
            "resolution_preview": {},
            "syntax_valid": True,
            "syntax_errors": [],
            "missing_variables": []
        }
        
        # Validate syntax first
        is_valid, syntax_errors = self.validate_variable_syntax(text)
        preview["syntax_valid"] = is_valid
        preview["syntax_errors"] = syntax_errors
        
        if not is_valid:
            return preview
        
        # Find all variables
        variables_found = self.VARIABLE_PATTERN.findall(text)
        preview["variables_found"] = list(set(variables_found))  # Remove duplicates
        
        # Try to resolve each variable
        all_variables = self.get_available_variables(variables)
        
        for var_name in preview["variables_found"]:
            try:
                resolved_value = self._resolve_variable_reference(var_name, all_variables)
                preview["resolution_preview"][f"${{{var_name}}}"] = {
                    "value": resolved_value,
                    "type": type(resolved_value).__name__,
                    "resolved": True
                }
            except VariableResolutionError as e:
                preview["resolution_preview"][f"${{{var_name}}}"] = {
                    "value": None,
                    "type": "undefined",
                    "resolved": False,
                    "error": str(e)
                }
                preview["missing_variables"].append(var_name)
            except Exception as e:
                preview["resolution_preview"][f"${{{var_name}}}"] = {
                    "value": None,
                    "type": "error",
                    "resolved": False,
                    "error": f"Resolution error: {e}"
                }
        
        return preview
    
    def _has_method_call_or_attribute_access(self, arg: str) -> bool:
        """Check if the argument contains method calls, attribute access, or indexing operations."""
        if not arg.startswith('${') or not arg.endswith('}'):
            return False
        
        # Extract content between ${ and }
        content = arg[2:-1]
        
        # Check for method calls (contains parentheses), attribute access (contains dots), 
        # or indexing inside the expression (contains brackets)
        has_method_call = '(' in content
        has_attribute_access = '.' in content and not content.replace('.', '').isdigit()
        has_internal_indexing = '[' in content and ']' in content
        
        return has_method_call or has_attribute_access or has_internal_indexing
    
    def _resolve_enhanced_variable_expression(self, arg: str, variables: Dict[str, Any]) -> Any:
        """
        Resolve enhanced variable expressions with method calls and attribute access.
        
        Examples:
        - ${response.json()}
        - ${response.json()[0]['bookingid']}
        - ${obj.method().attribute}
        
        Args:
            arg: Variable expression like ${var.method()}
            variables: Available variables
            
        Returns:
            Result of the expression evaluation
        """
        try:
            # Extract the expression content
            expression = arg[2:-1]  # Remove ${ and }
            logger.debug(f"Resolving enhanced expression: {expression}")
            
            # Parse the expression to find the base variable and operations
            return self._evaluate_expression(expression, variables)
            
        except Exception as e:
            logger.warning(f"Failed to resolve enhanced expression {arg}: {e}")
            # Fall back to treating it as a simple variable name
            var_name = arg[2:-1]
            available_vars = [name.replace('${', '').replace('}', '') 
                             for name in variables.keys() 
                             if name.startswith('${')]
            raise VariableResolutionError(var_name, available_vars)
    
    def _evaluate_expression(self, expression: str, variables: Dict[str, Any]) -> Any:
        """
        Safely evaluate a variable expression with method calls and attribute access.
        
        Args:
            expression: Expression to evaluate (without ${ })
            variables: Available variables
            
        Returns:
            Result of evaluation
        """
        import re
        
        # Find the base variable name (everything before first . or [)
        match = re.match(r'^([^.\[]+)', expression)
        if not match:
            raise ValueError(f"Cannot parse base variable from expression: {expression}")
        
        base_var_name = match.group(1)
        
        # Get the base variable value
        normalized_name = self._normalize_variable_name(base_var_name)
        if normalized_name not in variables:
            available_vars = [name.replace('${', '').replace('}', '') 
                             for name in variables.keys() 
                             if name.startswith('${')]
            raise VariableResolutionError(base_var_name, available_vars)
        
        base_value = variables[normalized_name]
        logger.debug(f"Base variable {base_var_name} resolved to {type(base_value).__name__}")
        
        # Get the operations part (everything after base variable)
        operations = expression[len(base_var_name):]
        
        # Apply operations sequentially
        result = base_value
        current_pos = 0
        
        while current_pos < len(operations):
            char = operations[current_pos]
            
            if char == '.':
                # Method call or attribute access
                current_pos += 1
                result, consumed = self._apply_method_or_attribute(result, operations[current_pos:])
                current_pos += consumed
                
            elif char == '[':
                # Index access
                result, consumed = self._apply_index_access(result, operations[current_pos:])
                current_pos += consumed
                
            else:
                current_pos += 1
        
        logger.debug(f"Enhanced expression result: {result}")
        return result
    
    def _apply_method_or_attribute(self, obj: Any, operations: str) -> tuple:
        """
        Apply method call or attribute access to an object.
        
        Returns:
            (result, consumed_chars)
        """
        import re
        
        # Match method call: method_name() or method_name(args)
        method_match = re.match(r'^([^.\[\(]+)\(([^)]*)\)', operations)
        if method_match:
            method_name = method_match.group(1)
            args_str = method_match.group(2).strip()
            
            if hasattr(obj, method_name):
                method = getattr(obj, method_name)
                if callable(method):
                    # Parse arguments if any
                    if args_str:
                        # Simple argument parsing - could be enhanced
                        args = [arg.strip().strip('"\'') for arg in args_str.split(',')]
                        result = method(*args)
                    else:
                        result = method()
                    
                    consumed = len(method_match.group(0))
                    logger.debug(f"Called method {method_name}() on {type(obj).__name__}")
                    return result, consumed
                else:
                    raise ValueError(f"'{method_name}' is not callable on {type(obj).__name__}")
            else:
                raise AttributeError(f"'{type(obj).__name__}' object has no attribute '{method_name}'")
        
        # Match attribute access: attribute_name
        attr_match = re.match(r'^([^.\[\(]+)', operations)
        if attr_match:
            attr_name = attr_match.group(1)
            
            if hasattr(obj, attr_name):
                result = getattr(obj, attr_name)
                consumed = len(attr_name)
                logger.debug(f"Accessed attribute {attr_name} on {type(obj).__name__}")
                return result, consumed
            else:
                raise AttributeError(f"'{type(obj).__name__}' object has no attribute '{attr_name}'")
        
        raise ValueError(f"Cannot parse method/attribute from: {operations}")
    
    def _apply_index_access(self, obj: Any, operations: str) -> tuple:
        """
        Apply index access to an object.
        
        Returns:
            (result, consumed_chars)
        """
        import re
        
        # Match index access: [index] or ['key'] or ["key"]
        index_match = re.match(r'^\[([^\]]+)\]', operations)
        if index_match:
            index_str = index_match.group(1).strip()
            
            # Handle different index types
            if index_str.startswith('"') and index_str.endswith('"'):
                # String key
                index = index_str[1:-1]  # Remove quotes
            elif index_str.startswith("'") and index_str.endswith("'"):
                # String key
                index = index_str[1:-1]  # Remove quotes
            elif index_str.isdigit():
                # Integer index
                index = int(index_str)
            else:
                # Try as string key first, then integer
                try:
                    index = int(index_str)
                except ValueError:
                    index = index_str
            
            # Apply index access
            try:
                result = obj[index]
                consumed = len(index_match.group(0))
                logger.debug(f"Accessed index [{index}] on {type(obj).__name__}")
                return result, consumed
            except (KeyError, IndexError, TypeError) as e:
                raise ValueError(f"Cannot access index [{index}] on {type(obj).__name__}: {e}")
        
        raise ValueError(f"Cannot parse index access from: {operations}")
    
    def _normalize_variable_name(self, var_name: str) -> str:
        """Normalize variable name to standard format with ${} wrapper."""
        if not var_name.startswith('${'):
            return f"${{{var_name}}}"
        return var_name