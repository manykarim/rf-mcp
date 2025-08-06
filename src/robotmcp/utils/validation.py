"""Validation utilities for Robot Framework MCP Server."""

import re
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass

class InputValidator:
    """Validates inputs for MCP tools."""
    
    @staticmethod
    def validate_scenario(scenario: str) -> Tuple[bool, Optional[str]]:
        """Validate scenario input."""
        if not scenario:
            return False, "Scenario cannot be empty"
        
        if not isinstance(scenario, str):
            return False, "Scenario must be a string"
        
        if len(scenario.strip()) < 5:
            return False, "Scenario too short - provide more detailed description"
        
        if len(scenario) > 5000:
            return False, "Scenario too long - maximum 5000 characters"
        
        return True, None
    
    @staticmethod
    def validate_context(context: str) -> Tuple[bool, Optional[str]]:
        """Validate context input."""
        valid_contexts = ["web", "mobile", "api", "database"]
        
        if context not in valid_contexts:
            return False, f"Context must be one of: {', '.join(valid_contexts)}"
        
        return True, None
    
    @staticmethod
    def validate_keyword(keyword: str) -> Tuple[bool, Optional[str]]:
        """Validate Robot Framework keyword."""
        if not keyword:
            return False, "Keyword cannot be empty"
        
        if not isinstance(keyword, str):
            return False, "Keyword must be a string"
        
        # Check for invalid characters
        if re.search(r'[<>"|*?]', keyword):
            return False, "Keyword contains invalid characters"
        
        if len(keyword) > 255:
            return False, "Keyword too long - maximum 255 characters"
        
        return True, None
    
    @staticmethod
    def validate_arguments(arguments: List[str]) -> Tuple[bool, Optional[str]]:
        """Validate keyword arguments."""
        if not isinstance(arguments, list):
            return False, "Arguments must be a list"
        
        if len(arguments) > 50:
            return False, "Too many arguments - maximum 50"
        
        for i, arg in enumerate(arguments):
            if not isinstance(arg, str):
                return False, f"Argument {i} must be a string"
            
            if len(arg) > 1000:
                return False, f"Argument {i} too long - maximum 1000 characters"
        
        return True, None
    
    @staticmethod
    def validate_session_id(session_id: str) -> Tuple[bool, Optional[str]]:
        """Validate session ID."""
        if not session_id:
            return False, "Session ID cannot be empty"
        
        if not isinstance(session_id, str):
            return False, "Session ID must be a string"
        
        # Allow alphanumeric, hyphens, underscores
        if not re.match(r'^[a-zA-Z0-9_-]+$', session_id):
            return False, "Session ID can only contain letters, numbers, hyphens, and underscores"
        
        if len(session_id) > 100:
            return False, "Session ID too long - maximum 100 characters"
        
        return True, None
    
    @staticmethod
    def validate_test_name(test_name: str) -> Tuple[bool, Optional[str]]:
        """Validate test case name."""
        if not test_name:
            return False, "Test name cannot be empty"
        
        if not isinstance(test_name, str):
            return False, "Test name must be a string"
        
        if len(test_name.strip()) < 3:
            return False, "Test name too short - minimum 3 characters"
        
        if len(test_name) > 200:
            return False, "Test name too long - maximum 200 characters"
        
        # Check for invalid characters in Robot Framework test names
        if re.search(r'[<>"|*?\\/:]', test_name):
            return False, "Test name contains invalid characters"
        
        return True, None
    
    @staticmethod
    def validate_tags(tags: List[str]) -> Tuple[bool, Optional[str]]:
        """Validate test tags."""
        if not isinstance(tags, list):
            return False, "Tags must be a list"
        
        if len(tags) > 20:
            return False, "Too many tags - maximum 20"
        
        for i, tag in enumerate(tags):
            if not isinstance(tag, str):
                return False, f"Tag {i} must be a string"
            
            if not tag.strip():
                return False, f"Tag {i} cannot be empty"
            
            if len(tag) > 50:
                return False, f"Tag {i} too long - maximum 50 characters"
            
            # Tags cannot contain spaces in Robot Framework
            if ' ' in tag:
                return False, f"Tag '{tag}' cannot contain spaces"
        
        return True, None

class SecurityValidator:
    """Security validation for inputs."""
    
    # Potentially dangerous patterns
    DANGEROUS_PATTERNS = [
        r'__import__',
        r'exec\s*\(',
        r'eval\s*\(',
        r'subprocess',
        r'os\.system',
        r'os\.popen',
        r'file\s*\(',
        r'open\s*\(',
        r'input\s*\(',
        r'raw_input\s*\(',
    ]
    
    @staticmethod
    def check_for_injection(text: str) -> Tuple[bool, Optional[str]]:
        """Check for potential code injection attempts."""
        if not isinstance(text, str):
            return True, None
        
        text_lower = text.lower()
        
        for pattern in SecurityValidator.DANGEROUS_PATTERNS:
            if re.search(pattern, text_lower):
                return False, f"Potentially dangerous pattern detected: {pattern}"
        
        return True, None
    
    @staticmethod
    def validate_file_path(path: str) -> Tuple[bool, Optional[str]]:
        """Validate file paths for security."""
        if not isinstance(path, str):
            return False, "Path must be a string"
        
        # Check for path traversal attempts
        if '..' in path or path.startswith('/'):
            return False, "Path traversal not allowed"
        
        # Check for absolute paths on Windows
        if re.match(r'^[A-Za-z]:', path):
            return False, "Absolute paths not allowed"
        
        return True, None

class RobotFrameworkValidator:
    """Validates Robot Framework specific inputs."""
    
    # Common Robot Framework libraries and their typical keywords
    LIBRARY_KEYWORDS = {
        'BuiltIn': [
            'Log', 'Set Variable', 'Should Be Equal', 'Should Contain',
            'Sleep', 'Fail', 'Pass Execution', 'Return From Keyword'
        ],
        'SeleniumLibrary': [
            'Open Browser', 'Close Browser', 'Go To', 'Click Element',
            'Click Button', 'Input Text', 'Page Should Contain',
            'Element Should Be Visible', 'Wait Until Element Is Visible'
        ],
        'RequestsLibrary': [
            'Create Session', 'Get Request', 'Post Request', 'Put Request',
            'Delete Request', 'Response Should Be Json', 'Should Be Equal As Numbers'
        ],
        'DatabaseLibrary': [
            'Connect To Database', 'Disconnect From Database', 'Execute Sql String',
            'Query', 'Row Count', 'Description'
        ]
    }
    
    @staticmethod
    def validate_library_name(library: str) -> Tuple[bool, Optional[str]]:
        """Validate Robot Framework library name."""
        if not library:
            return False, "Library name cannot be empty"
        
        if not isinstance(library, str):
            return False, "Library name must be a string"
        
        # Basic validation for library names
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9_.]*$', library):
            return False, "Invalid library name format"
        
        return True, None
    
    @staticmethod
    def suggest_corrections(keyword: str) -> List[str]:
        """Suggest corrections for misspelled keywords."""
        suggestions = []
        keyword_lower = keyword.lower()
        
        # Check against known keywords
        for library, keywords in RobotFrameworkValidator.LIBRARY_KEYWORDS.items():
            for known_keyword in keywords:
                # Simple similarity check
                if (keyword_lower in known_keyword.lower() or 
                    known_keyword.lower() in keyword_lower):
                    suggestions.append(f"{known_keyword} (from {library})")
        
        return suggestions[:5]  # Limit to 5 suggestions
    
    @staticmethod
    def validate_keyword_syntax(keyword: str, arguments: List[str]) -> Tuple[bool, Optional[str]]:
        """Validate keyword syntax and argument compatibility."""
        keyword_lower = keyword.lower()
        
        # Check specific keyword patterns
        if 'input text' in keyword_lower:
            if len(arguments) < 2:
                return False, "Input Text requires at least 2 arguments: locator and text"
        
        elif 'click' in keyword_lower:
            if len(arguments) < 1:
                return False, "Click keywords require at least 1 argument: locator"
        
        elif 'open browser' in keyword_lower:
            if len(arguments) < 1:
                return False, "Open Browser requires at least 1 argument: url"
        
        elif 'should be equal' in keyword_lower:
            if len(arguments) < 2:
                return False, "Should Be Equal requires 2 arguments: actual and expected"
        
        return True, None

def validate_mcp_input(tool_name: str, **kwargs) -> Dict[str, Any]:
    """Comprehensive validation for MCP tool inputs."""
    errors = []
    warnings = []
    
    try:
        if tool_name == "analyze_scenario":
            scenario = kwargs.get("scenario", "")
            context = kwargs.get("context", "web")
            
            # Validate scenario
            valid, error = InputValidator.validate_scenario(scenario)
            if not valid:
                errors.append(f"Scenario validation: {error}")
            
            # Security check
            safe, security_error = SecurityValidator.check_for_injection(scenario)
            if not safe:
                errors.append(f"Security validation: {security_error}")
            
            # Validate context
            valid, error = InputValidator.validate_context(context)
            if not valid:
                errors.append(f"Context validation: {error}")
        
        elif tool_name == "execute_step":
            keyword = kwargs.get("keyword", "")
            arguments = kwargs.get("arguments", [])
            session_id = kwargs.get("session_id", "default")
            
            # Validate keyword
            valid, error = InputValidator.validate_keyword(keyword)
            if not valid:
                errors.append(f"Keyword validation: {error}")
            
            # Validate arguments
            valid, error = InputValidator.validate_arguments(arguments)
            if not valid:
                errors.append(f"Arguments validation: {error}")
            
            # Validate session ID
            valid, error = InputValidator.validate_session_id(session_id)
            if not valid:
                errors.append(f"Session ID validation: {error}")
            
            # Robot Framework specific validation
            valid, error = RobotFrameworkValidator.validate_keyword_syntax(keyword, arguments)
            if not valid:
                warnings.append(f"Keyword syntax: {error}")
            
            # Security checks
            for arg in arguments:
                safe, security_error = SecurityValidator.check_for_injection(arg)
                if not safe:
                    errors.append(f"Argument security: {security_error}")
        
        elif tool_name == "build_test_suite":
            test_name = kwargs.get("test_name", "")
            tags = kwargs.get("tags", [])
            session_id = kwargs.get("session_id", "default")
            
            # Validate test name
            valid, error = InputValidator.validate_test_name(test_name)
            if not valid:
                errors.append(f"Test name validation: {error}")
            
            # Validate tags
            valid, error = InputValidator.validate_tags(tags)
            if not valid:
                errors.append(f"Tags validation: {error}")
            
            # Validate session ID
            valid, error = InputValidator.validate_session_id(session_id)
            if not valid:
                errors.append(f"Session ID validation: {error}")
        
        # Add more tool-specific validations as needed
        
    except Exception as e:
        errors.append(f"Validation error: {str(e)}")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }