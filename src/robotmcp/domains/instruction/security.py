"""Instruction Domain Security Module.

This module provides comprehensive security validation for MCP instructions,
including input validation, path security, and prompt injection prevention.

Security Controls:
- CVE-INST-001: Input validation for environment variables (mode allowlist)
- CVE-INST-002: Path traversal prevention with prefix validation
- CVE-INST-003: Prompt injection detection and content sanitization
- CVE-INST-004: Secure logging with redaction

References:
- docs/security/MCP_INSTRUCTIONS_SECURITY_ANALYSIS.md
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, FrozenSet, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Base exception for security-related errors."""

    pass


class PathTraversalError(SecurityError):
    """Raised when path traversal attempt is detected."""

    pass


class PromptInjectionError(SecurityError):
    """Raised when prompt injection attempt is detected."""

    pass


class InvalidInstructionModeError(SecurityError):
    """Raised when an invalid instruction mode is provided."""

    pass


class ContentValidationError(SecurityError):
    """Raised when content validation fails."""

    pass


class ValidInstructionMode(Enum):
    """Valid instruction modes with strict enum enforcement.

    CVE-INST-001 Mitigation: Only these modes are accepted.
    """

    OFF = "off"
    DEFAULT = "default"
    MINIMAL = "minimal"
    STRICT = "strict"
    CUSTOM = "custom"

    @classmethod
    def from_string(cls, value: str) -> "ValidInstructionMode":
        """Parse mode from string with validation.

        Args:
            value: Mode string (case-insensitive).

        Returns:
            ValidInstructionMode enum value.

        Raises:
            InvalidInstructionModeError: If value is not a valid mode.
        """
        normalized = value.strip().lower()
        try:
            return cls(normalized)
        except ValueError:
            valid_modes = [m.value for m in cls]
            raise InvalidInstructionModeError(
                f"Invalid instruction mode: '{value}'. "
                f"Must be one of: {', '.join(valid_modes)}"
            )

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Check if a string is a valid mode."""
        try:
            cls.from_string(value)
            return True
        except InvalidInstructionModeError:
            return False


@dataclass(frozen=True)
class SecurityValidationResult:
    """Result of security validation.

    Attributes:
        is_safe: Whether the input passed all security checks.
        issues: List of security issues found.
        warnings: List of non-critical warnings.
        sanitized_value: Sanitized version of the input (if applicable).
    """

    is_safe: bool
    issues: Tuple[str, ...] = field(default_factory=tuple)
    warnings: Tuple[str, ...] = field(default_factory=tuple)
    sanitized_value: Optional[str] = None

    def __bool__(self) -> bool:
        """Allow boolean evaluation."""
        return self.is_safe

    @property
    def has_issues(self) -> bool:
        """Check if there are security issues."""
        return len(self.issues) > 0

    @property
    def has_warnings(self) -> bool:
        """Check if there are warnings."""
        return len(self.warnings) > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_safe": self.is_safe,
            "issues": list(self.issues),
            "warnings": list(self.warnings),
            "sanitized_value": self.sanitized_value,
        }


class InstructionContentValidator:
    """Validates instruction content for security threats.

    CVE-INST-003 Mitigation: Detects and blocks prompt injection attempts.

    Attributes:
        MAX_CONTENT_LENGTH: Maximum allowed content length (2000 chars).
        MAX_LINES: Maximum allowed line count.
        DANGEROUS_PATTERNS: Compiled regex patterns for injection detection.
    """

    # Content limits
    MAX_CONTENT_LENGTH: ClassVar[int] = 2000
    MAX_LINES: ClassVar[int] = 100

    # Dangerous patterns that indicate potential prompt injection
    DANGEROUS_PATTERN_DEFINITIONS: ClassVar[List[Tuple[str, str]]] = [
        # Jailbreak patterns
        (r"ignore\s+(all\s+)?(previous|prior)\s+(instructions?|prompts?)", "jailbreak_ignore"),
        (r"you\s+are\s+(now|DAN|a\s+new)", "jailbreak_identity"),
        (r"pretend\s+(to\s+be|you\s+are)", "jailbreak_pretend"),
        (r"act\s+as\s+if", "jailbreak_act"),
        (r"disregard\s+(all|previous|prior)", "jailbreak_disregard"),
        (r"override\s+(all|previous|prior)", "jailbreak_override"),
        (r"forget\s+(all|everything|previous)", "jailbreak_forget"),
        (r"new\s+rules?\s*:", "jailbreak_new_rules"),
        (r"system\s+prompt\s+(is|was|override)", "jailbreak_system"),
        # Data exfiltration via variable interpolation
        (r"\$\{[A-Z_]*KEY[A-Z_]*\}", "exfil_key"),
        (r"\$\{[A-Z_]*SECRET[A-Z_]*\}", "exfil_secret"),
        (r"\$\{[A-Z_]*TOKEN[A-Z_]*\}", "exfil_token"),
        (r"\$\{[A-Z_]*PASSWORD[A-Z_]*\}", "exfil_password"),
        (r"\$\{[A-Z_]*CREDENTIAL[A-Z_]*\}", "exfil_credential"),
        (r"\$\{[A-Z_]*API[A-Z_]*\}", "exfil_api"),
        (r"\$\{ANTHROPIC_[A-Z_]+\}", "exfil_anthropic"),
        (r"\$\{OPENAI_[A-Z_]+\}", "exfil_openai"),
        # Python dangerous patterns
        (r"os\.environ", "code_environ"),
        (r"getenv\s*\(", "code_getenv"),
        (r"subprocess", "code_subprocess"),
        (r"\beval\s*\(", "code_eval"),
        (r"\bexec\s*\(", "code_exec"),
        (r"__import__", "code_import"),
        (r"__builtins__", "code_builtins"),
        # Tool abuse patterns
        (r"execute[_\s]*(step|keyword)\s*.*\$\{", "tool_abuse_execute"),
        (r"run[_\s]*keyword\s*.*\$\{", "tool_abuse_run"),
        # URL injection for data exfiltration
        (r"https?://[^\s]+\$\{", "url_injection"),
        (r"(attacker|evil|malicious)\.com", "suspicious_domain"),
        # Shell injection patterns
        (r"\$\([^)]+\)", "shell_command_sub"),
        (r"`[^`]+`", "shell_backtick"),
        (r"\|\s*(bash|sh|cmd|powershell)", "shell_pipe"),
    ]

    # Forbidden control characters (except newline, tab, carriage return)
    CONTROL_CHAR_PATTERN: ClassVar[re.Pattern] = re.compile(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"
    )

    # Compile patterns on class load
    _compiled_patterns: ClassVar[List[Tuple[re.Pattern, str]]] = []

    @classmethod
    def _ensure_patterns_compiled(cls) -> None:
        """Ensure patterns are compiled (lazy initialization)."""
        if not cls._compiled_patterns:
            cls._compiled_patterns = [
                (re.compile(pattern, re.IGNORECASE), name)
                for pattern, name in cls.DANGEROUS_PATTERN_DEFINITIONS
            ]

    @classmethod
    def validate(
        cls,
        content: str,
        strict_mode: bool = False,
    ) -> SecurityValidationResult:
        """Validate instruction content for security threats.

        Args:
            content: The instruction content to validate.
            strict_mode: If True, any detected pattern is a blocking issue.
                        If False, some patterns may be warnings only.

        Returns:
            SecurityValidationResult with validation outcome.
        """
        cls._ensure_patterns_compiled()

        issues: List[str] = []
        warnings: List[str] = []

        # Check for empty content
        if not content or not content.strip():
            issues.append("Content is empty or whitespace-only")
            return SecurityValidationResult(
                is_safe=False, issues=tuple(issues), warnings=tuple(warnings)
            )

        # Check content length
        if len(content) > cls.MAX_CONTENT_LENGTH:
            issues.append(
                f"Content exceeds maximum length: {len(content)} > {cls.MAX_CONTENT_LENGTH}"
            )

        # Check line count
        line_count = len(content.splitlines())
        if line_count > cls.MAX_LINES:
            warnings.append(
                f"Content has many lines: {line_count} > {cls.MAX_LINES}"
            )

        # Check for control characters
        control_matches = cls.CONTROL_CHAR_PATTERN.findall(content)
        if control_matches:
            issues.append(
                f"Content contains {len(control_matches)} forbidden control characters"
            )

        # Check for dangerous patterns
        for pattern, pattern_name in cls._compiled_patterns:
            if pattern.search(content):
                issue_msg = f"Dangerous pattern detected: {pattern_name}"
                if strict_mode:
                    issues.append(issue_msg)
                else:
                    # Some patterns are always blocking
                    if pattern_name.startswith(("jailbreak_", "exfil_", "code_")):
                        issues.append(issue_msg)
                    else:
                        warnings.append(issue_msg)

        return SecurityValidationResult(
            is_safe=len(issues) == 0,
            issues=tuple(issues),
            warnings=tuple(warnings),
        )

    @classmethod
    def sanitize(
        cls,
        content: str,
        redact_patterns: bool = True,
        truncate: bool = True,
    ) -> SecurityValidationResult:
        """Sanitize instruction content by removing dangerous patterns.

        Args:
            content: The content to sanitize.
            redact_patterns: If True, replace dangerous patterns with [REDACTED].
            truncate: If True, truncate to MAX_CONTENT_LENGTH.

        Returns:
            SecurityValidationResult with sanitized content.
        """
        cls._ensure_patterns_compiled()

        warnings: List[str] = []
        sanitized = content

        # Remove control characters
        control_count = len(cls.CONTROL_CHAR_PATTERN.findall(sanitized))
        if control_count > 0:
            sanitized = cls.CONTROL_CHAR_PATTERN.sub("", sanitized)
            warnings.append(f"Removed {control_count} control characters")

        # Redact dangerous patterns
        if redact_patterns:
            for pattern, pattern_name in cls._compiled_patterns:
                matches = pattern.findall(sanitized)
                if matches:
                    sanitized = pattern.sub("[REDACTED]", sanitized)
                    warnings.append(f"Redacted pattern: {pattern_name}")

        # Truncate if needed
        if truncate and len(sanitized) > cls.MAX_CONTENT_LENGTH:
            original_len = len(sanitized)
            sanitized = sanitized[: cls.MAX_CONTENT_LENGTH]
            warnings.append(
                f"Truncated from {original_len} to {cls.MAX_CONTENT_LENGTH} chars"
            )

        return SecurityValidationResult(
            is_safe=True,  # Sanitized content is considered safe
            issues=(),
            warnings=tuple(warnings),
            sanitized_value=sanitized,
        )


class SecurePathValidator:
    """Secure path validation for instruction files.

    CVE-INST-002 Mitigation: Prevents path traversal attacks.

    Attributes:
        ALLOWED_EXTENSIONS: File extensions allowed for instruction files.
        MAX_FILE_SIZE: Maximum allowed file size in bytes.
        MAX_PATH_LENGTH: Maximum allowed path length.
    """

    ALLOWED_EXTENSIONS: ClassVar[FrozenSet[str]] = frozenset(
        {".txt", ".md", ".instruction", ".instructions"}
    )

    MAX_FILE_SIZE: ClassVar[int] = 10 * 1024  # 10KB
    MAX_PATH_LENGTH: ClassVar[int] = 256

    # Path traversal patterns
    TRAVERSAL_PATTERNS: ClassVar[List[re.Pattern]] = [
        re.compile(r"\.\."),  # Parent directory reference
        re.compile(r"^~"),  # Home directory expansion
        re.compile(r"^/"),  # Absolute path (Unix)
        re.compile(r"^[A-Za-z]:"),  # Absolute path (Windows)
        re.compile(r"\\\\"),  # UNC path
        re.compile(r"\.\.[\\/]"),  # Encoded traversal
        re.compile(r"%2e%2e", re.IGNORECASE),  # URL-encoded ..
        re.compile(r"%252e%252e", re.IGNORECASE),  # Double URL-encoded ..
    ]

    @classmethod
    def validate(
        cls,
        user_path: str,
        base_dir: Optional[str] = None,
        check_exists: bool = True,
    ) -> Tuple[bool, Optional[str], Optional[Path]]:
        """Validate and resolve instruction file path securely.

        Args:
            user_path: The user-provided path to validate.
            base_dir: Base directory for relative paths. Defaults to CWD.
            check_exists: Whether to verify the file exists.

        Returns:
            Tuple of (is_valid, error_message, resolved_path).
        """
        # Check for empty/invalid input
        if not user_path or not isinstance(user_path, str):
            return False, "Path must be a non-empty string", None

        user_path = user_path.strip()

        # Check path length
        if len(user_path) > cls.MAX_PATH_LENGTH:
            return False, f"Path exceeds maximum length ({cls.MAX_PATH_LENGTH})", None

        # Check for traversal patterns BEFORE any path resolution
        for pattern in cls.TRAVERSAL_PATTERNS:
            if pattern.search(user_path):
                return False, "Path traversal attempt detected", None

        # Determine base directory
        if base_dir is None:
            base_dir = os.getcwd()

        try:
            base_path = Path(base_dir).resolve()
        except (OSError, ValueError) as e:
            return False, f"Invalid base directory: {e}", None

        # Resolve the full path
        try:
            # Do not use expanduser to prevent ~ expansion
            full_path = (base_path / user_path).resolve()
        except (OSError, ValueError) as e:
            return False, f"Invalid path: {e}", None

        # CRITICAL: Verify resolved path is under base directory
        try:
            full_path.relative_to(base_path)
        except ValueError:
            return False, "Path traversal detected after resolution", None

        # Verify extension
        if full_path.suffix.lower() not in cls.ALLOWED_EXTENSIONS:
            allowed = ", ".join(sorted(cls.ALLOWED_EXTENSIONS))
            return False, f"Invalid extension. Allowed: {allowed}", None

        # Check file existence and type
        if check_exists:
            if not full_path.exists():
                return False, "File does not exist", None
            if not full_path.is_file():
                return False, "Path does not point to a regular file", None

            # Check file size
            try:
                file_size = full_path.stat().st_size
                if file_size > cls.MAX_FILE_SIZE:
                    return (
                        False,
                        f"File exceeds maximum size ({cls.MAX_FILE_SIZE} bytes)",
                        None,
                    )
            except OSError as e:
                return False, f"Cannot read file stats: {e}", None

        return True, None, full_path

    @classmethod
    def safe_read(
        cls,
        user_path: str,
        base_dir: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Safely read instruction file content.

        Args:
            user_path: The user-provided path.
            base_dir: Base directory for relative paths.

        Returns:
            Tuple of (content, error_message). Content is None on error.
        """
        valid, error, resolved_path = cls.validate(
            user_path, base_dir, check_exists=True
        )

        if not valid:
            return None, error

        try:
            content = resolved_path.read_text(encoding="utf-8")
            return content, None
        except UnicodeDecodeError:
            return None, "File is not valid UTF-8 text"
        except OSError as e:
            return None, f"Failed to read file: {e}"


class SecureEnvironmentValidator:
    """Validates environment variable access for instruction configuration.

    CVE-INST-001 Mitigation: Validates env var values against allowlists.
    """

    # Environment variable rules
    ENV_VAR_RULES: ClassVar[Dict[str, Dict[str, Any]]] = {
        "ROBOTMCP_INSTRUCTIONS": {
            "type": "enum",
            "values": frozenset({"off", "default", "minimal", "strict", "custom"}),
            "default": "default",
        },
        "ROBOTMCP_INSTRUCTIONS_FILE": {
            "type": "path",
            "max_length": 256,
            "validators": ["path_security"],
        },
    }

    @classmethod
    def get_validated_mode(cls) -> ValidInstructionMode:
        """Get validated instruction mode from environment.

        Returns:
            ValidInstructionMode enum value.
        """
        raw_value = os.environ.get("ROBOTMCP_INSTRUCTIONS", "default")
        normalized = raw_value.strip().lower()

        if not ValidInstructionMode.is_valid(normalized):
            logger.warning(
                "Invalid ROBOTMCP_INSTRUCTIONS value '%s', using 'default'",
                raw_value,
            )
            return ValidInstructionMode.DEFAULT

        return ValidInstructionMode.from_string(normalized)

    @classmethod
    def get_validated_file_path(
        cls,
        base_dir: Optional[str] = None,
    ) -> Tuple[Optional[Path], Optional[str]]:
        """Get validated instruction file path from environment.

        Args:
            base_dir: Base directory for path validation.

        Returns:
            Tuple of (resolved_path, error_message). Path is None on error.
        """
        raw_path = os.environ.get("ROBOTMCP_INSTRUCTIONS_FILE")

        if not raw_path:
            return None, "ROBOTMCP_INSTRUCTIONS_FILE not set"

        valid, error, resolved = SecurePathValidator.validate(
            raw_path, base_dir, check_exists=True
        )

        if not valid:
            logger.error("Invalid instruction file path: %s", error)
            return None, error

        return resolved, None


class SecureLogger:
    """Logger that redacts sensitive information.

    CVE-INST-004 Mitigation: Prevents sensitive data exposure in logs.
    """

    SENSITIVE_KEYS: ClassVar[FrozenSet[str]] = frozenset(
        {
            "token",
            "password",
            "secret",
            "key",
            "credential",
            "api_key",
            "apikey",
            "auth",
            "bearer",
            "private",
            "passphrase",
        }
    )

    @classmethod
    def redact_value(cls, key: str, value: Any) -> str:
        """Redact value if key indicates sensitivity.

        Args:
            key: The configuration key name.
            value: The value to potentially redact.

        Returns:
            Redacted or original string representation.
        """
        if value is None:
            return "<not set>"

        key_lower = key.lower()
        for sensitive in cls.SENSITIVE_KEYS:
            if sensitive in key_lower:
                if isinstance(value, str) and len(value) > 4:
                    return f"{value[:2]}***{value[-2:]}"
                return "***"

        return str(value)

    @classmethod
    def log_config(
        cls,
        target_logger: logging.Logger,
        config: Dict[str, Any],
        level: int = logging.INFO,
    ) -> None:
        """Log configuration with sensitive values redacted.

        Args:
            target_logger: The logger to use.
            config: Configuration dictionary.
            level: Logging level.
        """
        redacted = {key: cls.redact_value(key, value) for key, value in config.items()}
        target_logger.log(level, "Configuration: %s", redacted)

    @classmethod
    def redact_instruction_content(cls, content: str, max_preview: int = 50) -> str:
        """Create a safe preview of instruction content for logging.

        Args:
            content: Full instruction content.
            max_preview: Maximum characters to show.

        Returns:
            Truncated content safe for logging.
        """
        if not content:
            return "<empty>"

        # Remove any potential secrets in the preview
        preview = content[:max_preview]

        # Check for and redact any variable-like patterns in preview
        preview = re.sub(r"\$\{[^}]+\}", "[VAR]", preview)
        preview = re.sub(r"[A-Za-z0-9]{32,}", "[HASH]", preview)

        if len(content) > max_preview:
            return f"{preview}... ({len(content)} chars total)"
        return preview


class InstructionSecurityService:
    """High-level service for instruction security validation.

    Combines all security validators into a single interface.
    """

    def __init__(
        self,
        event_publisher: Optional[Callable[[object], None]] = None,
        strict_mode: bool = False,
    ) -> None:
        """Initialize the security service.

        Args:
            event_publisher: Optional callback for security events.
            strict_mode: If True, apply stricter validation rules.
        """
        self._event_publisher = event_publisher
        self._strict_mode = strict_mode
        self._logger = logging.getLogger(__name__)

    def validate_mode(self, mode_str: str) -> Tuple[bool, Optional[ValidInstructionMode], Optional[str]]:
        """Validate instruction mode string.

        Args:
            mode_str: Mode string to validate.

        Returns:
            Tuple of (is_valid, mode_enum, error_message).
        """
        try:
            mode = ValidInstructionMode.from_string(mode_str)
            return True, mode, None
        except InvalidInstructionModeError as e:
            return False, None, str(e)

    def validate_content(self, content: str) -> SecurityValidationResult:
        """Validate instruction content.

        Args:
            content: Content to validate.

        Returns:
            SecurityValidationResult with validation outcome.
        """
        return InstructionContentValidator.validate(content, self._strict_mode)

    def validate_path(
        self,
        path: str,
        base_dir: Optional[str] = None,
    ) -> Tuple[bool, Optional[str], Optional[Path]]:
        """Validate instruction file path.

        Args:
            path: Path to validate.
            base_dir: Base directory for relative paths.

        Returns:
            Tuple of (is_valid, error_message, resolved_path).
        """
        return SecurePathValidator.validate(path, base_dir)

    def sanitize_content(self, content: str) -> SecurityValidationResult:
        """Sanitize instruction content.

        Args:
            content: Content to sanitize.

        Returns:
            SecurityValidationResult with sanitized content.
        """
        return InstructionContentValidator.sanitize(content)

    def load_secure_instructions(
        self,
        path: str,
        base_dir: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Securely load and validate instruction content from file.

        Args:
            path: Path to instruction file.
            base_dir: Base directory for path resolution.

        Returns:
            Tuple of (sanitized_content, error_message).
        """
        # Validate and read path
        content, error = SecurePathValidator.safe_read(path, base_dir)
        if error:
            return None, error

        # Validate content
        validation = self.validate_content(content)
        if not validation.is_safe:
            # Try sanitization
            sanitized = self.sanitize_content(content)
            if sanitized.warnings:
                self._logger.warning(
                    "Content sanitized: %s",
                    ", ".join(sanitized.warnings),
                )
            return sanitized.sanitized_value, None

        return content, None

    def get_config_from_environment(
        self,
        base_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get validated configuration from environment variables.

        Args:
            base_dir: Base directory for file path validation.

        Returns:
            Dictionary with validated configuration.
        """
        mode = SecureEnvironmentValidator.get_validated_mode()

        config = {
            "mode": mode,
            "mode_value": mode.value,
            "content": None,
            "source": f"env:{mode.value}",
            "errors": [],
        }

        if mode == ValidInstructionMode.CUSTOM:
            path, error = SecureEnvironmentValidator.get_validated_file_path(base_dir)
            if error:
                config["errors"].append(error)
                # Fall back to default mode
                config["mode"] = ValidInstructionMode.DEFAULT
                config["mode_value"] = ValidInstructionMode.DEFAULT.value
                config["source"] = "fallback:invalid_path"
            else:
                content, read_error = SecurePathValidator.safe_read(
                    str(path), base_dir
                )
                if read_error:
                    config["errors"].append(read_error)
                    config["mode"] = ValidInstructionMode.DEFAULT
                    config["mode_value"] = ValidInstructionMode.DEFAULT.value
                    config["source"] = "fallback:read_error"
                else:
                    # Validate and sanitize content
                    sanitized = self.sanitize_content(content)
                    config["content"] = sanitized.sanitized_value
                    if sanitized.warnings:
                        config["sanitization_warnings"] = list(sanitized.warnings)

        return config
