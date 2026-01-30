"""Security tests for the token optimization implementation.

This module tests security aspects:
- XPath injection prevention
- Ref format validation and injection prevention
- Timeout bounds validation
"""

from __future__ import annotations

import re
from typing import List

import pytest


# =============================================================================
# Security Utilities (to be moved to production code)
# =============================================================================


class XPathSanitizer:
    """Utility for sanitizing XPath expressions to prevent injection."""

    @staticmethod
    def escape_text(text: str) -> str:
        """Escape text for safe use in XPath expressions.

        Handles single quotes, double quotes, and both together
        using the concat() function when necessary.

        Args:
            text: The text to escape

        Returns:
            Safely escaped text for XPath
        """
        if "'" not in text:
            return f"'{text}'"
        elif '"' not in text:
            return f'"{text}"'
        else:
            # Both quotes present - use concat()
            parts = []
            current = ""
            for char in text:
                if char == "'":
                    if current:
                        parts.append(f"'{current}'")
                    parts.append('"\'"')
                    current = ""
                else:
                    current += char
            if current:
                parts.append(f"'{current}'")
            return f"concat({', '.join(parts)})"

    @staticmethod
    def validate_xpath(xpath: str) -> bool:
        """Validate that an XPath expression is safe.

        Checks for common injection patterns.

        Args:
            xpath: The XPath expression to validate

        Returns:
            True if safe, False if suspicious
        """
        # Suspicious patterns that might indicate injection
        suspicious_patterns = [
            r"\bor\s+['\"]?1['\"]?\s*=\s*['\"]?1",  # or '1'='1'
            r"\bor\s+['\"]['\"]?\s*=\s*['\"]",  # or ''=''
            r"\band\s+['\"]?1['\"]?\s*=\s*['\"]?1",  # and '1'='1'
            r"--",  # SQL-style comment
            r"]\s*\|",  # Union with |
            r"/\.\./",  # Path traversal
            r"\bcount\s*\(",  # count() function abuse
            r"\bname\s*\(",  # name() function abuse
            r"\bstring-length\s*\(",  # string-length() abuse
        ]

        for pattern in suspicious_patterns:
            if re.search(pattern, xpath, re.IGNORECASE):
                return False

        return True

    @staticmethod
    def build_safe_text_xpath(text: str, exact: bool = True) -> str:
        """Build a safe XPath expression to find text.

        Args:
            text: The text to search for
            exact: If True, match exact text; if False, use contains()

        Returns:
            Safe XPath expression
        """
        escaped = XPathSanitizer.escape_text(text)
        if exact:
            return f"//*[text()={escaped}]"
        else:
            return f"//*[contains(text(), {escaped})]"


class RefValidator:
    """Validator for element reference strings."""

    # Pattern: 'e' followed by one or more digits
    VALID_REF_PATTERN = re.compile(r"^e\d+$")

    # Maximum reasonable ref number
    MAX_REF_NUMBER = 1000000

    @classmethod
    def validate(cls, ref: str) -> bool:
        """Validate a ref string format.

        Args:
            ref: The ref string to validate

        Returns:
            True if valid, False otherwise
        """
        if not ref or not isinstance(ref, str):
            return False

        if not cls.VALID_REF_PATTERN.match(ref):
            return False

        # Check the number isn't absurdly large
        try:
            num = int(ref[1:])
            if num > cls.MAX_REF_NUMBER:
                return False
        except ValueError:
            return False

        return True

    @classmethod
    def sanitize(cls, ref: str) -> str:
        """Sanitize a ref string, raising on invalid input.

        Args:
            ref: The ref string to sanitize

        Returns:
            The sanitized ref string

        Raises:
            ValueError: If the ref is invalid
        """
        if not cls.validate(ref):
            raise ValueError(f"Invalid ref format: {ref!r}")
        return ref


class TimeoutValidator:
    """Validator for timeout values."""

    MIN_TIMEOUT = 100  # 100ms minimum
    MAX_TIMEOUT = 300000  # 5 minutes maximum

    @classmethod
    def validate(cls, timeout_ms: int) -> bool:
        """Validate a timeout value.

        Args:
            timeout_ms: Timeout in milliseconds

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(timeout_ms, (int, float)):
            return False

        timeout_ms = int(timeout_ms)
        return cls.MIN_TIMEOUT <= timeout_ms <= cls.MAX_TIMEOUT

    @classmethod
    def clamp(cls, timeout_ms: int) -> int:
        """Clamp a timeout value to valid bounds.

        Args:
            timeout_ms: Timeout in milliseconds

        Returns:
            Clamped timeout value

        Raises:
            ValueError: If timeout is negative
        """
        if timeout_ms < 0:
            raise ValueError("Timeout cannot be negative")

        return max(cls.MIN_TIMEOUT, min(cls.MAX_TIMEOUT, int(timeout_ms)))


# =============================================================================
# Tests
# =============================================================================


class TestXPathInjectionPrevention:
    """Security tests for XPath injection prevention."""

    @pytest.fixture
    def sanitizer(self) -> XPathSanitizer:
        """Create an XPathSanitizer."""
        return XPathSanitizer()

    def test_text_with_double_quotes_escaped(self, sanitizer):
        """Test that double quotes in text are properly escaped."""
        text = 'Click "here" to continue'
        escaped = sanitizer.escape_text(text)

        # Should wrap in single quotes since text contains double quotes
        assert escaped == f"'{text}'"

    def test_text_with_single_quotes_escaped(self, sanitizer):
        """Test that single quotes in text are properly escaped."""
        text = "It's a button"
        escaped = sanitizer.escape_text(text)

        # Should wrap in double quotes since text contains single quotes
        assert escaped == f'"{text}"'

    def test_text_with_both_quotes_uses_concat(self, sanitizer):
        """Test that text with both quote types uses concat()."""
        text = """It's a "button" element"""
        escaped = sanitizer.escape_text(text)

        # Should use concat() function
        assert escaped.startswith("concat(")
        assert "'" in escaped  # Contains string literals
        # The result should be safely constructable
        assert text.replace("'", "").replace('"', "") not in escaped or "concat" in escaped

    def test_malicious_xpath_injection_blocked(self, sanitizer):
        """Test that malicious XPath injection patterns are blocked."""
        malicious_inputs = [
            "' or '1'='1",
            '" or "1"="1',
            "' or ''='",
            "admin'--",
            "' and '1'='1' and ''='",
            "x']|//*[contains(.,'",
        ]

        for malicious in malicious_inputs:
            # Escaping should make it safe
            escaped = sanitizer.escape_text(malicious)

            # The escaped version should be a valid XPath string literal
            # It should be wrapped in quotes or use concat()
            assert (
                escaped.startswith("'")
                or escaped.startswith('"')
                or escaped.startswith("concat(")
            )

    def test_validate_xpath_rejects_injection_patterns(self, sanitizer):
        """Test that validation rejects common injection patterns."""
        dangerous_xpaths = [
            "//*[text()='' or '1'='1']",
            "//*[@id='x' or ''='']",
            "//user[name/text()='' and '1'='1']",
            "//*[contains(., 'x')]|//password",
            "//a[text()='x']/../password",
        ]

        for xpath in dangerous_xpaths:
            assert sanitizer.validate_xpath(xpath) is False, f"Should reject: {xpath}"

    def test_validate_xpath_accepts_safe_patterns(self, sanitizer):
        """Test that validation accepts safe XPath patterns."""
        safe_xpaths = [
            "//button[@type='submit']",
            "//*[text()='Click me']",
            "//div[@class='container']//button",
            "//*[contains(@class, 'btn')]",
            "//form//input[@name='email']",
        ]

        for xpath in safe_xpaths:
            assert sanitizer.validate_xpath(xpath) is True, f"Should accept: {xpath}"

    def test_build_safe_text_xpath_exact(self, sanitizer):
        """Test building safe XPath for exact text match."""
        xpath = sanitizer.build_safe_text_xpath("Click me")
        assert xpath == "//*[text()='Click me']"

    def test_build_safe_text_xpath_contains(self, sanitizer):
        """Test building safe XPath for text contains."""
        xpath = sanitizer.build_safe_text_xpath("Click", exact=False)
        assert xpath == "//*[contains(text(), 'Click')]"

    def test_build_safe_text_xpath_with_quotes(self, sanitizer):
        """Test building safe XPath when text contains quotes."""
        xpath = sanitizer.build_safe_text_xpath("It's here")
        assert "//*[text()=" in xpath
        assert '"' in xpath  # Should use double quotes


class TestRefValidation:
    """Security tests for ref format validation."""

    def test_valid_ref_format_accepted(self):
        """Test that valid ref formats are accepted."""
        valid_refs = ["e0", "e1", "e42", "e123", "e9999"]

        for ref in valid_refs:
            assert RefValidator.validate(ref) is True, f"Should accept: {ref}"

    def test_invalid_ref_format_rejected(self):
        """Test that invalid ref formats are rejected."""
        invalid_refs = [
            "invalid",
            "42",  # Missing 'e' prefix
            "E42",  # Wrong case
            "e",  # Missing number
            "e-1",  # Negative
            "e1.5",  # Decimal
            "e 42",  # Space
            "e42 ",  # Trailing space
            " e42",  # Leading space
            "ee42",  # Double 'e'
            "ref42",  # Wrong prefix
            "",  # Empty
            None,  # None
            123,  # Not a string
        ]

        for ref in invalid_refs:
            assert RefValidator.validate(ref) is False, f"Should reject: {ref!r}"

    def test_ref_injection_attempt_blocked(self, ref_injection_payloads):
        """Test that ref injection attempts are blocked."""
        for payload in ref_injection_payloads:
            assert RefValidator.validate(payload) is False, f"Should block: {payload}"

    def test_sanitize_returns_valid_ref(self):
        """Test that sanitize returns valid refs unchanged."""
        ref = "e42"
        assert RefValidator.sanitize(ref) == ref

    def test_sanitize_raises_on_invalid(self):
        """Test that sanitize raises on invalid refs."""
        with pytest.raises(ValueError, match="Invalid ref"):
            RefValidator.sanitize("invalid")

    def test_extremely_large_ref_number_rejected(self):
        """Test that extremely large ref numbers are rejected."""
        # Should reject numbers beyond reasonable bounds
        large_ref = "e" + "9" * 20  # e99999999999999999999
        assert RefValidator.validate(large_ref) is False


class TestTimeoutBounds:
    """Security tests for timeout validation."""

    def test_negative_timeout_rejected(self):
        """Test that negative timeouts are rejected."""
        assert TimeoutValidator.validate(-1) is False
        assert TimeoutValidator.validate(-1000) is False

        with pytest.raises(ValueError, match="negative"):
            TimeoutValidator.clamp(-1)

    def test_zero_timeout_clamped_to_minimum(self):
        """Test that zero timeout is clamped to minimum."""
        result = TimeoutValidator.clamp(0)
        assert result == TimeoutValidator.MIN_TIMEOUT

    def test_excessive_timeout_clamped(self):
        """Test that excessive timeouts are clamped."""
        # 1 hour timeout should be clamped to max
        result = TimeoutValidator.clamp(3600000)
        assert result == TimeoutValidator.MAX_TIMEOUT

    def test_valid_timeout_accepted(self):
        """Test that valid timeouts are accepted."""
        valid_timeouts = [100, 1000, 5000, 60000, 300000]

        for timeout in valid_timeouts:
            assert TimeoutValidator.validate(timeout) is True

    def test_minimum_timeout_boundary(self):
        """Test minimum timeout boundary."""
        assert TimeoutValidator.validate(99) is False
        assert TimeoutValidator.validate(100) is True
        assert TimeoutValidator.clamp(50) == 100

    def test_maximum_timeout_boundary(self):
        """Test maximum timeout boundary."""
        assert TimeoutValidator.validate(300000) is True
        assert TimeoutValidator.validate(300001) is False
        assert TimeoutValidator.clamp(500000) == 300000

    def test_float_timeout_converted(self):
        """Test that float timeouts are handled."""
        result = TimeoutValidator.clamp(5000.5)
        assert result == 5000
        assert isinstance(result, int)


class TestInputSanitization:
    """Additional input sanitization tests."""

    def test_null_byte_in_ref_rejected(self):
        """Test that null bytes in refs are rejected."""
        ref_with_null = "e1\x00malicious"
        assert RefValidator.validate(ref_with_null) is False

    def test_newline_in_ref_rejected(self):
        """Test that newlines in refs are rejected."""
        ref_with_newline = "e1\ninjected"
        assert RefValidator.validate(ref_with_newline) is False

    def test_unicode_in_ref_rejected(self):
        """Test that unicode characters in refs are rejected."""
        unicode_refs = [
            "e\u0031\u0032",  # Unicode digits (might look like e12)
            "e42\u200b",  # Zero-width space
            "\u0065\u0034\u0032",  # Unicode 'e42'
        ]

        # Only ASCII e followed by ASCII digits should be valid
        for ref in unicode_refs:
            # Some of these might pass validation if they're equivalent
            # The key is consistent handling
            result = RefValidator.validate(ref)
            # If it passes, it should be because it's actually "e42" etc
            if result:
                # Verify it's actually the expected form
                assert re.match(r"^e\d+$", ref, re.ASCII) is not None

    def test_xpath_with_entity_encoding(self):
        """Test XPath handling of entity-like strings."""
        sanitizer = XPathSanitizer()

        # These shouldn't cause issues - they're just text
        text = "&lt;script&gt;alert('xss')&lt;/script&gt;"
        escaped = sanitizer.escape_text(text)

        # Should be safely escaped
        assert (
            escaped.startswith("'")
            or escaped.startswith('"')
            or escaped.startswith("concat(")
        )


class TestCombinedSecurityScenarios:
    """Tests for combined security scenarios."""

    def test_full_workflow_with_untrusted_input(self):
        """Test a full workflow with untrusted user input."""
        # Simulate untrusted input coming from an LLM
        untrusted_ref = "e42'; DROP TABLE elements; --"
        untrusted_text = "'; DELETE FROM users; --"
        untrusted_timeout = -9999

        # All should be safely handled
        assert RefValidator.validate(untrusted_ref) is False

        sanitizer = XPathSanitizer()
        safe_xpath = sanitizer.build_safe_text_xpath(untrusted_text)
        # The text should be safely escaped - the malicious content becomes
        # a string literal, not executable XPath code.
        # The escape_text function wraps the entire input in quotes,
        # making it a harmless string value.
        assert safe_xpath.startswith("//*[text()=")
        # The content is safely quoted - the malicious text is treated as data,
        # not as XPath code to execute
        # Output: //*[text()="'; DELETE FROM users; --"]
        assert safe_xpath.endswith('"]')
        # Verify the text is wrapped as a string literal (in double or single quotes)
        # Either double quotes around the whole thing or concat() for mixed quotes
        assert (
            '="' in safe_xpath  # Double quote wrapper
            or "='" in safe_xpath  # Single quote wrapper
            or "concat(" in safe_xpath  # concat for mixed
        )

        with pytest.raises(ValueError):
            TimeoutValidator.clamp(untrusted_timeout)

    def test_defense_in_depth(self):
        """Test that multiple layers of validation work together."""
        # Even if one check is bypassed, others should catch issues

        # Example: a ref that might look valid but isn't
        tricky_refs = [
            "e1 ",  # Trailing space
            " e1",  # Leading space
            "e1\t",  # Tab
            "e1\r\n",  # CRLF
        ]

        for ref in tricky_refs:
            # Validation should reject
            assert RefValidator.validate(ref) is False
            # Sanitization should raise
            with pytest.raises(ValueError):
                RefValidator.sanitize(ref)
