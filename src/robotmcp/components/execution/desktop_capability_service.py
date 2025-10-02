"""Desktop capability service for PlatynUI sessions.

This service performs lightweight validation of the host environment before
running PlatynUI keywords. It checks for core Python dependencies, infers
supported backends by platform, and exposes diagnostics that can be surfaced to
MCP tools.
"""

from __future__ import annotations

import importlib.util
import os
import platform
from dataclasses import dataclass
from typing import Dict, List

from robotmcp.models.session_models import ExecutionSession, PlatformType, SessionType


@dataclass
class DesktopEnvironmentDiagnostics:
    """Diagnostics payload summarising environment readiness."""

    success: bool
    issues: List[str]
    warnings: List[str]
    detected_backends: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "success": self.success,
            "issues": self.issues,
            "warnings": self.warnings,
            "detected_backends": self.detected_backends,
        }


class DesktopAutomationService:
    """Service that validates prerequisites for desktop automation."""

    _PLATFORM_BACKEND_MAP: Dict[str, List[str]] = {
        "Windows": ["win32"],
        "Darwin": ["macos"],
        "Linux": ["x11", "at-spi2"],
    }

    def __init__(self) -> None:
        self.host_system = platform.system()

    def detect_supported_backends(self) -> List[str]:
        """Infer PlatynUI backends supported by the host OS."""
        return self._PLATFORM_BACKEND_MAP.get(self.host_system, [])

    def validate_environment(self, session: ExecutionSession | None = None) -> DesktopEnvironmentDiagnostics:
        """Return diagnostics about the current desktop automation environment."""
        issues: List[str] = []
        warnings: List[str] = []

        if not self._module_available("PlatynUI"):
            issues.append(
                "PlatynUI library not importable. Install the desktop dependency group (uv sync --group desktop)."
            )
        if not self._module_available("pythonnet") and not self._module_available("clr"):
            warnings.append(
                "pythonnet is missing; .NET interop will fail until it is installed."
            )
        if not self._module_available("platynui_spy"):
            warnings.append("platynui_spy module not found; Spy tooling will be unavailable.")

        backends = self.detect_supported_backends()
        if not backends:
            warnings.append(f"Unsupported desktop platform '{self.host_system}'.")

        if self.host_system == "Linux" and "DISPLAY" not in os.environ:
            warnings.append("DISPLAY environment variable not set; X11 backend may be unavailable.")

        success = not issues

        # Mark session as desktop if provided and diagnostics succeed partially.
        if session and success and not session.is_desktop_session():
            session.platform_type = PlatformType.DESKTOP
            if session.session_type == SessionType.UNKNOWN:
                session.session_type = SessionType.DESKTOP_TESTING

        return DesktopEnvironmentDiagnostics(
            success=success,
            issues=issues,
            warnings=warnings,
            detected_backends=backends,
        )

    @staticmethod
    def _module_available(name: str) -> bool:
        """Return True if the given module can be imported."""
        return importlib.util.find_spec(name) is not None
