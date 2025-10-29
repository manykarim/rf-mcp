"""Expose the Django ASGI application for external runners."""

from __future__ import annotations

from .config import FrontendConfig
from .django_app import get_django_application

# Default ASGI application used when uvicorn imports `robotmcp.frontend.asgi:application`.
application = get_django_application(FrontendConfig(enabled=True))
