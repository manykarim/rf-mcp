"""Utilities to configure and expose the Django ASGI application."""

from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import Any, Dict

from .config import FrontendConfig

_ASGI_APPLICATION = None


def _build_default_settings(config: FrontendConfig) -> Dict[str, Any]:
    base_dir = Path(__file__).resolve().parent
    secret_key = os.environ.get("ROBOTMCP_FRONTEND_SECRET_KEY")
    if not secret_key:
        # Deterministic enough for dev usage; overrides recommended for production.
        secret_key = "robotmcp-frontend-" + secrets.token_hex(16)

    static_url = (config.base_path or "/").rstrip("/") + "/static/"

    return {
        "DEBUG": config.debug,
        "SECRET_KEY": secret_key,
        "ALLOWED_HOSTS": ["*"],
        "INSTALLED_APPS": [
            "django.contrib.admin",
            "django.contrib.auth",
            "django.contrib.contenttypes",
            "django.contrib.sessions",
            "django.contrib.messages",
            "django.contrib.staticfiles",
            "robotmcp.frontend.apps.RobotMCPFrontendConfig",
        ],
        "MIDDLEWARE": [
            "django.middleware.security.SecurityMiddleware",
            "django.contrib.sessions.middleware.SessionMiddleware",
            "django.middleware.common.CommonMiddleware",
            "django.middleware.csrf.CsrfViewMiddleware",
            "django.contrib.auth.middleware.AuthenticationMiddleware",
            "django.contrib.messages.middleware.MessageMiddleware",
            "django.middleware.clickjacking.XFrameOptionsMiddleware",
        ],
        "ROOT_URLCONF": "robotmcp.frontend.urls",
        "TEMPLATES": [
            {
                "BACKEND": "django.template.backends.django.DjangoTemplates",
                "DIRS": [base_dir / "templates"],
                "APP_DIRS": True,
                "OPTIONS": {
                    "context_processors": [
                        "django.template.context_processors.debug",
                        "django.template.context_processors.request",
                        "django.contrib.auth.context_processors.auth",
                        "django.contrib.messages.context_processors.messages",
                    ]
                },
            }
        ],
        "ASGI_APPLICATION": "robotmcp.frontend.asgi.application",
        "DATABASES": {
            "default": {
                "ENGINE": "django.db.backends.sqlite3",
                "NAME": os.environ.get("ROBOTMCP_FRONTEND_DB", ":memory:"),
            }
        },
        "STATIC_URL": static_url,
        "STATICFILES_DIRS": [base_dir / "static"],
        "STATIC_ROOT": os.environ.get("ROBOTMCP_FRONTEND_STATIC_ROOT", str(base_dir / "static")),
        "DEFAULT_AUTO_FIELD": "django.db.models.AutoField",
        "USE_TZ": True,
        "TIME_ZONE": os.environ.get("ROBOTMCP_FRONTEND_TIME_ZONE", "UTC"),
        "LANGUAGE_CODE": "en-us",
        "CSRF_TRUSTED_ORIGINS": [
            f"http://{config.host}:{config.port}",
            f"https://{config.host}:{config.port}",
        ],
        "ROBOTMCP_FRONTEND_BASE_PATH": config.base_path,
    }


def get_django_application(config: FrontendConfig):
    """Return the configured Django ASGI application."""

    global _ASGI_APPLICATION
    if _ASGI_APPLICATION is not None:
        return _ASGI_APPLICATION

    import django
    from django.conf import settings

    if not settings.configured:
        base_settings = _build_default_settings(config)
        settings.configure(**base_settings)

    # Store config for later reuse (e.g., views)
    settings.ROBOTMCP_FRONTEND_CONFIG = config  # type: ignore[attr-defined]

    django.setup()

    from django.core.asgi import get_asgi_application
    from django.contrib.staticfiles.handlers import ASGIStaticFilesHandler

    base_app = get_asgi_application()
    _ASGI_APPLICATION = ASGIStaticFilesHandler(base_app)

    return _ASGI_APPLICATION
