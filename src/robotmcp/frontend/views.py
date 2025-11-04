from __future__ import annotations

from django.http import HttpRequest, HttpResponse
from django.shortcuts import render
from django.conf import settings
from django.views.decorators.csrf import ensure_csrf_cookie


@ensure_csrf_cookie
def index(request: HttpRequest) -> HttpResponse:
    """Render the landing page for the frontend."""

    base_path = getattr(settings, "ROBOTMCP_FRONTEND_BASE_PATH", "/")
    return render(
        request,
        "frontend/index.html",
        {
            "base_path": base_path,
        },
    )
