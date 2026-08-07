"""Security middleware for the optional Django frontend (change: frontend-dashboard-security-hardening).

Emits a Content-Security-Policy on every response. `script-src` deliberately omits
`'unsafe-inline'` (the page's former inline bootstrap script was removed; the base path is now read
from a `<body data-base-path>` attribute), so injected markup cannot execute script even if it reached
the DOM. The pinned CDN hosts are allowed for the icon/syntax-highlighting assets; vendoring them would
let this tighten to `'self'` only.
"""

from __future__ import annotations

_CDN = "https://unpkg.com https://cdn.jsdelivr.net"

CSP = "; ".join(
    [
        "default-src 'self'",
        f"script-src 'self' {_CDN}",
        f"style-src 'self' {_CDN} 'unsafe-inline'",
        f"font-src 'self' {_CDN} data:",
        "img-src 'self' data:",
        "connect-src 'self'",
        "base-uri 'self'",
        "frame-ancestors 'none'",
        "object-src 'none'",
    ]
)


class ContentSecurityPolicyMiddleware:
    """Attach a Content-Security-Policy header to every response."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        response.setdefault("Content-Security-Policy", CSP)
        response.setdefault("X-Content-Type-Options", "nosniff")
        response.setdefault("Referrer-Policy", "same-origin")
        return response
