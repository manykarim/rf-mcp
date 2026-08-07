## Why

The optional Django dashboard (`src/robotmcp/frontend/`) ships **critical, unauthenticated security
holes** in the shipped Docker path (validated 2026-08-07, see
`docs/frontend_dashboard_critical_validation.md`):

- **Unauthenticated arbitrary Robot Framework keyword execution (RCE).** `POST /api/sessions/<id>/execute/`
  runs any keyword (`Evaluate` = arbitrary Python) with no auth, and the SPA never calls it — pure
  attack surface. `docker/Dockerfile` + `supervisord.conf` bind it on `0.0.0.0:8001`.
- **`DEBUG=True` by default in production**, with `ALLOWED_HOSTS=["*"]` and no CSP. Any unhandled
  exception returns Django's technical 500 page — SECRET_KEY, the settings dict, **all environment
  variables (including model API keys)**, and source — to an unauthenticated remote client.
- **Stored XSS** — variable names/values scraped from the page-under-test are written via `innerHTML`;
  same-origin as the (unauthenticated) exec endpoint with a readable csrftoken = an RCE chain.
- CDN scripts loaded with **no SRI**, an unpinned `latest`, no offline fallback, and no CSP.

These are the four security items from the validation. They are the highest priority because they
convert a read-only observability UI into a network-reachable RCE and secret-exfiltration surface.

## What Changes

- **Remove the unauthenticated execute-keyword route** (`urls.py`) — the SPA has no consumer for it.
  (The bridge/api handlers may remain unused; the *route* is deleted so it is unreachable.)
- **Default `DEBUG=False`**; require an explicit opt-in. Scope `ALLOWED_HOSTS` to the configured host
  (+ loopback + env override) instead of `["*"]`. Add a **startup guard** that refuses to run with
  `DEBUG=True` on a non-loopback bind.
- **Escape untrusted data** — replace interpolating `innerHTML` on session/variable/step/event data
  with safe DOM construction; keep only static-string `innerHTML` (feather icon names).
- **Add a Content-Security-Policy** and pin the CDN assets with SRI (or vendor them locally so the
  dashboard has no third-party runtime dependency).

## Capabilities

### New Capabilities

- `frontend-dashboard`: the optional Django observability dashboard for rf-mcp sessions — this change
  establishes its **security posture** requirements (safe-by-default config, no unauthenticated
  mutation/exec, output escaping, CSP/asset integrity).

## Impact

- `src/robotmcp/frontend/config.py` (debug default), `django_app.py` (ALLOWED_HOSTS, CSP, debug guard),
  `urls.py` (remove exec route), `static/frontend/app.js` (escape `innerHTML`),
  `templates/frontend/layout.html` (CSP + SRI/vendored assets), `server.py` (non-loopback+DEBUG guard).
- No change to the MCP server surface or to sessions; the dashboard stays read-only.
- Docker deployment becomes safe-by-default (DEBUG off, host-scoped, no exec endpoint).
