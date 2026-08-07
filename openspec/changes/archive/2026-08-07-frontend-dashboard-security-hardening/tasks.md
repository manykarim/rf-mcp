## 1. Remove the unauthenticated exec route

- [x] 1.1 Delete the `api/sessions/<id>/execute/` path from `frontend/urls.py`. Confirm no SPA consumer
  (`grep -c execute static/frontend/app.js` → 0). Verify a POST to that URL now 404s.

## 2. Safe-by-default config

- [x] 2.1 `config.py`: default `FrontendConfig.debug=False` and the `_env_bool("ROBOTMCP_FRONTEND_DEBUG")`
  fallback to `False`; keep `--frontend-debug` as the explicit opt-in.
- [x] 2.2 `django_app.py`: set `ALLOWED_HOSTS` from `[config.host, localhost, 127.0.0.1, [::1]]` +
  `ROBOTMCP_FRONTEND_ALLOWED_HOSTS` env override, not `["*"]`. Keep `CSRF_TRUSTED_ORIGINS` consistent.
- [x] 2.3 Add a startup guard (config resolution or controller.start) that raises a clear error when
  `debug` is true and the host is not loopback.

## 3. Escape untrusted output (XSS)

- [x] 3.1 Add an `iconLabel(iconName, text)` DOM-construction helper in `app.js`; route every
  `innerHTML` that interpolates dynamic session/variable/step/event data through it or `textContent`.
  Leave static feather-only `innerHTML` as-is.
- [x] 3.2 Verify: seed a variable whose name/value contains `<img src=x onerror=...>`/`<script>` and
  confirm (via the Playwright probe) it renders as literal text and no script executes.

## 4. Constrain client assets (CSP + integrity)

- [x] 4.1 `layout.html`: pin the feather-icons + prismjs CDN tags with `integrity` + `crossorigin` and
  an exact feather version (not `latest`); move the inline bootstrap `<script>` off inline (data-attr or
  nonce).
- [x] 4.2 Emit a Content-Security-Policy (response header in `django_app.py`/middleware, or a `<meta>`),
  restricting `script-src`/`style-src` to `'self'` + the pinned CDN hosts, no `'unsafe-inline'` script.

## 5. Verify + wrap-up

- [x] 5.1 Re-run the dashboard (devserver) + the Playwright probe: page boots, icons + syntax
  highlighting still render, exec route 404s, CSP header present, injected markup escaped, DEBUG off.
- [x] 5.2 `uv run pytest` for any frontend tests still green; `openspec validate
  frontend-dashboard-security-hardening --strict` passes.
