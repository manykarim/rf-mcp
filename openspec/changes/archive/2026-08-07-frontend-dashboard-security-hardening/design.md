## Context

The dashboard is an optional Django app embedded in the rf-mcp process. It is read-only in intent
(observe sessions), but ships an unauthenticated exec route, DEBUG-on defaults, `ALLOWED_HOSTS=["*"]`,
`innerHTML` interpolation of page-under-test data, and CDN scripts with no SRI/CSP. The Docker path
binds it on `0.0.0.0:8001`. This change fixes the security posture only; live-runtime and rendering
correctness are separate changes.

## Goals / Non-Goals

**Goals:** safe-by-default config; remove the unauth exec/mutation surface; escape untrusted output;
constrain client assets (CSP + integrity/vendoring). Verifiable locally.

**Non-Goals:** building an auth system (the dashboard becomes read-only + safe defaults; a future
mutating feature would add auth); the SSE/live-feed rewrite; general rendering-correctness fixes; a
full asset-vendoring pipeline (SRI or local copy is sufficient).

## Decisions

**D1 — Remove the exec route, don't gate it.** `POST /api/sessions/<id>/execute/` has zero UI consumers
and is an unauthenticated RCE. Delete the URL entry; leave `api.execute_keyword`/`bridge.execute_keyword`
in place (unrouted) so no other caller breaks and a future authenticated re-introduction is easy.

**D2 — Debug off by default + non-loopback guard.** Flip `FrontendConfig.debug` default and the
`_env_bool` fallback to `False`; keep `--frontend-debug` as the explicit opt-in. Add a guard (in the
frontend controller/config resolution) that raises if `debug and host not in loopback`.

**D3 — Scope ALLOWED_HOSTS.** Derive from `[config.host, "localhost", "127.0.0.1", "[::1]"]` plus an
`ROBOTMCP_FRONTEND_ALLOWED_HOSTS` comma env override. `0.0.0.0` binds still need the real hostname/IP
via the override — documented — rather than silently allowing `*`.

**D4 — Escape via DOM construction.** Add a small `iconLabel(iconName, text)` helper that builds an
`<i data-feather>` + a text node, and route every `innerHTML` that interpolates dynamic data through it
(or through `textContent`). Static feather-only `innerHTML` stays. Verify by injecting markup in a
variable and asserting no script node / literal text.

**D5 — CSP + SRI (or vendor).** Add a CSP (`default-src 'self'`; allow the specific CDN hosts for
script/style, or 'self' if vendored; `script-src` without `'unsafe-inline'` where feasible). Pin the
CDN `<script>`/`<link>` with `integrity` + `crossorigin`, and pin feather to an exact version (not
`latest`). Minimal, non-breaking: keep the CDNs but add integrity + a CSP that allows them.

## Risks / Trade-offs

- **[CSP breaks inline bootstrap]** `layout.html` has an inline `<script>` setting `window.ROBOTMCP_FRONTEND`.
  Move it to a `data-` attribute / external file or allow a nonce, so `script-src` needn't include
  `'unsafe-inline'`. Verify the page still boots.
- **[0.0.0.0 + scoped ALLOWED_HOSTS needs the real host]** documented; the env override covers it. Safer
  than `*`.
- **[Removing the route]** confirmed no SPA consumer (`grep execute app.js` → 0); low risk.
