## Why

The Windows acid test proved the project-aware installer works end-to-end on Windows, but surfaced one
real cross-platform defect: when rf-mcp itself was installed from a local/unpublished source (a `file://`
direct URL), the uv-overlay resolver is supposed to layer *that same source* into the overlay via
`--with-editable <src>`. On Windows that silently degrades to a version pin (`--with rf-mcp==<ver>`),
because the `file://` URL is parsed with a naive `url[7:]` slice that yields `/C:/workspace/...` — not a
valid Windows path — so the source-directory existence check fails. The written config then only works if
that exact version happens to be resolvable from PyPI or a warm uv cache; on a clean Windows machine
installing rf-mcp from a local checkout, the overlay is unresolvable.

## What Changes

- Fix `installer.py::_rfmcp_with_args()` to convert a `file://` direct URL to a filesystem path with a
  correct, cross-platform routine (stdlib `urllib` URL→path conversion) instead of the `url[7:]`/`url[5:]`
  slice, so Windows drive-letter URLs (`file:///C:/...`) resolve to `C:\...` and the intended
  `--with-editable <src>` is emitted on every platform.
- Add unit tests covering POSIX and Windows-style `file://` direct URLs (and the published/no-direct-url
  fallback), so CI locks the behaviour in.
- Validate the fix on the real Windows machine over SSH by re-running the project-aware resolution and
  confirming the resolved launch now uses `--with-editable <local-checkout>` rather than the version pin.

## Capabilities

### New Capabilities
<!-- none -->

### Modified Capabilities
- `tool-install-onboarding`: the project-aware uv overlay must reference the *same* rf-mcp the user
  installed — including a local/unpublished source install — with correct cross-platform `file://` URL
  parsing, rather than silently falling back to a version pin on Windows.

## Impact

- Code: `src/robotmcp/onboarding/installer.py` (`_rfmcp_with_args`, ~2 lines of parsing logic).
- Tests: `tests/unit/test_installer_project_aware.py` (new cases for `file://` URL → `--with-editable`).
- No change to MCP tools, CLI flags, or the published-install path (a normal `uv tool install rf-mcp`
  from PyPI has no `file://` direct URL and already emits `--with rf-mcp==<ver>`).
- Behaviour only differs for a local/dev/unpublished rf-mcp install used to drive a project-aware install,
  which is the exact case the acid test exercised on Windows.
