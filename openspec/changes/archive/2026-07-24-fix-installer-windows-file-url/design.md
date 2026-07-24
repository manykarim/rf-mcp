## Context

`installer.py::_rfmcp_with_args()` builds the `uv --with…` argument that layers the user's installed
rf-mcp onto a project overlay. For a local/unpublished install it reads the `direct_url.json` recorded
by pip/uv and emits `--with-editable <source-dir>` so the overlay uses that exact source (whose version
may not be on PyPI). The path is currently extracted with a naive slice:

```python
path = url[7:] if url.startswith("file://") else url[5:] if url.startswith("file:") else url
```

On POSIX `file:///home/u/rf-mcp`[7:] → `/home/u/rf-mcp` (correct). On Windows `file:///C:/work/rf-mcp`[7:]
→ `/C:/work/rf-mcp`, which is **not** a valid Windows path, so `Path(path).exists()` is False and the
function falls through to `--with rf-mcp==<version>`. The acid test observed exactly this: a
local-checkout install on Windows produced a version-pinned overlay that only resolved because uv had a
cached 0.34.0 wheel; on a clean machine it would be unresolvable.

## Goals / Non-Goals

**Goals:**
- Emit `--with-editable <source>` for a local/unpublished rf-mcp install on every platform, including
  Windows drive-letter `file://` URLs.
- Lock the behaviour with unit tests (POSIX path, Windows drive-letter conversion, published fallback).
- Confirm the fix on the real Windows machine over SSH and in CI.

**Non-Goals:**
- Changing the published-install path (no `file://` URL → `--with rf-mcp==<version>` stays).
- Changing overlay strategy selection, verification, or any CLI/tool surface.
- Working around the box's uv managed-Python quirk (an environment issue, not rf-mcp's).

## Decisions

**Use stdlib URL→path conversion, not string slicing.** Replace the slice with
`urllib.request.url2pathname(urllib.parse.urlsplit(url).path)`, wrapped in a small helper
`_file_url_to_path(url)`. `url2pathname` is platform-dispatched (nt vs posix), so on Windows
`urlsplit('file:///C:/work/rf-mcp').path` → `/C:/work/rf-mcp` and `url2pathname('/C:/work/rf-mcp')` →
`C:\work\rf-mcp`; on POSIX `file:///home/u/rf-mcp` → `/home/u/rf-mcp`. `unquote` handling comes for free
(percent-encoded spaces etc.). Alternative considered: hand-roll a "strip leading slash before a
drive letter" special case — rejected as brittle and redundant with the stdlib.

**Keep the existing selection condition.** The branch still fires for `editable OR url.startswith("file:")`
and still guards on `Path(path).exists()`; only the path extraction changes. This preserves the
version-pin fallback whenever the source directory is genuinely absent.

**Test the Windows conversion cross-platform without a Windows CI runner.** `url2pathname` is
os-specific, and the Windows source dir won't exist on the Linux runner, so the full Windows happy-path
can't run natively on CI. Split the coverage:
- CI (any OS): (a) POSIX `file://` URL pointing at a real temp dir → `--with-editable <tmp>`; (b) the
  pure converter `_file_url_to_path` against a Windows drive-letter URL using `nturl2path.url2pathname`
  (imported directly) asserts `C:\work\rf-mcp`; (c) no `direct_url.json` → `--with rf-mcp==<ver>`.
- SSH Windows box: re-run the project-aware resolution against the local checkout and assert the resolved
  launch / written `.mcp.json` now contains `--with-editable C:\workspace\rf-mcp-experiments\rf-mcp`
  instead of `--with rf-mcp==0.34.0`.

## Risks / Trade-offs

- [`url2pathname` netloc handling for UNC paths (`file://server/share`)] → Out of scope; pip/uv record
  local installs as `file:///…` with an empty netloc. The helper passes only `urlsplit(url).path`, so a
  stray netloc is ignored rather than mis-joined; documented as a known non-goal.
- [The Windows scenario is validated on the SSH box, not the Linux CI runner] → Mitigated by unit-testing
  the pure converter with `nturl2path` so the drive-letter logic is still exercised in CI; the SSH run is
  the end-to-end confirmation.
- [Behaviour change is limited to local/dev installs] → Intended; the published path is byte-for-byte
  unchanged, keeping the blast radius minimal.

## Migration Plan

No config migration. Existing written entries are unaffected; the change only alters what a *new*
project-aware install writes for a local-source rf-mcp on Windows. Rollback = revert the one-function
diff.
