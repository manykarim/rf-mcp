## 1. Fix the file:// URL parse

- [x] 1.1 Add a `_file_url_to_path(url)` helper in `installer.py` using `urllib.parse.urlsplit` + `urllib.request.url2pathname` (platform-correct; handles percent-encoding).
- [x] 1.2 In `_rfmcp_with_args()` replace the `url[7:]`/`url[5:]` slice with `_file_url_to_path(url)`, keeping the `editable or url.startswith("file:")` selection and the `Path(path).exists()` guard unchanged.

## 2. Unit tests (CI coverage)

- [x] 2.1 Test: a `file://` direct URL pointing at a real temp dir → `_rfmcp_with_args()` returns `["--with-editable", <tmp>]` (POSIX happy path, runs natively on CI).
- [x] 2.2 Test: `_file_url_to_path("file:///C:/work/rf-mcp")` converts to `C:\work\rf-mcp` by asserting against `nturl2path.url2pathname` (drive-letter logic exercised cross-platform on the Linux runner).
- [x] 2.3 Test: no `direct_url.json` (published install) → `_rfmcp_with_args()` returns `["--with", "rf-mcp==<ver>"]` (fallback unchanged); and a `file://` URL whose dir does NOT exist falls back to the version pin.
- [x] 2.4 Run `uv run pytest tests/unit/test_installer_project_aware.py -q` and confirm green.

## 3. CI validation (full suite)

- [x] 3.1 Run the full unit suite (`uv run pytest tests/unit -q`) and confirm no regressions.

## 4. Windows validation over SSH (end-to-end)

- [x] 4.1 Sync the fix to the Windows box (`git pull` / copy the checkout) and reinstall the tool from the local checkout with `--python C:\Python\python.exe`.
- [x] 4.2 Re-run `robotmcp doctor --project-dir C:\workspace\rf-mcp-experiments\proj-acid` and `robotmcp install -C … --dry-run`; confirm the resolved launch now shows `--with-editable C:\workspace\rf-mcp-experiments\rf-mcp` (not `--with rf-mcp==0.34.0`).
- [x] 4.3 Run the real `robotmcp install -C …` and confirm the written `.mcp.json` uses the `--with-editable` overlay and the verify gate still passes (JSONLibrary reachable).

## 5. Wrap-up

- [x] 5.1 `openspec validate fix-installer-windows-file-url --strict` passes.
- [x] 5.2 Record the acid-test-fix result in the `windows-acid-test-2026-07-24` memory (bug → fixed + validated).
