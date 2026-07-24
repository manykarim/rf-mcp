# rf-mcp 0.35.0

A single-fix point release. It repairs one corner of the project-aware installer that shipped in
0.34.0: installing rf-mcp **from a local source checkout on Windows** and then wiring it into a
project. If you install rf-mcp from PyPI, nothing here changes for you.

---

## Fixes

- **Local-source install on Windows now writes a launch command that resolves.** When rf-mcp itself
  was installed from a local, unpublished checkout (a `file://` source), `robotmcp install` is meant
  to wire the project overlay to that checkout with `--with-editable <checkout>`. On Windows it
  silently didn't: the `file://` URL was parsed with a naive slice that turned `file:///C:/checkout`
  into `/C:/checkout` — not a valid Windows path — so the existence check failed and the command
  degraded to `--with rf-mcp==<version>`, a pin uv could not resolve for an unpublished build. rf-mcp
  now converts the `file://` URL with a proper cross-platform routine, so a Windows drive-letter URL
  resolves to `C:\checkout` (POSIX URLs resolve as before). Validated on real Windows: the written
  `.mcp.json` overlay flips to `--with-editable`, and the install-time verification gate still passes.

**Scope, honestly.** This only affects a local/unpublished checkout **on Windows**. A normal PyPI
install (`uv tool install rf-mcp`) never had a `file://` source — it already emitted the correct
`--with rf-mcp==<version>` pin and is unchanged, as are local installs on Linux and macOS, where the
old parsing already worked.

---

## Getting it

0.35.0 is published as a **GitHub release** — grab the wheel attached to the release and install it,
for example:

```
uv pip install ./rf_mcp-0.35.0-py3-none-any.whl
```

Upgrading from 0.34.x needs no configuration changes. No MCP tool names, parameters or return shapes
changed.
