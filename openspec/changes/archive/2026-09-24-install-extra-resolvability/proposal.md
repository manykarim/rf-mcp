# Proposal: install-extra-resolvability

## Why

**`uv tool install "rf-mcp[all]"` — the first of the three commands in the README's
Quick Start (`README.md:39`) — fails for every user, on every operating system.**

Measured on 2026-09-20 against the published rf-mcp 0.35.0 (full evidence:
`experiments/platynui_pin_resolution_evidence.md`):

```
$ uv tool install "rf-mcp[all]==0.35.0"
  × No solution found when resolving dependencies:
  ╰─▶ Because there is no version of robotframework-platynui==0.13.0.dev2 and
      rf-mcp[all]==0.35.0 depends on robotframework-platynui==0.13.0.dev2, we
      can conclude that rf-mcp[all]==0.35.0 cannot be used.
      hint: ... requested with a pre-release marker ... but pre-releases weren't
      enabled (try: `--prerelease=allow`)
```

24 of 24 resolutions of `[desktop]` and `[all]` fail — linux x86_64/aarch64, macOS
x86_64/arm64, Windows, musl, on Python 3.12 and 3.13. Bare `rf-mcp` always resolves.

### Root cause — it is NOT a missing version

`platynui-cli==0.13.0.dev2` **exists on PyPI** (5 wheels, not yanked), as do
`robotframework-PlatynUI==0.13.0.dev2` and `platynui-native==0.13.0.dev2`. All PlatynUI
dist names rf-mcp declares are real. uv's phrase *"there is no version of"* means
"not selectable under the current pre-release policy", not "does not exist".

uv's default policy is `if-necessary-or-explicit`, and **"explicit" only counts for
first-party requirements**. The same pin resolves or fails depending purely on who asks:

```
uv pip install -e ".[desktop]"     (CI ci.yml:299, maintainer)  -> first-party -> OK
uv sync --all-extras / uv lock     (CI everywhere)              -> first-party -> OK
uv tool install "rf-mcp[desktop]"  (every user)                 -> transitive  -> FAIL
```

Both halves were run on the same machine, same uv, same pin. This is why **no CI job can
see the bug**: every job installs rf-mcp as a local path or from the lock file, never from
PyPI the way a user does.

`pyproject.toml:44` and `:75` assert the opposite — *"resolvers accept it WITHOUT a global
--pre"* — which is true for pip and false for uv. `docs/RELEASE_NOTES_v0.34.0.md:160`
repeats the false claim (*"no `--pre` needed"*).

### Loosening the pin does not fix it

Four fixture wheels differing only in the specifier, resolved transitively:

| specifier | uv default | `--prerelease=allow` |
|---|---|---|
| `==0.13.0.dev2` (current) | FAIL | OK |
| `>=0.13.0.dev0` | **FAIL** | OK |
| `>=0.13.0.dev2` | **FAIL** | OK |
| `>=0.13.0.dev0,<0.14` | **FAIL** | OK |

The specifier shape is irrelevant; only the flag changes the outcome. A pin bump is not a fix.

### Affected installers: everything uv-backed — including pipx

| installer | backend | `rf-mcp[all]` |
|---|---|---|
| `uv pip/tool/add`, `uvx` | uv | **FAIL** |
| `pipx install` (1.17.4) | **uv** | **FAIL** |
| `pip`, `poetry`, `pdm`, conda(+pip) | own | OK |

### Second, independent defect: wheel platform coverage

`platynui-cli` / `platynui-native` publish **only** `manylinux_2_34_{x86_64,aarch64}`,
`macosx_11_0_arm64`, `win_{amd64,arm64}`, and **no sdist**. Coverage is identical from
`dev1` through `dev117`. So users on **macOS Intel, Alpine/musl, or glibc < 2.34**
(Ubuntu 20.04, Debian 11, RHEL/Rocky 8, Amazon Linux 2) cannot install `[desktop]` by any
resolver — and because `[all]` pulls `[desktop]`, they cannot install `[all]` either, even
though they only wanted web/API. A PEP 508 marker cannot express glibc or musl, so this can
only be contained, not fixed in-place.

## What Changes

- **Remove `rf-mcp[desktop]` from the `[all]` extra.** `[all]` becomes
  web + api + mobile + database + frontend + memory + tokens. Measured: this restores
  `uv tool install "rf-mcp[all]"` on linux x86_64/aarch64, macOS x86_64/arm64 and Windows,
  on Python 3.12 and 3.13. It simultaneously stops old-glibc / musl / macOS-Intel users
  being blocked by a dependency for a feature they did not ask for.
- **Keep the `==` pin on the `desktop` extra** (it is not the cause) and make the real
  requirement explicit in the docs: `uv tool install --prerelease=allow "rf-mcp[desktop]"`,
  or `UV_PRERELEASE=allow`. pip/poetry/pdm users need no flag.
- **Document the supported desktop platforms** (glibc >= 2.34 x86_64/aarch64, macOS arm64,
  Windows x64/arm64; no musl, no macOS Intel) next to the extras table.
- **Correct the three false statements**: `pyproject.toml:44`, `pyproject.toml:75`, and
  `docs/RELEASE_NOTES_v0.34.0.md:160`.
- **Update the docs that advertise the broken commands**: `README.md:39`, `README.md:165`
  (which promises `[all]` includes desktop), `README.md:174`, `docs/GETTING_STARTED.md:32`,
  `docs/RELEASE_NOTES_v0.34.0.md:32`.
- **Add a CI guard that installs rf-mcp the way a user does** — build the wheel, then
  `uv tool install` it from a local index in a clean env, asserting `[all]` resolves without
  `--prerelease` and that `[desktop]` resolves with it. A local-path/editable install cannot
  catch this class of bug; only a non-first-party install can.

## Non-Goals

- Fixing the upstream wheel coverage (macOS Intel / musl / glibc < 2.34) — that is a
  PlatynUI publishing decision, not an rf-mcp change.
- Changing the desktop runtime, PlatynUI version, or any desktop behaviour.
- Moving to a stable PlatynUI: none exists for the new Rust core (latest stable
  `robotframework-PlatynUI` is 0.9.2, pre-new-core).
