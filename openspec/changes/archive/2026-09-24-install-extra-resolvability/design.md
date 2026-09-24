# Design: install-extra-resolvability

## The mechanism, precisely

uv's pre-release policy defaults to `if-necessary-or-explicit`:

- **`if-necessary`** — allow a pre-release when the package's constraints exclude every
  stable release.
- **`explicit`** — allow a pre-release when the requirement carrying the pre-release
  marker is **first-party** (the workspace's own `pyproject.toml` / a requirement the user
  typed).

For a consumer running `uv tool install "rf-mcp[desktop]"`, rf-mcp's pin on
`robotframework-platynui==0.13.0.dev2` is **transitive package metadata**, so `explicit`
does not apply. Measured: `if-necessary` does not rescue it either, for any specifier shape
(§ proposal table) — uv evaluates the package's stable availability, and
`robotframework-PlatynUI` does have stable releases (0.9.2), so the range never reads as
"stable-free" in the way `if-necessary` requires.

```
                     ┌──────────────────────────────┐
                     │ robotframework-platynui       │
                     │   ==0.13.0.dev2               │
                     └──────────────┬───────────────┘
                                    │
             is this requirement FIRST-PARTY to the resolve?
                                    │
              ┌─────────── yes ─────┴───── no ───────────┐
              │                                          │
   uv lock / uv sync / uv pip install -e .    uv tool install "rf-mcp[...]"
   (maintainer + every CI job)                 uv add / uvx / pipx
              │                                          │
        "explicit" applies                      no rule applies
              ▼                                          ▼
         ✅ resolves                              ❌ "no solution found"
```

The asymmetry is the whole bug. It is invisible from inside the repository, which is why
it survived to a published release.

## Why not the obvious fixes

| candidate | verdict | why |
|---|---|---|
| `>=0.13.0.dev0` (as first reported) | **rejected — measured no-op** | Fails identically under uv. Specifier shape is irrelevant; only the flag matters. |
| Bump pin to `0.13.0.dev117` | rejected | Same pre-release class, and wheel platform coverage is byte-identical from dev1 to dev117. Fixes neither defect. |
| `[tool.uv] prerelease = "allow"` in rf-mcp's pyproject | **impossible** | uv reads `[tool.uv]` from the *consuming* project, never from an installed dependency's metadata. rf-mcp cannot set resolver policy for its users. |
| Move to stable PlatynUI | impossible today | No stable release exists for the Rust new-core line; latest stable `robotframework-PlatynUI` is 0.9.2 (pre-new-core, ADR-025 does not target it). |
| Gate desktop behind a platform marker | insufficient | PEP 508 markers cannot express glibc version or musl. `macOS x86_64` could be excluded via `platform_machine`, but Alpine and old-glibc Linux cannot be. |
| Keep desktop in `[all]`, document the flag everywhere | rejected by product decision | Honest, but makes the headline command `uv tool install --prerelease=allow "rf-mcp[all]"` and still hard-blocks macOS-Intel / musl / glibc<2.34 users from `[all]` entirely. |
| **Remove desktop from `[all]`** | **chosen** | Measured to restore `[all]` on all mainstream platforms, and contains the platform gap as a side effect. Desktop remains fully available, opt-in, with a documented flag. |

## Consequences

- `rf-mcp[all]` stops meaning *literally* everything. `README.md:162` already scopes desktop
  to "Windows/Linux", and `README.md:12` lists desktop separately from the library set, so
  the documentation shift is small — but it is a **behaviour change for anyone currently
  relying on `[all]` to pull desktop**, and must appear in the release notes with the
  one-line migration.
- Desktop users gain an explicit, working command instead of a silently broken one.
- Users on macOS Intel / Alpine / glibc < 2.34 gain a working `[all]`; they still cannot get
  desktop, but they now fail only when they ask for desktop specifically.
- On Python 3.10/3.11 the `python_version >= '3.12'` marker already dropped PlatynUI
  silently; removing it from `[all]` makes that consistent across all Python versions
  instead of version-dependent.

## Residual risk

The `desktop` extra still depends on a pre-release pin, so it stays uv-hostile by
construction. That is acceptable while PlatynUI new-core is pre-release, **provided the
documentation states the flag**. When PlatynUI publishes a stable 0.13.x, the pin should
move to a stable range and desktop can be reconsidered for `[all]` — at which point the
CI guard added here becomes the gate that proves it is safe to do so.

## Evidence

Full measured evidence: `experiments/platynui_pin_resolution_evidence.md` (untracked —
`experiments/` is git-excluded via `.git/info/exclude:17`); a tracked copy of the decisive
tables is reproduced in `proposal.md`. Commands were run on 2026-09-20 with uv 0.9.26,
pip 26.2.1, pipx 1.17.4 against the published rf-mcp 0.35.0.
