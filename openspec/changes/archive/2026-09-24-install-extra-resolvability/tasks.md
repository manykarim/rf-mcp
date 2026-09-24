# Tasks: install-extra-resolvability

## 1. pyproject — make `[all]` resolvable
- [x] 1.1 Remove the `"rf-mcp[desktop]"` self-extra from `[project.optional-dependencies].all`
- [x] 1.2 Keep the `desktop` extra and its `==0.13.0.dev2` pins unchanged (the pin is not the cause)
- [x] 1.3 Rewrite the `desktop` comment block: state that uv-backed resolvers (uv, uvx, pipx) REQUIRE `--prerelease=allow` because the pin is transitive for consumers, and that pip/poetry/pdm do not
- [x] 1.4 Delete the false sentence at `pyproject.toml:44` ("resolvers accept it WITHOUT a global --pre") and the matching claim at `:75`
- [x] 1.5 Record the wheel-platform limitation (glibc >= 2.34, no musl, no macOS x86_64, no sdist) in the same comment block

## 2. Documentation
- [x] 2.1 `README.md:165` extras table — `all` no longer includes `desktop`; add a `desktop` row note with the uv flag
- [x] 2.2 `README.md:39` Quick Start — verify `uv tool install "rf-mcp[all]"` is now correct as written
- [x] 2.3 `README.md:174` (`uv add "rf-mcp[all]"`) — verify; add the desktop variant with the flag
- [x] 2.4 `docs/GETTING_STARTED.md:32` — same treatment
- [x] 2.5 `docs/RELEASE_NOTES_v0.34.0.md:32` and `:160` — correct the "no `--pre` needed" claim (errata note; do not rewrite history silently)
- [x] 2.6 Add a short "Desktop: supported platforms" subsection (glibc >= 2.34 x86_64/aarch64, macOS arm64, Windows x64/arm64; NOT macOS Intel, NOT musl)
- [x] 2.7 Release notes for this version: state that `[all]` no longer implies `[desktop]` and give the one-line migration (`uv tool install --prerelease=allow "rf-mcp[desktop]"`)

## 3. CI guard — install as a user, not as a path
- [x] 3.1 New job: `uv build`, then in a clean env `uv tool install --find-links dist "rf-mcp[all]"` with DEFAULT prerelease settings; assert success
- [x] 3.2 Assert the resolved set contains NO `platynui*` distribution for `[all]`
- [x] 3.3 Second step on a PlatynUI-supported runner: `uv tool install --prerelease=allow --find-links dist "rf-mcp[desktop]"`; assert success and that `platynui-native` is present
- [x] 3.4 Run the guard on ubuntu-latest, macos-latest (arm64) and windows-latest
- [x] 3.5 Ensure the job does NOT use `uv sync`, the committed `uv.lock`, or any `-e .` install

## 4. Regression test (fast, offline)
- [x] 4.1 Unit test parsing `pyproject.toml`: the `all` extra SHALL NOT reference `rf-mcp[desktop]` nor any `*platynui*` distribution
- [x] 4.2 Unit test: every requirement in `all` has a specifier that admits at least one non-pre-release version (guards the whole class, not just PlatynUI)

## 5. Verification
- [x] 5.1 Re-run the resolution matrix from `experiments/platynui_pin_resolution_evidence.md` §2.1 against the rebuilt wheel; all `[all]` cells must flip FAIL -> OK
- [x] 5.2 Confirm `pipx install "rf-mcp[all]"` succeeds
- [x] 5.3 Confirm `uv tool install --prerelease=allow "rf-mcp[desktop]"` still installs a working desktop stack
