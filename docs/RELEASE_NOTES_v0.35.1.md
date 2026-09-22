# rf-mcp 0.35.1

An install-blocker point release. **`uv tool install "rf-mcp[all]"` — the first command in the
README Quick Start — failed for every user on every operating system in 0.34.0 and 0.35.0.**
If you install with `pip`, `poetry` or `pdm`, you were never affected.

---

## The fix

`desktop` is no longer part of the `[all]` extra. `[all]` now means *every extra that resolves
everywhere with default resolver settings*:

```bash
uv tool install "rf-mcp[all]"     # works again — no flag needed
pipx install "rf-mcp[all]"        # works again
pip install "rf-mcp[all]"         # unchanged, always worked
```

Desktop automation is unchanged and fully supported — it is now an explicit opt-in:

```bash
uv tool install --prerelease=allow "rf-mcp[desktop]"   # uv, uvx, pipx
pip install "rf-mcp[desktop]"                          # pip, poetry, pdm — no flag
```

### Migration

If you relied on `rf-mcp[all]` to pull desktop automation, install `rf-mcp[desktop]` as well using
one of the commands above. Nothing else changes; no configuration changes are needed.

---

## What was actually wrong

Not a missing version — `platynui-cli==0.13.0.dev2` exists on PyPI and is not yanked.

uv's default pre-release policy is `if-necessary-or-explicit`, and **"explicit" applies only to
first-party requirements**. For anyone installing rf-mcp from PyPI, rf-mcp's pin on
`robotframework-platynui==0.13.0.dev2` is *transitive*, so uv refused it and reported:

```
Because there is no version of robotframework-platynui==0.13.0.dev2 and
rf-mcp[all]==0.35.0 depends on robotframework-platynui==0.13.0.dev2, ...
```

which reads like the version does not exist. It does; it was simply not selectable.

The same pin resolved fine for maintainers and in CI (`uv sync`, `uv lock`,
`uv pip install -e ".[desktop]"` — all first-party) and failed for every user
(`uv tool install`, `uv add`, `uvx`, `pipx`). That asymmetry is why no CI job could see it.
Affected installers were exactly the uv-backed ones — **including pipx**, which delegates to uv.

Loosening the pin to `>=0.13.0.dev0` would *not* have fixed it: the specifier shape is irrelevant
to uv's policy, and only `--prerelease=allow` changes the outcome. The pin is therefore unchanged.

## Second issue this also fixes

PlatynUI publishes wheels only for Linux x86_64/aarch64 with **glibc ≥ 2.34**, macOS **arm64**, and
Windows x86_64/arm64 — and **no sdist**. Because `[all]` pulled `desktop`, users on macOS Intel,
musl/Alpine, or glibc < 2.34 (Ubuntu 20.04, Debian 11, RHEL/Rocky 8, Amazon Linux 2) could not
install `[all]` at all, even though they only wanted web or API testing. They can now.

Those platforms still cannot install `desktop` itself — that is an upstream PlatynUI packaging
limit, and it is now stated in the README extras table instead of surfacing as an opaque
resolver error.

## Preventing a recurrence

CI previously installed rf-mcp only as a local path (`-e .`) or from the committed lock file, both
of which make rf-mcp's own pins first-party and therefore mask this entire class of defect. A new
installability guard builds the wheel and installs it as a third-party dependency across operating
systems, asserting that `[all]` resolves with default settings and that no pre-release-only
distribution has crept into it.

---

## Also fixed: the launch rf-mcp writes now matches the environment your tests get

rf-mcp imports Robot Framework libraries into its own process and runs suites in that same
interpreter, so the launch command written into your agent's config decides which libraries
and which Robot Framework version your tests actually get. Four places where the report and
the reality disagreed:

- **A project-aware launch kept rf-mcp's own extras.** If you installed `rf-mcp[all]` and
  then ran `robotmcp install` in a project, the resulting overlay layered **bare** rf-mcp
  onto it - no Browser, no SeleniumLibrary - so web automation could not run, while
  `robotmcp doctor` correctly reported both as installed (they were, in rf-mcp's own
  environment). The overlay now carries the extras your installation actually provides.
- **An rf-mcp installed from a local wheel produced an invalid launch command**
  (`--with-editable` pointed at an archive, which uv rejects). Editable references are now
  used only for a real source directory.
- **The Robot Framework version that will execute your tests is now reported.** An overlay
  supplies its own Robot Framework, which shadows the project's; only a MAJOR mismatch was
  flagged before, so a project pinned to 7.2 was silently tested on 7.5. `robotmcp doctor`
  and the install-time note now say which version will run. Major mismatches still route to
  the attach bridge - that behaviour is unchanged.
- **Dry-run validation no longer reports "passed" when imports failed.** `run_test_suite`
  with `mode="dry"` returned `success: true` / `validation_status: "passed"` alongside
  error-severity import failures, because a zero return code short-circuited the check. An
  agent read that as a green light and proceeded to a full run. Status and `success` now
  reflect error-severity findings; warnings alone still do not fail a validation.

The README gained a short **"Which environment runs your tests"** section, and `robotmcp
init` now notes that its config snippet uses rf-mcp's own environment - for a project with
its own libraries, run `robotmcp install` from the project directory instead.

---

## Also fixed: your project's own keywords are now discoverable

rf-mcp could already **execute** keywords from your project's custom Python libraries and
`.resource` files - but `find_keywords` and `get_keyword_info` could not see them, because
discovery read a catalogue of rf-mcp's own libraries while execution went through Robot
Framework's namespace. Asking for a keyword by its exact name returned unrelated
Browser/SeleniumLibrary results, and `get_keyword_info` reported "not found in any loaded
library" for a keyword `execute_step` would happily run.

Since `find_keywords`' guidance is to look a keyword up *before* using it, an agent working
on a project with a curated keyword layer concluded those keywords did not exist and
re-implemented them from raw Browser/Selenium calls.

Now, whatever a session imports is also discoverable:

```
manage_session(action="import_resource", resource_path="resources/acme_kw.resource")
find_keywords(query="Acme Login")   ->  Acme Login  (acme_kw, confidence 1.0)
get_keyword_info("Acme Login")      ->  args: ['user', 'password=secret']
```

Registration covers libraries declared at session init, libraries added with
`import_library`, and resource files added with `import_resource`. Argument names,
defaults, type hints and documentation come from Robot Framework's own LibDoc, so what is
described matches what execution accepts. Registrations are **per session**, so two
sessions working on different projects never see each other's keywords. A keyword that
still cannot be found now says which import call would make it available.
