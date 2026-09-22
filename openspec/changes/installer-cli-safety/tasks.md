# Tasks: installer-cli-safety

Ordered by severity. Sections 1-3 are correctness/safety and should land first; they are
independently shippable from the rest.

## 1. `--dry-run` must not mutate (critical)
- [x] 1.1 Thread `dry_run` into `resolve_launch()` (`installer.py:338` call site) and down to `_install_into_project()` (`installer.py:181`)
- [x] 1.2 `_install_into_project()` returns a planned-not-performed result under dry-run; never executes `uv pip install`
- [x] 1.3 Result status/detail wording distinguishes "would install" from "installed"
- [x] 1.4 Audit every other side effect reachable from `resolve_launch` (dir creation, manifest writes) for dry-run safety
- [x] 1.5 Regression test: `install --dry-run --into-project` against a venv lacking rf-mcp leaves it unchanged (assert import still fails afterwards)
- [x] 1.6 Regression test: `--dry-run` creates no file and no directory for a fresh agent target

## 2. Exit codes
- [x] 2.1 `cli.py:_print_results` computes rc from result statuses instead of the hardcoded `rc = 0`; define which statuses are failures (`unverified`, `error`, `no-assets` on an explicit `--what`) vs benign (`already-present`, `absent`, `kept-user-modified`)
- [x] 2.2 `cmd_init` returns non-zero when browser init fails
- [x] 2.3 `cmd_doctor` gains `--strict` (non-zero when any checked capability is missing); default stays 0 so plain `doctor` remains a report
- [x] 2.4 Tests covering each status -> exit-code mapping

## 3. Argument validation (follow `--scope`'s model)
- [x] 3.1 `adapters.resolve()` returns unknown tokens instead of dropping them (`adapters.py:159`); callers fail with the token named + valid id list
- [x] 3.2 `--agents ""` is an error, not a silent alias for `detected`
- [x] 3.3 `--what` validates against the known kinds; distinguish "unknown kind" (error) from "known kind, no bundled assets yet"
- [x] 3.4 `_parse_env` rejects entries without `=` or with an empty key (`cli.py:20`); decide and document duplicate-key behaviour
- [x] 3.5 `--attach` validates `host[:port]`: port numeric and in range; reject otherwise
- [x] 3.6 `robotmcp list` gains an ID column (the value `--agents` accepts)
- [x] 3.7 Reject `--command` together with `--attach` (currently silently co-exist)
- [x] 3.8 Tests for each malformed-value case

## 3b. Declining / aborting must be honoured (critical)
- [x] 3b.1 `adapters.resolve_selection`: stop coercing an empty spec to `detected` (`adapters.py:150`); an empty selection stays empty
- [x] 3b.2 `_interactive_agents` returns an explicit sentinel for "declined" rather than `""`, and the caller aborts
- [x] 3b.3 A declined prompt prints an abort message and exits non-zero, writing nothing
- [x] 3b.4 `--agents ""` is rejected as an error (guards `--agents "$UNSET_VAR"`)
- [x] 3b.5 Catch `KeyboardInterrupt` in the CLI entry: print an abort message, exit 130, leave no partial write
- [x] 3b.6 Regression test: declining the prompt writes no config and exits non-zero
- [x] 3b.7 Regression test: `resolve_selection("")` returns no adapters

## 4. Config integrity
- [x] 4.1 Fix `codecs.py:48`: apply trailing-comma removal only outside string literals (do it inside the existing character scanner rather than as a post-pass regex)
- [x] 4.2 Regression test with the reproducer string: `"Close the brace like this: { a, } and the bracket like [ b, ]"` survives byte-identical
- [x] 4.3 Align the `kilo` adapter (`adapters.py:117`) with the format Kilo actually reads — filename, `type`, and `command` cardinality; cross-check against `.kilo/kilo.json` in this repo and upstream Kilo docs before changing
- [x] 4.4 goose YAML: preserve anchors/merge keys, or detect an anchored document and decline to rewrite with an explanatory message
- [x] 4.5 Document per-format fidelity (TOML: byte-identical incl. comments; JSON: data-preserving, re-indents; JSONC: comments lost; YAML: comments lost) and fix the contradicting claim at `design.md:56`
- [x] 4.6 Consider preserving JSON indentation to avoid noisy diffs in VCS-tracked `.mcp.json` (cosmetic; 4-space input currently becomes 2-space)
- [x] 4.7 Verify whether `copilot` needs the `"type": "stdio"` field that this repo's own `.vscode/mcp.json` carries

## 5. `uninstall` correctness
- [x] 5.1 Filter manifest entries by project dir/scope so `-C <dirA>` cannot remove `dirB` (`installer.py:407`)
- [x] 5.2 Stop defaulting `uninstall` to `--agents detected`; drive removal from the manifest
- [x] 5.3 Continue past an unparseable config instead of aborting the whole run
- [x] 5.4 Add `uninstall --force` so a `kept-user-modified` entry can be removed deliberately
- [x] 5.5 Fix the orphaned empty `{}` file after `install -> install --force -> uninstall` (`created_file` flipped at `installer.py:388`)
- [x] 5.6 Remove a directory the installer created when its only file is removed (e.g. `.cursor/`)

## 6. Readable failures
- [x] 6.1 Wrap `codecs.load` parse errors: report path + parse problem, no traceback
- [x] 6.2 Validate `--project-dir` in `doctor` and `list` too — move the check in `cli.py` ahead of the subcommand dispatch
- [x] 6.3 `-C ""` is an error rather than a silent alias for the CWD
- [x] 6.4 `-C <regular file>` is rejected up front (currently `FileExistsError` at write time)

## 7. Discoverability
- [x] 7.1 `entry.py:14`: route `-h`/`--help`, and any unrecognised first token, to the onboarding parser without importing the server module
- [x] 7.2 Add a did-you-mean suggestion for near-miss subcommands (`instal` -> `install`)
- [x] 7.3 Bare `robotmcp` on a TTY prints a one-line explanation (stdio server; see `robotmcp --help`) before blocking; unchanged when stdin is a pipe
- [x] 7.4 Ensure the fast path stays fast — assert the help path does not import `robotmcp.server` (currently 2.4s vs 0.4s)

## 8. `init` browser opt-in
- [x] 8.1 `diagnostics.py:152`: `want_browsers = browsers` only; drop the `or libs.get("Browser")`
- [x] 8.2 Without the flag, report the browser as uninitialized and print the exact init command
- [x] 8.3 Test: plain `init` with an importable-but-uninitialized `Browser` attempts no download

## 9. Diagnostics accuracy
- [x] 9.1 Add a desktop/PlatynUI row to `TEST_LIBRARIES` (`diagnostics.py:13`) using the working install command from `install-extra-resolvability`
- [x] 9.2 Intersect `project_env._BUNDLED_DISTS` with what this installation actually provides, so `doctor` cannot claim "bundle suffices" for a library it reported missing
- [x] 9.3 `list`'s REGISTERED column reads the agent config, not only rf-mcp's manifest (`installer.py:442`); keep the manifest for provenance
- [x] 9.4 Tests for the contradiction case and the hand-pasted-config case

## 10. Verification
- [x] 10.1 Re-run the three probe matrices (`experiments/installer_ux_init_track.md`, `installer_ux_install_track.md`, `adapter_config_audit.md`) and confirm each recorded defect flips
- [x] 10.2 Confirm the `codex`/TOML adapter's byte-identical round trip is preserved (it is the reference implementation — do not regress it)
