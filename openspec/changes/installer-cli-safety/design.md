# Design: installer-cli-safety

## The shape of these defects

None of them are exotic. Almost every one is the same mistake in a different place:
**a value that means "nothing" was treated as a value that means "everything", or an
outcome that means "failed" was reported as "succeeded".**

```
  user intent            what the code did                 result
  ───────────            ─────────────────                 ──────
  "n" at the prompt  ->  returns ""                     ->  "" or "detected"
  --agents ""        ->  falsy                          ->  "" or "detected"   -> installs into 5 agents
  --agents claude    ->  BY_ID.get() -> None -> dropped ->  empty selection    -> "Nothing to do." exit 0
  --what MCP         ->  not in WHAT_IMPLEMENTED        ->  "no-assets"        -> exit 0
  --env FOO          ->  no "=" -> skipped              ->  partial env        -> "installed"
  verify failed      ->  status "unverified"            ->  rc hardcoded 0     -> exit 0
  browser absent     ->  init failed                    ->  return 0           -> exit 0
```

The fix is uniform: **make "nothing" distinguishable from "everything", and let an
outcome reach the exit code.** `--scope` was already correct (`invalid choice: 'global'`,
exit 2) and is the model the rest now follows.

## Why `--dry-run` leaked

`install()` is structured as *resolve once, verify once, then write per agent*:

```
    plan = resolve_launch(...)              <-- (1) NOT pure: --into-project installs
    if not (no_verify or dry_run):          <-- (2) correctly gated
        verify_launch(plan)
    for what, adapter in ...:
        if dry_run: report; continue        <-- (3) correctly gated
        write()
```

Steps (2) and (3) were gated; step (1) was not, and nothing in its name suggests it
mutates. The fix threads `dry_run` into `resolve_launch` **and** makes
`_install_into_project` refuse on its own — belt and braces, because the leak came
precisely from a caller not knowing the callee mutates.

The dry-run statuses are now `would-install` / `would-update` / `would-remove` rather
than reusing `installed` / `removed`. A status that reads as past tense in a dry run is
the same class of bug as the exit code: the report claimed something that did not happen.

## Why the JSONC corruption existed

`_strip_jsonc` had a correct character scanner that tracked string state — and then
threw that knowledge away on the last line:

```python
return re.sub(r",(\s*[}\]])", r"\1", "".join(out))    # applies to strings too
```

The fix moves trailing-comma handling *into* the scanner, where string state is already
known. A comma is remembered as a removal candidate and cancelled by any non-whitespace
character, including a quote. This is strictly more correct than the regex and costs
nothing: the scan was already O(n).

## The kilo adapter: acting on one data point

This is the only change here that rests on a single piece of evidence rather than a
reproduction. The adapter wrote `.kilo/kilo.jsonc` with `{type: "stdio", command: "<str>"}`;
this repository contains a **working** `.kilo/kilo.json` using
`{type: "local", command: [argv...], enabled: true}` alongside a real `.kilo/` install.
Three independent disagreements (filename, `type` value, `command` cardinality) make a
coincidence unlikely, and the observed shape is exactly the `opencode` style the registry
already implements — so the adapter now reuses it.

A filesystem-wide search for `kilo.jsonc` afterwards found **no** such file anywhere
except the ones the audit's own sandbox runs had just written — i.e. no real Kilo
installation on this machine has ever produced the filename the adapter was writing,
while a real `kilo.json` does exist and is in use.

Upstream Kilo documentation was **not** consulted (no reliable access during this work).
If that shape is wrong, the failure mode is unchanged from today's: a file Kilo ignores.
Worth confirming against upstream before release.

## What was deliberately NOT changed

- **JSONC/YAML comment loss on write.** Preserving comments would mean adopting a
  round-tripping parser per format. Instead the behaviour is now documented honestly per
  format in `codecs.py`, and the one case where silent loss is *invisible but harmful* —
  YAML anchors and merge keys, where the data compares equal but the DRY link is severed —
  is refused outright rather than flattened.
- **The launch-strategy decision tree** (`uv-overlay` / `attach` / `own-shim` / `fallback`).
  Out of scope; only its dry-run purity changed.
- **`doctor`'s default exit code.** It stays 0 so `doctor` remains a report you can run
  without ceremony; `--strict` is the opt-in for CI.

## Risks

- **`uninstall` now removes more than before** in one respect (no longer gated on
  detection) and less in another (`-C` filters). Both are corrections, but a user who
  relied on the old `-C`-is-cosmetic behaviour to clean everything should use
  `--agents all` without `-C`.
- **Exit codes are now meaningful**, which will surface as newly-"failing" CI for anyone
  who was inadvertently depending on `robotmcp install` always returning 0.
- **The copilot entry gained `"type": "stdio"`**, matching this repo's own working
  `.vscode/mcp.json`. If VS Code rejects the field, this regresses copilot — it is the
  second-least-verified change here after kilo.

## Evidence

Reproductions and verbatim before/after output:
`experiments/installer_ux_init_track.md`, `experiments/installer_ux_install_track.md`,
`experiments/adapter_config_audit.md` (untracked — `experiments/` is git-excluded via
`.git/info/exclude:17`). The regression suite
`tests/unit/test_installer_cli_safety.py` pins every defect and is the durable record.
