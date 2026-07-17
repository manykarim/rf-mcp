# Tasks: desktop-turn-economy-guidance

> Reconciled 2026-07-17: this tasks.md was blank (0 bytes) — the apply-commit
> (a10b1b7) shipped the code but did not persist the task list. All items below
> are verified present in `src/` and covered by unit tests; restored as [x].

## 1. Init-time desktop guidance bundle
- [x] 1.1 `desktop_guidance.py` builds a process-cached 24-keyword one-line-signature cheat-sheet (derived once from `LibraryDocumentation('PlatynUI.BareMetal')`) + `_LOCATOR_CRIB` (scope to `/app:*`, `control:Frame` not `control:Window` on Linux, Take Screenshot descriptor-first) + batch-first steer
- [x] 1.2 `get_desktop_guidance()` soft-fails to None on any libdoc error (never crashes init)
- [x] 1.3 `manage_session(init)` attaches the `desktop_guidance` bundle for desktop sessions / when `libraries` include `PlatynUI[.BareMetal]` (server.py)

## 2. Process as a desktop core library
- [x] 2.1 `Process` added to `DESKTOP_TESTING` core libraries (`session_models.py`) so a bare desktop init can `Start Process` an AUT without an extra import

## 3. Discovery steering
- [x] 3.1 `desktop_focused` instruction template / crib points the agent at the guidance bundle instead of `find_keywords` (which is near-useless for PlatynUI)

## 4. Tests + validation
- [x] 4.1 Unit tests: guidance bundle content (24-kw surface + crib), init attach, Process-in-core — 8 tests, green
- [x] 4.2 `openspec validate --strict` clean

> Evidence (SPIKE_2 / EVAL_M27, this session): `find_keywords` collapsed 13→0/1
> per desktop run; M2.7 launched the calculator on its first batch with zero
> discovery — the bundle is load-bearing.
