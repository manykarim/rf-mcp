# Tasks: desktop-unscoped-locator-guardrail

## 1. Detection + guard

- [x] 1.1 Add unscoped-xpath detection (helper in `desktop_execution_signals.py`): `is_query_keyword` (`query`/`evaluate` basenames) and `is_unscoped_locator(xpath)` — True for leading `//` / `descendant-or-self::` / bare `*`/`//*`; False for `/app:`-anchored, relative (`control:`/`item:`/`.`/axis), and pure scalar-aggregate (`count(`/`string(`/`number(`/`boolean(` outer wrapper). Pure function, unit-tested in isolation.
- [x] 1.2 Add `_unscoped_locator_guard(session, keyword, arguments) -> Optional[dict]` to `keyword_executor.py`: desktop-only; returns the refusal error dict (hint type `unscoped_desktop_locator`, restated rule, app-scoped rewrite using a resolved/launched app name when available); None to proceed
- [x] 1.3 Wire into the desktop pre-flight block beside `_screenshot_path_guard` (~line 1726): refuse before focus/dispatch
- [x] 1.4 Escape hatch: `ROBOTMCP_PLATYNUI_ALLOW_UNSCOPED=1` + `session.platynui_allow_unscoped` → downgrade to a one-time warning (`desktop_unscoped_warned` one-shot field on `ExecutionSession`, mirroring `desktop_wayland_warned`)

## 2. Tests

- [x] 2.1 `tests/unit/test_unscoped_locator_guardrail.py`: detection truth table (`//control:Paragraph`, `//*` → unscoped; `/app:*…`, `control:Button`, `.//x`, `count(//…)` → allowed); guard refuses unscoped desktop Query with the rewrite hint; allows scoped/relative/count; opt-out (env + session) downgrades to one-time warning; web session untouched
- [x] 2.2 Regression: the report's exact `Query //control:Paragraph` shape is refused with a hint naming an `/app:*[@Name=…]//control:Paragraph` rewrite

## 3. Validation

- [x] 3.1 Full unit suite green (baseline 6817 passed + 1 skipped; no regressions)
- [x] 3.2 Live check on `:100`: confirm the guard refuses `Query //control:Paragraph` instantly (no 36 s walk) and that `/app:*[@Name='soffice']//control:Paragraph` still resolves
