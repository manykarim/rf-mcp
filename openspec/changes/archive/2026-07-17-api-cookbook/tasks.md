# Tasks: api-cookbook

## 1. Cookbook content
- [x] 1.1 Add `get_requests_guidance(error_message=None, keyword_name=None)` to `RobotFrameworkNativeConverter` (`rf_native_type_converter.py`), returning `{success, library:"RequestsLibrary", tips, warnings, examples}` — the same shape as `get_browser_locator_guidance`
- [x] 1.2 Encode the 8 recipes (design.md): session setup; response-object access (`${resp.json()["f"]}`, `${resp.status_code}`); `$resp`-in-Evaluate vs `${resp.json()}`-elsewhere; `Status Should Be`; `json=`/`headers=` body; `Cookie: token=` auth; `expected_status=` for non-2xx; named-args `name=value`
- [x] 1.3 Factor the recipe text shared with `utils/hints.py` (recipes 1/3/5/8) into a module-level constant reused by both, so proactive cookbook and reactive hint never drift
- [x] 1.4 Optional: when `error_message`/`keyword_name` indicate a 400/415 or an Evaluate misuse, order the matching recipe first (full cookbook still returned)

## 2. Tool dispatch
- [x] 2.1 Add a `requests` / `requestslibrary` / `api` branch to `get_locator_guidance` (`server.py:6613-6644`) dispatching to `converter.get_requests_guidance(...)`; set `result["library"]="RequestsLibrary"`, `setdefault("success", True)`
- [x] 2.2 Update the `get_locator_guidance` docstring to list `RequestsLibrary`/`api` as a supported library so the tool self-advertises the cookbook

## 3. Steering
- [x] 3.1 Add a one-line pointer in the API-session surface (session-init guidance / recommender note for the api context) telling the agent to call `get_locator_guidance(library="requests")` before writing `Evaluate`-based assertions

## 4. Tests
- [x] 4.1 `get_locator_guidance(library="requests")` returns success + the key recipes present (assert on: `${resp.json()}` access, `Status Should Be`, `Cookie: token=`, `expected_status=`, `$resp` Evaluate rule)
- [x] 4.2 Alias resolution: `api`, `requestslibrary`, `Requests` (case-insensitive) all resolve to the cookbook
- [x] 4.3 Shape parity: the returned dict has the same top-level keys as the Browser/Selenium guidance
- [x] 4.4 Unknown library still returns the existing error payload (no regression)
- [x] 4.5 Reactive `hints.py` RequestsLibrary hints unchanged (shared-constant refactor keeps existing hint text/tests green)

## 5. Acceptance validation (docker, LLM — key-gated)
- [x] 5.1 Re-run the restful-booker scenario (`docker/run_lab_scenario_cc.sh api-booker …`) and confirm it completes WITH artifacts (result.json + suite) inside the turn budget, with a materially lower `Evaluate` count than the 2026-07-17 baseline (178) — the empirical proof F-API1 is closed
