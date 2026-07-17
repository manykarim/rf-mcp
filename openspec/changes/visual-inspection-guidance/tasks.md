# Tasks: visual-inspection-guidance

## 1. Default path — screenshot response advertises the artifact (token-cheap)
- [x] 1.1 On a successful desktop/web screenshot step, add `screenshot_path` (absolute) + a one-line `visual_hint` to the response, extending the existing hint layer (`evidence_missing_hint` / `screenshot_request_path`, `desktop_execution_signals.py`)
- [x] 1.2 Hint text names the vision-only use cases + "prefer Get Text when the value is in the DOM"; no image bytes in the default response

## 2. visual_validation guidance topic
- [x] 2.1 `get_locator_guidance(library="visual")` branch (server.py:6639-6674) → new `get_visual_guidance` in `rf_native_type_converter.py`, mirroring the requests cookbook
- [x] 2.2 Content: the six vision-only case categories; the dual read-back pattern (Get Text first when a node has the value; screenshot to confirm; screenshot-primary only when no node exposes it); the multimodal + file-access caveats
- [x] 2.3 Docstring of `get_locator_guidance` advertises `visual`/`screenshot` as a topic

## 3. Opt-in visual_check tool (image escape hatch, OFF by default)
- [x] 3.1 New `visual_check` tool: unified capture over Browser/Selenium/Appium/PlatynUI (whole page or `selector`/`descriptor`), always saves to `ROBOTMCP_SCREENSHOT_DIR` and returns the path
- [x] 3.2 `return_image: bool = false` — false → `{path, size, dimensions}` text only; true → a FastMCP `Image` content block + path
- [x] 3.3 `ROBOTMCP_SCREENSHOT_MODE=file|image|auto` (default `file`) gates whether `return_image=true` is honored / defaulted; text-only deployments stay `file`
- [x] 3.4 Missing/failed capture degrades to the evidence-missing hint (never raises)

## 4. Cross-library naming + docstrings
- [x] 4.1 One shared sentence in the Browser/Selenium/Appium/PlatynUI screenshot surfaces + `visual_check`: "saved to <path>; read it to validate visually (multimodal) — use for canvas/image text, layout/overlap, obscured elements, color, charts"
- [ ] 4.2 README: short "Visual validation (multimodal)" section documenting the default (path) + opt-in (image) + the `ROBOTMCP_SCREENSHOT_MODE` knob + caveats (determinism, PII, file-access)

## 5. Tests
- [x] 5.1 `get_locator_guidance(library="visual")` returns the case list + dual-read-back + caveats; alias `screenshot` resolves
- [x] 5.2 A successful screenshot response carries `screenshot_path` + `visual_hint`; non-screenshot steps unaffected
- [x] 5.3 `visual_check` returns text (path) by default; an image content block only when `return_image=true` and mode allows
- [x] 5.4 `ROBOTMCP_SCREENSHOT_MODE` gating (file/image/auto); missing capture degrades cleanly

## 6. De-risking experiments (docker, key-gated)
- [x] 6.1 Multimodal driver (GPT-4o) on the Tricentis wizard: fed the rf-mcp Browser capture `artifacts/tricentis/wizard_vehicle.png`, image-only → answered all 4 correctly (Enter Vehicle Data active; 7 required-field errors; Make flagged; License Plate optional). Usage 859 prompt (image ≈765) + 51 completion. Proves a multimodal model extracts the DOM-impossible gestalt.
- [x] 6.2 Text-only driver (MiniMax-M3, `docker/run_lab_scenario_cc.sh visual-textonly-m3`): success/is_error=false/13 turns; agent self-called `get_locator_guidance` + `visual_check` (no return_image); response was `mode=file` + `screenshot_path=/artifacts/visual_check_tricentis_test.png` (path string, no image block); zero image-content errors → graceful degradation confirmed.
- [x] 6.3 Token-cost: default path+hint ≈73 tok/screenshot vs naive always-attach ≈765 tok (measured GPT-4o) → ~10× cheaper/screenshot; over an 8-screenshot run with 2 real visual checks ≈2,114 vs ≈6,120 tok (~2.9× saving), and the always-attach floor is fixed regardless of need.
