# Proposal: visual-inspection-guidance

## Why

rf-mcp drives four UI technologies (Browser, SeleniumLibrary, AppiumLibrary,
PlatynUI) whose agents validate almost exclusively through the DOM / page source /
ARIA / AT-SPI tree. But a class of checks is **provably impossible** through those
structured trees — and a multimodal coding agent could do them from a screenshot
rf-mcp *already captures to disk*. Two hands-on experiments (2026-07-17) confirm it:

- **Controlled DOM-blind page** (`artifacts/visual-exp/`): a value drawn on
  `<canvas>` ("TOTAL DUE: $42.00") has **no DOM text node at all**; a "Place order"
  button reported by the DOM as visible+enabled+clickable was **fully covered by a
  "Session expired" overlay** — a user can't click it, yet every structured check
  says it's actionable. Vision caught both (and that the canvas value was clipped).
- **Tricentis insurance wizard** (`artifacts/tricentis/`, the user's target): after
  a failed "Next", one screenshot conveys the entire state — active step (of 5), a
  red **"7" badge = seven validation errors**, which fields are required (red ✳) vs
  optional, and which field is actively flagged (pink "Make" + a "Select an option"
  callout). A DOM-only agent must introspect every field to reconstruct that. The
  run also failed first because `text=Get a quote` matched **4 identical buttons** —
  ambiguous from visible text alone.

The value is **real and in several cases unique** (a research sweep found six vision-
only classes: GTK/desktop text content, `<canvas>`/SVG charts, rendered-image
correctness, layout/overlap/clipping, color/theme, transient states). The sharpest
framing from this session: **verification failure ≠ steering failure** — the
LibreOffice run steered the app correctly (0 timeouts) but was *scored a failure*
purely because AT-SPI exposes no text-content attribute; a screenshot would have
confirmed the already-achieved success instantly (`PLATYNUI_STEERING_EVAL §2.3`).

**But the capability is invisible today, and the obvious implementation is
anti-token.** rf-mcp writes screenshots to disk as *artifacts* and says nothing
about reading them; the driving model never learns the pattern exists. This session
only used it because Claude Code has its own file-read tool and *I chose* to read
`calc.png`. Conversely, base64-encoding every screenshot into the MCP **response**
(the naive version) costs **~1,000–1,600 tokens/image** and would gut rf-mcp's
low-token identity — and silently breaks on text-only driver models (MiniMax-M3,
this session's driver, is text-only; the server cannot detect client modality).

The fix is mostly **discoverability + naming**, not new machinery — the same lever
as `desktop_guidance` / the API cookbook / `actionable_controls`.

## What Changes

**Default path stays token-cheap and works on every model (incl. text-only):**

- **Screenshot responses advertise the artifact + the visual-validation pattern.**
  When a desktop/web screenshot keyword succeeds, the response SHALL include the
  **absolute path** of the saved image plus a one-line hint: *"saved to <path> — if
  your model is multimodal, read it to validate visually (canvas/image text,
  layout/overlap, obscured elements, color, charts); prefer Get Text when the value
  is in the DOM."* Extends the existing screenshot-hint layer
  (`evidence_missing_hint` / `screenshot_request_path`,
  `desktop_execution_signals.py`). Zero image tokens by default — the agent spends
  them only when it decides to read the file.
- **Clear naming + docstrings across all 4 UI libraries' screenshot surfaces** so an
  agent knows the capability exists and *when* it beats the DOM. Today none mention
  visual validation.
- **A `visual_validation` guidance topic** exposed through the existing
  `get_locator_guidance` tool (a new `library="visual"` branch, mirroring the
  `requests` cookbook, `server.py:6639-6674`): the six vision-only case categories,
  the **dual read-back pattern** (assert with `Get Text` first when a node carries
  the value; screenshot to *confirm*; screenshot-as-primary *only* when no node
  exposes it), and the multimodal + file-access caveats.

**Opt-in image return for agents without file access (OFF by default):**

- **A dedicated `visual_check` tool** captures a screenshot (page or element, any of
  the 4 libraries) and, when explicitly asked, returns a FastMCP `Image` content
  block so a multimodal agent that *cannot* read the artifact file gets the pixels
  in-response. It always also returns the saved path. An `output_path`-style text
  mode returns only the path string (the chrome-devtools `take_screenshot` /
  `filePath` precedent) — the graceful degradation for text-only drivers.
- **Deployment knob `ROBOTMCP_SCREENSHOT_MODE=file|image|auto`** (default `file`):
  the server can't detect the client model, so operators driving with a multimodal
  model can flip the default; text-only deployments keep `file`. Mirrors the
  existing `ROBOTMCP_SCREENSHOT_DIR` / tool-profile env pattern.

Out of scope (v1): auto-attaching images to `execute_step` / `execute_batch` /
`get_session_state` (per-step images would wreck the budget); any server-side OCR or
vision model (the *driving* model does the seeing); a visual regression/baseline-diff
engine; PII redaction (documented caveat only). Appium is covered by the shared
naming/guidance but not separately exercised (flagged inference, no mobile scenario
ran this session).

## Capabilities

### New Capabilities

- `visual-inspection-guidance`: rf-mcp makes its already-captured screenshots
  *usable* for visual validation — screenshot responses surface the artifact path +
  a visual-validation hint, all four UI libraries' screenshot surfaces are clearly
  named/documented for the use case, a `visual_validation` guidance topic teaches
  when vision beats the DOM (and when it does not), and an opt-in `visual_check` tool
  returns an image content block for multimodal agents without file access — all
  token-cheap by default (path-as-text) and degrading gracefully on text-only models.

### Modified Capabilities

- None (additive over existing screenshot keywords and `get_locator_guidance`; no
  change to the default response shape beyond an added path + hint).

## Impact

- `src/robotmcp/components/execution/desktop_execution_signals.py` /
  `keyword_executor.py` — screenshot-success responses include the absolute path +
  visual-validation hint (extend `evidence_missing_hint`); docstring updates on the
  desktop screenshot path.
- `src/robotmcp/utils/rf_native_type_converter.py` + `server.py:6639-6674` — new
  `library="visual"` branch on `get_locator_guidance` → a `get_visual_guidance`
  cookbook (six case categories + dual-read-back + multimodal/file caveats).
- `src/robotmcp/server.py` — a new opt-in `visual_check` tool (capture across
  Browser/Selenium/Appium/PlatynUI; return FastMCP `Image` only when requested; else
  path text); `ROBOTMCP_SCREENSHOT_MODE` reader.
- Docstrings on the Browser/Selenium/Appium/PlatynUI screenshot surfaces (unified
  wording). README: a short "visual validation (multimodal)" section.
- Tests: `tests/unit/` — the guidance topic returns the case list + caveats;
  screenshot response carries a path + hint; `visual_check` returns text by default
  and an image block only when asked; `ROBOTMCP_SCREENSHOT_MODE` gating.
- Experiments (evidence, already captured): `artifacts/visual-exp/` (canvas +
  obscured button), `artifacts/tricentis/` (wizard validation-state orientation).
