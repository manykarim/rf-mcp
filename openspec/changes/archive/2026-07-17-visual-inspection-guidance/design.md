# Design: visual-inspection-guidance

## Guiding principle — token economy first

rf-mcp's identity is low-token agent steering. The naive "return the image in every
response" costs ~1,000–1,600 tokens/image and, if it fired per step, would dwarf the
entire tool-call budget. So the design inverts the default:

```
 DEFAULT (path-as-text, ~a few tokens; works on EVERY model)
   screenshot keyword succeeds → save to disk (already happens)
                               → response adds: absolute path + one-line visual hint
   multimodal agent WITH file access → reads the file ON DEMAND (image tokens only
                                        when it decides a visual check is worth it)
   text-only agent / no need        → ignores it, zero cost

 OPT-IN (visual_check tool, image block) — only for multimodal agents WITHOUT
   file access to the artifact dir (hosted/remote MCP). Never auto-attached.
```

This is exactly what happened this session by accident: PlatynUI saved `calc.png`;
the agent read it only to confirm "42". The image cost was paid once, by choice.

## Why not the obvious "always return the image"

- **Cost** (above) — anti-token, against the whole project ethos.
- **Gating is unsolvable server-side.** MCP exposes no field for the *client's*
  driving-model modality. The same rf-mcp process may be driven by Claude
  (multimodal), GPT-4o (multimodal), or MiniMax-M3 (**text-only** — this session).
  Returning an `Image` block to a text-only driver is silently dropped at best, an
  error at worst. You design *around* it (opt-in + text fallback + env knob +
  self-selecting docstring), you don't detect it.

## The dual read-back pattern (the honest contract)

The guidance must not oversell vision. Structured read-back is *better* whenever a
node exposes the value (exact, sub-ms, deterministic; OCR can misread `61.000,00` as
`81.000,00`). So the taught pattern is:

1. **Assert with `Get Text` / attribute read FIRST** when the value lives in the tree.
2. **Screenshot to CONFIRM** the structured pass (belt-and-suspenders), or for a
   human-auditable artifact.
3. **Screenshot as PRIMARY only when no node carries the answer** — the six cases
   below.

The calculator "42" success was *coincidental* (that toolkit surfaced the value as a
node `Name`); it does not generalize to a Writer body, a chart, or a rendered image.

## The six vision-only case categories (from the research sweep + experiments)

| Case | Why the tree can't answer | Evidence |
|---|---|---|
| Desktop/GTK text content | AT-SPI gives `Text.CharacterCount` (a count), never the glyphs | LibreOffice run: steered OK, scored FAIL for lack of a text attribute (§2.3) |
| `<canvas>` / unlabeled SVG | no addressable text node at all | `artifacts/visual-exp/` canvas "$42.00" — DOM-blind |
| rendered `<img>` correctness | DOM has `src`/`alt`; a 404/placeholder still has a valid `src` | carconfig rendered car |
| layout / overlap / clipping | visibility is style-truth, not composited-truth | `artifacts/visual-exp/` overlay-covered "clickable" button |
| color / theme / contrast | ARIA/AT-SPI carry no color; CSS gives declared, not composited | GTK theme env pinning |
| transient states (spinner/modal/focus/active-window) | pixel-only or vanish before a stable node | multi-window run: 208 focus mentions, unresolved |
| validation/error gestalt | error *count* + which field flagged is a visual summary | Tricentis wizard "7" badge + pink Make + callout |

## Integration shape (recommended, minimal)

1. **Response hint** — extend the existing screenshot-hint layer
   (`evidence_missing_hint` / `screenshot_request_path`) so a successful screenshot
   step returns `{screenshot_path, visual_hint}`. No new tool for the default path.
2. **Guidance topic** — reuse `get_locator_guidance` with `library="visual"` (mirrors
   the `requests` cookbook branch); content = the case table + dual-read-back + the
   multimodal/file-access caveats. Agents already pull this tool (9/9 in the API
   naming spike), so discoverability is proven.
3. **`visual_check` tool** — the only genuinely new surface: one unified capture over
   the 4 libraries (page or `selector`/`descriptor`), `return_image: bool = false`.
   `false` → `{path, size}` text; `true` → a FastMCP `Image` block + path. Honors
   `ROBOTMCP_SCREENSHOT_MODE`.
4. **Docstrings** — one shared sentence across the 4 screenshot keywords + the new
   tool, so the capability self-advertises.

## Naming rationale

The change is *named* "guidance" because the load-bearing 80% is naming + docstrings +
the guidance topic (token-cheap, model-agnostic). The `visual_check` image return is
the escape hatch, not the headline — keeping the framing honest about where the value
and the cost sit.

## Risks / boundaries

- **Determinism.** Vision judgments are non-deterministic (fonts, anti-aliasing) — it
  is a *fallback/complement oracle*, never a CI-stable primary assertion. The guidance
  says so.
- **Screenshot reliability.** Headless/Wayland capture has failed before (PlatynUI
  screenshot issues); the feature inherits those and must degrade (missing file →
  the existing evidence-missing hint).
- **PII in screenshots.** Documented caveat (screenshots may capture sensitive
  content); no redaction in v1.
- **Appium** covered by shared naming/guidance only — no mobile scenario ran; flagged
  inference.

## What v1 explicitly leaves out

Auto-attaching images to normal tool responses; server-side OCR/vision; visual
regression/baseline diffing; redaction. These are deliberate non-goals so v1 stays a
thin, token-honest discoverability layer over capture that already exists.

## De-risking experiments (before/with implementation)

1. A multimodal-driver docker run (Claude/GPT-4o via a key) on the Tricentis wizard:
   does the agent, *given the guidance*, pull the screenshot and correctly report "7
   required fields, Make flagged" — and does it *avoid* over-using vision where
   `Get Text` suffices? (measures both value and cost discipline)
2. A text-only-driver run (MiniMax-M3): confirm `visual_check` returns a path string
   and the run does not break (graceful degradation).
3. Token-cost measurement: average tokens/run with vision on-demand vs a naive
   always-attach baseline, to confirm the default stays cheap.
