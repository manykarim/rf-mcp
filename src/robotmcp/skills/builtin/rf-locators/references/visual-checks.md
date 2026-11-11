# Visual checks (screenshots)

Ported from rf-mcp 0.36.0. Legacy source: main:src/robotmcp/utils/visual_guidance.py (`VISUAL_HINT_ONE_LINER`, `build_visual_cookbook`); server.py (`get_locator_guidance` with `library="visual"`, `visual_check`, `_screenshot_mode`); rf_native_type_converter.py (`get_platynui_locator_guidance` → `display_state_reading`); components/execution/desktop_guidance.py (`_LOCATOR_CRIB`).

## Rule
Prefer `Get Text` or an attribute read whenever the value is in the DOM, ARIA or AT-SPI tree. Use a screenshot only for checks the tree cannot answer.

## When a screenshot beats the tree
| Case | Why the tree fails |
|---|---|
| Desktop/GTK text | AT-SPI exposes `Text.CharacterCount`, never the glyphs (e.g. a LibreOffice or GTK body) |
| Canvas, unlabeled SVG charts | There is no addressable text node, so vision is the only channel |
| Rendered `<img>` correctness | A 404 or placeholder still has a valid `src` and `alt`. Only pixels prove the right image painted |
| Layout, overlap, clipping | "visible + clickable" describes style, not the composited result, so an element can sit under a modal the user cannot click through |
| Color, theme, contrast | ARIA and AT-SPI carry no color. CSS gives the declared color, not the composited one |
| Transient states | Spinners, progress overlays, modal dimming, focus rings, which window is in front. These are pixel-only or gone before a stable node exists |
| Validation gestalt | An error-count badge plus the flagged field is a one-glance summary, compared with inspecting every field |

## Dual read-back pattern
1. Assert with a structured read FIRST when a node holds the value. It is exact and deterministic, whereas OCR can misread `61.000,00` as `81.000,00`.
2. Screenshot to CONFIRM the structured pass, or to leave a human-auditable artifact.
3. Use the screenshot as the PRIMARY check only when no node exposes the answer.

## Capture keywords per library
| Library | Keyword line |
|---|---|
| Browser | `Take Screenshot    filename=/tmp/shot.png` |
| PlatynUI.BareMetal | `Take Screenshot    filename=/tmp/shot.png`. The first positional argument is a node descriptor, so always pass `filename=`. `filename=EMBED` embeds the image in the log |
| SeleniumLibrary, AppiumLibrary | `Capture Page Screenshot    /tmp/shot.png` |

The keyword saves to disk and reports the path; a multimodal agent with file access reads that file.

## Desktop last resort (PlatynUI)
If neither result Labels nor `native:Text.CharacterCount` can be read, take `Take Screenshot` and OCR it, e.g. tesseract via `Run Process`.

## Warnings
- Visual checks need a multimodal driving model. A text-only model cannot see the image.
- The agent needs file access to the screenshot directory.
- Vision is a fallback or complementary check, not a deterministic CI gate, because fonts and anti-aliasing vary. Gate CI on structured assertions.
- Screenshots can capture PII or sensitive content. Handle the artifacts accordingly.
