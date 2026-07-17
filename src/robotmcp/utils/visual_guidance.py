"""Visual-validation guidance (change: visual-inspection-guidance).

Single source for the ``get_locator_guidance(library="visual")`` topic: teaches an
agent WHEN a screenshot beats the DOM/ARIA/AT-SPI tree, the dual read-back pattern,
and the multimodal + file-access caveats. Token-cheap text; no images here.

The value is delivered mostly through discoverability (naming + this guidance),
not new machinery: rf-mcp already captures screenshots to disk — the agent just
needs to know it can read them for the checks structured trees cannot answer.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# Shared canonical text (also referenced by the screenshot success-hint so the two
# cannot drift).
VISUAL_HINT_ONE_LINER = (
    "read it to validate visually (multimodal model) — use for checks the DOM/ARIA "
    "can't do: canvas/image-rendered text, layout/overlap, obscured-but-'clickable' "
    "elements, color/theme, charts, validation gestalt; prefer Get Text when the "
    "value is in the DOM."
)


def build_visual_cookbook(
    error_message: Optional[str] = None, keyword_name: Optional[str] = None
) -> Dict[str, Any]:
    """Return the visual-validation cookbook payload (tips/warnings/cases)."""
    cases = [
        "DESKTOP/GTK TEXT: AT-SPI exposes Text.CharacterCount (a count), never the "
        "glyphs — a screenshot reads the actual text (e.g. a LibreOffice/GTK body).",
        "CANVAS / unlabeled SVG charts: no addressable text node at all — vision is "
        "the only channel.",
        "RENDERED <img> correctness: the DOM has src/alt, but a 404/placeholder still "
        "has a valid src — only a screenshot confirms the right image painted.",
        "LAYOUT / OVERLAP / CLIPPING: visibility checks are style-truth, not "
        "composited-truth — an element can be 'visible+clickable' yet covered by a "
        "modal/overlay a user can't click through.",
        "COLOR / THEME / CONTRAST: ARIA/AT-SPI carry no color; CSS gives declared, "
        "not composited, color.",
        "TRANSIENT STATES: spinners, progress overlays, modal dimming, focus rings, "
        "which-window-is-front — pixel-only or gone before a stable node exists.",
        "VALIDATION GESTALT: an error-count badge + which field is flagged is a "
        "one-glance visual summary vs. introspecting every field.",
    ]
    dual_read_back = [
        "1. Assert with Get Text / an attribute read FIRST when the value lives in a "
        "node (exact, deterministic; OCR can misread 61.000,00 as 81.000,00).",
        "2. Screenshot to CONFIRM the structured pass, or for a human-auditable artifact.",
        "3. Screenshot as PRIMARY only when NO node exposes the answer (the cases above).",
    ]
    warnings = [
        "Requires a MULTIMODAL driving model — a text-only model cannot see the image.",
        "The default path is disk: your agent must have FILE ACCESS to the screenshot "
        "directory (ROBOTMCP_SCREENSHOT_DIR). If it does not, call the visual_check "
        "tool with the image-return option instead.",
        "Vision is a fallback/complement oracle, NOT a deterministic CI primary — "
        "fonts/anti-aliasing make it non-deterministic. Gate CI on structured asserts.",
        "Screenshots may capture PII/sensitive content — treat artifacts accordingly.",
    ]
    return {
        "when_vision_beats_the_tree": cases,
        "dual_read_back": dual_read_back,
        "how_to_get_the_image": (
            "Screenshot keywords save to disk and the response reports the path — read "
            "that file if your model is multimodal and has file access. Otherwise call "
            "visual_check(..., return_image=true) for an inline image content block."
        ),
        "warnings": warnings,
    }
