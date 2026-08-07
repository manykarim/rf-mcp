## Why

The core flow is unusable without a mouse (validation 2026-08-07): the session list is not focusable and
the stylesheet has **zero focus indicators**, so a keyboard/screen-reader user can never select a session
(WCAG 2.1.1 / 4.1.2 / 2.4.7). `touch-action: none` on the whole step card blocks touch scrolling, and
there is no `prefers-reduced-motion` support.

## What Changes

- Make session cards keyboard-operable: `role="button"`, `tabindex=0`, `aria-pressed`, `aria-label`, and
  Enter/Space activation.
- Add a global `:focus-visible` outline so keyboard focus is visible everywhere.
- Relax `touch-action` to `pan-y` so the panel scrolls on touch.
- Add a `prefers-reduced-motion` block.

Deeper a11y items (pass/fail text alternative; `aria-live` re-announcing whole rebuilt containers) are
recorded as follow-ups — they depend on the keyed-render rewrite tracked in the rendering change.

## Capabilities

### Modified Capabilities

- `frontend-dashboard`: add an accessibility requirement — the core flow is keyboard-operable with visible
  focus, and motion/scroll respect user constraints.

## Impact

- `static/frontend/app.js` (card semantics + keyboard), `static/frontend/base.css` (focus-visible,
  reduced-motion, touch-action).
