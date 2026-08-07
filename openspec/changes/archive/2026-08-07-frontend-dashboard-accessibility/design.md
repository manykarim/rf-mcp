## Context
The session list is a set of non-focusable `<article>`s selected only by mouse click; base.css has no
focus styling; the step card sets `touch-action: none`; no reduced-motion support.

## Goals / Non-Goals
**Goals:** keyboard-select a session, visible focus, reduced-motion, touch scroll — verifiable via the
probe. **Non-Goals:** pass/fail text alternative and aria-live verbosity (depend on the keyed-render
rewrite; recorded as follow-ups).

## Decisions
**D1 —** cards become `role="button" tabindex=0` with `aria-pressed`/`aria-label` and an Enter/Space
keydown handler. **D2 —** one global `:focus-visible` outline rule. **D3 —** `touch-action: none` →
`pan-y`. **D4 —** a `prefers-reduced-motion` block neutralizing animations/transitions.

## Risks / Trade-offs
- **[pan-y vs drag]** vertical scroll is restored; the custom pointer-drag still captures horizontal
  movement. **[Deferred a11y items]** explicitly recorded.
