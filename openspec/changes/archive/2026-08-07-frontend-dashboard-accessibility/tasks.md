## 1. Keyboard operability
- [x] 1.1 Session cards: `role=button`, `tabindex=0`, `aria-pressed`, `aria-label`, Enter/Space activation. (H5)

## 2. Focus, motion, touch
- [x] 2.1 Global `:focus-visible` outline in base.css.
- [x] 2.2 `touch-action: none` -> `pan-y` (restore touch scroll).
- [x] 2.3 `prefers-reduced-motion` block.

## 3. Deferred (recorded follow-ups)
- [~] 3.1 Pass/fail text alternative + non-outcome events not painted success-green.
- [~] 3.2 aria-live re-announces whole rebuilt containers (fix with keyed render).

## 4. Verify + wrap-up
- [x] 4.1 Live probe: card tabindex=0/role=button; Enter selects the session; focus-visible + reduced-motion
  rules present. PASS.
- [x] 4.2 `openspec validate frontend-dashboard-accessibility --strict` passes.
