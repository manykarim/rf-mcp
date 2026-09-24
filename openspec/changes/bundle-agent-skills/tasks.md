## 1. Packaging (depend on rf-agentskills)

- [ ] 1.1 Add an optional `skills` extra in `pyproject.toml` depending on `rf-agentskills` (pin a
  compatible range); document `pip install rf-mcp[skills]` / `uv sync --extra skills`.
- [ ] 1.2 Add a resolver that locates the installed rf-agentskills canonical `SKILL.md` folders
  (package resources path), with a clear degrade when the extra isn't installed (skills simply absent).

## 2. Installer: co-install skills per agent (files-first, 10/10 coverage)

- [ ] 2.1 Extend `src/robotmcp/onboarding/adapters.py` + `manifest.py` so the installer optionally writes
  skill folders per agent: `.agents/skills/` (Codex, Cursor, opencode, Kilo, Goose, pi) and
  `.claude/skills/` (Claude Code). Prefer calling rf-agentskills' installer/transforms as a library over
  re-implementing the matrix.
- [ ] 2.2 Preserve safety guarantees: opt-in, project/user scope selectable, non-clobbering merge, and a
  tracked hash manifest for clean uninstall. Add tests for the co-install + uninstall round-trip.

## 3. Serve skills over MCP (agent-agnostic)

- [ ] 3.1 Register `SkillsDirectoryProvider` (stable fastmcp v3) in `server.py` to serve the rf-agentskills
  folders as `skill://…` resources with sha256 `_manifest` digests (progressive disclosure).
- [ ] 3.2 Add a tools bridge for tools-only clients — `ResourcesAsTools`, or a minimal
  `list_skills`/`get_skill` pair — and signpost the skill index from the WORKFLOW-GUIDE instructions.
- [ ] 3.3 Gate all of §3 behind the `skills` extra / a config flag; no behavior change when skills absent.

## 4. Scope to non-duplicative knowledge

- [ ] 4.1 Audit each skill against rf-mcp's existing guidance (`get_locator_guidance`, tool docstrings,
  WORKFLOW GUIDE); exclude/reduce material overlaps so the bundle adds workflow knowledge, not repeats.
  - _Evidence from 5.3:_ the skills empirically **substitute** for rf-mcp (strong overlap — each recovers
    ~½ of rf-mcp's correctness gap standalone) and the **browser skill actively conflicts** with rf-mcp's
    live guidance via a concrete doc bug (`references/iframes-shadow-dom.md` teaches `>>` for iframes; must
    be `>>>`). Concrete audit action: fix that doc; sweep skills for stale keywords (a `Run Keyword If`
    deprecation in the requests skill sank one session); prefer rf-mcp's live guidance where they overlap.

## 5. Validate uplift (eval-gate)

- [x] 5.1 Ran a **cross-agent** control-vs-treatment harness (generalizes `rf-skill-eval`): control = rf-mcp
  only, treatment = rf-mcp + native SKILL.md; 3 agents (claude-fable-5, codex/gpt-5.6-sol@xhigh,
  opencode/MiniMax-M3@max), 4 scenarios, 4 reps = 96 real headless sessions; deterministic graders.
  Harness + report: `experiments/bundle-agent-skills-eval/` (`harness.py`, `report.md`, `summary.json`).
- [x] 5.2 Measured uplift **recorded, not asserted**, across **two model tiers** (`report.md` = strong:
  fable / gpt-5.6-sol@xhigh / MiniMax-M3@max; `report_small.md` = weak: haiku / gpt-5.6-luna / MiniMax-M2.5).
  **No primary-metric (correctness) uplift on any of the 24 cells in either eval** — control = 1.00 in every
  cell of both (joint **96/96 control reps**; ceiling robust to model tier) → **no capability gain claimed**.
  The weak-model run **refuted** the "weaker model reveals headroom" hypothesis (control moved 0.00) → the
  ceiling is **task-driven, not model-driven**. Net correctness effect over **192 sessions = −1** (one
  treatment regression, never a gain). One *conditional* efficiency pocket: the **browser** skill tames
  control over-exploration on models that over-explore (sol@xhigh −63% in_tok; haiku −35% turns / −41% cost)
  but is a wash on lean-control luna. keyword/results skills = net cost → prune candidates. **Central question
  still unanswerable *from those two saturated evals*: it needed an **rf-mcp-ABSENT arm + harder tasks**
  → run in 5.3 below (see `report_small.md` "Recommendation").
- [x] 5.3 **Decisive 2×2 factorial** `{rf-mcp on/off} × {skill on/off}` on **hard, non-saturated** tasks
  (iframe frame-pierce · nested-JSON RequestsLibrary · teardown-failure output.xml; weak models; 3 agents ×
  4 arms × 3 scenarios × 3 reps = **108 sessions**; `report_hard2x2.md`, `results_hard2x2.jsonl`). With real
  headroom (4 cells at `neither`=0.25) the effects finally separate: **rf-mcp has real correctness value**
  (main +0.38 headroom; 0.25→0.92, positive in all 4 headroom cells); **the skill has genuine *standalone*
  value** (`skill_only−neither` +0.33; 0.25→0.58) — the value the saturated evals masked; but **the skill is
  neutral-to-harmful *on top of* rf-mcp** (`both−mcp_only` ≤0 in all 9 cells, <0 in 3; −0.25 headroom;
  interaction −0.58). **rf-mcp and the skill are SUBSTITUTES, not complements** — stacking them is worse than
  rf-mcp alone. Root cause of the on-top harm traced from logs: the **browser skill's
  `references/iframes-shadow-dom.md` teaches the WRONG iframe combinator `>>` (must be `>>>`)**, displacing
  rf-mcp's correct live guidance. Direction robust (skill|mcp never >0 across 9 cells); magnitude directional
  at N=3 (5 `both` fails vs 1 `mcp_only`). **Answer to the change's core question: value *on top of* rf-mcp
  is ≤0 — do NOT co-bundle skill+rf-mcp; position the skill as an rf-mcp-ABSENT substitute.**
- [x] 5.4 **Confirmatory pass** (`report_conf.md`, `results_conf.jsonl`; 5 reps, 2 agents haiku+M2.5, codex
  dropped, + a NEW *non-iframe* hard Browser task `browser_dynamic` = **160 sessions**). **All eval#3 signs
  REPLICATE** (rf-mcp +0.10, skill‑standalone +0.05, skill‑on‑top −0.05, interaction −0.10) but every
  magnitude **shrinks** (rf-mcp effect to ~⅔, the skill/interaction effects to ~⅓): eval#3 overstated sizes
  because its iframe `neither` baseline was small‑N noise (claude 0.00@N3 → **0.80@N5**; the shrink *survives*
  restricting eval#3 to the same 2 agents, so it's not the codex drop). **The on‑top harm is proven
  iframe‑specific**: `both−mcp_only` = **−0.40 on iframe (both agents)** but **0.00 on the new non‑iframe
  `browser_dynamic`** (same Browser skill loaded → not a general on‑top penalty; it's the `>>` doc bug
  displacing the page's correct `>>>`). **Cleanest substitution demo — `opencode/browser_dynamic`: neither
  0.00 → mcp_only 1.00 AND skill_only 1.00 AND both 1.00** (rf-mcp OR skill each fully recovers a hard task;
  stacking adds zero). Skill standalone value is **content‑dependent** (+1.00 where correct+needed; −0.40
  where wrong). **Verdict: the §5.3 recommendation is hardened, not changed — ship rf-mcp; do NOT co-bundle;
  fix the browser skill's `>>>` iframe doc; position skills as rf-mcp-absent substitutes.** (Fact-check:
  trustworthy/minor-fixes, applied.)

## 6. Docs + wrap-up

- [ ] 6.1 Document the bundle (install, which agents get files vs MCP resources vs tool bridge, uninstall,
  the SEP-2640 forward-compat note, and the "no fastmcp-4 dependency" stance).
- [ ] 6.2 Full pytest green (skills off by default → no regression); `openspec validate bundle-agent-skills
  --strict` passes.

## 7. Track the standard (follow-on, not blocking)

- [ ] 7.1 When SEP-2640 merges: align `skill://` URI/method/digest details and declare the
  `io.modelcontextprotocol/skills` extension capability.
- [ ] 7.2 When fastmcp 4 reaches GA: take the modest migration (httpx2, pydantic≥2.12, import renames) for
  path-security hardening + response caching.
