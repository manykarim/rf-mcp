# Proposal: lazy-offline-embeddings

## Why

`find_keywords(strategy="semantic")` is wired to a **~2 GB `sentence-transformers`
(torch)** backend, and installing it via `uv sync --all-extras` makes the server
**download a model from HuggingFace at startup** (eager, module-import time, gated
only by package presence). A dependency-hygiene review + an empirical ranking
benchmark (2026-07-17) show the torch dependency is **not justified** and the
startup download is a bug.

**The benchmark (`experiments/keyword-ranking-benchmark/`, 544 real RF keywords,
30 intent queries; recall@5 is the headline — agents scan the returned list, they
do not trust rank #1):**

| Method | Deps / size | top-1 | recall@5 | MRR |
|---|---|---:|---:|---:|
| `difflib` (current live fallback) | 0 (stdlib) | 0.57 | 0.80 | 0.67 |
| `token_overlap` (no-dep Jaccard) | 0 (stdlib) | 0.70 | 0.90 | 0.80 |
| **`model2vec`** (static, already in `[memory]`) | **292 KB**, no torch | 0.67 | **0.97** | 0.78 |
| `fastembed` (bge-small ONNX) | ~150 MB, no torch | 0.83 | 0.97 | 0.90 |
| `sentence-transformers` | **~2 GB (torch)** | *≈ fastembed (same weights via ONNX)* |

Reads:
- **Embeddings earn a place** over the no-dep floor (recall@5 0.80 → 0.97).
- **Torch is strictly dominated:** `fastembed` matches it (same model, no torch), and
  **`model2vec` ties recall@5 (0.97) at 292 KB** — 1/500th the size, already vendored,
  ~1 s load. For the list-scanning usage pattern, model2vec is as good as the best
  embedder.
- The `semantic` feature is a **1.6 %-of-calls, non-load-bearing path** (usage
  analysis across 48 eval runs); OBS-30 already records that "the docstring
  over-promised." And a torch-free 3-tier backend abstraction
  (`EmbeddingBackend.detect()`: model2vec → fastembed → sentence-transformers)
  **already exists and ships** in the memory domain — the keyword-ranking path just
  doesn't use it, hardcoding `SentenceTransformer` instead.

An **external MCP server is not the answer** (evaluated, rejected): MCP has no
server-to-server call path, so a core in-process tool consulting an external vector
MCP would *add* a client dependency + a hot-path network hop, break the single-call
contract (candidates fused with RF metadata), fragment the dynamic per-session
corpus, and add turns — the opposite of the goal.

**Latent regression:** because torch/ST are not in the default tree, keyword ranking
is *already silently on difflib*. Swapping in model2vec is therefore a **net upgrade**
over today's live behavior, not merely a size reduction.

## What Changes

- **Swap the keyword-ranking embedder to the torch-free in-tree backend (primary).**
  `keyword_matcher.py` SHALL obtain its embedder from the existing
  `EmbeddingBackend.detect()` / memory encode dispatch instead of hardcoding
  `SentenceTransformer('all-MiniLM-L6-v2')`. Default to **model2vec** (292 KB, already
  a `[memory]` dependency, recall@5 = 0.97); use `fastembed` if present; fall back to
  the lexical path (difflib/token-overlap) when no embedder is available.
- **Demote `sentence-transformers`/torch to an optional "max-quality" opt-in.** It is
  no longer the required or default semantic path; the `semantic` extra's docstring is
  corrected (fastembed/model2vec are the recommended torch-free backends). No hard
  torch dependency remains on the ranking path.
- **Lazy + flag-gated + offline-safe (still applies to whatever backend is chosen).**
  No embedding model is constructed at module import or in `KeywordMatcher.__init__`;
  it loads on first `find_keywords(strategy="semantic")` use, only when enabled
  (`ROBOTMCP_SEMANTIC_KEYWORDS`, off by default), and every HF-backed load
  (model2vec/fastembed/ST + the memory model2vec load) fails fast + degrades when the
  model can't be fetched (honor `HF_HUB_OFFLINE`) — never blocking startup or a call.
- **Upgrade the zero-dep floor (secondary, cheap):** the no-embedding fallback SHOULD
  use the token-overlap scorer (recall@5 0.90) alongside difflib (0.80) — a free lift
  when no embedder is enabled.
- **Packaging + docs hygiene (secondary):** note that `uv sync --all-extras` pulls the
  heavy `semantic`/torch extra that `[all]` deliberately excludes; dedupe the two `dev`
  lists; trim the `pydantic-ai` dev extras (dev-only ~150 MB from
  bedrock/cohere/temporal/…).

Out of scope: removing the semantic feature (it earns +17 pts recall@5 — kept, on a
light backend); calibrating cosine-vs-lexical pooling or embedding-aware reranking
(usage evidence says it is not on the critical path); Browser/Playwright footprint;
the memory feature's off-by-default state.

## Capabilities

### New Capabilities

- `lazy-offline-embeddings`: semantic keyword ranking runs on a torch-free embedder
  (default model2vec, already in-tree; fastembed optional) selected via the shared
  `EmbeddingBackend.detect()` abstraction, loaded lazily and only when explicitly
  enabled (never at import, never from mere package presence), with every model load
  offline-safe (fail-fast + degrade, honoring `HF_HUB_OFFLINE`) — so no default or
  `--all-extras` startup pulls torch or blocks on a HuggingFace download, and
  `sentence-transformers` is an optional max-quality opt-in rather than the required
  path.

### Modified Capabilities

- None (the difflib/token fallback and returned metadata contract are unchanged; the
  change replaces the embedding backend and its activation semantics).

## Impact

- `src/robotmcp/components/keyword_matcher.py:13-18, 359-392` — remove the hardcoded
  `SentenceTransformer` import/construct; obtain the embedder from
  `EmbeddingBackend.detect()` (reuse `domains/memory/services.py` encode dispatch);
  lazy `_ensure_embeddings()` + `ROBOTMCP_SEMANTIC_KEYWORDS` gate; token-overlap floor.
- `src/robotmcp/server.py:1041` — `KeywordMatcher()` global performs no model load /
  network at import.
- `src/robotmcp/domains/memory/aggregates.py:183` / `services.py:88-91` — offline-safe
  `from_pretrained` (honor `HF_HUB_OFFLINE`, fail fast) for the shared backend load.
- `pyproject.toml` — `semantic` extra demoted/re-described (torch-free backends
  preferred); `memory` (model2vec) is the effective ranking backend; dedupe `dev`;
  trim `pydantic-ai`. `README.md` — `--all-extras` note + corrected semantic docs.
- Tests: `tests/unit/` — `KeywordMatcher()` constructs no model; ranking uses model2vec
  when available and the lexical floor otherwise; flag-off = no torch import; offline
  degrades without hanging; the OBS-30 docstring test updated for the new backend.
- Evidence: `experiments/keyword-ranking-benchmark/` (harness + RESULTS).
- Acceptance: `HF_HUB_OFFLINE=1` + `--all-extras` startup makes **zero** HF calls and
  no hang; `find_keywords(strategy="semantic")` ranks via model2vec (or the lexical
  floor) with the measured recall — with **no torch installed**.
