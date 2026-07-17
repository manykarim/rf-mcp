# Design: lazy-offline-embeddings

## Empirical basis (why the backend swap, not just lazy-gating)

Benchmark (`experiments/keyword-ranking-benchmark/`, 544 real RF keywords, 30 intent
queries; recall@5 headline — agents scan the list):

| Method | size | top-1 | recall@5 | MRR |
|---|---|---:|---:|---:|
| difflib (current live fallback) | 0 | 0.57 | 0.80 | 0.67 |
| token_overlap | 0 | 0.70 | 0.90 | 0.80 |
| **model2vec** (in `[memory]`) | **292 KB** | 0.67 | **0.97** | 0.78 |
| fastembed (bge ONNX) | ~150 MB | 0.83 | 0.97 | 0.90 |
| sentence-transformers | ~2 GB torch | ≈ fastembed (same weights) |

Conclusions that drive the design:
- Embeddings beat the lexical floor (recall@5 0.80 → 0.97) → keep semantic ranking.
- model2vec ties fastembed on recall@5 at 1/500th the size and is already vendored →
  **default to model2vec**; torch buys nothing → **demote sentence-transformers**.
- token_overlap > difflib at zero cost → **upgrade the no-embedding floor** too.

## Primary change — reuse the memory domain's backend abstraction

The memory domain already ships the exact abstraction the keyword matcher needs:
`EmbeddingBackend.detect()` (aggregates.py:183) probes **model2vec → fastembed →
sentence-transformers** and returns a configured backend; `EmbeddingService` (services.py)
dispatches `.encode()` per backend. `keyword_matcher.py` currently ignores this and
hardcodes `SentenceTransformer('all-MiniLM-L6-v2')` gated on an ST import.

The swap: `KeywordMatcher` obtains its embedder from `detect()` (or a thin shared
`encode(texts)` helper factored from the memory service), so both domains share one
torch-free backend path. Default resolves to model2vec (present via `[memory]`); if
only fastembed is installed, that; if neither, the lexical floor. `sentence-transformers`,
if present, is still usable (last in the detect chain) as a max-quality opt-in — but is
never required and never the default.

## Activation model — mirror the memory subsystem (lazy + flag)

The defect today: `KeywordMatcher.__init__` (keyword_matcher.py:369) eagerly builds the
model whenever the package imports, and `server.py:1041` instantiates the matcher at
module import → import-time network + torch load. Fix mirrors memory
(`ROBOTMCP_MEMORY_ENABLED`, off by default, lazy `create_memory_services`):

```
  find_keywords(strategy="semantic")
     ├─ ROBOTMCP_SEMANTIC_KEYWORDS not truthy?  → token_overlap + difflib floor (no model, no HF)
     └─ enabled → _ensure_embeddings() (lazy, once)
                   ├─ detect() backend available? no → lexical floor + one-time hint
                   ├─ HF reachable / model cached?  no → fail fast + floor (never hang; honor HF_HUB_OFFLINE)
                   └─ yes → load once, cache on the instance
```

Two switches (installed AND enabled) + laziness (load on first semantic use, never at
import). With model2vec already in `[memory]`, "installed" is usually satisfied, so the
`ROBOTMCP_SEMANTIC_KEYWORDS` flag is the deliberate on-switch; default stays on the fast
lexical floor.

## Offline safety (all HF-backed loads)

model2vec `StaticModel.from_pretrained`, fastembed `TextEmbedding`, and (if used) ST all
hit HF on a cache miss. Guard each: honor `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE`
(local-cache-only), catch network/load errors fast → lexical floor with a one-line log,
never a hang. After first successful load the model is cached, so this only bites the
first enabled run — and the guard keeps even that from hanging.

## Secondary — packaging / hygiene

- **Docs**: `uv sync --all-extras` pulls the heavy `semantic`/torch extra that `[all]`
  deliberately excludes; recommend `[all]` or specific extras for a torch-free install.
  Correct the `semantic` docstring: torch-free model2vec/fastembed are the recommended
  backends; sentence-transformers is an optional max-quality path.
- **dev de-dup**: `[project.optional-dependencies].dev` and `[dependency-groups].dev`
  duplicate `pytest/openai/pydantic-ai/...` → one source of truth.
- **pydantic-ai trim**: used only in 3 test files but resolves with
  `[bedrock,cohere,temporal,google,groq,vertexai,…]` (boto3/botocore, temporalio,
  fastavro, cohere, grpc ~150 MB dev-only). Pin to the extras the tests actually use.

## Why NOT an external MCP (rejected option)

MCP has no server-to-server call path; a core in-process tool consulting an external
vector MCP would need an embedded MCP **client** + a hot-path network hop (adding, not
removing, a dependency), break the single-call "candidates + fused RF metadata"
contract, add turns (against turn-economy goals), and still own embedding + ingestion
of the **dynamic per-session** keyword corpus. It relocates and fragments the burden.
Option A (in-process light-backend swap) removes the weight cleanly with none of this.

## Risks / boundaries

- **Behavior change for `semantic` users relying on package-presence activation**: after
  this change they set `ROBOTMCP_SEMANTIC_KEYWORDS`; a one-time log explains it. The
  fallback ranking (now token_overlap + difflib) is unchanged in contract and improved.
- **model2vec vs fastembed default**: benchmark says model2vec ties on recall@5 → default
  model2vec (smallest, already present). If a future corpus shows a material recall gap,
  the detect() order can prefer fastembed; no code-shape change.
- No feature removed, no external server, no torch required.

## Acceptance

`HF_HUB_OFFLINE=1` + fresh `--all-extras` env: `python -m robotmcp.server` starts with
zero HF calls and no hang; `find_keywords(strategy="semantic")` (flag on) ranks via
model2vec with recall matching the benchmark; flag off uses the token/difflib floor; no
torch is installed or imported anywhere on the path.
