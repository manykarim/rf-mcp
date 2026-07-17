# Tasks: lazy-offline-embeddings

## 1. Torch-free backend swap (primary)
- [x] 1.1 `KeywordMatcher` obtains its embedder from the shared `EmbeddingBackend.detect()` abstraction (aggregates.py:183) / a thin `encode(texts)` helper factored from `domains/memory/services.py`, instead of hardcoding `SentenceTransformer` (keyword_matcher.py:13-18, 369)
- [x] 1.2 Default backend resolves to **model2vec** (already in `[memory]`; benchmark recall@5=0.97); fastembed if present; lexical floor if neither. `sentence-transformers` stays last in the detect chain as an optional max-quality opt-in — never required, never default
- [x] 1.3 Remove the hard `sentence-transformers` gate as the semantic path; no torch import remains on the ranking path
- [x] 1.4 `pyproject.toml`: demote/re-describe the `semantic` extra (torch-free backends preferred); ensure model2vec is the effective ranking backend (it's in `[memory]`); update the OBS-30 docstring

## 2. Lazy + flag-gated activation
- [x] 2.1 No embedder constructed in `KeywordMatcher.__init__` or at module import; lazy `_ensure_embeddings()` fire-once on first semantic use
- [x] 2.2 `ROBOTMCP_SEMANTIC_KEYWORDS` gate (default off): unset/false → token_overlap + difflib floor, no model load / import; one-time "installed but not enabled" hint
- [x] 2.3 Confirm `server.py:1041` `KeywordMatcher()` global triggers no model load / network at import

## 3. Offline safety (all HF-backed loads)
- [x] 3.1 Honor `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE`; catch load/network errors fast → lexical floor (never hang); one-line log — for model2vec/fastembed/ST in the matcher
- [x] 3.2 Same guard for the memory `model2vec.StaticModel.from_pretrained` (aggregates.py:187-200 / services.py:88-91)

## 4. Upgrade the zero-dep floor (secondary, cheap)
- [x] 4.1 Use the token-overlap scorer (Jaccard; benchmark recall@5=0.90) alongside difflib (0.80) in the no-embedding fallback path

## 5. Packaging + docs hygiene (secondary)
- [x] 5.1 README/pyproject note: `uv sync --all-extras` pulls the heavy `semantic`/torch extra that `[all]` excludes; recommend `[all]` or specific extras for a torch-free install
- [x] 5.2 Dedupe the two `dev` definitions into one source of truth
- [x] 5.3 Trim `pydantic-ai` to only the extras the 3 test files use (drops boto3/botocore, temporalio, fastavro, cohere, grpc from dev/CI)

## 6. Tests
- [x] 6.1 `KeywordMatcher()` constructs WITHOUT loading a model (assert no model post-init even when a backend is importable)
- [x] 6.2 `find_keywords(strategy="semantic")` with `ROBOTMCP_SEMANTIC_KEYWORDS` unset → token/difflib floor, no embedder construction, no torch import
- [x] 6.3 Flag on → ranks via model2vec (detect default); assert the backend used is torch-free
- [x] 6.4 Offline (simulated) with flag on → degrades to floor, does not raise/hang
- [x] 6.5 Memory `from_pretrained` failure → degrades cleanly (offline path)
- [x] 6.6 OBS-30 docstring test updated for the new default backend; existing semantic-ranking tests pass with a stub backend

## 7. Acceptance validation (offline)
- [x] 7.1 `HF_HUB_OFFLINE=1` in an `--all-extras` env with NO torch installed: `python -m robotmcp.server` starts with ZERO HF calls and no hang; `find_keywords(strategy="semantic")` ranks via model2vec at the benchmark recall — capture as evidence
