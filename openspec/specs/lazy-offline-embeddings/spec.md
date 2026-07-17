# lazy-offline-embeddings Specification

## Purpose
TBD - created by archiving change lazy-offline-embeddings. Update Purpose after archive.
## Requirements
### Requirement: Semantic keyword ranking uses a torch-free backend by default
Semantic `find_keywords` ranking SHALL use a torch-free embedding backend by
default, selected via the shared `EmbeddingBackend.detect()` abstraction
(model2vec → fastembed → sentence-transformers). `sentence-transformers` (torch)
SHALL NOT be required for semantic ranking and SHALL NOT be the default path; it
remains an optional max-quality backend only. When no embedding backend is
available, ranking SHALL fall back to the lexical path (token-overlap + difflib).

#### Scenario: default ranking runs without torch
- **WHEN** semantic ranking is enabled and `model2vec` (or `fastembed`) is available but `sentence-transformers`/torch is not installed
- **THEN** ranking runs on the torch-free backend and produces embedding-based results — no torch import occurs

#### Scenario: no embedder available falls back to the lexical floor
- **WHEN** semantic ranking is enabled but no embedding backend is importable
- **THEN** ranking uses the token-overlap + difflib lexical path and returns results without error

### Requirement: The semantic embedding model is never loaded at import or construction
The server SHALL NOT construct the semantic `find_keywords` embedding model
(`SentenceTransformer`) at module import or in `KeywordMatcher.__init__`. The
model is loaded lazily on the first actual use of the semantic strategy, so
importing the server or constructing the matcher performs no model load and no
network access — even when the `sentence-transformers` extra is installed.

#### Scenario: constructing the matcher loads no model
- **WHEN** `KeywordMatcher()` is constructed (with `sentence-transformers` importable)
- **THEN** no embedding model is loaded and no network call is made; the model handle is unset until first semantic use

#### Scenario: server import performs no embedding network call
- **WHEN** the server module is imported (which instantiates the module-level matcher)
- **THEN** no HuggingFace download or model load occurs at import time

### Requirement: Semantic embedding activation requires an explicit flag, not mere installation
Semantic embedding ranking SHALL activate only when explicitly enabled via an
environment flag (`ROBOTMCP_SEMANTIC_KEYWORDS`), not from the `sentence-transformers`
package merely being present. When the extra is installed but the flag is off,
`find_keywords(strategy="semantic")` SHALL use the existing pattern + tag +
difflib fallback ranking and SHALL NOT import torch or load the model.

#### Scenario: installed but not enabled uses the fallback
- **WHEN** `sentence-transformers` is installed, `ROBOTMCP_SEMANTIC_KEYWORDS` is unset, and `find_keywords(strategy="semantic")` runs
- **THEN** ranking uses the difflib/tag fallback, no embedding model is loaded, and a one-time hint notes the extra is installed but not enabled

#### Scenario: enabled loads lazily on first use
- **WHEN** `ROBOTMCP_SEMANTIC_KEYWORDS` is truthy and `find_keywords(strategy="semantic")` is called for the first time
- **THEN** the embedding model loads once at that point (not before) and is reused for subsequent calls

### Requirement: HuggingFace model loads are offline-safe
Every HuggingFace-backed model load (the semantic `SentenceTransformer` and the
memory `model2vec` backend) SHALL fail fast and degrade to the non-embedding
fallback when the model cannot be fetched (offline, air-gapped, unreachable, or
`HF_HUB_OFFLINE` set with no local cache) — it SHALL NOT block startup or a tool
call on a network round-trip.

#### Scenario: offline enablement degrades without hanging
- **WHEN** `ROBOTMCP_SEMANTIC_KEYWORDS` is on, the model is not cached locally, and HuggingFace is unreachable (or `HF_HUB_OFFLINE=1`)
- **THEN** the load fails fast, a clear one-line message is logged, and `find_keywords(strategy="semantic")` returns via the difflib fallback — no hang

#### Scenario: memory load offline degrades cleanly
- **WHEN** memory is enabled but the `potion-base-8M` model cannot be fetched (offline)
- **THEN** memory initialization degrades (returns unavailable) and automation tool calls proceed normally, without blocking on the download
