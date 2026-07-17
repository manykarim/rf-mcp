"""Unit tests: lazy + torch-free + offline-safe keyword embeddings
(change: lazy-offline-embeddings)."""
import sys
import pytest

# numpy is a transitive dep of the (optional) embedding extras. The optional
# dependency matrix installs a single extra without it, so skip cleanly rather
# than erroring at collection — the cosine path under test is inert without it.
np = pytest.importorskip("numpy")

from robotmcp.components.keyword_matcher import KeywordMatcher


def test_construct_loads_no_model_and_no_torch():
    # No embedder at construction; and importing/using the matcher pulls no torch.
    m = KeywordMatcher()
    assert m.embeddings_model is None
    assert m._embeddings_attempted is False
    assert "torch" not in sys.modules
    assert "sentence_transformers" not in sys.modules


def test_flag_off_uses_lexical_fallback(monkeypatch):
    monkeypatch.delenv("ROBOTMCP_SEMANTIC_KEYWORDS", raising=False)
    m = KeywordMatcher()
    assert m._ensure_embeddings() is None      # no embedder
    assert m.embeddings_model is None
    assert m._embeddings_attempted is True      # fire-once recorded
    assert "torch" not in sys.modules


def test_flag_variants_enable(monkeypatch):
    for v in ("1", "true", "YES", "on"):
        monkeypatch.setenv("ROBOTMCP_SEMANTIC_KEYWORDS", v)
        assert KeywordMatcher._semantic_enabled() is True
    monkeypatch.setenv("ROBOTMCP_SEMANTIC_KEYWORDS", "false")
    assert KeywordMatcher._semantic_enabled() is False


class _FakeBackend:
    is_available = True
    backend_name = "model2vec"


class _FakeSvc:
    def __init__(self, backend): self._b = backend
    def _ensure_model(self): pass
    def encode_texts(self, texts): return np.ones((len(texts), 4), dtype=float)


def _patch_backend(monkeypatch, *, available=True, raise_on_load=False):
    import robotmcp.domains.memory.aggregates as agg
    import robotmcp.domains.memory.services as svcmod
    b = _FakeBackend(); b.is_available = available
    monkeypatch.setattr(agg.EmbeddingBackend, "detect", classmethod(lambda cls, *a, **k: b))
    if raise_on_load:
        class _Boom(_FakeSvc):
            def _ensure_model(self): raise RuntimeError("offline: model not cached")
        monkeypatch.setattr(svcmod, "EmbeddingService", _Boom)
    else:
        monkeypatch.setattr(svcmod, "EmbeddingService", _FakeSvc)


def test_flag_on_selects_torch_free_backend(monkeypatch):
    monkeypatch.setenv("ROBOTMCP_SEMANTIC_KEYWORDS", "1")
    _patch_backend(monkeypatch)
    m = KeywordMatcher()
    svc = m._ensure_embeddings()
    assert svc is not None
    assert m._embed_backend == "model2vec"       # torch-free default
    assert "torch" not in sys.modules


def test_flag_on_no_backend_degrades(monkeypatch):
    monkeypatch.setenv("ROBOTMCP_SEMANTIC_KEYWORDS", "1")
    _patch_backend(monkeypatch, available=False)
    m = KeywordMatcher()
    assert m._ensure_embeddings() is None
    assert m.embeddings_model is None


def test_flag_on_offline_load_error_degrades(monkeypatch):
    monkeypatch.setenv("ROBOTMCP_SEMANTIC_KEYWORDS", "1")
    _patch_backend(monkeypatch, raise_on_load=True)
    m = KeywordMatcher()
    # load raises (simulated offline) -> caught -> None, no hang, no raise
    assert m._ensure_embeddings() is None
    assert m.embeddings_model is None


def test_cosine():
    a = np.array([1.0, 0, 0]); b = np.array([1.0, 0, 0]); c = np.array([0.0, 1, 0])
    assert KeywordMatcher._cosine(a, b) == pytest.approx(1.0)
    assert KeywordMatcher._cosine(a, c) == pytest.approx(0.0)
    assert KeywordMatcher._cosine(a, np.zeros(3)) == 0.0
