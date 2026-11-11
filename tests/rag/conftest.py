"""Tests de l'assistant RAG : aucun appel réseau, aucune clé, index et sources dans des dossiers temporaires."""
import hashlib

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def isolated_rag(tmp_path, monkeypatch):
    """Index et sources temporaires ; clés retirées : aucun test ne peut appeler Voyage ou le proxy LiteLLM."""
    monkeypatch.setenv("RAG_INDEX_DIR", str(tmp_path / "index"))
    monkeypatch.setenv("RAG_SOURCES_DIR", str(tmp_path / "sources"))
    monkeypatch.setenv("AIDGE_ASSISTANT_AUTO_INGEST", "0")
    for var in ("VOYAGE_API_KEY", "LITELLM_MASTER_KEY", "EMBEDDINGS_BACKEND", "AIDGE_UI_BIND", "AIDGE_UI_HOST",
                "AIDGE_ASSISTANT_TRUSTED_AUTH", "VOYAGE_RPM", "VOYAGE_TPM"):
        monkeypatch.delenv(var, raising=False)
    import rag.retrieval
    monkeypatch.setattr(rag.retrieval, "_RETRIEVER", None)
    yield


class FakeEmbedder:
    """Embedder déterministe hors ligne : sac de mots haché sur 64 dimensions (des textes proches se ressemblent)."""

    backend = "local"

    def __init__(self, model="faux-modele"):
        self.model = model
        self.calls = 0

    def _vec(self, text):
        from rag.store import tokenize
        v = np.zeros(64, dtype=np.float32)
        for w in tokenize(text):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1
        n = np.linalg.norm(v)
        return v / n if n else v + 1e-3

    def embed_documents(self, texts):
        self.calls += 1
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()
