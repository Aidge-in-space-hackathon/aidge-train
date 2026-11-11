"""Embeddings : limiteur de débit Voyage, client Voyage simulé (429, quotas), cache par empreinte."""
import json

import httpx
import numpy as np
import pytest

from rag.embeddings import (
    EmbeddingCache,
    EmbeddingError,
    QuotaExceeded,
    RateLimiter,
    VoyageClient,
    VoyageEmbedder,
    embed_with_cache,
)


class Clock:
    def __init__(self):
        self.t = 1000.0
        self.waits = []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.waits.append(s)
        self.t += s


def test_rate_limiter_requests_per_minute():
    clock = Clock()
    lim = RateLimiter(rpm=3, tpm=10_000, clock=clock, sleep=clock.sleep)
    for _ in range(3):
        assert lim.acquire(10)
    assert clock.waits == []
    assert lim.acquire(10)                       # 4e requête : attend que la 1re sorte de la fenêtre
    assert clock.waits == [pytest.approx(60)]


def test_rate_limiter_tokens_per_minute_and_max_wait():
    clock = Clock()
    lim = RateLimiter(rpm=100, tpm=1000, clock=clock, sleep=clock.sleep)
    assert lim.acquire(800)
    clock.t += 10
    assert not lim.acquire(400, max_wait=3)      # il faudrait attendre 50 s : refus immédiat
    assert clock.waits == []
    assert lim.acquire(400)
    assert clock.waits == [pytest.approx(50)]


def voyage(handler, rpm=1000, tpm=1_000_000):
    clock = Clock()
    http = httpx.Client(base_url="https://voyage.test/v1", transport=httpx.MockTransport(handler))
    client = VoyageClient("cle-factice", limiter=RateLimiter(rpm, tpm, clock=clock, sleep=clock.sleep),
                          http=http, sleep=clock.sleep)
    return client, clock


def embeddings_handler(log, fail_first=0):
    state = {"n": 0}

    def handler(request):
        body = json.loads(request.content)
        log.append((request.url.path, body))
        state["n"] += 1
        if state["n"] <= fail_first:
            return httpx.Response(429, json={"detail": "rate limit"})
        if request.url.path.endswith("/rerank"):
            docs = body["documents"]
            order = sorted(range(len(docs)), key=lambda i: -len(docs[i]))[: body["top_k"]]
            return httpx.Response(200, json={"data": [{"index": i, "relevance_score": 1 - r / 10}
                                                      for r, i in enumerate(order)]})
        data = [{"index": i, "embedding": [float(len(t)), 1.0]} for i, t in enumerate(body["input"])]
        return httpx.Response(200, json={"data": data[::-1]})       # ordre quelconque : remis par index
    return handler


def test_voyage_embed_batches_by_token_budget_and_keeps_order():
    log = []
    client, _ = voyage(embeddings_handler(log), tpm=1000)           # lots de 600 tokens estimés
    texts = ["a" * 1200, "b" * 1200, "c" * 40]                       # 300, 300, 10 tokens
    vecs = client.embed(texts, "document")
    assert [v[0] for v in vecs] == [1200.0, 1200.0, 40.0]
    assert [len(b["input"]) for _, b in log] == [2, 1]
    assert {b["input_type"] for _, b in log} == {"document"} and log[0][1]["model"] == "voyage-4"


def test_voyage_backs_off_on_429_then_succeeds_during_ingestion():
    log = []
    client, clock = voyage(embeddings_handler(log, fail_first=2))
    assert len(client.embed(["texte"], "document")) == 1
    assert clock.waits == [20.0, 40.0]


def test_voyage_quota_exceeded_after_retries_and_at_query_time():
    client, _ = voyage(lambda r: httpx.Response(429, json={"detail": "quota"}))
    with pytest.raises(QuotaExceeded):
        client.embed(["texte"], "document")
    hits = []
    client, _ = voyage(lambda r: hits.append(1) or httpx.Response(429, json={"detail": "quota"}))
    with pytest.raises(QuotaExceeded):                               # question : aucune nouvelle tentative
        VoyageEmbedder(client).embed_query("question")
    assert hits == [1]


def test_voyage_other_errors():
    client, _ = voyage(lambda r: httpx.Response(401, json={"detail": "clé invalide"}))
    with pytest.raises(EmbeddingError, match="401"):
        client.embed(["x"], "query")
    with pytest.raises(EmbeddingError):
        VoyageClient("")


def test_voyage_rerank():
    log = []
    client, _ = voyage(embeddings_handler(log))
    order = client.rerank("q", ["court", "plus long", "le plus long de tous"], top_k=2)
    assert [i for i, _ in order] == [2, 1]
    assert log[0][1]["model"] == "rerank-2.5-lite" and log[0][1]["top_k"] == 2


def test_cache_embeds_only_new_texts(fake_embedder):
    cache = EmbeddingCache()
    v1, fresh1 = embed_with_cache(fake_embedder, ["un texte", "deux textes"], cache)
    v2, fresh2 = embed_with_cache(fake_embedder, ["un texte", "deux textes", "trois"], cache)
    assert (fresh1, fresh2) == (2, 1)
    assert np.allclose(v1[0], v2[0]) and np.allclose(v1[1], v2[1])
    _, fresh3 = embed_with_cache(fake_embedder, ["un texte"], cache)
    assert fresh3 == 0
    other = type(fake_embedder)(model="autre-modele")                # autre modèle : autre clé
    _, fresh4 = embed_with_cache(other, ["un texte"], cache)
    assert fresh4 == 1
