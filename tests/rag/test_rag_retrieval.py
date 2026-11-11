"""Recherche hybride : fusion BM25 + vecteurs, rerank, replis quand Voyage est indisponible."""
import pytest

from rag import store
from rag.chunking import Block, chunk_blocks
from rag.embeddings import QuotaExceeded
from rag.retrieval import IndexMissing, Retriever

TEXTS = [("2 GDN", "La normalisation divisive généralisée GDN divise chaque canal par les énergies voisines."),
         ("3 Quantification", "La quantification uniforme est remplacée par un bruit additif pendant l'entraînement."),
         ("4 Entropie", "Le codeur arithmétique utilise un modèle de densité par morceaux linéaires."),
         ("Export", "Aidge exporte le graphe en C pour la cible embarquée, puis LoCod génère les accélérateurs."),
         ("Mémoire", "Le plan mémoire statique compare le pic d'activations à la BRAM du Zynq.")]


def build(fake_embedder, backends=("local",)):
    chunks = []
    for i, (section, text) in enumerate(TEXTS):
        chunks += chunk_blocks([Block(section, i + 1, text)], "balle.pdf", "balle2017")
    vecs = fake_embedder.embed_documents([c.embed_text() for c in chunks])
    store.write_index(chunks, {b: vecs for b in backends},
                      {"sources_hash": "h", "backends": {b: {"model": "m"} for b in backends}})
    return chunks


class FakeVoyage:
    def __init__(self, embedder, fail_embed=False, fail_rerank=False):
        self.embedder, self.fail_embed, self.fail_rerank = embedder, fail_embed, fail_rerank
        self.rerank_calls = []

    def embed(self, texts, input_type, model=None, max_wait=None):
        if self.fail_embed:
            raise QuotaExceeded("quota")
        return [self.embedder.embed_query(t) for t in texts]

    def rerank(self, query, documents, top_k, model=None, max_wait=None):
        self.rerank_calls.append(len(documents))
        if self.fail_rerank:
            raise QuotaExceeded("quota")
        return [(i, 1.0 - i / 10) for i in reversed(range(len(documents)))][:top_k]


def test_missing_index():
    with pytest.raises(IndexMissing):
        Retriever().search("GDN")


def test_local_only_search_finds_relevant_passage(fake_embedder):
    chunks = build(fake_embedder)
    result = Retriever(local=fake_embedder).search("Comment fonctionne la GDN ?", k=2)
    assert result.methode == ["bm25", "local", "rrf"]
    assert result.passages[0].chunk.chunk_id == chunks[0].chunk_id
    assert len(result.passages) == 2
    tool = result.passages[0].to_tool()
    assert tool["source"] == "balle.pdf, p. 1 (2 GDN)" and tool["page"] == 1 and "GDN" in tool["texte"]


def test_voyage_embeddings_and_rerank_at_most_8_candidates(fake_embedder, monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "factice")
    build(fake_embedder, backends=("voyage", "local"))
    voyage = FakeVoyage(fake_embedder)
    result = Retriever(voyage=voyage, local=fake_embedder).search("GDN canal", k=3)
    assert result.methode == ["bm25", "voyage", "rrf", "rerank"]
    assert voyage.rerank_calls and max(voyage.rerank_calls) <= 8
    assert len(result.passages) == 3


def test_falls_back_to_local_vectors_then_rrf_order(fake_embedder, monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "factice")
    chunks = build(fake_embedder, backends=("voyage", "local"))
    voyage = FakeVoyage(fake_embedder, fail_embed=True, fail_rerank=True)
    result = Retriever(voyage=voyage, local=fake_embedder).search("GDN", k=2)
    assert result.methode == ["bm25", "local", "rrf"]
    assert len(result.notes) == 2 and "repli local" in result.notes[0] and "ordre RRF" in result.notes[1]
    assert result.passages[0].chunk.chunk_id == chunks[0].chunk_id


def test_bm25_only_when_no_vectors_available(fake_embedder):
    build(fake_embedder)

    class Broken:
        def embed_query(self, text):
            raise OSError("modèle absent")

    result = Retriever(local=Broken()).search("quantification bruit additif", k=1)
    assert result.methode == ["bm25", "rrf"] and "BM25 seul" in result.notes[0]
    assert result.passages[0].chunk.section == "3 Quantification"


def test_retriever_reloads_new_version(fake_embedder):
    build(fake_embedder)
    r = Retriever(local=fake_embedder)
    first = r.index()
    chunks = chunk_blocks([Block("Nouveau", 1, "Un passage tout neuf sur le rover lunaire et sa caméra embarquée.")],
                          "x.md", "s")
    store.write_index(chunks, {"local": fake_embedder.embed_documents([c.embed_text() for c in chunks])},
                      {"sources_hash": "h2", "backends": {"local": {"model": "m"}}})
    assert r.index() is not first and r.search("rover lunaire", k=1).passages[0].chunk.section == "Nouveau"
