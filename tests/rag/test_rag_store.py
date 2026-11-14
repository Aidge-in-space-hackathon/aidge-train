"""Index : fusion RRF, BM25, Chroma, versions et manifeste."""
import pytest

from rag import config, store
from rag.chunking import Block, chunk_blocks


def test_rrf_scores_and_order():
    fused = store.rrf([["a", "b", "c"], ["b", "c", "d"]], k=60)
    scores = dict(fused)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["d"] == pytest.approx(1 / 63)
    assert [i for i, _ in fused] == ["b", "c", "a", "d"]


def test_rrf_ties_keep_first_seen_and_empty_inputs():
    assert [i for i, _ in store.rrf([["x"], ["y"]])] == ["x", "y"]
    assert store.rrf([]) == [] and store.rrf([[], []]) == []
    assert store.rrf([["a"]])[0][1] == pytest.approx(1 / (config.RRF_K + 1))


def test_tokenize_strips_accents_and_stopwords():
    assert store.tokenize("Comment fonctionne la GDN ? Le réseau généralisé") == \
        ["fonctionne", "gdn", "reseau", "generalise"]


def corpus():
    blocks = [Block("2 GDN", 3, "La normalisation divisive généralisée GDN divise chaque canal par une combinaison "
                                "des énergies des canaux voisins, comme dans le cortex visuel."),
              Block("4 Entropie", 6, "Le codeur arithmétique compresse les coefficients quantifiés avec un modèle "
                                     "de densité appris par morceaux linéaires."),
              Block("Export", None, "Aidge exporte le graphe en C pour la cible embarquée avec LoCod.")]
    chunks = []
    for b, name in zip(blocks, ["balle.pdf", "balle.pdf", "docs/x.md"], strict=True):
        chunks += chunk_blocks([b], name, "s")
    return chunks


def test_write_and_load_index(fake_embedder):
    chunks = corpus()
    vecs = fake_embedder.embed_documents([c.embed_text() for c in chunks])
    version = store.write_index(chunks, {"local": vecs}, {"sources_hash": "h", "backends": {"local": {"model": "m"}}})
    manifest = store.read_manifest()
    assert manifest["sources_hash"] == "h" and manifest["chunks"] == 3
    assert manifest["backends"] == {"local": {"model": "m"}}
    assert store.current_version() == version
    index = store.Index(version)
    assert index.backends() == ["local"]
    assert index.bm25_search("Comment fonctionne la GDN ?", 5)[0] == chunks[0].chunk_id
    assert index.bm25_search("le la les", 5) == []                 # que des mots vides
    q = fake_embedder.embed_query("codeur arithmétique densité")
    assert index.vector_search("local", q, 2)[0] == chunks[1].chunk_id
    assert index.by_id[chunks[2].chunk_id].source == "docs/x.md"


def test_new_version_replaces_current_and_old_ones_are_pruned(fake_embedder):
    chunks = corpus()
    vecs = fake_embedder.embed_documents([c.embed_text() for c in chunks])
    versions = []
    for i in range(4):
        versions.append(store.write_index(chunks[: i % 3 + 1], {"local": vecs[: i % 3 + 1]}, {"sources_hash": str(i)}))
    assert store.read_manifest()["sources_hash"] == "3"
    kept = sorted(p.name for p in config.index_dir().glob("v-*"))
    assert versions[-1].name in kept and len(kept) <= store.KEEP_VERSIONS


def test_no_index():
    assert store.current_version() is None and store.read_manifest() is None
