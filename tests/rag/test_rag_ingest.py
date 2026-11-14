"""Ingestion de bout en bout (sources du dépôt, embedder simulé) et détection d'un index à jour."""
import pytest

import rag.ingest as ingest
from rag import config, store

EXPORT = "Aidge exporte le graphe en C pour la cible embarquée, puis LoCod génère les accélérateurs."


@pytest.fixture
def small_catalog(tmp_path, monkeypatch, fake_embedder):
    root = tmp_path / "repo"
    (root / "docs").mkdir(parents=True)
    (root / "docs" / "gdn.md").write_text("# GDN\n\nLa GDN divise chaque canal par les énergies voisines.\n\n"
                                          f"# Export\n\n{EXPORT}\n", encoding="utf-8")
    private = config.sources_dir() / "private"
    private.mkdir(parents=True)
    (private / "plaquette.txt").write_text("Document privé du challenge, à ne jamais publier.", encoding="utf-8")
    catalog = tmp_path / "sources.yaml"
    catalog.write_text("""sources:
  - {id: projet, visibility: public, kind: repo, files: ["docs/*.md"]}
  - {id: challenge, visibility: private, kind: local, dir: private, files: ["*.txt"]}
""", encoding="utf-8")
    monkeypatch.setattr(config, "CATALOG", catalog)
    monkeypatch.setattr(config, "ROOT", root)
    fake_embedder.model = config.LOCAL_EMBED_MODEL          # même nom de modèle que le vrai backend local
    monkeypatch.setattr(ingest, "LocalEmbedder", lambda: fake_embedder)
    return root


def test_ingest_builds_index_then_skips_when_fresh(small_catalog, fake_embedder):
    logs = []
    manifest = ingest.run("local", log=logs.append)
    assert manifest["chunks"] == 3 and set(manifest["backends"]) == {"local"}
    assert {s["visibility"] for s in manifest["sources"]} == {"public", "private"}
    calls = fake_embedder.calls
    again = ingest.run("local", if_stale=True, log=logs.append)
    assert again["created"] == manifest["created"] and fake_embedder.calls == calls
    assert "rien à faire" in logs[-1]
    # source modifiée : reconstruction, seul le passage changé est recalculé
    modified = f"# GDN\n\nTexte modifié : la GDN normalise chaque canal du réseau.\n\n# Export\n\n{EXPORT}\n"
    (small_catalog / "docs" / "gdn.md").write_text(modified, encoding="utf-8")
    ingest.run("local", if_stale=True, log=logs.append)
    assert any("dont 1 calculés" in line for line in logs)


def test_public_only_never_contains_private_sources(small_catalog):
    manifest = ingest.run("local", public_only=True, log=lambda *_: None)
    assert manifest["public_only"] is True
    assert all(s["visibility"] == "public" for s in manifest["sources"])
    index = store.Index(store.current_version())
    assert all(c.visibility == "public" for c in index.chunks)
    assert not any("privé" in c.text for c in index.chunks)


def test_seed_holds_public_vectors_only_and_spares_the_next_ingestion(small_catalog, fake_embedder, tmp_path,
                                                                      monkeypatch):
    import sqlite3

    from rag.embeddings import EmbeddingCache
    ingest.run("local", log=lambda *_: None)                     # cache : passages publics et privé
    written, total = ingest.export_seed(model=config.LOCAL_EMBED_MODEL, log=lambda *_: None)
    assert written == total == 2                                 # le passage privé n'est jamais exporté
    seed = config.seed_path()
    db = sqlite3.connect(seed)
    keys = {k for (k,) in db.execute("SELECT key FROM vectors")}
    db.close()
    private = [c for c in store.Index(store.current_version()).chunks if c.visibility == "private"]
    assert private and not keys & {EmbeddingCache.key(config.LOCAL_EMBED_MODEL, "document", c.embed_text())
                                   for c in private}
    # nouveau poste (cache vide) : la graine évite tout calcul pour les sources publiques
    monkeypatch.setenv("RAG_INDEX_DIR", str(tmp_path / "autre-index"))
    calls, logs = fake_embedder.calls, []
    ingest.run("local", public_only=True, log=logs.append)
    assert fake_embedder.calls == calls
    assert any("2 vecteurs repris de la graine" in line for line in logs)
    assert any("dont 0 calculés" in line for line in logs)


def test_cache_merge_keeps_known_vectors(tmp_path):
    import numpy as np

    from rag.embeddings import EmbeddingCache
    a, b = EmbeddingCache(tmp_path / "a.sqlite"), EmbeddingCache(tmp_path / "b.sqlite")
    a.put_many("m", {"k1": np.ones(4), "k2": np.ones(4)})
    b.put_many("m", {"k1": np.zeros(4)})
    assert b.merge(tmp_path / "a.sqlite") == 1
    vecs = b.get_many(["k1", "k2"])
    assert vecs["k1"].sum() == 0 and vecs["k2"].sum() == 4


def test_voyage_backend_requires_key():
    with pytest.raises(SystemExit):
        ingest.wanted_backends("voyage")
    assert ingest.wanted_backends("auto") == ["local"]


def test_is_fresh():
    m = {"sources_hash": "h", "backends": {"voyage": {"model": config.VOYAGE_EMBED_MODEL},
                                           "local": {"model": config.LOCAL_EMBED_MODEL}}}
    assert ingest.is_fresh(m, "h", ["local"]) and ingest.is_fresh(m, "h", ["voyage", "local"])
    assert not ingest.is_fresh(m, "autre", ["local"])
    assert not ingest.is_fresh({"sources_hash": "h", "backends": {"local": {"model": "ancien"}}}, "h", ["local"])
    assert not ingest.is_fresh({"sources_hash": "h", "backends": {"local": {"model": config.LOCAL_EMBED_MODEL}}},
                               "h", ["voyage", "local"])
    assert not ingest.is_fresh(None, "h", ["local"])


def test_cli_reports_refused_source(monkeypatch, capsys):
    from rag import sources

    def refuse(**kwargs):
        raise sources.SourceError("SHA-256 différent")

    monkeypatch.setattr(sources, "documents", refuse)
    assert ingest.main(["--backend", "local"]) == 2
    assert "Source refusée" in capsys.readouterr().err
