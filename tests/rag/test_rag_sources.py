"""Catalogue des sources : validité, téléchargement épinglé (SHA-256), sources privées, empreinte."""
import hashlib

import httpx
import pytest

from rag import config, sources


def client_for(routes, hits):
    def handler(request):
        hits.append(str(request.url))
        body = routes.get(str(request.url))
        return httpx.Response(200, content=body) if body is not None else httpx.Response(404)
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_real_catalog_is_valid_and_pinned():
    entries = sources.load_catalog()
    ids = [e["id"] for e in entries]
    assert {"balle2017", "erts2026", "aidge", "locod", "projet", "challenge"} <= set(ids)
    by = {e["id"]: e for e in entries}
    assert by["balle2017"]["url"].endswith("1611.01704v3")              # version arXiv épinglée
    assert by["erts2026"]["url"].endswith("hal-05513843v1/document")     # version HAL épinglée
    assert by["aidge"]["version"] == "v0.10.1"                           # même version que les paquets pip
    for e in entries:
        if e["kind"] in ("url", "git"):
            assert e["visibility"] == "public"
            shas = [e["sha256"]] if e["kind"] == "url" else [f["sha256"] for f in e["files"]]
            assert all(len(s) == 64 for s in shas)
        if e["kind"] == "git":
            assert len(e["commit"]) == 40
    assert by["challenge"]["visibility"] == "private" and by["challenge"]["kind"] == "local"


def test_catalog_rejects_incomplete_entries(tmp_path):
    bad = tmp_path / "s.yaml"
    bad.write_text("sources:\n  - {id: x, visibility: public, kind: url, url: http://e/x}\n", encoding="utf-8")
    with pytest.raises(sources.SourceError, match="manquants"):
        sources.load_catalog(bad)
    bad.write_text("sources:\n  - {id: x, visibility: secret, kind: repo, files: [a]}\n", encoding="utf-8")
    with pytest.raises(sources.SourceError, match="visibility"):
        sources.load_catalog(bad)


def test_download_verifies_sha_and_skips_intact_file(tmp_path):
    body = b"%PDF contenu"
    sha = hashlib.sha256(body).hexdigest()
    hits = []
    client = client_for({"https://e/a.pdf": body}, hits)
    dest = tmp_path / "a.pdf"
    sources.download("https://e/a.pdf", dest, sha, client)
    assert dest.read_bytes() == body and len(hits) == 1
    sources.download("https://e/a.pdf", dest, sha, client)              # déjà là et intact : aucune requête
    assert len(hits) == 1
    with pytest.raises(sources.SourceError, match="SHA-256"):
        sources.download("https://e/a.pdf", tmp_path / "b.pdf", "0" * 64, client)
    assert not (tmp_path / "b.pdf").exists()
    with pytest.raises(sources.SourceError, match="impossible"):
        sources.download("https://e/absent.pdf", tmp_path / "c.pdf", sha, client)


def test_documents_public_only_excludes_private(tmp_path):
    root = tmp_path / "repo"
    (root / "docs").mkdir(parents=True)
    (root / "README.md").write_text("# Projet\n\ntexte\n", encoding="utf-8")
    (root / "docs" / "guide.md").write_text("# Guide\r\n\r\ntexte\r\n", encoding="utf-8")
    private = config.sources_dir() / "private"
    private.mkdir(parents=True)
    (private / "plaquette.pdf").write_bytes(b"%PDF prive")
    body = b"# Aidge\n\ndoc\n"
    catalog = tmp_path / "s.yaml"
    catalog.write_text(f"""sources:
  - {{id: projet, visibility: public, kind: repo, files: [README.md, "docs/*.md"]}}
  - id: aidge
    visibility: public
    kind: git
    version: v0
    commit: {"a" * 40}
    raw: "https://e/{{commit}}/{{path}}"
    files: [{{path: README.md, sha256: {hashlib.sha256(body).hexdigest()}}}]
  - {{id: challenge, visibility: private, kind: local, dir: private, files: ["*.pdf"]}}
""", encoding="utf-8")
    hits = []
    client = client_for({f"https://e/{'a' * 40}/README.md": body}, hits)
    docs = sources.documents(catalog=catalog, root=root, client=client)
    assert [(d.source_id, d.name, d.visibility) for d in docs] == [
        ("projet", "README.md", "public"), ("projet", "docs/guide.md", "public"),
        ("aidge", "aidge/README.md", "public"), ("challenge", "plaquette.pdf", "private")]
    public = sources.documents(include_private=False, catalog=catalog, root=root, client=client)
    assert all(d.visibility == "public" for d in public) and len(public) == 3
    # sans téléchargement : le cache suffit
    assert len(sources.documents(include_private=False, fetch=False, catalog=catalog, root=root)) == 3


def test_text_hash_ignores_line_endings_and_sources_hash_tracks_changes(tmp_path):
    a, b = tmp_path / "a.md", tmp_path / "b.md"
    a.write_bytes(b"ligne\r\nautre\r\n")
    b.write_bytes(b"ligne\nautre\n")
    assert sources.file_sha256(a) == sources.file_sha256(b)
    d1 = sources.Document("s", "a.md", a, sources.file_sha256(a), "public")
    d2 = sources.Document("s", "a.md", a, "autre", "public")
    assert sources.sources_hash([d1]) != sources.sources_hash([d2])
    assert sources.sources_hash([d1]) != sources.sources_hash([d1], extra="taille=900")
    d3 = sources.Document("s", "b.md", b, sources.file_sha256(b), "public")
    assert sources.sources_hash([d1, d3]) == sources.sources_hash([d3, d1])
