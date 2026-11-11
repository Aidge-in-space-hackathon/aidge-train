"""Registre d'outils : enregistrement, appels, erreurs rendues au LLM, outil search_docs."""
import json

import pytest

from rag import store
from rag.chunking import Block, chunk_blocks
from rag.tools import REGISTRY, Tool, ToolRegistry

SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}


def test_register_and_schema():
    reg = ToolRegistry()

    @reg.tool("double", "Double un entier.", SCHEMA)
    def double(a):
        return {"r": 2 * a}

    assert reg.names() == ["double"] and reg.get("double").func is double
    assert reg.schemas() == [{"type": "function", "function": {"name": "double", "description": "Double un entier.",
                                                                "parameters": SCHEMA}}]
    assert reg.call("double", '{"a": 4}') == {"r": 8}
    assert reg.call("double", {"a": 1, "invente": 3}) == {"r": 2}            # argument inconnu ignoré


def test_register_rejects_duplicates_and_bad_definitions():
    reg = ToolRegistry()
    reg.register(Tool("t", "d", SCHEMA, lambda a: a))
    with pytest.raises(ValueError, match="déjà"):
        reg.register(Tool("t", "d", SCHEMA, lambda a: a))
    with pytest.raises(ValueError, match="invalide"):
        reg.register(Tool("mauvais nom", "d", SCHEMA, lambda a: a))
    with pytest.raises(ValueError, match="object"):
        reg.register(Tool("u", "d", {"type": "string"}, lambda a: a))


def test_call_errors_are_returned_not_raised():
    reg = ToolRegistry()
    reg.register(Tool("boom", "d", SCHEMA, lambda a: 1 / 0))
    assert "inconnu" in reg.call("absent", "{}")["erreur"]
    assert "JSON" in reg.call("boom", "{pas du json")["erreur"]
    assert "objet" in reg.call("boom", "[1, 2]")["erreur"]
    assert "manquants : a" in reg.call("boom", "{}")["erreur"]
    assert reg.call("boom", '{"a": 1}')["erreur"].startswith("ZeroDivisionError")


def test_default_registry_has_only_search_docs_for_now():
    assert REGISTRY.names() == ["search_docs"]
    params = REGISTRY.get("search_docs").parameters
    assert params["required"] == ["query"] and set(params["properties"]) == {"query", "k"}


def test_search_docs_returns_sources(fake_embedder, monkeypatch):
    chunks = chunk_blocks([Block("2 GDN", 3, "La GDN divise chaque canal par les énergies voisines.")],
                          "balle.pdf", "balle2017")
    store.write_index(chunks, {"local": fake_embedder.embed_documents([c.embed_text() for c in chunks])},
                      {"sources_hash": "h", "backends": {"local": {"model": "m"}}})
    import rag.retrieval
    monkeypatch.setattr(rag.retrieval, "_RETRIEVER", rag.retrieval.Retriever(local=fake_embedder))
    out = REGISTRY.call("search_docs", json.dumps({"query": "GDN", "k": 50}))
    assert out["passages"][0]["source"] == "balle.pdf, p. 3 (2 GDN)" and out["passages"][0]["page"] == 3
    assert REGISTRY.call("search_docs", '{"query": "  "}') == {"erreur": "requête vide"}
    assert REGISTRY.call("search_docs", '{"query": "zzzz"}')["passages"] in ([], out["passages"][:1])


def test_search_docs_without_index_reports_error_to_llm():
    out = REGISTRY.call("search_docs", '{"query": "GDN"}')
    assert out["erreur"].startswith("IndexMissing")
