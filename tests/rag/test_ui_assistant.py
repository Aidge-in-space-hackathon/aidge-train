"""Bulle de l'assistant : garde-fou d'exposition, historique, sources, streaming, montage dans les Blocks."""
import gradio as gr
import pytest

from aidge_in_space import ui_assistant as ua
from rag.agent import Event


def test_status_requires_local_exposure_or_trusted_auth(monkeypatch):
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-test")
    assert ua.status("127.0.0.1", docker=False) == (True, "")
    assert ua.status("localhost", docker=False)[0]
    assert not ua.status("0.0.0.0", docker=False)[0]
    # Docker : seule compte l'adresse de l'hôte qui publie le port
    assert ua.status("0.0.0.0", docker=True)[0]
    monkeypatch.setenv("AIDGE_UI_BIND", "192.168.1.10")
    enabled, why = ua.status("0.0.0.0", docker=True)
    assert not enabled and "authentification" in why
    monkeypatch.setenv("AIDGE_ASSISTANT_TRUSTED_AUTH", "1")
    assert ua.status("0.0.0.0", docker=True)[0]


def test_status_requires_master_key():
    enabled, why = ua.status("127.0.0.1", docker=False)
    assert not enabled and "LITELLM_MASTER_KEY" in why


def test_history_for_agent_drops_sources_and_handles_blocks():
    history = [{"role": "user", "content": [{"type": "text", "text": "Question ?"}]},
               {"role": "assistant", "content": "Réponse." + ua.SOURCES_MARK + "\n- balle.pdf, p. 3"},
               {"role": "assistant", "content": ""}]
    assert ua.history_for_agent(history) == [{"role": "user", "content": "Question ?"},
                                             {"role": "assistant", "content": "Réponse."}]


def test_format_sources():
    assert ua.format_sources([]) == ""
    md = ua.format_sources([{"source": "balle.pdf, p. 3 (2 GDN)"}, {"source": "docs/x.md (Install)"}])
    assert md.startswith(ua.SOURCES_MARK) and "- balle.pdf, p. 3 (2 GDN)\n- docs/x.md (Install)" in md


class FakeAgent:
    def __init__(self, *args, **kwargs):
        pass

    def run(self, question, history=None):
        yield Event("tool", {"name": "search_docs", "arguments": "{}"})
        yield Event("text", "La GDN normalise ")
        yield Event("text", "[balle.pdf, p. 3].")
        yield Event("sources", [{"source": "balle.pdf, p. 3 (2 GDN)", "fichier": "balle.pdf"},
                                {"source": "docs/x.md (Autre)", "fichier": "docs/x.md"}])


def test_respond_streams_then_adds_cited_sources(monkeypatch):
    import rag.agent
    monkeypatch.setattr(rag.agent, "Agent", FakeAgent)
    steps = list(ua.respond("Comment fonctionne la GDN ?", []))
    assert steps[0][1][-1]["content"] == "…"
    assert any("Recherche" in s[1][-1]["content"] for s in steps)
    assert steps[-2][1][-1]["content"] == "La GDN normalise [balle.pdf, p. 3]."
    final = steps[-1][1]
    assert final[0] == {"role": "user", "content": "Comment fonctionne la GDN ?"}
    assert final[-1]["content"] == "La GDN normalise [balle.pdf, p. 3]." + ua.SOURCES_MARK + \
        "\n- balle.pdf, p. 3 (2 GDN)"                       # seule la source citée est affichée
    assert all(s[0] == "" for s in steps)


def test_respond_without_configuration_says_so():
    steps = list(ua.respond("Bonjour", []))
    assert "LITELLM_MASTER_KEY" in steps[-1][1][-1]["content"]
    assert list(ua.respond("   ", [])) == [("", [])]


@pytest.mark.parametrize("key", [None, "sk-test"])
def test_mount_in_blocks(monkeypatch, key):
    if key:
        monkeypatch.setenv("LITELLM_MASTER_KEY", key)
    with gr.Blocks() as demo:
        with gr.Tabs():
            with gr.Tab("1"):
                gr.Markdown("onglet")
        fab, panel = ua.mount("127.0.0.1")
    assert fab.elem_id == "chat-fab" and panel.elem_id == "chat-panel"
    chatbots = [b for b in demo.blocks.values() if isinstance(b, gr.Chatbot)]
    assert len(chatbots) == (1 if key else 0)                # désactivé : aucune route de chat
    assert "#chat-panel.open" in ua.CSS and "chat-panel" in ua.TOGGLE_JS


def test_warm_never_raises_and_respects_auto_ingest(monkeypatch):
    import threading

    from aidge_in_space import ui_assistant
    assert ui_assistant.warm() is False                        # pas d'index : aucune exception
    started = []
    monkeypatch.setattr(threading.Thread, "start", lambda self: started.append(self.name))
    monkeypatch.setattr(ui_assistant, "_warm_started", False)
    ui_assistant.start_warm()                                  # AIDGE_ASSISTANT_AUTO_INGEST=0 (conftest)
    assert started == []
    monkeypatch.setenv("AIDGE_ASSISTANT_AUTO_INGEST", "1")
    ui_assistant.start_warm()
    ui_assistant.start_warm()                                  # une seule fois par processus
    assert started == ["rag-warm"]
