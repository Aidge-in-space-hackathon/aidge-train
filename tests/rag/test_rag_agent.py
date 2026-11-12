"""Agent : boucle d'appel d'outils en streaming, limite d'itérations, sources, erreurs. LLM simulé."""
import json
from types import SimpleNamespace as NS

import pytest

from rag.agent import SYSTEM_PROMPT, Agent, cited, friendly_error
from rag.tools import Tool, ToolRegistry


def text_chunks(text):
    return [NS(choices=[NS(delta=NS(content=part, tool_calls=None))]) for part in text.split("|")]


def tool_chunks(name, args, call_id="c1"):
    half = len(args) // 2                                          # arguments découpés sur deux morceaux
    return [NS(choices=[NS(delta=NS(content=None, tool_calls=[
                NS(index=0, id=call_id, function=NS(name=name, arguments=args[:half]))]))]),
            NS(choices=[NS(delta=NS(content=None, tool_calls=[
                NS(index=0, id=None, function=NS(name=None, arguments=args[half:]))]))]),
            NS(choices=[])]


class FakeClient:
    """Client OpenAI simulé : rend les tours prévus, garde les requêtes."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.requests = []
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})     # instantané de la requête
        if isinstance(self.turns[0], Exception):
            raise self.turns.pop(0)
        return iter(self.turns.pop(0))


def registry(log):
    reg = ToolRegistry()

    def search(query, k=None):
        log.append(query)
        return {"passages": [{"source": "balle.pdf, p. 3 (2 GDN)", "fichier": "balle.pdf", "page": 3,
                              "texte": "La GDN..."},
                             {"source": "docs/x.md (Autre)", "fichier": "docs/x.md", "page": None, "texte": "..."}]}

    reg.register(Tool("search_docs", "cherche", {"type": "object", "properties": {"query": {"type": "string"}},
                                                 "required": ["query"]}, search))
    return reg


def test_tool_call_then_streamed_answer_with_sources():
    log = []
    client = FakeClient([tool_chunks("search_docs", '{"query": "GDN"}'),
                         text_chunks("La GDN |normalise [balle.pdf, p. 3].")])
    events = list(Agent(client, "assistant", registry(log)).run("Comment fonctionne la GDN ?"))
    kinds = [e.kind for e in events]
    assert kinds == ["tool", "text", "text", "sources"]
    assert events[0].data == {"name": "search_docs", "arguments": '{"query": "GDN"}'}
    assert log == ["GDN"]
    assert "".join(e.data for e in events if e.kind == "text") == "La GDN normalise [balle.pdf, p. 3]."
    assert [p["source"] for p in events[-1].data] == ["balle.pdf, p. 3 (2 GDN)", "docs/x.md (Autre)"]
    first, second = client.requests
    assert first["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert first["tools"][0]["function"]["name"] == "search_docs" and first["stream"] is True
    tool_msg = second["messages"][-1]
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "c1" and "balle.pdf" in tool_msg["content"]
    assert second["messages"][-2]["tool_calls"][0]["function"]["arguments"] == '{"query": "GDN"}'


def test_at_most_four_llm_calls_and_last_one_without_tools():
    log = []
    turns = [tool_chunks("search_docs", f'{{"query": "q{i}"}}', f"c{i}") for i in range(3)] + [text_chunks("Fin.")]
    client = FakeClient(turns)
    events = list(Agent(client, "assistant", registry(log), max_iterations=4).run("question"))
    assert len(client.requests) == 4 and log == ["q0", "q1", "q2"]
    assert "tools" in client.requests[2] and "tools" not in client.requests[3]
    assert events[-1].kind == "sources"


def test_history_is_passed_and_unknown_tool_is_reported_to_llm():
    client = FakeClient([tool_chunks("inexistant", "{}"), text_chunks("Je ne sais pas.")])
    history = [{"role": "user", "content": "avant"}, {"role": "assistant", "content": "réponse"},
               {"role": "system", "content": "ignoré"}]
    events = list(Agent(client, "assistant", registry([])).run("question", history))
    msgs = client.requests[0]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert "inconnu" in client.requests[1]["messages"][-1]["content"]
    assert events[-1].kind == "sources" and events[-1].data == []


def test_proxy_errors_become_readable_messages():
    import httpx
    import openai
    req = httpx.Request("POST", "http://litellm:4000/chat/completions")
    client = FakeClient([openai.APIConnectionError(request=req)])
    events = list(Agent(client, "assistant", registry([])).run("q"))
    assert events == [events[0]] and events[0].kind == "error" and "injoignable" in events[0].data
    resp = httpx.Response(429, request=req)
    assert "Quotas" in friendly_error(openai.RateLimitError("x", response=resp, body=None))
    resp = httpx.Response(401, request=req)
    assert "LITELLM_MASTER_KEY" in friendly_error(openai.AuthenticationError("x", response=resp, body=None))
    assert friendly_error(KeyError("secret")) == "Erreur de l'assistant : KeyError"   # pas de détail


def test_missing_master_key():
    with pytest.raises(RuntimeError, match="LITELLM_MASTER_KEY"):
        Agent()


def test_cited_keeps_only_cited_files():
    passages = [{"source": "a", "fichier": "balle.pdf"}, {"source": "b", "fichier": "docs/x.md"}]
    assert cited("voir [balle.pdf, p. 3]", passages) == passages[:1]
    assert cited("sans citation", passages) == passages


def test_empty_answer_is_reported_not_disguised():
    """Un flux sans aucun texte (constaté après un quota Groq) donne une erreur lisible, pas « Je ne sais pas »."""
    from rag.agent import EMPTY_ANSWER
    client = FakeClient([tool_chunks("search_docs", '{"query": "GDN"}'), [NS(choices=[])]])
    events = list(Agent(client, "assistant", registry([])).run("Comment fonctionne la GDN ?"))
    assert [e.kind for e in events] == ["tool", "error"]
    assert events[-1].data == EMPTY_ANSWER


XML_CALL = ("<tool_call>\n<function=search_docs>\n<parameter=k>\n10\n</parameter>\n"
            "<parameter=query>\nBallé λ\n</parameter>\n</function>\n</tool_call>")


def test_text_filter_hides_a_written_tool_call_even_split_across_chunks():
    from rag.agent import TextCallFilter
    f = TextCallFilter()
    shown = "".join(f.feed(p) for p in ["Je cherche. <to", "ol_call>\n<function=search_docs>", "...</tool_call>"])
    assert shown + f.flush() == "Je cherche. " and f.captured.startswith("<tool_call>")
    g = TextCallFilter()                                    # « < » ordinaire : rendu, seulement retardé
    assert "".join(g.feed(p) for p in ["a <", "b et x < y"]) + g.flush() == "a <b et x < y"
    assert g.captured == ""


def test_parse_text_calls_decodes_json_values():
    from rag.agent import parse_text_calls
    (call,) = parse_text_calls(XML_CALL)
    assert call.name == "search_docs" and call.id == "texte0"
    assert json.loads(call.arguments) == {"k": 10, "query": "Ballé λ"}
    assert parse_text_calls("rien") == []


def test_written_tool_call_is_executed_and_never_shown():
    log = []
    client = FakeClient([text_chunks(XML_CALL[:20] + "|" + XML_CALL[20:]), text_chunks("Réponse [balle.pdf, p. 3].")])
    events = list(Agent(client, "assistant", registry(log)).run("question"))
    assert log == ["Ballé λ"]
    text = "".join(e.data for e in events if e.kind == "text")
    assert text == "Réponse [balle.pdf, p. 3]." and "<" not in text
    assert events[-1].kind == "sources"


def test_last_turn_asks_for_an_answer_and_drops_a_written_call():
    from rag.agent import FINAL_TURN
    turns = [tool_chunks("search_docs", '{"query": "q"}'), text_chunks("Voici. " + XML_CALL)]
    client = FakeClient(turns)
    events = list(Agent(client, "assistant", registry([]), max_iterations=2).run("question"))
    assert client.requests[1]["messages"][-1] == {"role": "user", "content": FINAL_TURN}
    assert "".join(e.data for e in events if e.kind == "text") == "Voici. "
    assert events[-1].kind == "sources"
