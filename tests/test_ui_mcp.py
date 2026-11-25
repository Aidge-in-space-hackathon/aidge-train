"""Tests du serveur MCP de l'interface (ui_mcp) : python -m pytest tests/test_ui_mcp.py

L'interface est construite sans être lancée, sauf dans le dernier test (vrai serveur et vrai client MCP, sauté sans
le paquet mcp). Dossiers temporaires uniquement.
"""
import asyncio
import importlib.util
import inspect
import json

import matplotlib
import pytest

matplotlib.use("Agg")
gr = pytest.importorskip("gradio")
ui_mcp = pytest.importorskip("aidge_in_space.ui_mcp")

from gradio.utils import get_function_description   # noqa: E402

from aidge_in_space import config, ui_training   # noqa: E402

HAS_MCP = importlib.util.find_spec("mcp") is not None


@pytest.fixture
def demo(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(config, "CKPT_DIR", tmp_path / "ckpts")
    monkeypatch.setattr(config, "TRAIN_DIR", tmp_path / "train")
    monkeypatch.setattr(ui_training, "_codecs_loaded", None)
    from aidge_in_space import app
    return app.build("127.0.0.1")


def _tools(demo):
    return {f.api_name: f.fn for f in demo.fns.values() if getattr(f.fn, "_aidge_tool", False)}


def test_seuls_les_outils_sont_publics(demo):
    assert set(demo.get_api_info()["named_endpoints"]) == {f"/{name}" for name in ui_mcp.TOOLS}
    publics = [f.api_name for f in demo.fns.values() if f.api_visibility != "private"]
    assert sorted(publics) == sorted(ui_mcp.TOOLS)


def test_chaque_outil_est_nomme_et_tous_ses_parametres_decrits(demo):
    tools = _tools(demo)
    assert set(tools) == set(ui_mcp.TOOLS)
    for name, fn in tools.items():
        assert fn.__name__ == name
        description, params, returns = get_function_description(fn)
        assert description and returns, name
        sans_description = [p for p in inspect.signature(fn).parameters if not params.get(p)]
        assert not sans_description, f"{name} : paramètres non décrits dans ui_mcp.TOOLS : {sans_description}"


def test_parametres_simples_et_facultatifs(demo):
    """Les défauts trouvés en essai : niveau entier, aucun champ d'image ou de référence obligatoire."""
    sig = inspect.signature(_tools(demo)["test_model"])
    assert sig.parameters["level"].annotation is int
    assert [p for p, v in sig.parameters.items() if v.default is inspect.Parameter.empty] == ["model"]
    assert "upload" not in sig.parameters
    bench_sig = inspect.signature(_tools(demo)["benchmark"])
    assert bench_sig.parameters["reference"].default is None
    assert inspect.signature(_tools(demo)["launch_optim"]).parameters["tile"].annotation is int


def test_list_options_lit_les_valeurs_a_l_appel(demo):
    options = json.loads(_tools(demo)["list_options"]())
    assert {"models", "codecs", "datasets", "backends", "sizes", "campaigns", "optim"} <= set(options)
    assert "jpeg" in [c["codec"] for c in options["codecs"]["gris"]]
    assert "512 x 512" in options["sizes"] and "torch" in options["backends"]
    assert options["optim"]["tiles"] == [32, 64, 128, 256] and options["optim"]["default_recipe"] == "mixte"


def test_compare_encoders_et_suivi_des_jobs(demo):
    tools = _tools(demo)
    assert isinstance(json.loads(tools["compare_encoders"]()), list)
    assert json.loads(tools["list_optim_jobs"]()) == []


def test_les_erreurs_de_valeur_arrivent_a_l_agent(demo):
    tools = _tools(demo)
    with pytest.raises(gr.Error, match="list_options"):
        tools["test_model"]("modele-inexistant")
    with pytest.raises(gr.Error, match="Taille inconnue"):
        tools["compare_encoders"]("3 x 3")
    with pytest.raises(gr.Error, match="checkpoint"):
        tools["launch_optim"]("serie", [], "jeu")


def test_l_enveloppe_garde_la_fonction_d_origine_intacte():
    def additionne(a: int, b: int = 2) -> int:
        """! @brief Doxygen."""
        return a + b

    tool = ui_mcp._as_tool(additionne, "somme", "Somme.\n\nArgs:\n    a: x\n    b: y")
    assert tool(1) == 3 and tool(1, b=5) == 6
    assert tool.__name__ == "somme" and str(inspect.signature(tool)) == "(a: int, b: int = 2) -> int"
    assert additionne.__name__ == "additionne" and additionne.__doc__.startswith("!")


def test_l_enveloppe_refuse_generateurs_et_coroutines():
    def gen():
        """! @brief Générateur."""
        yield 1

    async def coro():
        """! @brief Coroutine."""

    for fn in (gen, coro):
        with pytest.raises(TypeError):
            ui_mcp._as_tool(fn, "x", "")


def test_restrict_api_signale_un_outil_absent():
    with gr.Blocks() as demo:
        gr.api(ui_mcp._as_tool(ui_mcp.list_options, "list_options", ui_mcp.TOOLS["list_options"]),
               api_name="list_options")
    with pytest.raises(ValueError, match="benchmark"):
        ui_mcp.restrict_api(demo)


@pytest.fixture
def env_mcp(monkeypatch):
    for var in (ui_mcp.ENABLE_VAR, ui_mcp.TRUSTED_AUTH_VAR, "AIDGE_UI_BIND", "AIDGE_UI_HOST"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(ui_mcp.importlib.util, "find_spec", lambda name: object())
    return monkeypatch


def test_status_desactive_par_defaut(env_mcp):
    assert ui_mcp.status("127.0.0.1", docker=False) == (False, "")


def test_status_local_actif(env_mcp):
    env_mcp.setenv(ui_mcp.ENABLE_VAR, "1")
    assert ui_mcp.status("127.0.0.1", docker=False) == (True, "")
    env_mcp.setenv("AIDGE_UI_BIND", "127.0.0.1")
    assert ui_mcp.status("0.0.0.0", docker=True) == (True, "")


def test_status_refuse_une_interface_exposee_sans_authentification(env_mcp):
    env_mcp.setenv(ui_mcp.ENABLE_VAR, "1")
    on, message = ui_mcp.status("0.0.0.0", docker=False)
    assert not on and ui_mcp.TRUSTED_AUTH_VAR in message
    env_mcp.setenv("AIDGE_UI_BIND", "0.0.0.0")
    assert not ui_mcp.status("0.0.0.0", docker=True)[0]
    env_mcp.setenv(ui_mcp.TRUSTED_AUTH_VAR, "1")
    assert ui_mcp.status("0.0.0.0", docker=True) == (True, "")


def test_status_sans_paquet_mcp(env_mcp):
    env_mcp.setenv(ui_mcp.ENABLE_VAR, "1")
    env_mcp.setattr(ui_mcp.importlib.util, "find_spec", lambda name: None)
    on, message = ui_mcp.status("127.0.0.1", docker=False)
    assert not on and "mcp" in message


@pytest.mark.skipif(not HAS_MCP, reason="paquet mcp absent")
def test_vrai_serveur_mcp(demo):
    from mcp import ClientSession
    try:
        from mcp.client.streamable_http import streamable_http_client as streamablehttp_client
    except ImportError:   # mcp plus ancien
        from mcp.client.streamable_http import streamablehttp_client

    demo.queue().launch(server_name="127.0.0.1", mcp_server=True, prevent_thread_lock=True, quiet=True)
    try:
        async def session():
            async with streamablehttp_client(f"{demo.local_url}gradio_api/mcp/") as (read, write, _):
                async with ClientSession(read, write) as s:
                    await s.initialize()
                    tools = {t.name: t for t in (await s.list_tools()).tools}
                    jobs = await s.call_tool("list_optim_jobs", {})
                    unknown = await s.call_tool("test_model", {"model": "modele-inexistant", "level": 3})
                    return tools, jobs, unknown

        tools, jobs, unknown = asyncio.run(session())
    finally:
        demo.close()
    assert set(tools) == set(ui_mcp.TOOLS)
    schema = tools["test_model"].inputSchema
    assert schema["properties"]["level"]["type"] == "integer", schema
    assert schema["required"] == ["model"], schema
    assert tools["launch_optim"].inputSchema["properties"]["n_calibration"]["description"]
    assert not jobs.isError, jobs.content
    assert json.loads(jobs.content[0].text) == []
    assert unknown.isError, unknown.content
    assert "list_options" in unknown.content[0].text, unknown.content
