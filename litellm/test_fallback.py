"""! @brief Bascule entre fournisseurs du proxy LiteLLM : le fournisseur principal tombe, la réponse vient d'un secours.

Deux modes :
- hors ligne (pytest, CI) : un Router LiteLLM est construit à partir de litellm/config.yaml, mais chaque fournisseur
  pointe vers un faux serveur local (panne 503, quota 429, délai dépassé, ou réponse correcte). Aucun appel réseau,
  aucune clé : python -m pytest litellm/test_fallback.py
- réel : python litellm/test_fallback.py --live interroge le proxy lancé (docker compose up litellm) avec
  mock_testing_fallbacks, qui simule une panne du principal, et vérifie que la réponse vient de Groq.
  Demande LITELLM_MASTER_KEY et une clé Groq valide ; consomme une requête gratuite.

Ne pas ajouter de __init__.py dans litellm/ : ce dossier masquerait alors le paquet litellm.
"""
import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

## Configuration du proxy, source unique des modèles et des règles de bascule.
CONFIG = Path(__file__).with_name("config.yaml")
## Groupes de modèles, dans l'ordre de bascule attendu.
ORDER = ["assistant", "assistant-groq"]

# La table des prix de LiteLLM est téléchargée à l'import sinon : les tests restent hors ligne
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")


def load_config():
    """! @brief Lit litellm/config.yaml.

    @return dict de la configuration.
    """
    import yaml
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- faux fournisseurs

class FakeProvider:
    """! @brief Faux fournisseur compatible OpenAI (/chat/completions), sur un port local libre.

    @param behavior  "ok" (réponse normale), "503" (panne), "429" (quota épuisé) ou "slow" (ne répond pas à temps).
    @param name      Texte de la réponse, pour savoir quel fournisseur a répondu.
    """

    def __init__(self, behavior, name):
        """! @brief Démarre le serveur dans un thread.

        @param behavior  Comportement du serveur (voir la classe).
        @param name      Nom renvoyé dans la réponse.
        """
        ## Comportement à chaque requête.
        self.behavior = behavior
        ## Nombre de requêtes reçues.
        self.hits = 0
        provider = self

        class Handler(BaseHTTPRequestHandler):
            """! @brief Gestionnaire HTTP du faux fournisseur."""

            def log_message(self, *args):
                """! @brief Silence : pas de journal par requête."""

            def do_POST(self):
                """! @brief Répond à /chat/completions selon le comportement choisi."""
                length = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(length)
                provider.hits += 1
                if provider.behavior == "slow":
                    time.sleep(3)
                if provider.behavior in ("503", "429"):
                    body = json.dumps({"error": {"message": f"panne simulée {provider.behavior}",
                                                 "type": "server_error", "code": provider.behavior}}).encode()
                    self.send_response(int(provider.behavior))
                else:
                    body = json.dumps({
                        "id": "chatcmpl-faux", "object": "chat.completion", "created": int(time.time()),
                        "model": name, "service_tier": "on_demand", "system_fingerprint": "faux",
                        "choices": [{"index": 0, "finish_reason": "stop",
                                     "message": {"role": "assistant", "content": f"réponse de {name}"}}],
                        "usage": {"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6},
                    }).encode()
                    self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except OSError:                       # le client a abandonné (délai dépassé)
                    pass

        ## Serveur HTTP local.
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        ## URL de base à donner à LiteLLM (api_base).
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        """! @brief Arrête le serveur."""
        self.server.shutdown()
        self.server.server_close()


def build_router(urls, timeout=None):
    """! @brief Construit un Router LiteLLM à partir de config.yaml, chaque fournisseur redirigé vers un faux serveur.

    @param urls     dict groupe de modèles -> URL du faux serveur.
    @param timeout  Délai par appel (s) ; None : celui de la configuration.
    @return litellm.Router.
    """
    from litellm import Router
    cfg = load_config()
    model_list = []
    for entry in cfg["model_list"]:
        params = dict(entry["litellm_params"])
        params["api_key"] = "cle-factice"         # la configuration lit os.environ/... : jamais de vraie clé ici
        params["api_base"] = urls[entry["model_name"]]
        model_list.append({"model_name": entry["model_name"], "litellm_params": params})
    settings = dict(cfg["router_settings"])
    if timeout is not None:
        settings["timeout"] = timeout
    return Router(model_list=model_list, **settings)


# --------------------------------------------------------------------------- tests hors ligne

def _litellm():
    """! @brief Importe le paquet litellm, ou saute le test s'il n'est pas installé.

    @return Module litellm.
    """
    import pytest
    try:
        import litellm
        litellm.Router  # noqa: B018 (le dossier litellm/ seul, sans le paquet, n'a pas de Router)
    except (ImportError, AttributeError):
        pytest.skip("paquet litellm absent : pip install litellm==1.106.8")
    return litellm


def test_config_declares_fallback_chain():
    cfg = load_config()
    groups = [m["model_name"] for m in cfg["model_list"]]
    assert groups == ORDER
    providers = [m["litellm_params"]["model"].split("/")[0] for m in cfg["model_list"]]
    assert providers == ["nvidia_nim", "groq"]
    assert cfg["model_list"][0]["litellm_params"]["model"] == "nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b"
    router = cfg["router_settings"]
    assert router["fallbacks"] == [{"assistant": ["assistant-groq"]}]
    assert router["num_retries"] >= 1 and router["timeout"] > 0 and router["cooldown_time"] > 0
    assert router["retry_policy"]["RateLimitErrorRetries"] == 0
    # Aucune clé en clair : tout vient de l'environnement
    for m in cfg["model_list"]:
        assert m["litellm_params"]["api_key"].startswith("os.environ/")
    assert cfg["general_settings"]["master_key"] == "os.environ/LITELLM_MASTER_KEY"


def _servers(primary, groq="ok"):
    """! @brief Démarre les deux faux fournisseurs.

    @param primary  Comportement du principal (NVIDIA NIM).
    @param groq     Comportement du secours (Groq).
    @return dict groupe -> FakeProvider.
    """
    return {"assistant": FakeProvider(primary, "nvidia"),
            "assistant-groq": FakeProvider(groq, "groq")}


def _ask(router):
    """! @brief Pose une question au groupe « assistant ».

    @param router  litellm.Router.
    @return Texte de la réponse.
    """
    resp = router.completion(model="assistant", messages=[{"role": "user", "content": "ping"}])
    return resp.choices[0].message.content


def test_fallback_on_provider_failure():
    import pytest
    _litellm()
    for failure in ("503", "429", "slow"):
        servers = _servers(failure)
        try:
            router = build_router({k: s.url for k, s in servers.items()}, timeout=1)
            answer = _ask(router)
            assert answer == "réponse de groq", f"panne {failure} : {answer}"
            assert servers["assistant"].hits >= 1, f"panne {failure} : le principal n'a pas été essayé"
        except AssertionError:
            raise
        except Exception as e:                       # erreur inattendue : message lisible
            pytest.fail(f"panne {failure} : {type(e).__name__}: {e}")
        finally:
            for s in servers.values():
                s.close()


def test_error_when_every_provider_fails():
    import pytest
    _litellm()
    servers = _servers("503", groq="429")
    try:
        router = build_router({k: s.url for k, s in servers.items()}, timeout=2)
        with pytest.raises(Exception):  # noqa: B017 (l'erreur exacte dépend de LiteLLM)
            _ask(router)
        assert servers["assistant"].hits >= 1 and servers["assistant-groq"].hits >= 1
    finally:
        for s in servers.values():
            s.close()


def test_rate_limited_primary_is_cooled_down():
    _litellm()
    servers = _servers("429")
    try:
        router = build_router({k: s.url for k, s in servers.items()}, timeout=2)
        assert _ask(router) == "réponse de groq"
        hits = servers["assistant"].hits
        assert _ask(router) == "réponse de groq"
        assert servers["assistant"].hits == hits, "le principal en cooldown ne doit pas être rappelé"
    finally:
        for s in servers.values():
            s.close()


# --------------------------------------------------------------------------- mode réel

def live(base_url):
    """! @brief Vérifie la bascule sur le proxy réel : panne simulée du principal, réponse d'un secours attendue.

    @param base_url  URL du proxy (ex. http://127.0.0.1:4000).
    @return 0 si la réponse vient d'un secours, 1 sinon.
    """
    from openai import OpenAI
    key = os.environ.get("LITELLM_MASTER_KEY")
    if not key:
        print("LITELLM_MASTER_KEY manquante (voir .env.example)", file=sys.stderr)
        return 1
    client = OpenAI(base_url=base_url, api_key=key)
    raw = client.chat.completions.with_raw_response.create(
        model="assistant", messages=[{"role": "user", "content": "Réponds juste : ok"}], max_tokens=20,
        extra_body={"mock_testing_fallbacks": True})
    resp = raw.parse()
    group = raw.headers.get("x-litellm-model-group", "?")
    print(f"modèle : {resp.model} ; groupe : {group} ; réponse : {resp.choices[0].message.content!r}")
    if group == "assistant-groq" or "gpt-oss" in resp.model:
        print("OK : la réponse vient d'un secours")
        return 0
    print("ÉCHEC : la réponse ne vient pas d'un secours", file=sys.stderr)
    return 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Vérifie la bascule du proxy LiteLLM vers un fournisseur de secours")
    ap.add_argument("--live", action="store_true", help="interroge le proxy réel (consomme une requête gratuite)")
    ap.add_argument("--url", default=os.environ.get("LITELLM_BASE_URL", "http://127.0.0.1:4000"))
    args = ap.parse_args()
    if args.live:
        raise SystemExit(live(args.url))
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
