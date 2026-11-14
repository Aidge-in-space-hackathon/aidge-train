"""! @brief Bulle de l'assistant RAG : bouton rond et panneau de chat flottants, par-dessus tous les onglets.

mount() se place dans les Blocks de l'application, après les onglets ; CSS s'ajoute à la feuille passée à launch().
Le bouton ouvre et ferme le panneau en JavaScript, sans aller-retour serveur. Les réponses arrivent en streaming,
avec leurs sources sous chaque réponse.

Garde-fou : l'assistant dépense des quotas d'API gratuits. Il est désactivé (et aucune route n'est créée) dès que
l'interface est joignable depuis une autre machine, sauf si AIDGE_ASSISTANT_TRUSTED_AUTH=1 atteste qu'une
authentification la protège. Il l'est aussi sans LITELLM_MASTER_KEY.
"""
import ipaddress
import os
import subprocess
import sys
import threading
from pathlib import Path

import gradio as gr

## Feuille de style de la bulle (ajoutée à ui_style.CSS dans launch()). Couleurs du thème : le mode sombre suit.
CSS = """
#chat-fab { position: fixed !important; right: 24px; bottom: 24px; z-index: 1000;
            width: 60px !important; min-width: 60px !important; max-width: 60px; height: 60px;
            border-radius: 50% !important; padding: 0 !important; font-size: 26px;
            box-shadow: 0 6px 20px rgba(0, 0, 0, .25); }
#chat-panel { position: fixed !important; right: 24px; bottom: 96px; z-index: 999;
              width: min(400px, calc(100vw - 32px)); max-height: 72vh; overflow: auto; display: none;
              flex-direction: column; gap: 8px; padding: 10px;
              background: var(--background-fill-primary); border: 1px solid var(--border-color-primary);
              border-radius: 16px; box-shadow: 0 12px 40px rgba(0, 0, 0, .25); }
#chat-panel.open { display: flex; }
#chat-panel .chat-title p { margin: 0; }
#chat-panel .chat-status { font-size: 0.85rem; color: var(--body-text-color-subdued); }
@media (max-width: 480px) { #chat-fab { right: 16px; bottom: 16px; } #chat-panel { right: 16px; bottom: 84px; } }
"""

## Ouvre ou ferme le panneau, côté navigateur.
TOGGLE_JS = "() => { const p = document.querySelector('#chat-panel'); if (p) { p.classList.toggle('open'); } }"
## Séparateur entre la réponse et ses sources (retiré de l'historique renvoyé au LLM).
SOURCES_MARK = "\n\n---\n**Sources**"

_ingest_started = False
_ingest_lock = threading.Lock()
_warm_started = False


def _is_loopback(host):
    """! @brief Adresse joignable seulement depuis cette machine (même règle qu'app.is_loopback).

    @param host  Nom ou adresse.
    @return bool.
    """
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def in_docker():
    """! @brief Le processus tourne-t-il dans un conteneur Docker ?

    @return bool.
    """
    return Path("/.dockerenv").exists()


def status(host=None, docker=None):
    """! @brief L'assistant peut-il être activé ?

    Dans Docker, l'exposition dépend de l'adresse de l'hôte qui publie le port (AIDGE_UI_BIND), pas de l'adresse
    d'écoute du conteneur (0.0.0.0).
    @param host    Adresse d'écoute de l'interface ; None : AIDGE_UI_HOST, sinon 127.0.0.1.
    @param docker  Force la détection de Docker (tests) ; None : in_docker().
    @return (activé, message à afficher si désactivé).
    """
    docker = in_docker() if docker is None else docker
    exposed_on = (os.environ.get("AIDGE_UI_BIND") or "127.0.0.1") if docker \
        else (host or os.environ.get("AIDGE_UI_HOST") or "127.0.0.1")
    if not _is_loopback(exposed_on) and os.environ.get("AIDGE_ASSISTANT_TRUSTED_AUTH") != "1":
        return False, ("Assistant désactivé : l'interface est exposée sur le réseau sans authentification devant. "
                       "Il consomme des quotas d'API : placez une authentification devant, puis définissez "
                       "AIDGE_ASSISTANT_TRUSTED_AUTH=1.")
    if not os.environ.get("LITELLM_MASTER_KEY"):
        return False, ("Assistant non configuré : renseignez LITELLM_MASTER_KEY et les clés des fournisseurs dans "
                       ".env (voir .env.example), puis relancez docker compose up.")
    return True, ""


def _text(content):
    """! @brief Texte d'un message du Chatbot (chaîne, ou liste de blocs {"type": "text", "text": ...}).

    @param content  Contenu d'un message.
    @return str.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return str(content.get("text") or "")
    if isinstance(content, (list, tuple)):
        return "".join(_text(c) for c in content)
    return ""


def history_for_agent(history):
    """! @brief Historique du Chatbot réduit à ce que le LLM doit relire : rôles et textes, sans les sources.

    @param history  Messages du Chatbot.
    @return Liste de {"role", "content"}.
    """
    out = []
    for m in history or []:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
        text = _text(content).split(SOURCES_MARK)[0].strip()
        if role in ("user", "assistant") and text:
            out.append({"role": role, "content": text})
    return out


def format_sources(passages):
    """! @brief Bloc des sources affiché sous une réponse.

    @param passages  Liste de dict (forme rag.retrieval.Passage.to_tool) ; vide : aucun bloc.
    @return Markdown (commence par SOURCES_MARK), ou "".
    """
    if not passages:
        return ""
    lines = [f"- {p['source']}" for p in passages if p.get("source")]
    return SOURCES_MARK + "\n" + "\n".join(lines) if lines else ""


def index_message():
    """! @brief État de l'index pour l'en-tête du panneau.

    @return Markdown court.
    """
    try:
        from rag import store
        manifest = store.read_manifest()
    except Exception as e:                             # dépendances RAG absentes de l'environnement
        return f"Index indisponible ({type(e).__name__})."
    if not manifest:
        return "Index en construction (premier lancement : quelques minutes)."
    return f"{manifest.get('chunks', '?')} passages indexés ({', '.join(manifest.get('backends', {}))})."


def start_ingest():
    """! @brief Construit ou met à jour l'index en arrière-plan, une fois par processus.

    Deux passes : d'abord le backend local (rapide, sans clé), pour que l'assistant réponde vite, puis le backend
    demandé (Voyage, régulé par l'offre gratuite). Chaque passe ne fait rien si l'index est déjà à jour.
    Journal : <index>/ingest.log. AIDGE_ASSISTANT_AUTO_INGEST=0 désactive ce lancement.
    """
    global _ingest_started
    if os.environ.get("AIDGE_ASSISTANT_AUTO_INGEST", "1") == "0":
        return
    with _ingest_lock:
        if _ingest_started:
            return
        _ingest_started = True

    def work():
        """! @brief Lance les deux passes d'ingestion, l'une après l'autre."""
        from rag import config
        root = config.ROOT
        log = config.index_dir() / "ingest.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_DISABLE_SYMLINKS_WARNING="1")
        passes = [["--backend", "local"]]
        if config.embeddings_backend() != "local":
            passes.append([])
        with open(log, "a", encoding="utf-8") as f:
            for extra in passes:
                subprocess.run([sys.executable, "-m", "rag.ingest", "--if-stale", *extra], cwd=root, env=env,
                               stdout=f, stderr=subprocess.STDOUT, check=False)
                if extra:                                # index local prêt (premier lancement) : on le précharge
                    warm()

    threading.Thread(target=work, name="rag-ingest", daemon=True).start()


def warm():
    """! @brief Précharge l'index et le modèle d'embeddings local du processus ; sans effet s'il n'y a pas d'index.

    Sans cela, la première question paie les imports de chromadb et fastembed et le chargement du modèle
    (15 à 25 s mesurées dans le conteneur). Ne lève jamais : un échec laisse la première question s'en charger.
    @return True si l'index est chargé.
    """
    try:
        from rag.retrieval import get_retriever
        return get_retriever().warm()
    except Exception:                                    # dépendances absentes, index illisible : sans gravité
        return False


def start_warm():
    """! @brief Lance warm() en arrière-plan, une fois par processus. AIDGE_ASSISTANT_AUTO_INGEST=0 le désactive
    aussi (tests, environnements sans index)."""
    global _warm_started
    if os.environ.get("AIDGE_ASSISTANT_AUTO_INGEST", "1") == "0":
        return
    with _ingest_lock:
        if _warm_started:
            return
        _warm_started = True
    threading.Thread(target=warm, name="rag-warm", daemon=True).start()


def respond(message, history):
    """! @brief Répond à un message, en streaming (générateur Gradio).

    @param message  Question saisie.
    @param history  Messages du Chatbot.
    @return Générateur de (texte de la zone de saisie, messages du Chatbot).
    """
    message = (message or "").strip()
    history = list(history or [])
    if not message:
        yield "", history
        return
    previous = history_for_agent(history)
    history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": "…"}]
    yield "", list(history)
    try:
        from rag.agent import Agent, cited
        agent = Agent()
    except Exception as e:                                       # dépendances absentes, clé manquante
        history[-1] = {"role": "assistant", "content": f"Assistant indisponible : {e}"}
        yield "", history
        return
    answer, searching = "", False
    for event in agent.run(message, previous):
        if event.kind == "text":
            answer += event.data
            history[-1] = {"role": "assistant", "content": answer}
        elif event.kind == "tool" and not answer:
            searching = True
            history[-1] = {"role": "assistant", "content": "_Recherche dans la documentation…_"}
        elif event.kind == "error":
            answer = (answer + "\n\n" if answer else "") + f"⚠️ {event.data}"
            history[-1] = {"role": "assistant", "content": answer}
        elif event.kind == "sources":
            history[-1] = {"role": "assistant",
                           "content": (answer or "Je ne sais pas.") + format_sources(cited(answer, event.data))}
        yield "", list(history)                                  # copie : chaque étape du streaming reste intacte
    if not answer and not searching:
        history[-1] = {"role": "assistant", "content": "Je ne sais pas."}
        yield "", list(history)


def mount(host=None):
    """! @brief Ajoute la bulle et son panneau aux Blocks en cours (à appeler dans le bloc @c with gr.Blocks()).

    @param host  Adresse d'écoute de l'interface (garde-fou d'exposition) ; None : AIDGE_UI_HOST.
    @return (bouton, panneau) : gr.Button et gr.Column.
    """
    enabled, why = status(host)
    with gr.Column(elem_id="chat-panel") as panel:
        gr.Markdown("**🛰️ Assistant AIDGE** · papier de Ballé, Aidge, LoCod, docs du projet",
                    elem_classes="chat-title")
        if not enabled:
            gr.Markdown(why, elem_classes="chat-status")
        else:
            start_ingest()
            start_warm()
            note = gr.Markdown(index_message(), elem_classes="chat-status")
            chat = gr.Chatbot(height=380, show_label=False, buttons=["copy"],
                              placeholder="Posez une question : « Comment fonctionne la GDN ? »")
            msg = gr.Textbox(placeholder="Votre question…", show_label=False, lines=1, max_lines=4,
                             submit_btn=True)
            msg.submit(respond, [msg, chat], [msg, chat], concurrency_limit=4, api_visibility="private")
            msg.submit(index_message, None, note, queue=False, api_visibility="private")
    fab = gr.Button("💬", elem_id="chat-fab", variant="primary")
    fab.click(fn=None, js=TOGGLE_JS)
    return fab, panel
