"""! @brief Réglages de l'assistant RAG : chemins, découpage, modèles, limites des offres gratuites.

Tout ce qui dépend de l'environnement est lu ici, une seule fois par appel de fonction (pas au chargement du module),
pour que les tests puissent le changer.
"""
import os
from pathlib import Path

## Dossier du paquet rag/.
RAG_DIR = Path(__file__).resolve().parent
## Racine du dépôt (chemins relatifs des documents du projet).
ROOT = RAG_DIR.parent
## Catalogue des sources, source unique de vérité pour l'ingestion.
CATALOG = RAG_DIR / "sources.yaml"

## Taille visée d'un passage, en tokens (estimation, voir chunking.estimate_tokens).
CHUNK_TOKENS = 800
## Chevauchement entre deux passages consécutifs d'une même section, en tokens.
CHUNK_OVERLAP = 100
## Caractères par token pour l'estimation sans tokenizer (anglais et français, ordre de grandeur de Voyage).
CHARS_PER_TOKEN = 4

## Candidats retenus par chaque méthode de recherche (BM25 et vecteurs) avant fusion.
RETRIEVE_CANDIDATES = 20
## Candidats envoyés au rerank : 8 x 800 tokens tiennent sous les 10K tokens/min de Voyage sans moyen de paiement.
RERANK_CANDIDATES = 8
## Passages rendus par search_docs.
TOP_K = 5
## Caractères de texte au plus par passage rendu au LLM (coupé sur un espace) : 5 passages pleins coûtaient ~3 700
## tokens par question, au-delà du budget de 8K tokens/min du secours Groq pour deux questions rapprochées.
PASSAGE_MAX_CHARS = 1200
## Constante de la fusion RRF (valeur usuelle de Cormack et al., 2009).
RRF_K = 60

## Modèle d'embeddings Voyage.
VOYAGE_EMBED_MODEL = "voyage-4"
## Modèle de rerank Voyage.
VOYAGE_RERANK_MODEL = "rerank-2.5-lite"
## API Voyage.
VOYAGE_URL = "https://api.voyageai.com/v1"
## Modèle local de repli (fastembed, multilingue, 384 dimensions, ~220 Mo, sans clé).
LOCAL_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

## Itérations maximales de la boucle d'appel d'outils de l'agent.
MAX_TOOL_ITERATIONS = 4
## Attente maximale (s) d'un créneau Voyage pendant une question : au-delà, repli local plutôt que faire attendre.
QUERY_MAX_WAIT_S = 3.0


def index_dir():
    """! @brief Dossier de l'index (RAG_INDEX_DIR, sinon rag/index).

    @return Path.
    """
    return Path(os.environ.get("RAG_INDEX_DIR") or RAG_DIR / "index")


def sources_dir():
    """! @brief Dossier des sources téléchargées (cache/) et privées (private/) : RAG_SOURCES_DIR, sinon rag/sources.

    @return Path.
    """
    return Path(os.environ.get("RAG_SOURCES_DIR") or RAG_DIR / "sources")


def embeddings_backend():
    """! @brief Backend d'embeddings demandé (EMBEDDINGS_BACKEND).

    @return "auto" (Voyage si la clé est là, plus le repli local), "voyage" ou "local".
    @throws ValueError si la valeur est inconnue.
    """
    value = (os.environ.get("EMBEDDINGS_BACKEND") or "auto").strip().lower()
    if value not in ("auto", "voyage", "local"):
        raise ValueError(f"EMBEDDINGS_BACKEND={value!r} : attendu auto, voyage ou local")
    return value


def voyage_key():
    """! @brief Clé Voyage (VOYAGE_API_KEY), ou None.

    @return str ou None.
    """
    return os.environ.get("VOYAGE_API_KEY") or None


def voyage_limits():
    """! @brief Limites Voyage à respecter : sans moyen de paiement, 3 requêtes et 10K tokens par minute.

    VOYAGE_RPM et VOYAGE_TPM les relèvent si l'offre change.
    @return (requêtes par minute, tokens par minute).
    """
    return int(os.environ.get("VOYAGE_RPM") or 3), int(os.environ.get("VOYAGE_TPM") or 10_000)


def litellm_settings():
    """! @brief Adresse du proxy LiteLLM, clé et nom du modèle.

    @return (base_url, master_key ou None, modèle).
    """
    base = os.environ.get("LITELLM_BASE_URL") or "http://127.0.0.1:4000"
    return base.rstrip("/"), os.environ.get("LITELLM_MASTER_KEY") or None, \
        os.environ.get("RAG_ASSISTANT_MODEL") or "assistant"
