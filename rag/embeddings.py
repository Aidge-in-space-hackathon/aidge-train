"""! @brief Embeddings et rerank : Voyage AI (régulé pour l'offre gratuite) et repli local fastembed, avec cache.

Sans moyen de paiement, Voyage accepte 3 requêtes et 10K tokens par minute : RateLimiter fait patienter l'ingestion
(lots d'au plus 60 % du budget de tokens, attente exponentielle sur 429), et une question n'attend jamais plus de
config.QUERY_MAX_WAIT_S : au-delà, QuotaExceeded déclenche le repli local. EmbeddingCache garde chaque vecteur par
empreinte du texte et du modèle : une réingestion ne renvoie à Voyage que les passages qui ont changé.
"""
import hashlib
import sqlite3
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from rag import config
from rag.chunking import estimate_tokens

## Part du budget de tokens par minute utilisée par un lot d'ingestion (l'estimation peut sous-compter).
BATCH_TOKEN_SHARE = 0.6
## Textes au plus par requête Voyage.
MAX_BATCH_ITEMS = 128
## Nouvelles tentatives sur 429 / 5xx pendant l'ingestion, attente doublée à chaque fois.
MAX_RETRIES = 5
## Première attente après un 429 (s).
BACKOFF_S = 20.0
## Délai d'une requête Voyage (s).
HTTP_TIMEOUT_S = 60


class QuotaExceeded(RuntimeError):
    """! @brief Quota ou débit Voyage atteint (ou attente trop longue) : il faut se replier sur le local."""


class EmbeddingError(RuntimeError):
    """! @brief Erreur d'embeddings non liée au quota (clé refusée, réponse invalide)."""


class RateLimiter:
    """! @brief Fenêtre glissante d'une minute : requêtes et tokens.

    @param rpm    Requêtes par minute.
    @param tpm    Tokens par minute.
    @param clock  Horloge (tests) ; None : time.monotonic.
    @param sleep  Attente (tests) ; None : time.sleep.
    """

    def __init__(self, rpm, tpm, clock=None, sleep=None):
        """! @brief Crée le limiteur.

        @param rpm    Requêtes par minute.
        @param tpm    Tokens par minute.
        @param clock  Horloge.
        @param sleep  Fonction d'attente.
        """
        ## Requêtes par minute.
        self.rpm = rpm
        ## Tokens par minute.
        self.tpm = tpm
        ## Horloge.
        self.clock = clock or time.monotonic
        ## Fonction d'attente.
        self.sleep = sleep or time.sleep
        ## (instant, tokens) des requêtes de la dernière minute.
        self.events = deque()
        ## Verrou : un limiteur partagé entre les threads de l'interface.
        self.lock = threading.Lock()

    def wait_time(self, tokens):
        """! @brief Attente nécessaire avant une requête de @p tokens tokens.

        @param tokens  Tokens de la requête.
        @return Secondes (0 si possible tout de suite).
        """
        now = self.clock()
        while self.events and now - self.events[0][0] >= 60:
            self.events.popleft()
        wait = 0.0
        if len(self.events) >= self.rpm:
            wait = max(wait, 60 - (now - self.events[len(self.events) - self.rpm][0]))
        used = sum(t for _, t in self.events)
        if used + tokens > self.tpm:
            acc = used
            for ts, t in self.events:             # attendre que assez de tokens sortent de la fenêtre
                acc -= t
                if acc + tokens <= self.tpm:
                    wait = max(wait, 60 - (now - ts))
                    break
        return max(0.0, wait)

    def acquire(self, tokens, max_wait=None):
        """! @brief Réserve un créneau, en attendant au besoin.

        @param tokens    Tokens de la requête.
        @param max_wait  Attente maximale (s) ; None : sans limite.
        @return Vrai si le créneau est réservé, faux si l'attente dépassait @p max_wait.
        """
        with self.lock:
            wait = self.wait_time(tokens)
            if max_wait is not None and wait > max_wait:
                return False
            if wait > 0:
                self.sleep(wait)
            self.events.append((self.clock(), tokens))
            return True


class EmbeddingCache:
    """! @brief Cache SQLite des vecteurs, par empreinte (modèle, type d'entrée, texte).

    @param path  Fichier SQLite ; None : <index>/embeddings.sqlite.
    """

    def __init__(self, path=None):
        """! @brief Ouvre (et crée) le cache.

        @param path  Fichier SQLite.
        """
        ## Fichier SQLite.
        self.path = Path(path or config.index_dir() / "embeddings.sqlite")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, model TEXT, dim INTEGER, vec BLOB)")

    def _connect(self):
        """! @brief Connexion SQLite.

        @return sqlite3.Connection.
        """
        return sqlite3.connect(self.path, timeout=30)

    @staticmethod
    def key(model, input_type, text):
        """! @brief Clé d'un vecteur.

        @param model       Modèle.
        @param input_type  "document" ou "query".
        @param text        Texte embarqué.
        @return Empreinte hexadécimale.
        """
        return hashlib.sha256(f"{model}\0{input_type}\0{text}".encode()).hexdigest()

    def get_many(self, keys):
        """! @brief Vecteurs déjà connus.

        @param keys  Clés.
        @return dict clé -> np.ndarray float32.
        """
        out = {}
        with self._connect() as db:
            for i in range(0, len(keys), 500):
                part = keys[i:i + 500]
                rows = db.execute(f"SELECT key, vec FROM vectors WHERE key IN ({','.join('?' * len(part))})", part)
                out.update({k: np.frombuffer(v, dtype=np.float32) for k, v in rows})
        return out

    def put_many(self, model, items):
        """! @brief Enregistre des vecteurs.

        @param model  Modèle.
        @param items  dict clé -> vecteur.
        """
        with self._connect() as db:
            db.executemany("INSERT OR REPLACE INTO vectors VALUES (?, ?, ?, ?)",
                           [(k, model, len(v), np.asarray(v, dtype=np.float32).tobytes()) for k, v in items.items()])


class VoyageClient:
    """! @brief Client HTTP minimal de l'API Voyage (embeddings et rerank), régulé.

    @param api_key  Clé Voyage.
    @param limiter  RateLimiter partagé ; None : un limiteur aux limites de config.voyage_limits().
    @param http     httpx.Client (tests) ; None : un client créé ici.
    @param sleep    Attente entre deux tentatives (tests) ; None : time.sleep.
    """

    def __init__(self, api_key, limiter=None, http=None, sleep=None):
        """! @brief Crée le client.

        @param api_key  Clé Voyage.
        @param limiter  Limiteur.
        @param http     Client HTTP.
        @param sleep    Fonction d'attente.
        """
        import httpx
        if not api_key:
            raise EmbeddingError("VOYAGE_API_KEY absente")
        ## Limiteur de débit.
        self.limiter = limiter or RateLimiter(*config.voyage_limits())
        ## Client HTTP (la clé n'est jamais journalisée : elle n'est que dans l'en-tête).
        self.http = http or httpx.Client(base_url=config.VOYAGE_URL, timeout=HTTP_TIMEOUT_S,
                                         headers={"Authorization": f"Bearer {api_key}"})
        ## Fonction d'attente.
        self.sleep = sleep or time.sleep

    def _post(self, path, payload, tokens, max_wait=None):
        """! @brief Envoie une requête régulée, avec nouvelles tentatives sur 429 et 5xx si l'attente est permise.

        @param path      Route (/embeddings, /rerank).
        @param payload   Corps JSON.
        @param tokens    Tokens estimés.
        @param max_wait  Attente maximale (s) ; None : ingestion, attente permise.
        @return Réponse JSON.
        @throws QuotaExceeded si le débit ou le quota est atteint (ou l'attente trop longue).
        @throws EmbeddingError pour une autre erreur.
        """
        import httpx
        retries = 0 if max_wait is not None else MAX_RETRIES
        for attempt in range(retries + 1):
            if not self.limiter.acquire(tokens, max_wait):
                raise QuotaExceeded("créneau Voyage indisponible dans le délai")
            try:
                resp = self.http.post(path, json=payload)
            except httpx.HTTPError as e:
                raise EmbeddingError(f"Voyage injoignable : {type(e).__name__}") from None
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt < retries:
                    self.sleep(BACKOFF_S * 2 ** attempt)
                    continue
                if resp.status_code == 429:
                    raise QuotaExceeded(f"Voyage : quota ou débit atteint ({_detail(resp)})")
                raise EmbeddingError(f"Voyage : erreur {resp.status_code}")
            if resp.status_code >= 400:
                raise EmbeddingError(f"Voyage : erreur {resp.status_code} ({_detail(resp)})")
            return resp.json()
        raise QuotaExceeded("Voyage : nouvelles tentatives épuisées")

    def embed(self, texts, input_type, model=None, max_wait=None):
        """! @brief Embeddings d'une liste de textes, par lots régulés.

        @param texts       Textes.
        @param input_type  "document" ou "query".
        @param model       Modèle ; None : config.VOYAGE_EMBED_MODEL.
        @param max_wait    Attente maximale par lot (s) ; None : sans limite.
        @return Liste de vecteurs (np.ndarray float32), dans l'ordre des textes.
        """
        model = model or config.VOYAGE_EMBED_MODEL
        budget = max(1, int(self.limiter.tpm * BATCH_TOKEN_SHARE))
        out = []
        for batch in _batches(texts, budget, MAX_BATCH_ITEMS):
            tokens = sum(estimate_tokens(t) for t in batch)
            data = self._post("/embeddings", {"input": batch, "model": model, "input_type": input_type},
                              tokens, max_wait)
            rows = sorted(data["data"], key=lambda r: r["index"])
            if len(rows) != len(batch):
                raise EmbeddingError("Voyage : nombre de vecteurs inattendu")
            out += [np.asarray(r["embedding"], dtype=np.float32) for r in rows]
        return out

    def rerank(self, query, documents, top_k, model=None, max_wait=None):
        """! @brief Reclasse des passages par pertinence pour la question.

        @param query      Question.
        @param documents  Textes des passages candidats.
        @param top_k      Nombre de passages à garder.
        @param model      Modèle ; None : config.VOYAGE_RERANK_MODEL.
        @param max_wait   Attente maximale (s).
        @return Liste de (indice dans documents, score), du plus pertinent au moins pertinent.
        """
        tokens = estimate_tokens(query) * len(documents) + sum(estimate_tokens(d) for d in documents)
        data = self._post("/rerank", {"query": query, "documents": documents, "top_k": top_k,
                                      "model": model or config.VOYAGE_RERANK_MODEL}, tokens, max_wait)
        return [(r["index"], float(r["relevance_score"])) for r in data["data"]]


def _detail(resp):
    """! @brief Message d'erreur court d'une réponse Voyage (jamais la clé, qui n'est que dans la requête).

    @param resp  httpx.Response.
    @return str.
    """
    try:
        return str(resp.json().get("detail", ""))[:200]
    except ValueError:
        return resp.text[:200]


def _batches(texts, token_budget, max_items):
    """! @brief Regroupe des textes en lots d'au plus @p token_budget tokens estimés et @p max_items textes.

    @param texts         Textes.
    @param token_budget  Tokens par lot.
    @param max_items     Textes par lot.
    @return Itérateur de listes.
    """
    batch, used = [], 0
    for t in texts:
        n = estimate_tokens(t)
        if batch and (used + n > token_budget or len(batch) >= max_items):
            yield batch
            batch, used = [], 0
        batch.append(t)
        used += n
    if batch:
        yield batch


class VoyageEmbedder:
    """! @brief Embeddings Voyage (voyage-4), type d'entrée document ou query.

    @param client  VoyageClient.
    """

    ## Nom du backend dans l'index.
    backend = "voyage"

    def __init__(self, client):
        """! @brief Crée l'embedder.

        @param client  VoyageClient.
        """
        ## Client Voyage.
        self.client = client
        ## Modèle.
        self.model = config.VOYAGE_EMBED_MODEL

    def embed_documents(self, texts):
        """! @brief Vecteurs de passages (ingestion : attend les créneaux).

        @param texts  Textes.
        @return Liste de vecteurs.
        """
        return self.client.embed(texts, "document", self.model)

    def embed_query(self, text):
        """! @brief Vecteur d'une question (n'attend pas plus de config.QUERY_MAX_WAIT_S).

        @param text  Question.
        @return Vecteur.
        """
        return self.client.embed([text], "query", self.model, max_wait=config.QUERY_MAX_WAIT_S)[0]


class LocalEmbedder:
    """! @brief Embeddings locaux fastembed (repli sans clé). Le modèle est téléchargé au premier usage.

    @param model  Modèle fastembed ; None : config.LOCAL_EMBED_MODEL.
    """

    ## Nom du backend dans l'index.
    backend = "local"

    def __init__(self, model=None):
        """! @brief Crée l'embedder (le modèle est chargé au premier appel).

        @param model  Modèle fastembed.
        """
        ## Modèle.
        self.model = model or config.LOCAL_EMBED_MODEL
        self._impl = None

    def _load(self):
        """! @brief Charge le modèle (dossier de cache : <index>/models).

        @return fastembed.TextEmbedding.
        """
        if self._impl is None:
            from fastembed import TextEmbedding
            cache = config.index_dir() / "models"
            cache.mkdir(parents=True, exist_ok=True)
            self._impl = TextEmbedding(self.model, cache_dir=str(cache), threads=1)
        return self._impl

    def embed_documents(self, texts):
        """! @brief Vecteurs de passages.

        @param texts  Textes.
        @return Liste de vecteurs.
        """
        return [np.asarray(v, dtype=np.float32) for v in self._load().passage_embed(list(texts))]

    def embed_query(self, text):
        """! @brief Vecteur d'une question.

        @param text  Question.
        @return Vecteur.
        """
        return np.asarray(next(iter(self._load().query_embed([text]))), dtype=np.float32)


def embed_with_cache(embedder, texts, cache, input_type="document"):
    """! @brief Vecteurs des textes, en n'appelant l'embedder que pour ceux absents du cache.

    Les vecteurs obtenus sont enregistrés au fur et à mesure (par lots) : une ingestion interrompue par le quota
    garde ce qu'elle a déjà payé.
    @param embedder    VoyageEmbedder ou LocalEmbedder.
    @param texts       Textes.
    @param cache       EmbeddingCache.
    @param input_type  Type d'entrée (fait partie de la clé).
    @return (liste de vecteurs dans l'ordre des textes, nombre de textes réellement embarqués).
    """
    keys = [cache.key(embedder.model, input_type, t) for t in texts]
    known = cache.get_many(list(dict.fromkeys(keys)))
    todo = [i for i, k in enumerate(keys) if k not in known]
    step = MAX_BATCH_ITEMS if embedder.backend == "local" else 16
    for start in range(0, len(todo), step):
        idx = todo[start:start + step]
        vecs = embedder.embed_documents([texts[i] for i in idx])
        fresh = {keys[i]: v for i, v in zip(idx, vecs, strict=True)}
        cache.put_many(embedder.model, fresh)
        known.update(fresh)
    return [known[k] for k in keys], len(todo)
