"""! @brief Recherche hybride : BM25 + vecteurs (Chroma), fusion RRF, rerank Voyage des 8 meilleurs, top 5.

Chaque étape a un repli : question non embarquée par Voyage (quota) -> vecteurs locaux ; aucun vecteur -> BM25 seul ;
rerank impossible -> ordre RRF. Le résultat dit quelles étapes ont servi (champ @c methode), pour le diagnostic.
"""
import threading
from dataclasses import dataclass, field
from pathlib import Path

from rag import config, store
from rag.embeddings import EmbeddingError, LocalEmbedder, QuotaExceeded, VoyageClient, VoyageEmbedder


class IndexMissing(RuntimeError):
    """! @brief Aucun index en service : lancer python -m rag.ingest."""


@dataclass
class Passage:
    """! @brief Passage trouvé pour une question.

    @param chunk  rag.chunking.Chunk.
    @param score  Score final (rerank, ou RRF).
    """

    chunk: object
    score: float

    def to_tool(self):
        """! @brief Forme renvoyée au LLM par l'outil search_docs.

        @return dict (source à citer, qui contient déjà la section ; fichier ; page ; texte, tronqué à
                config.PASSAGE_MAX_CHARS caractères pour ménager les quotas du LLM).
        """
        c = self.chunk
        return {"source": c.citation(), "fichier": c.source, "page": c.page,
                "texte": truncate(c.text, config.PASSAGE_MAX_CHARS)}


def truncate(text, limit):
    """! @brief Coupe un texte trop long sur le dernier espace avant la limite, en le signalant.

    @param text   Texte.
    @param limit  Nombre de caractères au plus (hors marque de coupure).
    @return Le texte, ou son début suivi de « […] ».
    """
    if len(text) <= limit:
        return text
    head = text[:limit]
    cut = head.rfind(" ")
    return (head[:cut] if cut > limit // 2 else head).rstrip() + " […]"


@dataclass
class SearchResult:
    """! @brief Résultat d'une recherche.

    @param passages  Liste de Passage, du plus au moins pertinent.
    @param methode   Étapes utilisées (ex. ["bm25", "voyage", "rrf", "rerank"]).
    @param notes     Replis rencontrés (texte lisible).
    """

    passages: list
    methode: list = field(default_factory=list)
    notes: list = field(default_factory=list)


class Retriever:
    """! @brief Recherche dans l'index en service ; se recharge quand une ingestion publie une nouvelle version.

    @param root      Dossier de l'index ; None : config.index_dir().
    @param voyage    VoyageClient (tests) ; None : créé si VOYAGE_API_KEY est définie et le backend le permet.
    @param local     LocalEmbedder (tests) ; None : créé au premier besoin.
    """

    def __init__(self, root=None, voyage=None, local=None):
        """! @brief Crée le moteur (l'index est chargé au premier appel).

        @param root    Dossier de l'index.
        @param voyage  Client Voyage.
        @param local   Embedder local.
        """
        ## Dossier de l'index.
        self.root = Path(root or config.index_dir())
        self._voyage = voyage
        self._local = local
        self._index = None
        self._lock = threading.Lock()

    def index(self):
        """! @brief Index en service (rechargé si CURRENT a changé).

        @return store.Index.
        @throws IndexMissing s'il n'y a pas d'index.
        """
        with self._lock:
            version = store.current_version(self.root)
            if version is None:
                raise IndexMissing("aucun index : lancer python -m rag.ingest")
            if self._index is None or self._index.version != version:
                self._index = store.Index(version)
            return self._index

    def voyage(self):
        """! @brief Client Voyage, ou None (pas de clé, ou EMBEDDINGS_BACKEND=local).

        @return VoyageClient ou None.
        """
        if self._voyage is None and config.voyage_key() and config.embeddings_backend() != "local":
            self._voyage = VoyageClient(config.voyage_key())
        return self._voyage

    def local(self):
        """! @brief Embedder local (créé au premier besoin).

        @return LocalEmbedder.
        """
        if self._local is None:
            self._local = LocalEmbedder()
        return self._local

    def warm(self):
        """! @brief Charge d'avance l'index, ses collections Chroma et le modèle local (une dizaine de secondes de
        CPU), pour que la première question n'attende pas. Le modèle local sert de repli même avec Voyage : il est
        toujours chargé. Chroma ne lit l'index vectoriel d'une collection qu'à sa première requête (~4 s chacune).

        @return True si l'index a pu être chargé ; False s'il n'existe pas encore.
        """
        try:
            index = self.index()
        except IndexMissing:
            return False
        if "local" in index.backends():
            self.local().embed_query("préchauffage")
        for backend, info in index.manifest.get("backends", {}).items():
            if info.get("dim"):
                index.vector_search(backend, [1.0] + [0.0] * (int(info["dim"]) - 1), 1)
        return True

    def _dense(self, index, query, n, result):
        """! @brief Classement vectoriel, Voyage d'abord, local en repli.

        @param index   store.Index.
        @param query   Question.
        @param n       Nombre de passages.
        @param result  SearchResult complété (méthode, notes).
        @return Liste d'identifiants, vide si aucun backend n'a répondu.
        """
        backends = index.backends()
        if "voyage" in backends and self.voyage() is not None:
            try:
                vec = VoyageEmbedder(self.voyage()).embed_query(query)
                result.methode.append("voyage")
                return index.vector_search("voyage", vec, n)
            except (QuotaExceeded, EmbeddingError) as e:
                result.notes.append(f"embeddings Voyage indisponibles ({e}) : repli local")
        if "local" in backends:
            try:
                vec = self.local().embed_query(query)
                result.methode.append("local")
                return index.vector_search("local", vec, n)
            except Exception as e:                    # modèle absent hors ligne, etc. : BM25 seul
                result.notes.append(f"embeddings locaux indisponibles ({type(e).__name__}) : BM25 seul")
        return []

    def search(self, query, k=None):
        """! @brief Passages les plus pertinents pour la question.

        @param query  Question.
        @param k      Nombre de passages ; None : config.TOP_K.
        @return SearchResult.
        @throws IndexMissing s'il n'y a pas d'index.
        """
        k = k or config.TOP_K
        index = self.index()
        result = SearchResult([])
        n = config.RETRIEVE_CANDIDATES
        rankings = [index.bm25_search(query, n)]
        result.methode.append("bm25")
        dense = self._dense(index, query, n, result)
        if dense:
            rankings.append(dense)
        fused = store.rrf(rankings)[:max(config.RERANK_CANDIDATES, k)]
        result.methode.append("rrf")
        candidates = [(index.by_id[i], s) for i, s in fused if i in index.by_id]
        candidates = candidates[:config.RERANK_CANDIDATES] if len(candidates) > k else candidates
        ranked = None
        if self.voyage() is not None and len(candidates) > 1:
            try:
                order = self.voyage().rerank(query, [c.embed_text() for c, _ in candidates], top_k=k,
                                             max_wait=config.QUERY_MAX_WAIT_S)
                ranked = [Passage(candidates[i][0], s) for i, s in order]
                result.methode.append("rerank")
            except (QuotaExceeded, EmbeddingError) as e:
                result.notes.append(f"rerank Voyage indisponible ({e}) : ordre RRF")
        if ranked is None:
            ranked = [Passage(c, s) for c, s in candidates]
        result.passages = ranked[:k]
        return result


## Moteur partagé par l'outil search_docs (créé au premier appel).
_RETRIEVER = None
## Verrou de création du moteur partagé.
_RETRIEVER_LOCK = threading.Lock()


def get_retriever():
    """! @brief Moteur de recherche partagé du processus.

    @return Retriever.
    """
    global _RETRIEVER
    with _RETRIEVER_LOCK:
        if _RETRIEVER is None:
            _RETRIEVER = Retriever()
        return _RETRIEVER
