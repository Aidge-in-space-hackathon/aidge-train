"""! @brief Index de l'assistant : passages (JSONL), BM25, Chroma persistant, fusion RRF, manifeste.

Disposition de rag/index/ :
- embeddings.sqlite : cache des vecteurs (rag.embeddings.EmbeddingCache), conservé d'une version à l'autre ;
- models/ : modèle local fastembed ;
- v-<horodatage>/ : une version de l'index (chunks.jsonl, manifest.json, chroma/) ;
- CURRENT : nom de la version en service.
Une ingestion écrit une nouvelle version puis met CURRENT à jour : l'interface ne lit jamais un index à moitié écrit,
et recharge l'index quand CURRENT change.
"""
import json
import os
import re
import shutil
import time
import unicodedata
from pathlib import Path

from rag import config
from rag.chunking import Chunk

## Fichier qui désigne la version en service.
CURRENT = "CURRENT"
## Versions gardées (la courante et la précédente).
KEEP_VERSIONS = 2
## Mots vides français et anglais ignorés par BM25.
STOPWORDS = frozenset("""
le la les un une des du de d l et ou en au aux a à est sont pour par sur dans avec ce cette ces qui que quoi
comment il elle on nous vous ils elles se sa son ses leur leurs ne pas plus fait faire être avoir y
the a an and or of to in on for with is are be by as at it this that from which how what
""".split())


def tokenize(text):
    """! @brief Mots pour BM25 : minuscules, sans accents, sans mots vides ni mots d'une lettre.

    @param text  Texte.
    @return Liste de mots.
    """
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return [w for w in re.findall(r"\w+", text) if len(w) > 1 and w not in STOPWORDS]


def rrf(rankings, k=None):
    """! @brief Fusion de classements par Reciprocal Rank Fusion : score = somme de 1 / (k + rang).

    @param rankings  Listes d'identifiants, chacune du plus au moins pertinent (rang 1 en tête).
    @param k         Constante RRF ; None : config.RRF_K.
    @return Liste de (identifiant, score), score décroissant ; à égalité, ordre de première apparition.
    """
    k = config.RRF_K if k is None else k
    scores, first = {}, {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
            first.setdefault(item, len(first))
    return sorted(scores.items(), key=lambda kv: (-kv[1], first[kv[0]]))


def current_version(root=None):
    """! @brief Dossier de la version en service, ou None.

    @param root  Dossier de l'index ; None : config.index_dir().
    @return Path ou None.
    """
    root = Path(root or config.index_dir())
    try:
        name = (root / CURRENT).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    path = root / name
    return path if (path / "manifest.json").exists() else None


def read_manifest(root=None):
    """! @brief Manifeste de la version en service.

    @param root  Dossier de l'index.
    @return dict, ou None s'il n'y a pas d'index.
    """
    version = current_version(root)
    if version is None:
        return None
    return json.loads((version / "manifest.json").read_text(encoding="utf-8"))


def write_index(chunks, vectors, manifest, root=None):
    """! @brief Écrit une nouvelle version de l'index et la met en service.

    @param chunks    Liste de Chunk.
    @param vectors   dict backend -> liste de vecteurs (même ordre que chunks).
    @param manifest  dict du manifeste (complété ici par la date, le nombre de passages et les backends).
    @param root      Dossier de l'index ; None : config.index_dir().
    @return Dossier de la version écrite.
    """
    import chromadb
    root = Path(root or config.index_dir())
    root.mkdir(parents=True, exist_ok=True)
    # Écrite sous son nom définitif : seule la mise à jour de CURRENT, à la fin, la rend visible (pas de renommage
    # de dossier, impossible sous Windows tant que Chroma garde ses fichiers ouverts).
    stamp = time.strftime("v-%Y%m%d-%H%M%S") + f"-{os.getpid()}"
    for n in range(1000):                             # plusieurs versions dans la même seconde (tests)
        name = stamp + (f"-{n}" if n else "")
        tmp = root / name
        try:
            tmp.mkdir()
            break
        except FileExistsError:
            continue
    with open(tmp / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    client = chromadb.PersistentClient(path=str(tmp / "chroma"),
                                       settings=chromadb.config.Settings(anonymized_telemetry=False))
    for backend, vecs in vectors.items():
        col = client.create_collection(f"docs-{backend}", metadata={"hnsw:space": "cosine"},
                                       embedding_function=None)
        for i in range(0, len(chunks), 500):
            part = chunks[i:i + 500]
            col.add(ids=[c.chunk_id for c in part],
                    embeddings=[list(map(float, v)) for v in vecs[i:i + 500]],
                    metadatas=[{"source": c.source, "page": c.page if c.page is not None else -1,
                                "section": c.section[:500]} for c in part])
    client.close()                                    # libère les fichiers (sinon Windows interdit le nettoyage)
    manifest = dict(manifest, created=time.strftime("%Y-%m-%dT%H:%M:%S%z"), chunks=len(chunks),
                    backends={b: manifest.get("backends", {}).get(b, {}) for b in vectors})
    (tmp / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    final = tmp
    pointer = root / (CURRENT + ".tmp")
    pointer.write_text(name, encoding="utf-8")
    os.replace(pointer, root / CURRENT)
    _prune(root, keep=name)
    return final


def _prune(root, keep):
    """! @brief Supprime les anciennes versions (au mieux : une version ouverte ailleurs reste).

    @param root  Dossier de l'index.
    @param keep  Version en service.
    """
    versions = sorted((p for p in root.glob("v-*") if p.is_dir()), key=lambda p: p.name, reverse=True)
    for old in [v for v in versions if v.name != keep][KEEP_VERSIONS - 1:]:
        shutil.rmtree(old, ignore_errors=True)


class Index:
    """! @brief Version en service de l'index, chargée en mémoire : passages, BM25, collections Chroma.

    @param version  Dossier de la version.
    """

    def __init__(self, version):
        """! @brief Charge la version.

        @param version  Dossier.
        """
        from rank_bm25 import BM25Okapi
        ## Dossier de la version.
        self.version = Path(version)
        ## Manifeste.
        self.manifest = json.loads((self.version / "manifest.json").read_text(encoding="utf-8"))
        with open(self.version / "chunks.jsonl", encoding="utf-8") as f:
            ## Passages, dans l'ordre d'écriture.
            self.chunks = [Chunk(**json.loads(line)) for line in f if line.strip()]
        ## Passage par identifiant.
        self.by_id = {c.chunk_id: c for c in self.chunks}
        ## Index BM25 (texte, document et section).
        self.bm25 = BM25Okapi([tokenize(c.embed_text()) for c in self.chunks]) if self.chunks else None
        self._chroma = None

    def backends(self):
        """! @brief Backends d'embeddings présents dans cette version.

        @return Liste de noms ("voyage", "local").
        """
        return list(self.manifest.get("backends", {}))

    def bm25_search(self, query, n):
        """! @brief Meilleurs passages BM25.

        @param query  Question.
        @param n      Nombre de passages.
        @return Liste d'identifiants (meilleur en tête), sans les passages de score nul.
        """
        if self.bm25 is None:
            return []
        words = tokenize(query)
        if not words:
            return []
        scores = self.bm25.get_scores(words)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:n]
        return [self.chunks[i].chunk_id for i in order if scores[i] > 0]

    def vector_search(self, backend, vector, n):
        """! @brief Plus proches voisins dans la collection Chroma d'un backend.

        @param backend  "voyage" ou "local".
        @param vector   Vecteur de la question.
        @param n        Nombre de passages.
        @return Liste d'identifiants (plus proche en tête).
        """
        import chromadb
        if self._chroma is None:
            self._chroma = chromadb.PersistentClient(
                path=str(self.version / "chroma"), settings=chromadb.config.Settings(anonymized_telemetry=False))
        col = self._chroma.get_collection(f"docs-{backend}", embedding_function=None)
        res = col.query(query_embeddings=[list(map(float, vector))], n_results=min(n, len(self.chunks)))
        return res["ids"][0]
