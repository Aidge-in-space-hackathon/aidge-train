"""! @brief Ingestion : télécharge et vérifie les sources, découpe, calcule les embeddings, publie une version d'index.

Usage :
    python -m rag.ingest                     toutes les sources (privées comprises), backend EMBEDDINGS_BACKEND
    python -m rag.ingest --if-stale          ne fait rien si l'index en service correspond déjà aux sources
    python -m rag.ingest --public-only --backend voyage    (CI : jamais de source privée dans l'artefact)
Backends : voyage (Voyage AI, clé VOYAGE_API_KEY), local (fastembed), auto (voyage si la clé est là, plus local
en repli). En auto, un quota Voyage épuisé n'arrête pas l'ingestion : l'index est publié avec le seul backend local,
et le cache garde les vecteurs Voyage déjà obtenus pour la prochaine fois.
Codes de sortie : 0 index publié ou à jour, 1 erreur, 2 source refusée (téléchargement, SHA-256).
"""
import argparse
import json
import sys
import time

from rag import config, sources, store
from rag.chunking import chunk_document
from rag.embeddings import (
    EmbeddingCache,
    EmbeddingError,
    LocalEmbedder,
    QuotaExceeded,
    VoyageClient,
    VoyageEmbedder,
    embed_with_cache,
)


def wanted_backends(backend):
    """! @brief Backends à construire.

    @param backend  "auto", "voyage" ou "local".
    @return Liste ordonnée de noms.
    @throws SystemExit (1) si voyage est demandé sans clé.
    """
    if backend == "local":
        return ["local"]
    if backend == "voyage":
        if not config.voyage_key():
            print("EMBEDDINGS_BACKEND=voyage mais VOYAGE_API_KEY est absente", file=sys.stderr)
            raise SystemExit(1)
        return ["voyage"]
    return (["voyage"] if config.voyage_key() else []) + ["local"]


## Modèle de chaque backend (enregistré dans le manifeste).
MODELS = {"voyage": config.VOYAGE_EMBED_MODEL, "local": config.LOCAL_EMBED_MODEL}


def params_signature(public_only):
    """! @brief Paramètres de découpage qui, s'ils changent, imposent de reconstruire l'index.

    @param public_only  Vrai : index sans sources privées.
    @return str.
    """
    return json.dumps({"chunk_tokens": config.CHUNK_TOKENS, "overlap": config.CHUNK_OVERLAP,
                       "chars_per_token": config.CHARS_PER_TOKEN, "public_only": public_only}, sort_keys=True)


def is_fresh(manifest, digest, backends):
    """! @brief L'index en service correspond-il aux sources et contient-il les backends demandés (mêmes modèles) ?

    Un index Voyage + local reste donc valable pour une demande « local » seule : il n'est pas reconstruit.
    @param manifest  Manifeste en service, ou None.
    @param digest    Empreinte des sources et des paramètres de découpage.
    @param backends  Backends demandés.
    @return bool.
    """
    if not manifest or manifest.get("sources_hash") != digest:
        return False
    have = manifest.get("backends", {})
    return all(b in have and have[b].get("model") == MODELS[b] for b in backends)


def run(backend=None, public_only=False, if_stale=False, fetch=True, log=print):
    """! @brief Ingestion complète.

    @param backend      "auto", "voyage", "local" ; None : EMBEDDINGS_BACKEND.
    @param public_only  Vrai : sources privées exclues.
    @param if_stale     Vrai : ne reconstruit pas un index déjà à jour.
    @param fetch        Faux : aucun téléchargement (sources déjà en cache).
    @param log          Fonction d'affichage.
    @return Manifeste de l'index en service.
    """
    backend = backend or config.embeddings_backend()
    backends = wanted_backends(backend)
    t0 = time.time()
    docs = sources.documents(include_private=not public_only, fetch=fetch)
    digest = sources.sources_hash(docs, params_signature(public_only))
    manifest = store.read_manifest()
    if if_stale and is_fresh(manifest, digest, backends):
        log(f"Index à jour ({manifest['chunks']} passages, {', '.join(manifest['backends'])}) : rien à faire")
        return manifest
    log(f"{len(docs)} documents ({sum(d.visibility == 'private' for d in docs)} privés)")

    chunks = []
    for d in docs:
        part = chunk_document(d)
        chunks += part
        log(f"  {d.name} : {len(part)} passages")
    if not chunks:
        raise RuntimeError("aucun passage : vérifier rag/sources.yaml")
    texts = [c.embed_text() for c in chunks]

    cache = EmbeddingCache()
    vectors, infos = {}, {}
    for b in backends:
        if b == "voyage":
            embedder = VoyageEmbedder(VoyageClient(config.voyage_key()))
        else:
            embedder = LocalEmbedder()
        log(f"Embeddings {b} ({embedder.model})...")
        try:
            vecs, fresh = embed_with_cache(embedder, texts, cache)
        except (QuotaExceeded, EmbeddingError) as e:
            if backend == "auto" and b == "voyage":
                log(f"  Voyage indisponible ({e}) : index publié avec le backend local seul, "
                    "les vecteurs déjà obtenus restent en cache")
                continue
            raise
        vectors[b] = vecs
        infos[b] = {"model": embedder.model, "dim": int(len(vecs[0]))}
        log(f"  {len(vecs)} vecteurs, dont {fresh} calculés (le reste vient du cache)")
    if not vectors:
        raise RuntimeError("aucun backend d'embeddings disponible")

    manifest = {
        # Un backend manquant (quota Voyage) rend l'index incomplet : is_fresh() le reconstruira la fois suivante
        "sources_hash": digest,
        "public_only": public_only,
        "chunk_tokens": config.CHUNK_TOKENS, "chunk_overlap": config.CHUNK_OVERLAP,
        "backends": infos,
        "sources": [{"name": d.name, "source_id": d.source_id, "sha256": d.sha256, "visibility": d.visibility,
                     "url": d.url} for d in docs],
    }
    version = store.write_index(chunks, vectors, manifest)
    log(f"Index publié : {version.name} ({len(chunks)} passages, {', '.join(vectors)}) en {time.time() - t0:.0f} s")
    return store.read_manifest()


def main(argv=None):
    """! @brief Point d'entrée en ligne de commande.

    @param argv  Arguments ; ceux de la ligne de commande si None.
    @return Code de sortie.
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Construit l'index de l'assistant RAG à partir de rag/sources.yaml")
    ap.add_argument("--backend", choices=["auto", "voyage", "local"], default=None,
                    help="embeddings (défaut : EMBEDDINGS_BACKEND, sinon auto)")
    ap.add_argument("--public-only", action="store_true", help="exclut les sources privées (CI, artefact)")
    ap.add_argument("--if-stale", action="store_true", help="ne reconstruit que si les sources ont changé")
    ap.add_argument("--no-fetch", action="store_true", help="aucun téléchargement : sources déjà en cache")
    args = ap.parse_args(argv)
    try:
        run(args.backend, args.public_only, args.if_stale, fetch=not args.no_fetch)
    except sources.SourceError as e:
        print(f"Source refusée : {e}", file=sys.stderr)
        return 2
    except (RuntimeError, ValueError, EmbeddingError, QuotaExceeded) as e:
        print(f"Échec de l'ingestion : {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
