"""! @brief Catalogue des sources (rag/sources.yaml) : téléchargement épinglé, contrôle SHA-256, liste des documents.

Un document est un fichier local prêt à découper, avec le nom affiché dans les citations. Les fichiers téléchargés
sont vérifiés avant usage ; un fichier modifié à la source est refusé (SourceError) au lieu d'être ingéré en silence.
"""
import fnmatch
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from rag import config

## Extensions prises en charge par le découpage.
SUFFIXES = (".pdf", ".md", ".rst", ".txt")
## Délai d'un téléchargement (s).
DOWNLOAD_TIMEOUT_S = 120


class SourceError(RuntimeError):
    """! @brief Source introuvable, téléchargement en échec ou SHA-256 différent de celui du catalogue."""


@dataclass(frozen=True)
class Document:
    """! @brief Fichier à ingérer.

    @param source_id   Identifiant de la source dans le catalogue.
    @param name        Nom affiché dans les citations (ex. balle2017_end2end_1611.01704v3.pdf, aidge/docs/...).
    @param path        Chemin local du fichier.
    @param sha256      Empreinte du contenu (texte : fins de ligne normalisées).
    @param visibility  "public" ou "private".
    @param url         Adresse d'origine, ou "" pour un fichier du dépôt ou privé.
    """

    source_id: str
    name: str
    path: Path
    sha256: str
    visibility: str
    url: str = ""


def load_catalog(path=None):
    """! @brief Lit et valide le catalogue.

    @param path  Fichier YAML ; None : rag/sources.yaml.
    @return Liste des entrées (dict).
    @throws SourceError si une entrée est incomplète.
    """
    path = Path(path or config.CATALOG)
    entries = yaml.safe_load(path.read_text(encoding="utf-8"))["sources"]
    seen = set()
    for e in entries:
        sid = e.get("id")
        if not sid or sid in seen:
            raise SourceError(f"{path.name} : identifiant absent ou en double ({sid!r})")
        seen.add(sid)
        if e.get("visibility") not in ("public", "private"):
            raise SourceError(f"{sid} : visibility doit valoir public ou private")
        kind = e.get("kind")
        needed = {"url": ("url", "sha256", "file", "version"), "git": ("raw", "commit", "files", "version"),
                  "repo": ("files",), "local": ("dir", "files")}.get(kind)
        if needed is None:
            raise SourceError(f"{sid} : kind inconnu {kind!r}")
        missing = [k for k in needed if not e.get(k)]
        if missing:
            raise SourceError(f"{sid} : champs manquants {missing}")
        if kind == "git" and any(not f.get("path") or not f.get("sha256") for f in e["files"]):
            raise SourceError(f"{sid} : chaque fichier doit avoir path et sha256")
        if kind in ("url", "git") and e["visibility"] != "public":
            raise SourceError(f"{sid} : une source téléchargée est publique par nature")
    return entries


def file_sha256(path):
    """! @brief Empreinte d'un fichier : octets bruts pour un PDF, texte aux fins de ligne normalisées sinon.

    La normalisation donne la même empreinte à un fichier du dépôt extrait sous Windows (CRLF) ou Linux (LF).
    @param path  Fichier.
    @return Empreinte hexadécimale.
    """
    data = Path(path).read_bytes()
    if Path(path).suffix.lower() != ".pdf":
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def _raw_sha256(path):
    """! @brief Empreinte des octets bruts (comparée au catalogue pour un téléchargement).

    @param path  Fichier.
    @return Empreinte hexadécimale.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download(url, dest, sha256, client=None):
    """! @brief Télécharge un fichier épinglé et vérifie son SHA-256 ; ne fait rien s'il est déjà là et intact.

    Écriture atomique (fichier temporaire puis renommage) : un téléchargement interrompu ne laisse rien.
    @param url     Adresse.
    @param dest    Chemin local.
    @param sha256  Empreinte attendue des octets.
    @param client  httpx.Client ; None : un client créé pour l'appel.
    @return dest.
    @throws SourceError si le téléchargement échoue ou si l'empreinte diffère.
    """
    import httpx
    dest = Path(dest)
    if dest.exists() and _raw_sha256(dest) == sha256:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    own = client is None
    client = client or httpx.Client(follow_redirects=True, timeout=DOWNLOAD_TIMEOUT_S,
                                    headers={"User-Agent": "aidge-train-rag"})
    try:
        resp = client.get(url)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        raise SourceError(f"téléchargement impossible : {url} ({e})") from None
    finally:
        if own:
            client.close()
    got = hashlib.sha256(resp.content).hexdigest()
    if got != sha256:
        raise SourceError(f"{url} : SHA-256 {got} au lieu de {sha256}. La source a changé : vérifier puis mettre "
                          "à jour rag/sources.yaml (version et sha256).")
    tmp = dest.with_name(dest.name + f".tmp-{os.getpid()}")
    tmp.write_bytes(resp.content)
    os.replace(tmp, dest)
    return dest


def _repo_files(patterns, root):
    """! @brief Fichiers du dépôt correspondant aux motifs (relatifs à la racine), triés.

    @param patterns  Motifs glob (ex. docs/*.md).
    @param root      Racine du dépôt.
    @return Liste de Path.
    """
    found = []
    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        found += [p for p in matches if p.is_file() and p.suffix.lower() in SUFFIXES and p not in found]
    return found


def documents(include_private=True, fetch=True, catalog=None, root=None, client=None):
    """! @brief Liste les documents à ingérer, en téléchargeant au besoin les sources épinglées.

    @param include_private  Faux : sources privées exclues (CI, artefact d'index publié).
    @param fetch            Faux : ne télécharge rien, une source absente du cache lève SourceError.
    @param catalog          Fichier du catalogue ; None : rag/sources.yaml.
    @param root             Racine du dépôt ; None : celle du paquet.
    @param client           httpx.Client à utiliser pour les téléchargements (tests).
    @return Liste de Document, dans l'ordre du catalogue.
    @throws SourceError (source absente, empreinte différente).
    """
    root = Path(root or config.ROOT)
    cache = config.sources_dir() / "cache"
    docs = []
    for e in load_catalog(catalog):
        if e["visibility"] == "private" and not include_private:
            continue
        sid, vis = e["id"], e["visibility"]
        if e["kind"] == "url":
            dest = cache / sid / e["file"]
            if fetch:
                download(e["url"], dest, e["sha256"], client)
            elif not dest.exists() or _raw_sha256(dest) != e["sha256"]:
                raise SourceError(f"{sid} : {dest} absent ou modifié (relancer sans --no-fetch)")
            docs.append(Document(sid, e["file"], dest, file_sha256(dest), vis, e["url"]))
        elif e["kind"] == "git":
            for f in e["files"]:
                url = e["raw"].format(commit=e["commit"], path=f["path"])
                dest = cache / sid / e["commit"][:12] / f["path"]
                if fetch:
                    download(url, dest, f["sha256"], client)
                elif not dest.exists() or _raw_sha256(dest) != f["sha256"]:
                    raise SourceError(f"{sid} : {dest} absent ou modifié (relancer sans --no-fetch)")
                docs.append(Document(sid, f"{sid}/{f['path']}", dest, file_sha256(dest), vis, url))
        elif e["kind"] == "repo":
            for p in _repo_files(e["files"], root):
                docs.append(Document(sid, p.relative_to(root).as_posix(), p, file_sha256(p), vis))
        else:                                                  # local (privé, non versionné)
            folder = config.sources_dir() / e["dir"]
            files = sorted(p for p in folder.glob("*") if p.is_file()
                           and any(fnmatch.fnmatch(p.name, pat) for pat in e["files"])) if folder.is_dir() else []
            for p in files:
                docs.append(Document(sid, p.name, p, file_sha256(p), vis))
    return docs


def sources_hash(docs, extra=""):
    """! @brief Empreinte de l'ensemble des sources : change dès qu'un document est ajouté, retiré ou modifié.

    @param docs   Liste de Document.
    @param extra  Texte ajouté à l'empreinte (paramètres de découpage, modèles).
    @return Empreinte hexadécimale.
    """
    h = hashlib.sha256(extra.encode())
    for d in sorted(docs, key=lambda d: d.name):
        h.update(f"{d.name}\0{d.sha256}\0{d.visibility}\n".encode())
    return h.hexdigest()
