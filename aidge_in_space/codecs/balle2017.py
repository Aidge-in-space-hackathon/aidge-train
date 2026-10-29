"""! @brief Ballé 2017 entraîné par le notebook : un codec par architecture trouvée dans outputs/checkpoints.

Chaque fichier bls2017_c{canaux}_N{N}_lambda{λ}.pt devient un niveau ; les fichiers de même
architecture (canaux, N, GDN finale) sont regroupés dans un même codec, triés par λ croissant
(donc par débit croissant).

Les checkpoints écrits par la plateforme d'entraînement portent une clé @c run (nom, auteur, job) :
ils sont regroupés en une série par architecture et par nom de run, avec un niveau par λ.
"""
import re
import unicodedata

import torch

from ..config import CKPT_DIR
from ..nn import Balle2017
from .base import LearnedCodec, Level

## Motif des checkpoints écrits par le notebook.
PATTERN = "bls2017_c*_N*_lambda*.pt"


def slug(name):
    """! @brief Nom de run réduit à un identifiant ASCII : minuscules, tirets, 40 caractères au plus.

    Exemple : « Lune LRO (été) » donne @c lune-lro-ete.
    @param name  Nom du run.
    @return Identifiant, ou @c "run" s'il est vide.
    """
    ascii_name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")[:40].rstrip("-") or "run"


class Balle2017Codec(LearnedCodec):
    """! @brief Modèles Ballé 2017 d'une même architecture (et d'un même run, pour une série), un niveau par λ."""

    def __init__(self, in_ch, N, gdn_last, entries, series=None):
        """! @brief Crée le codec d'une architecture, ou d'une série de la plateforme d'entraînement.

        @param in_ch     Canaux de l'image (1 = gris, 3 = RGB).
        @param N         Canaux des couches.
        @param gdn_last  Présence de la dernière GDN.
        @param entries   Checkpoints, dictionnaires avec les clés @c path, @c lmbda et @c step ; pour une
                         série aussi @c job_id, @c name, @c author et @c complete.
        @param series    Nom du run d'une série (checkpoints écrits par la plateforme), ou None pour les
                         checkpoints du notebook, qui n'ont pas de run.
        """
        super().__init__()
        ## Architecture commune aux checkpoints : canaux de l'image, canaux des couches, GDN finale.
        self.in_ch, self.N, self.gdn_last = in_ch, N, gdn_last
        ## Nom du run de la série, ou None pour les checkpoints du notebook.
        self.series = series
        ## Checkpoints triés par λ croissant (donc par débit croissant), puis par numéro de job pour une série.
        self.entries = sorted(entries, key=lambda e: e["lmbda"]) if series is None else sorted(
            entries, key=lambda e: (e["lmbda"], -1 if e.get("job_id") is None else e["job_id"]))
        suffix = "" if gdn_last else "-sans-gdn3"
        ## Identifiant construit à partir de l'architecture, par exemple @c balle2017-c1-n128 (puis du run pour une série).
        self.key = f"balle2017-c{in_ch}-n{N}{suffix}" + ("" if series is None else f"-{slug(series)}")
        color = "gris" if in_ch == 1 else "RGB"
        ## Nom affiché.
        self.label = (f"Ballé 2017 · N={N} · {color}" + ("" if gdn_last else " · sans GDN finale")
                      + ("" if series is None else f" · {series}"))
        if series is None:
            ## Description affichée, avec le nombre de checkpoints.
            self.description = ("Modèle du notebook (conv 9x9/4, 5x5/2, 5x5/2 + GDN, prior factorisé), "
                                f"entraîné localement : {len(self.entries)} checkpoint(s).")
        else:
            authors = sorted({e["author"] for e in self.entries if e.get("author")})
            self.description = (f"Run « {series} » ({', '.join(authors) or 'auteur inconnu'}), "
                                f"entraîné par la plateforme : {len(self.entries)} checkpoint(s).")
        ## Un modèle gris n'accepte que le mode gris ; un modèle RGB accepte les deux.
        self.modes = ("gris",) if in_ch == 1 else ("rgb", "gris")

    def levels(self, mode):
        """! @copydoc aidge_in_space.codecs.base.Codec.levels

        La valeur de chaque niveau est le chemin de son checkpoint.
        """
        if self.series is None:
            return [Level(str(e["path"]), f"λ={e['lmbda']} ({e['step']:,} pas)".replace(",", " "))
                    for e in self.entries]
        return [Level(str(e["path"]), f"λ={e['lmbda']:g} ({e['step']:,} pas)".replace(",", " ")
                      + f" · {e.get('name') or self.series}"
                      + ("" if e.get("job_id") is None else f" · job {e['job_id']}")
                      + ("" if e.get("complete", True) else " · en cours"))
                for e in self.entries]

    def make_model(self, level):
        """! @copydoc aidge_in_space.codecs.base.LearnedCodec.make_model"""
        return Balle2017(self.in_ch, self.N, self.gdn_last)

    def load_weights(self, model, level):
        """! @copydoc aidge_in_space.codecs.base.LearnedCodec.load_weights"""
        ck = torch.load(level.value, map_location="cpu")
        model.load_state_dict(ck["state_dict"])

    def architecture(self, level):
        """! @copydoc aidge_in_space.codecs.base.LearnedCodec.architecture"""
        return f"N={self.N}" + ("" if self.gdn_last else ", sans GDN finale")


def codecs():
    """! @brief Un codec par architecture de checkpoint trouvée, et par run pour ceux de la plateforme.

    Les checkpoints sans clé @c run (notebook) sont regroupés par architecture ; ceux qui en ont une
    le sont par architecture et nom de run. Les fichiers illisibles sont signalés et ignorés. Une clé
    de série déjà prise reçoit un suffixe @c -2, @c -3, etc.
    """
    groups = {}
    for path in sorted(CKPT_DIR.glob(PATTERN)):
        try:
            ck = torch.load(path, map_location="cpu")
            cfg = ck["cfg"]
        except Exception as e:
            print(f"Checkpoint ignoré ({path.name}) : {e}")
            continue
        entry = {"path": path, "lmbda": cfg["lmbda"], "step": ck.get("step", 0)}
        run = ck.get("run")
        arch = (cfg["in_ch"], cfg["N"], cfg.get("gdn_last", True))
        if isinstance(run, dict):
            name = str(run.get("name") or "")
            entry.update(job_id=run.get("job_id"), name=name, author=str(run.get("author") or ""),
                         complete=ck.get("step", 0) >= cfg.get("steps", 0))
            group = groups.setdefault(arch + (slug(name),), {"series": name or "run", "entries": []})
        else:
            group = groups.setdefault(arch + (None,), {"series": None, "entries": []})
        group["entries"].append(entry)
    found = [Balle2017Codec(*arch[:3], g["entries"], g["series"]) for arch, g in groups.items()]
    used = {c.key for c in found if c.series is None}
    for c in (c for c in found if c.series is not None):
        base, k = c.key, 2
        while c.key in used:
            c.key, k = f"{base}-{k}", k + 1
        used.add(c.key)
    return found
