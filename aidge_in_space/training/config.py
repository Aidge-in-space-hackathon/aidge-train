"""! @brief Configuration d'entraînement : champs, validation, lecture et écriture YAML, préréglages.

Module sans dépendance à torch (seulement dataclasses, yaml et pathlib) : il est importable à faible
coût par l'interface et par la file d'attente. Le contrat est dans docs/conception_entrainement.md (§ 2.2).
"""
import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path

import yaml

from .. import config as project_config


class ConfigError(ValueError):
    """! @brief Configuration d'entraînement invalide (code de retour 2 de la CLI train)."""


## Champs entiers de TrainConfig.
INT_FIELDS = ("in_ch", "N", "crop", "batch_size", "steps", "eval_every", "n_patches", "seed",
              "log_every", "save_every", "n_eval_images")
## Champs flottants de TrainConfig.
FLOAT_FIELDS = ("lmbda", "lr", "lr_decay_at", "grad_clip")
## Champs textuels de TrainConfig.
STR_FIELDS = ("name", "author", "train_dir", "eval_dir")
## Nombre maximal de caractères du nom et de l'auteur.
MAX_LABEL_LEN = 60
## Taille maximale de la banque de patchs en octets (uint8) : 2 Gio.
MAX_BANK_BYTES = 2 ** 31


def _int(field, v):
    """! @brief Entier d'un champ, ou ConfigError (un flottant de valeur entière est accepté, jamais un booléen)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ConfigError(f"{field} : entier attendu (reçu {v!r})")
    if isinstance(v, float):
        if not math.isfinite(v) or v != int(v):
            raise ConfigError(f"{field} : entier attendu (reçu {v!r})")
        return int(v)
    return v


def _float(field, v):
    """! @brief Flottant fini d'un champ, ou ConfigError (jamais un booléen)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ConfigError(f"{field} : nombre attendu (reçu {v!r})")
    if not math.isfinite(v):
        raise ConfigError(f"{field} : nombre fini attendu (reçu {v!r})")
    return float(v)


def _str(field, v):
    """! @brief Chaîne d'un champ, ou ConfigError."""
    if not isinstance(v, str):
        raise ConfigError(f"{field} : texte attendu (reçu {v!r})")
    return v


def _check(cond, field, problem, value):
    """! @brief Lève ConfigError("<champ> : <problème> (reçu <valeur>)") si la condition est fausse."""
    if not cond:
        raise ConfigError(f"{field} : {problem} (reçu {value!r})")


@dataclass
class TrainConfig:
    """! @brief Réglages d'un entraînement ; les défauts reprennent le notebook (cellule 4, mode GPU complet)."""

    ## Nom du run (facultatif ici ; obligatoire pour un job) : regroupe les checkpoints en série dans le banc.
    name: str = ""
    ## Auteur (facultatif).
    author: str = ""
    ## Dossier des images d'entraînement, relatif à config.ROOT ou absolu.
    train_dir: str = "data/train"
    ## Dossier des images d'évaluation (Kodak par défaut), relatif à config.ROOT ou absolu.
    eval_dir: str = "data/kodak"
    ## Canaux de l'image : 1 = niveaux de gris, 3 = RGB.
    in_ch: int = 1
    ## Canaux internes et du latent.
    N: int = 128
    ## GDN après la 3e convolution (article) ; False = variante sans dernière GDN.
    gdn_last: bool = True
    ## Compromis débit / distorsion : perte = λ·MSE + bpp.
    lmbda: float = 2048
    ## Côté des patchs d'entraînement (multiple de 16).
    crop: int = 256
    ## Patchs par lot.
    batch_size: int = 16
    ## Pas d'apprentissage d'Adam.
    lr: float = 1e-4
    ## Nombre de pas d'entraînement.
    steps: int = 50_000
    ## Fraction des pas après laquelle le pas d'apprentissage est divisé par 10.
    lr_decay_at: float = 0.8
    ## Écrêtage des valeurs des gradients.
    grad_clip: float = 5.0
    ## Évaluation tous les eval_every pas.
    eval_every: int = 2_500
    ## Patchs pré-extraits gardés sur le device (8000 de 256 x 256 en uint8 = 500 Mo).
    n_patches: int = 8_000
    ## Graine des générateurs aléatoires.
    seed: int = 1234
    ## Journal tous les log_every pas.
    log_every: int = 200
    ## Sauvegarde du checkpoint tous les save_every pas.
    save_every: int = 5_000
    ## Nombre d'images d'évaluation.
    n_eval_images: int = 4

    def __post_init__(self) -> None:
        """! @brief Normalise name et author (strip) et les types numériques, puis appelle validate().

        @throws ConfigError si un champ est invalide.
        """
        for f in ("name", "author"):
            v = getattr(self, f)
            if isinstance(v, str):
                setattr(self, f, v.strip())
        for f in INT_FIELDS:
            setattr(self, f, _int(f, getattr(self, f)))
        for f in FLOAT_FIELDS:
            setattr(self, f, _float(f, getattr(self, f)))
        self.validate()

    def validate(self) -> None:
        """! @brief Vérifie les types et les plages de chaque champ.

        @throws ConfigError au premier champ invalide, avec un message « champ : problème (reçu valeur) ».
        """
        for f in ("name", "author"):
            v = _str(f, getattr(self, f)).strip()
            _check(len(v) <= MAX_LABEL_LEN, f, f"{MAX_LABEL_LEN} caractères au plus", v)
            _check(not any(ord(c) < 32 or ord(c) == 127 for c in v), f, "caractère de contrôle interdit", v)
        for f in ("train_dir", "eval_dir"):
            _check(bool(_str(f, getattr(self, f)).strip()), f, "chemin non vide attendu", getattr(self, f))
        _check(isinstance(self.gdn_last, bool), "gdn_last", "booléen attendu", self.gdn_last)
        ints = {f: _int(f, getattr(self, f)) for f in INT_FIELDS}
        floats = {f: _float(f, getattr(self, f)) for f in FLOAT_FIELDS}
        _check(ints["in_ch"] in (1, 3), "in_ch", "1 ou 3 attendu", ints["in_ch"])
        _check(8 <= ints["N"] <= 512, "N", "entre 8 et 512 attendu", ints["N"])
        _check(0 < floats["lmbda"] <= 100_000, "lmbda", "entre 0 (exclu) et 100000 attendu", floats["lmbda"])
        crop = ints["crop"]
        _check(crop % 16 == 0 and 32 <= crop <= 1024, "crop", "multiple de 16 entre 32 et 1024 attendu", crop)
        _check(1 <= ints["batch_size"] <= 256, "batch_size", "entre 1 et 256 attendu", ints["batch_size"])
        _check(0 < floats["lr"] <= 0.01, "lr", "entre 0 (exclu) et 0,01 attendu", floats["lr"])
        _check(1 <= ints["steps"] <= 10_000_000, "steps", "entre 1 et 10 000 000 attendu", ints["steps"])
        _check(0 < floats["lr_decay_at"] <= 1, "lr_decay_at", "entre 0 (exclu) et 1 attendu", floats["lr_decay_at"])
        _check(floats["grad_clip"] > 0, "grad_clip", "strictement positif attendu", floats["grad_clip"])
        for f in ("eval_every", "log_every", "save_every"):
            _check(ints[f] >= 1, f, "au moins 1 attendu", ints[f])
        n = ints["n_patches"]
        _check(1 <= n <= 200_000, "n_patches", "entre 1 et 200000 attendu", n)
        _check(n * ints["in_ch"] * crop * crop <= MAX_BANK_BYTES, "n_patches",
               "banque de patchs de plus de 2 Gio (réduisez n_patches ou crop)", n)
        _check(ints["seed"] >= 0, "seed", "positif ou nul attendu", ints["seed"])
        _check(1 <= ints["n_eval_images"] <= 24, "n_eval_images", "entre 1 et 24 attendu", ints["n_eval_images"])

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        """! @brief Construit une configuration depuis un dictionnaire ; les champs absents prennent leur défaut.

        @param d  Dictionnaire des champs.
        @return La configuration validée.
        @throws ConfigError si une clé est inconnue, un type ou une valeur invalide.
        """
        if not isinstance(d, dict):
            raise ConfigError(f"configuration : dictionnaire attendu (reçu {type(d).__name__})")
        known = {f.name for f in dataclasses.fields(cls)}
        for k in d:
            if k not in known:
                raise ConfigError(f"clé de configuration inconnue : {k!r}")
        return cls(**d)

    @classmethod
    def from_yaml(cls, path) -> "TrainConfig":
        """! @brief Lit une configuration YAML (UTF-8).

        @param path  Fichier YAML.
        @return La configuration validée.
        @throws FileNotFoundError si le fichier est absent.
        @throws ConfigError si le YAML est illisible ou si sa racine n'est pas un dictionnaire.
        """
        text = Path(path).read_text(encoding="utf-8")
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as e:
            raise ConfigError(f"YAML illisible ({Path(path).name}) : {e}") from e
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ConfigError(f"YAML invalide ({Path(path).name}) : un dictionnaire est attendu à la racine")
        return cls.from_dict(data)

    def to_dict(self) -> dict:
        """! @brief Tous les champs en types Python natifs (lmbda entier s'il l'est), sérialisable en JSON.

        @return Dictionnaire dans l'ordre des champs.
        """
        d = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        for f in FLOAT_FIELDS:
            d[f] = float(d[f])
        if d["lmbda"] == int(d["lmbda"]):
            d["lmbda"] = int(d["lmbda"])
        return d

    def to_yaml(self, path=None) -> str:
        """! @brief YAML de to_dict() (ordre des champs, accents conservés).

        @param path  Si donné, le texte est aussi écrit dans ce fichier (UTF-8).
        @return Le texte YAML.
        @throws OSError si l'écriture échoue.
        """
        text = yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def to_cfg_dict(self) -> dict:
        """! @brief Dictionnaire @c cfg du checkpoint : to_dict() sans name ni author (ils vont dans @c run)."""
        d = self.to_dict()
        d.pop("name")
        d.pop("author")
        return d

    @property
    def mode(self) -> str:
        """! @brief Mode d'image de aidge_in_space.images : @c "gris" si in_ch vaut 1, sinon @c "rgb"."""
        return "gris" if self.in_ch == 1 else "rgb"


## Préréglages : champs de TrainConfig (sans name, author, train_dir, eval_dir, lmbda).
PRESETS = {
    "test_rapide": dict(in_ch=1, N=128, gdn_last=True, crop=128, batch_size=4, lr=1e-4, steps=300,
                        lr_decay_at=0.8, grad_clip=5.0, eval_every=150, n_patches=200, seed=1234,
                        log_every=10, save_every=100, n_eval_images=2),
    "standard": dict(in_ch=1, N=128, gdn_last=True, crop=256, batch_size=16, lr=1e-4, steps=50_000,
                     lr_decay_at=0.8, grad_clip=5.0, eval_every=2_500, n_patches=8_000, seed=1234,
                     log_every=200, save_every=5_000, n_eval_images=4),
}


def schedule_for(steps: int) -> dict:
    """! @brief Réglages non exposés dans l'interface, déduits du nombre de pas.

    @param steps  Nombre de pas d'entraînement.
    @return Dictionnaire @c log_every, @c eval_every, @c save_every et @c n_eval_images.
    """
    return {"log_every": min(200, max(10, steps // 30)),
            "eval_every": min(2500, max(50, steps // 20)) if steps >= 1000 else max(1, steps // 2),
            "save_every": min(5000, max(100, steps // 10)),
            "n_eval_images": 2 if steps < 1000 else 4}


def resolve_path(p) -> Path:
    """! @brief Chemin absolu tel quel, sinon relatif à config.ROOT (lu à l'appel).

    @param p  Chemin absolu ou relatif au dépôt.
    @return Chemin absolu.
    """
    p = Path(p)
    return p if p.is_absolute() else project_config.ROOT / p
