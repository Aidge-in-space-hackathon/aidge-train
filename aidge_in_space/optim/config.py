"""! @brief Configuration d'une optimisation Aidge : champs, validation, lecture et écriture YAML.

Module sans dépendance à torch ni à Aidge (seulement dataclasses, yaml et pathlib) : l'interface et la file d'attente
l'importent à faible coût. L'existence des fichiers et du jeu de calibration se vérifie au démarrage du pipeline, pas
ici. Contrat : docs/conception_optimisation.md (§ 4).
"""
import dataclasses
import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..training.config import ConfigError

## Recettes de précision : FP32 de référence, tentative INT8 sur le graphe entier, convolutions INT8 et GDN flottante.
RECIPES = ("fp32", "int8", "mixte")
## Recette proposée par défaut dans l'interface.
DEFAULT_RECIPE = "mixte"
## Modes d'écrêtage de la calibration, noms de aidge_quantization.Clipping.
CLIPPINGS = ("MAX", "MSE", "KL")
## Multiple obligatoire des côtés d'une tuile : le sous-échantillonnage de l'encodeur Ballé 2017.
TILE_MULTIPLE = 16
## Plus petit et plus grand côté de tuile acceptés.
TILE_MIN, TILE_MAX = 16, 1024
## Nombre maximal de caractères du nom et de l'auteur (comme un entraînement).
MAX_LABEL_LEN = 60
## Clés ajoutées à la configuration d'un job pour les colonnes de la base ; absentes d'OptimConfig.
JOB_KEYS = ("in_ch", "N", "lmbda", "steps")
## Opérateurs qui restent en flottant dans la recette mixte (la GDN gelée).
GDN_OPS = ["Sqrt", "Div", "Mul"]


def _check(cond, name, problem, value):
    """! @brief Lève ConfigError(« champ : problème (reçu valeur) ») si la condition est fausse."""
    if not cond:
        raise ConfigError(f"{name} : {problem} (reçu {value!r})")


def _int(name, v):
    """! @brief Entier d'un champ, ou ConfigError (un flottant de valeur entière est accepté, jamais un booléen)."""
    ok = isinstance(v, int) or (isinstance(v, float) and math.isfinite(v) and v == int(v))
    _check(ok and not isinstance(v, bool), name, "entier attendu", v)
    return int(v)


def _bool(name, v):
    """! @brief Booléen d'un champ, ou ConfigError."""
    _check(isinstance(v, bool), name, "booléen attendu", v)
    return v


def _text(name, v):
    """! @brief Chaîne non vide d'un champ, ou ConfigError."""
    _check(isinstance(v, str) and bool(v.strip()), name, "texte non vide attendu", v)
    return v.strip()


def _section(cls, name, value):
    """! @brief Construit une section depuis un dictionnaire, ou la rend telle quelle si elle est déjà construite.

    @param cls    Dataclass de la section.
    @param name   Nom de la section, pour les messages.
    @param value  Dictionnaire des champs, ou instance de @p cls.
    @return L'instance de @p cls.
    @throws ConfigError si une clé est inconnue ou si @p value n'est pas un dictionnaire.
    """
    if isinstance(value, cls):
        return value
    _check(isinstance(value, dict), name, "dictionnaire attendu", value)
    known = {f.name for f in dataclasses.fields(cls)}
    for k in value:
        if k not in known:
            raise ConfigError(f"clé de configuration inconnue : '{name}.{k}'")
    return cls(**value)


@dataclass
class SourceConfig:
    """! @brief Modèle à optimiser."""

    ## Identifiant du codec d'origine (aidge_in_space.codecs), pour l'affichage et le regroupement ; peut être vide.
    codec: str = ""
    ## Checkpoint Ballé 2017, relatif à config.ROOT ou absolu.
    checkpoint: str = ""

    def __post_init__(self) -> None:
        """! @brief Vérifie les champs.

        @throws ConfigError si le checkpoint est vide.
        """
        _check(isinstance(self.codec, str), "source.codec", "texte attendu", self.codec)
        self.checkpoint = _text("source.checkpoint", self.checkpoint)


@dataclass
class PtqConfig:
    """! @brief Quantification après entraînement."""

    ## Bits des convolutions : 8 seulement, Aidge 0.10.1 n'a pas de noyau CPU Pad en int16.
    bits: int = 8
    ## Écrêtage de la calibration, parmi CLIPPINGS.
    clipping: str = "MAX"
    ## Quantification des poids par canal.
    per_channel: bool = False
    ## Nombre de tuiles de calibration.
    n_calibration: int = 32
    ## Jeu de calibration : sous-dossier de data/train.
    calibration_set: str = ""
    ## Graine du tirage des tuiles de calibration.
    seed: int = 0
    ## Opérateurs gardés en flottant ; documentaire et contrôlé : seule la GDN est acceptée.
    float_ops: list = field(default_factory=lambda: list(GDN_OPS))

    def __post_init__(self) -> None:
        """! @brief Vérifie les champs.

        @throws ConfigError au premier champ invalide.
        """
        self.bits = _int("ptq.bits", self.bits)
        _check(self.bits == 8, "ptq.bits", "8 seulement : Aidge 0.10.1 n'a pas de noyau CPU Pad en int16", self.bits)
        _check(self.clipping in CLIPPINGS, "ptq.clipping", f"parmi {', '.join(CLIPPINGS)} attendu", self.clipping)
        self.per_channel = _bool("ptq.per_channel", self.per_channel)
        self.n_calibration = _int("ptq.n_calibration", self.n_calibration)
        _check(1 <= self.n_calibration <= 512, "ptq.n_calibration", "entre 1 et 512 attendu", self.n_calibration)
        self.calibration_set = _text("ptq.calibration_set", self.calibration_set)
        self.seed = _int("ptq.seed", self.seed)
        _check(self.seed >= 0, "ptq.seed", "positif ou nul attendu", self.seed)
        _check(isinstance(self.float_ops, list) and sorted(self.float_ops) == sorted(GDN_OPS), "ptq.float_ops",
               f"{GDN_OPS} attendu : seule la GDN reste en flottant", self.float_ops)


@dataclass
class EvaluateConfig:
    """! @brief Évaluation débit-distorsion."""

    ## Dossier d'évaluation, relatif à config.ROOT ou absolu.
    eval_dir: str = "data/kodak"
    ## Nombre d'images évaluées.
    n_images: int = 24
    ## La série de niveaux n'a pas le même entraînement : le BD-rate est signalé comme indicatif.
    indicative_series: bool = False
    ## Raison affichée pour une série indicative.
    indicative_reason: str = ""

    def __post_init__(self) -> None:
        """! @brief Vérifie les champs.

        @throws ConfigError au premier champ invalide.
        """
        self.eval_dir = _text("evaluate.eval_dir", self.eval_dir)
        self.n_images = _int("evaluate.n_images", self.n_images)
        _check(1 <= self.n_images <= 24, "evaluate.n_images", "entre 1 et 24 attendu", self.n_images)
        self.indicative_series = _bool("evaluate.indicative_series", self.indicative_series)
        _check(isinstance(self.indicative_reason, str), "evaluate.indicative_reason", "texte attendu",
               self.indicative_reason)


@dataclass
class ExportConfig:
    """! @brief Exports C."""

    ## Génère l'export C++ d'Aidge (et, pour la recette mixte, le fichier C pour LoCod).
    generate: bool = True
    ## Compile et vérifie les exports si g++ est disponible ; sans effet si generate est faux.
    compile: bool = True

    def __post_init__(self) -> None:
        """! @brief Vérifie les champs.

        @throws ConfigError si un champ n'est pas un booléen.
        """
        self.generate = _bool("export.generate", self.generate)
        self.compile = _bool("export.compile", self.compile)


@dataclass
class OptimConfig:
    """! @brief Une optimisation : un checkpoint, une tuile, des recettes. Les sections sont des dataclasses."""

    ## Nom (facultatif ici ; obligatoire pour un job) : regroupe les niveaux en une courbe.
    name: str = ""
    ## Auteur (facultatif).
    author: str = ""
    ## Modèle à optimiser (SourceConfig).
    source: SourceConfig = None
    ## Taille de tuile [C, H, W].
    tile: list = field(default_factory=lambda: [3, 64, 64])
    ## Recettes demandées ; fp32 est ajoutée si elle manque, car elle sert de référence.
    recipes: list = field(default_factory=lambda: list(RECIPES))
    ## Quantification (PtqConfig).
    ptq: PtqConfig = None
    ## Évaluation (EvaluateConfig).
    evaluate: EvaluateConfig = field(default_factory=EvaluateConfig)
    ## Exports (ExportConfig).
    export: ExportConfig = field(default_factory=ExportConfig)

    def __post_init__(self) -> None:
        """! @brief Construit les sections, normalise les recettes, puis valide.

        @throws ConfigError si un champ est invalide.
        """
        _check(self.source is not None, "source", "section obligatoire", self.source)
        _check(self.ptq is not None, "ptq", "section obligatoire", self.ptq)
        self.source = _section(SourceConfig, "source", self.source)
        self.ptq = _section(PtqConfig, "ptq", self.ptq)
        self.evaluate = _section(EvaluateConfig, "evaluate", self.evaluate)
        self.export = _section(ExportConfig, "export", self.export)
        for f in ("name", "author"):
            v = getattr(self, f)
            _check(isinstance(v, str), f, "texte attendu", v)
            v = v.strip()
            _check(len(v) <= MAX_LABEL_LEN, f, f"{MAX_LABEL_LEN} caractères au plus", v)
            _check(not any(ord(c) < 32 or ord(c) == 127 for c in v), f, "caractère de contrôle interdit", v)
            setattr(self, f, v)
        _check(isinstance(self.tile, (list, tuple)) and len(self.tile) == 3, "tile", "[C, H, W] attendu", self.tile)
        self.tile = [_int("tile", v) for v in self.tile]
        c, h, w = self.tile
        _check(c in (1, 3), "tile", "C vaut 1 ou 3", self.tile)
        for side in (h, w):
            _check(side % TILE_MULTIPLE == 0 and TILE_MIN <= side <= TILE_MAX, "tile",
                   f"H et W multiples de {TILE_MULTIPLE} entre {TILE_MIN} et {TILE_MAX} attendus", self.tile)
        _check(isinstance(self.recipes, (list, tuple)) and len(self.recipes) > 0, "recipes",
               f"liste non vide parmi {', '.join(RECIPES)} attendue", self.recipes)
        for r in self.recipes:
            _check(r in RECIPES, "recipes", f"parmi {', '.join(RECIPES)} attendu", r)
        self.recipes = [r for r in RECIPES if r == "fp32" or r in self.recipes]

    @classmethod
    def from_dict(cls, d: dict) -> "OptimConfig":
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
    def from_yaml(cls, path) -> "OptimConfig":
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
        if not isinstance(data, dict):
            raise ConfigError(f"YAML invalide ({Path(path).name}) : un dictionnaire est attendu à la racine")
        return cls.from_dict(data)

    @classmethod
    def from_job(cls, d: dict) -> "OptimConfig":
        """! @brief Relit la configuration stockée dans un job, sans les clés réservées aux colonnes de la base.

        @param d  Configuration du job (voir to_job()).
        @return La configuration validée.
        @throws ConfigError si elle est invalide.
        """
        return cls.from_dict({k: v for k, v in d.items() if k not in JOB_KEYS})

    def to_dict(self) -> dict:
        """! @brief Tous les champs en types Python natifs, sérialisable en JSON.

        @return Dictionnaire dans l'ordre des champs.
        """
        return dataclasses.asdict(self)

    def to_job(self, in_ch: int, N: int, lmbda: float, steps: int) -> dict:
        """! @brief Configuration à soumettre à JobStore.submit, avec les valeurs des colonnes de la base.

        @param in_ch  Canaux du checkpoint source.
        @param N      Largeur du checkpoint source.
        @param lmbda  λ du checkpoint source.
        @param steps  Nombre d'étapes prévues (voir n_steps()).
        @return Dictionnaire sérialisable en JSON.
        """
        return {**self.to_dict(), "in_ch": int(in_ch), "N": int(N), "lmbda": lmbda, "steps": int(steps)}

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

    @property
    def mode(self) -> str:
        """! @brief Mode d'image de aidge_in_space.images : @c "gris" si la tuile a un canal, sinon @c "rgb"."""
        return "gris" if self.tile[0] == 1 else "rgb"


def n_steps(cfg: OptimConfig) -> int:
    """! @brief Nombre d'étapes du pipeline, pour la colonne @c steps du job et la progression.

    Import, rapport, mémoire et calibration, puis pour chaque recette : quantification, fidélité, débit-distorsion,
    et si l'export est demandé : export, compilation et vérification (et le fichier LoCod pour la recette mixte).
    @param cfg  Configuration.
    @return Nombre d'étapes, au moins 1.
    """
    per_recipe = 3 + (3 if cfg.export.generate else 0)
    locod = 1 if cfg.export.generate and "mixte" in cfg.recipes else 0
    return 4 + per_recipe * len(cfg.recipes) + locod
