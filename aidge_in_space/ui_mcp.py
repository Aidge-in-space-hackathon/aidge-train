"""! @brief Serveur MCP de l'interface : les actions de la plateforme, outils d'un agent (Claude Code, Claude Desktop…).

Gradio transforme chaque endpoint public de l'API en outil MCP, servi à http://ADRESSE:7860/gradio_api/mcp/. Les
événements de l'interface ne s'y prêtent pas : leur schéma vient des composants (entiers annoncés comme du texte, champs
facultatifs rendus obligatoires, valeurs des menus figées au démarrage). mount() déclare donc, avec gr.api, des fonctions
écrites pour un agent : types simples, valeurs par défaut, et un outil list_options qui lit les valeurs permises à
l'appel. Gradio tire le schéma des annotations et la description de TOOLS (format Google). restrict_api() rend privés
tous les autres événements : l'interface les appelle toujours, l'API et le MCP ne les voient pas.

Activation : AIDGE_UI_MCP=1, avec le paquet mcp installé (requirements.txt). Garde-fou : launch_optim met des jobs
dans la file et l'API n'a pas d'authentification à elle ; le MCP est donc refusé quand l'interface est joignable
depuis une autre machine, sauf si AIDGE_UI_MCP_TRUSTED_AUTH=1 atteste qu'une authentification la protège.
"""
import functools
import importlib.util
import inspect
import json
import os

import gradio as gr

from . import bench, ui_optim
from .codecs import discover, get
from .jobs import JobStore
from .ui_assistant import exposed_locally_only

## Variable d'environnement qui active le MCP.
ENABLE_VAR = "AIDGE_UI_MCP"
## Variable qui atteste qu'une authentification protège une interface exposée au réseau.
TRUSTED_AUTH_VAR = "AIDGE_UI_MCP_TRUSTED_AUTH"
## Taille d'image par défaut des outils de coût à bord (clé de app.PROFILE_PRESETS).
DEFAULT_SIZE = "512 x 512"
## Nombre de noms d'images donnés en exemple par jeu dans list_options.
EXAMPLE_IMAGES = 10
## Rappel ajouté aux erreurs de valeur.
OPTIONS_HINT = "Les valeurs permises sont données par l'outil list_options."

## Description de chaque outil, lue par Gradio (format Google : Args:, Returns:). Mêmes noms que les fonctions.
TOOLS = {
    "list_options": """Valeurs permises des autres outils, lues à l'instant : à appeler en premier.

Returns:
    JSON : modèles appris (identifiant, mode, niveaux), codecs par mode, jeux d'images (nombre et exemples de noms),
    moteurs, tailles d'image, campagnes enregistrées, et pour l'optimisation Aidge : checkpoints, jeux de calibration,
    recettes, tuiles et écrêtages.
""",
    "test_model": """Teste un modèle de compression sur une image et le compare à des codecs de référence au débit le plus proche.

Args:
    model: Identifiant du modèle appris (list_options, models).
    level: Indice du niveau de compression du modèle (list_options, levels ; 0 par défaut).
    image: Nom de l'image dans le jeu (kodim01.png par défaut).
    dataset: Jeu d'images (kodak par défaut).
    references: Identifiants des codecs de comparaison, du même mode que le modèle ; absent : JPEG et JPEG 2000.
    backend: Moteur de l'encodeur : torch (par défaut), onnxruntime ou aidge (bien plus lent).
    real_rate: Vrai (par défaut) : débit réel mesuré par le codeur arithmétique ; faux : débit estimé, plus rapide.

Returns:
    JSON : une mesure par codec (bpp, plus bas = plus compressé ; PSNR et MS-SSIM, plus haut = plus fidèle ; temps).
""",
    "benchmark": """Lance une campagne de mesures débit-distorsion de plusieurs codecs sur un jeu d'images et l'enregistre (CSV).

Peut durer plusieurs minutes. Une campagne n'en écrase jamais une autre.

Args:
    codecs: Identifiants des codecs à mesurer (list_options, codecs du mode choisi).
    dataset: Jeu d'images (kodak par défaut).
    n_images: Nombre d'images du jeu à utiliser (4 par défaut).
    mode: gris (par défaut) ou rgb.
    points: Niveaux mesurés par codec, de 2 à 12 (6 par défaut ; au moins 4 pour un BD-rate).
    backend: Moteur de l'encodeur des modèles : torch (par défaut), onnxruntime ou aidge.
    real_rate: Vrai (par défaut) : débit réel ; faux : débit estimé, plus rapide.
    name: Nom de la campagne (facultatif).
    reference: Courbe de référence du BD-rate (facultatif : JPEG, sinon la première courbe).
    metric: Qualité du BD-rate : psnr (par défaut) ou ms_ssim_db.

Returns:
    JSON : statut (nom, nombre de mesures, erreurs), BD-rate de chaque courbe (négatif = moins de bits que la
    référence à qualité égale), points moyens des courbes et chemin du CSV.
""",
    "profile": """Profil couche par couche de la partie d'un modèle qui tourne à bord (encodeur) : calcul et mémoire.

Calculé sur l'architecture, sans poids ni image.

Args:
    model: Identifiant du modèle appris (list_options, models).
    level: Indice du niveau de compression (0 par défaut).
    size: Taille de l'image (list_options, sizes ; 512 x 512 par défaut).

Returns:
    JSON : résumé (GMAC, paramètres, mémoire) et tableau par couche.
""",
    "compare_encoders": """Compare le coût à bord (calcul, poids, activations) de l'encodeur de chaque modèle appris.

Args:
    size: Taille de l'image (list_options, sizes ; 512 x 512 par défaut).
    mode: gris (par défaut) ou rgb.

Returns:
    JSON : une ligne par architecture (GMAC, opérations sqrt+div, paramètres, poids fp32 et int8, plus grande activation,
    pic mémoire entrée+sortie).
""",
    "check_backends": """Vérifie que l'encodeur d'un modèle donne les mêmes latents sur PyTorch, ONNX Runtime et Aidge.

Entrée aléatoire à graine fixe, 1 thread ; le premier passage (export ONNX, compilation) est chronométré à part.

Args:
    model: Identifiant du modèle appris (list_options, models).
    level: Indice du niveau de compression (0 par défaut).
    size: Taille de l'image (list_options, sizes ; 512 x 512 par défaut).

Returns:
    JSON : une ligne par moteur (écart des latents face à PyTorch, latents identiques en %, temps).
""",
    "analyze_optim": """Analyse, sans rien lancer ni enregistrer, l'optimisation Aidge d'un modèle entraîné : tiendra-t-il dans le FPGA ?

Quelques secondes. Seul le premier checkpoint est analysé.

Args:
    checkpoints: Checkpoints à optimiser (list_options, optim.checkpoints).
    tile: Côté des tuiles traitées, en pixels : 32, 64 (par défaut), 128 ou 256 ; plus petit = moins de mémoire.
    recipes: Recettes parmi mixte, fp32, int8 ; absent : mixte (convolutions en entiers, GDN en flottant).
    clipping: Écrêtage de la calibration : MAX (par défaut), MSE ou KL.
    per_channel: Vrai : une échelle de quantification par filtre au lieu d'une par couche (faux par défaut).
    n_calibration: Nombre de tuiles de calibration, de 1 à 512 (32 par défaut).
    calibration_set: Jeu de calibration (list_options, optim.calibration_sets) ; facultatif pour l'analyse.
    n_images: Nombre d'images Kodak d'évaluation, de 1 à 24 (24 par défaut).

Returns:
    JSON : rapport Markdown (mémoire par recette face à celle du FPGA, opérateurs pris en charge).
""",
    "launch_optim": """Met dans la file un job d'optimisation Aidge par checkpoint (quantification, mesures, code C pour LoCod).

Chaque job dure plusieurs minutes, en arrière-plan sur le serveur : suivre avec list_optim_jobs.

Args:
    name: Nom de la série, qui regroupe les λ en une courbe (ex. lune-mixte-n64), 60 caractères au plus.
    checkpoints: Checkpoints à optimiser (list_options, optim.checkpoints) : un job chacun.
    calibration_set: Jeu de calibration (list_options, optim.calibration_sets), proche des images du rover.
    recipes: Recettes parmi mixte, fp32, int8 ; absent : mixte. fp32 est toujours calculée comme référence.
    tile: Côté des tuiles traitées, en pixels : 32, 64 (par défaut), 128 ou 256.
    clipping: Écrêtage de la calibration : MAX (par défaut), MSE ou KL.
    per_channel: Vrai : une échelle de quantification par filtre (faux par défaut).
    n_calibration: Nombre de tuiles de calibration, de 1 à 512 (32 par défaut).
    n_images: Nombre d'images Kodak d'évaluation, de 1 à 24 (24 par défaut).
    export: Vrai (par défaut) : générer le code C (export C++ et fichier pour LoCod).
    compile_export: Vrai (par défaut) : compiler et vérifier le code généré (si g++ est installé).
    author: Auteur (facultatif).
    indicative: Vrai : signaler la série comme indicative, si les modèles n'ont pas eu le même entraînement.
    indicative_reason: Raison affichée pour une série indicative.

Returns:
    JSON : numéros des jobs déposés, dans l'ordre des λ.
""",
    "list_optim_jobs": """Optimisations Aidge lancées, de la plus récente à la plus ancienne : statut, étape, progression.

Returns:
    JSON : une ligne par job (numéro, nom, auteur, statut, λ, N, tuile, recettes, progression, étape, dates).
""",
}


def _json(obj):
    """! @brief Texte JSON lisible renvoyé à l'agent (accents gardés, objets inconnus convertis en texte)."""
    return json.dumps(obj, ensure_ascii=False, default=str)


def _records(df):
    """! @brief Lignes d'un DataFrame en dictionnaires (NaN devient null) ; liste vide pour None ou un tableau vide."""
    if df is None or df.empty:
        return []
    return json.loads(df.to_json(orient="records", force_ascii=False))


def _no_progress(*args, **kwargs):
    """! @brief Remplace la barre de progression Gradio des fonctions de l'interface : un agent ne la voit pas."""


def _codec(key):
    """! @brief Vérifie qu'un codec existe ; relit le registre une fois s'il manque (modèle entraîné depuis le démarrage).

    @param key  Identifiant du codec.
    @throws KeyError si le codec reste inconnu.
    """
    try:
        get(key)
    except KeyError:
        discover()
        get(key)


def _app():
    """! @brief Module app, importé à l'appel : app importe ce module."""
    from . import app
    return app


def list_options() -> str:
    """! @brief Valeurs permises des autres outils, lues à l'appel (le registre des codecs est relu).

    @return Texte JSON.
    """
    app = _app()
    discover()
    models = []
    for label, key in app.model_choices():
        mode = app.model_mode(key)
        models.append({"model": key, "label": label, "mode": mode,
                       "levels": [{"level": i, "label": lvl} for lvl, i in app.level_choices(key, mode)]})
    datasets = {}
    for name in app.dataset_choices():
        images = app.image_choices(name)
        datasets[name] = {"images": len(images), "examples": images[:EXAMPLE_IMAGES]}
    return _json({
        "models": models,
        "codecs": {m: [{"codec": key, "label": label} for label, key in app.codec_choices(m)] for m in ("gris", "rgb")},
        "datasets": datasets,
        "backends": [key for _, key in app.backend_choices()],
        "sizes": list(app.PROFILE_PRESETS),
        "campaigns": bench.list_runs(),
        "optim": {
            "checkpoints": [{"checkpoint": s["checkpoint"], "label": s["label"]} for s in ui_optim.list_sources()],
            "calibration_sets": [value for _, value in ui_optim.dataset_choices()],
            "recipes": [r for _, r in ui_optim.RECIPE_CHOICES], "default_recipe": ui_optim.DEFAULT_RECIPE,
            "tiles": ui_optim.TILE_CHOICES, "clippings": [c for _, c in ui_optim.CLIPPING_CHOICES],
        },
    })


def test_model(model: str, level: int = 0, image: str = "kodim01.png", dataset: str = "kodak",
               references: list[str] | None = None, backend: str = "torch", real_rate: bool = True) -> str:
    """! @brief Outil test_model : app.run_test() sur une image d'un jeu, sans galeries.

    @param model       Identifiant du modèle appris.
    @param level       Indice du niveau.
    @param image       Nom de l'image dans le jeu.
    @param dataset     Jeu d'images.
    @param references  Codecs de comparaison ; None : ceux cochés par défaut dans l'onglet (JPEG, JPEG 2000).
    @param backend     Moteur de l'encodeur.
    @param real_rate   Débit réel ou estimé.
    @return Texte JSON : @c mesures.
    """
    app = _app()
    _codec(model)
    refs = app.reference_choices(model)[1] if references is None else references
    _, _, table = app.run_test(dataset, image, None, model, level, refs, backend, real_rate, 50, 50,
                               progress=_no_progress)
    return _json({"mesures": _records(table)})


def benchmark(codecs: list[str], dataset: str = "kodak", n_images: int = 4, mode: str = "gris", points: int = 6,
              backend: str = "torch", real_rate: bool = True, name: str | None = None, reference: str | None = None,
              metric: str = "psnr") -> str:
    """! @brief Outil benchmark : app.run_benchmark(), sans figures.

    @param codecs     Identifiants des codecs.
    @param dataset    Jeu d'images.
    @param n_images   Nombre d'images.
    @param mode       @c gris ou @c rgb.
    @param points     Niveaux par codec.
    @param backend    Moteur de l'encodeur.
    @param real_rate  Débit réel ou estimé.
    @param name       Nom de la campagne, ou None.
    @param reference  Courbe de référence du BD-rate, ou None.
    @param metric     Qualité du BD-rate.
    @return Texte JSON : @c statut, @c bd_rate, @c points, @c csv.
    """
    out = _app().run_benchmark(dataset, n_images, mode, codecs, points, backend, real_rate, name or "", reference,
                               metric, progress=_no_progress)
    status, bd, table, csv = out[0], out[5], out[6], out[8]
    return _json({"statut": status, "bd_rate": _records(bd), "points": _records(table), "csv": csv})


def profile(model: str, level: int = 0, size: str = DEFAULT_SIZE) -> str:
    """! @brief Outil profile : app.profile(), sans figure.

    @param model  Identifiant du modèle appris.
    @param level  Indice du niveau.
    @param size   Clé de app.PROFILE_PRESETS.
    @return Texte JSON : @c résumé, @c couches.
    """
    _codec(model)
    summary, table, _ = _app().profile(model, level, size)
    return _json({"résumé": summary, "couches": _records(table)})


def compare_encoders(size: str = DEFAULT_SIZE, mode: str = "gris") -> str:
    """! @brief Outil compare_encoders : app.compare_encoders().

    @param size  Clé de app.PROFILE_PRESETS.
    @param mode  @c gris ou @c rgb.
    @return Texte JSON : une ligne par architecture.
    """
    app = _app()
    if size not in app.PROFILE_PRESETS:
        raise ValueError(f"Taille inconnue : {size}")
    return _json(_records(app.compare_encoders(size, mode)))


def check_backends(model: str, level: int = 0, size: str = DEFAULT_SIZE) -> str:
    """! @brief Outil check_backends : app.check_backends().

    @param model  Identifiant du modèle appris.
    @param level  Indice du niveau.
    @param size   Clé de app.PROFILE_PRESETS.
    @return Texte JSON : une ligne par moteur.
    """
    _codec(model)
    return _json(_records(_app().check_backends(model, level, size, progress=_no_progress)))


def analyze_optim(checkpoints: list[str], tile: int = ui_optim.DEFAULT_TILE, recipes: list[str] | None = None,
                  clipping: str = "MAX", per_channel: bool = False, n_calibration: int = 32,
                  calibration_set: str | None = None, n_images: int = 24) -> str:
    """! @brief Outil analyze_optim : ui_optim.analysis_config() puis ui_optim.analyze().

    @param checkpoints      Chemins des checkpoints ; seul le premier est analysé.
    @param tile             Côté de la tuile.
    @param recipes          Recettes ; None : la recette par défaut.
    @param clipping         Écrêtage de la calibration.
    @param per_channel      Quantification par canal.
    @param n_calibration    Nombre de tuiles de calibration.
    @param calibration_set  Jeu de calibration, ou None.
    @param n_images         Nombre d'images Kodak évaluées.
    @return Texte JSON : @c rapport.
    @throws ValueError si un champ est invalide ou si l'analyse échoue.
    """
    cfg = ui_optim.analysis_config(checkpoints, tile, recipes, clipping, per_channel, n_calibration, calibration_set,
                                   n_images)
    return _json({"rapport": ui_optim.analysis_markdown(ui_optim.analyze(cfg), len(checkpoints))})


def launch_optim(name: str, checkpoints: list[str], calibration_set: str, recipes: list[str] | None = None,
                 tile: int = ui_optim.DEFAULT_TILE, clipping: str = "MAX", per_channel: bool = False,
                 n_calibration: int = 32, n_images: int = 24, export: bool = True, compile_export: bool = True,
                 author: str | None = None, indicative: bool = False, indicative_reason: str | None = None) -> str:
    """! @brief Outil launch_optim : ui_optim.launch_jobs() sur la file par défaut.

    @param name               Nom de la série.
    @param checkpoints        Chemins des checkpoints : un job chacun.
    @param calibration_set    Jeu de calibration.
    @param recipes            Recettes ; None : la recette par défaut.
    @param tile               Côté de la tuile.
    @param clipping           Écrêtage de la calibration.
    @param per_channel        Quantification par canal.
    @param n_calibration      Nombre de tuiles de calibration.
    @param n_images           Nombre d'images Kodak évaluées.
    @param export             Générer le code C.
    @param compile_export     Compiler et vérifier le code.
    @param author             Auteur, ou None.
    @param indicative         Série indicative.
    @param indicative_reason  Raison affichée, ou None.
    @return Texte JSON : @c jobs.
    """
    ids = ui_optim.launch_jobs(JobStore(), name, author, checkpoints, tile, recipes or [ui_optim.DEFAULT_RECIPE],
                               clipping, per_channel, n_calibration, calibration_set, n_images, export, compile_export,
                               indicative, indicative_reason or "")
    return _json({"jobs": ids, "suivi": "outil list_optim_jobs, ou onglet 5. Optimisation Aidge de l'interface"})


def list_optim_jobs() -> str:
    """! @brief Outil list_optim_jobs : ui_optim.optim_jobs_table() sur la file par défaut.

    @return Texte JSON : une ligne par job.
    """
    return _json(_records(ui_optim.optim_jobs_table(JobStore())))


def _as_tool(fn, name, doc):
    """! @brief Enveloppe @p fn sous le nom @p name avec la description @p doc, que Gradio lit pour l'outil MCP.

    Gradio nomme l'outil d'après le __name__ de la fonction et le décrit d'après sa docstring : la fonction d'origine
    garde sa docstring Doxygen. inspect.signature suit __wrapped__ : paramètres et annotations restent ceux de @p fn.
    Les erreurs de valeur (identifiant inconnu, champ invalide) deviennent des gr.Error, que l'agent reçoit avec leur
    message.
    @param fn    Fonction (ni générateur ni coroutine).
    @param name  Nom de l'outil.
    @param doc   Description au format Google.
    @return Fonction enveloppe, marquée @c _aidge_tool.
    @throws TypeError si @p fn est un générateur ou une coroutine (l'enveloppe ne le serait plus).
    """
    if inspect.isgeneratorfunction(fn) or inspect.iscoroutinefunction(fn) or inspect.isasyncgenfunction(fn):
        raise TypeError(f"{name} : un générateur ou une coroutine ne peut pas être enveloppé")

    @functools.wraps(fn)
    def tool(*args, **kwargs):
        """! @brief Appelle la fonction enveloppée ; erreurs de valeur traduites en gr.Error."""
        try:
            return fn(*args, **kwargs)
        except KeyError as e:
            raise gr.Error(f"Valeur inconnue : {e}. {OPTIONS_HINT}") from None
        except ValueError as e:
            raise gr.Error(f"{e} {OPTIONS_HINT}") from None

    tool.__name__ = tool.__qualname__ = name
    tool.__doc__ = doc
    tool._aidge_tool = True
    return tool


def mount():
    """! @brief Déclare les outils de TOOLS comme endpoints d'API (gr.api), dans les Blocks de l'application.

    À appeler avant les onglets : un événement de l'interface créé ensuite avec le même nom de fonction reçoit un
    nom suffixé, et l'outil garde le sien.
    """
    for name, doc in TOOLS.items():
        gr.api(_as_tool(globals()[name], name, doc), api_name=name)


def restrict_api(demo):
    """! @brief Ne laisse publics que les outils déclarés par mount() ; rend privés tous les autres événements.

    @param demo  gr.Blocks construit, pas encore lancé.
    @return @p demo.
    @throws ValueError si un outil de TOOLS manque dans @p demo, ou n'a pas reçu son nom.
    """
    found = set()
    for block_fn in demo.fns.values():
        if getattr(block_fn.fn, "_aidge_tool", False):
            found.add(block_fn.api_name)
        else:
            block_fn.api_visibility = "private"
    missing = set(TOOLS) - found
    if missing:
        raise ValueError(f"Outils MCP introuvables dans l'interface : {', '.join(sorted(missing))}")
    return demo


def status(host=None, docker=None):
    """! @brief Le serveur MCP doit-il être lancé ?

    @param host    Adresse d'écoute de l'interface ; None : AIDGE_UI_HOST, sinon 127.0.0.1.
    @param docker  Force la détection de Docker (tests) ; None : détection automatique.
    @return (activé, message à afficher, vide si le MCP n'est pas demandé).
    """
    if os.environ.get(ENABLE_VAR) != "1":
        return False, ""
    if importlib.util.find_spec("mcp") is None:
        return False, ("MCP non lancé : le paquet mcp manque. Installez les dépendances "
                       "(pip install -r requirements.txt).")
    if not exposed_locally_only(host, docker) and os.environ.get(TRUSTED_AUTH_VAR) != "1":
        return False, ("MCP refusé : l'interface est exposée sur le réseau sans authentification devant, et un agent "
                       f"pourrait lancer des jobs. Placez une authentification devant, puis définissez {TRUSTED_AUTH_VAR}=1.")
    return True, ""
