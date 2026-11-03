"""! @brief Onglet « 5. Optimisation Aidge » de l'interface Gradio.

L'interface ne quantifie jamais elle-même : « Lancer » dépose un job, et « Analyser » passe par un sous-processus court, car
la capture du rapport d'Aidge agit sur tout le processus et un plantage C++ tuerait l'interface. Toute la logique est dans
des fonctions pures ; les callbacks de build_tab() n'en sont que de minces adaptateurs. Le parcours est en cinq étapes
numérotées : 1. modèle, 2. recette, 3. analyser ou lancer, 4. suivre (à côté du formulaire, pour voir le job déposé sans
chercher), 5. comparer les variantes et télécharger le code. Les jobs d'optimisation se suivent ici, pas dans l'onglet
Suivre, réservé aux entraînements. Contrat : docs/conception_optimisation.md (§ 9).
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import gradio as gr
import pandas as pd

from . import config, ui_style
from .codecs import all_codecs
from .codecs.balle2017 import Balle2017Codec
from .jobs import JobStore
from .optim import results
from .optim.config import DEFAULT_RECIPE, CLIPPINGS, RECIPES, OptimConfig, n_steps
from .ui_training import (REFRESH_S, cancel_job, choice_update, dataset_choices, format_date, job_action, job_choices,
                          job_progress, lmbda_text, log_tail, requeue_job, status_label, visible_jobs, worker_banner)

## Tailles de tuile proposées (côté).
TILE_CHOICES = [32, 64, 128, 256]
## Tuile par défaut : à N=64, poids et activations y tiennent ensemble dans la BRAM.
DEFAULT_TILE = 64
## Choix du menu des tuiles : (libellé, côté).
TILE_LABELS = [(f"{t} x {t}" + (" (conseillée)" if t == DEFAULT_TILE else ""), t) for t in TILE_CHOICES]
## Explication de chaque recette, dans l'ordre du formulaire.
RECIPE_HELP = {"mixte": "convolutions en entiers 8 bits, GDN en virgule flottante. Conseillée",
               "fp32": "rien n'est quantifié : c'est la référence, toujours calculée",
               "int8": "tout le modèle en entiers. Sans effet avec Aidge 0.10.1, qui ne sait pas quantifier la GDN"}
## Choix des recettes : (libellé expliqué, recette).
RECIPE_CHOICES = [(f"{r} : {RECIPE_HELP[r]}", r) for r in ("mixte", "fp32", "int8")]
## Explication de chaque mode d'écrêtage de la calibration.
CLIPPING_HELP = {"MAX": "garde toute la plage observée (défaut)", "MSE": "coupe les valeurs rares, pour plus de précision",
                 "KL": "réglage statistique, moins bon lors des essais"}
## Choix de l'écrêtage : (libellé expliqué, mode).
CLIPPING_CHOICES = [(f"{c} : {CLIPPING_HELP.get(c, '')}".rstrip(" :"), c) for c in CLIPPINGS]
## Délai maximal de l'analyse, en secondes.
ANALYZE_TIMEOUT_S = 300
## Commande de l'analyse. Remplaçable dans les tests.
ANALYZE_CMD = [sys.executable, "-m", "aidge_in_space.optimize", "--analyze"]
## Message quand Aidge manque dans l'environnement de l'interface.
AIDGE_MISSING = ("Aidge n'est pas installé dans l'environnement de l'interface : **Analyser** ne peut pas fonctionner. "
                 "Installez les dépendances (`requirements.txt`) ou utilisez l'image Docker.")
## Colonnes du tableau des BD-rate.
BD_COLUMNS = ["série", "recette", "niveaux", "BD-rate (%)", "raison", "indicative"]
## Colonnes des écarts point par point.
POINT_COLUMNS = ["série", "recette", "λ", "Δbpp (%)", "ΔPSNR (dB)"]
## Colonnes du tableau synthétique des variantes.
SUMMARY_COLUMNS = ["origine", "série", "λ", "modèle", "recette", "verdict", "mémoire de travail / BRAM (%)",
                   "poids dans le FPGA / BRAM (%)", "Δ débit (%)", "Δ PSNR (dB)", "export C++", "fichier LoCod"]
## Alerte quand la tentative INT8 n'a rien changé.
NO_EFFECT_ALERT = ("ℹ️ **int8 : SANS EFFET.** Aidge 0.10.1 ne sait pas quantifier la GDN : le graphe n'a pas changé. "
                   "Ce n'est pas une erreur de votre part : la recette **mixte** garde la GDN en flottant et quantifie les "
                   "convolutions.")
## Colonnes du tableau des optimisations lancées.
JOBS_COLUMNS = ["n°", "nom", "auteur", "statut", "λ", "N", "tuile", "recettes", "progression (%)", "étape", "créé le",
                "terminé le"]
## Statuts d'un job qui n'est pas encore allé au bout de sa tentative.
ACTIVE_STATUSES = ("queued", "running")
## Message quand aucune optimisation n'est choisie.
NO_JOB_SELECTED = "Sélectionnez d'abord une optimisation."
## Texte du détail quand aucune optimisation n'a encore été lancée.
NO_JOB_YET = ("Aucune optimisation lancée pour l'instant. Réglez les étapes 1 à 3 à gauche, puis cliquez sur **Lancer** : "
              "le job apparaîtra ici.")
## Texte initial de la carte Analyse.
ANALYSIS_PLACEHOLDER = ("Cliquez sur **Analyser** pour savoir, en quelques secondes et sans rien lancer, si le premier "
                        "modèle coché tiendra dans la mémoire du FPGA.")
## Aide de l'onglet : le vocabulaire, pour des lecteurs qui ne connaissent ni la quantification ni Aidge.
HELP_MD = """
**Quantifier** : remplacer les nombres à virgule (32 bits, « fp32 ») d'un modèle par de petits entiers (8 bits, « int8 »).
Le modèle prend 4 fois moins de place et calcule plus simplement, au prix d'une petite perte de qualité. Ici, la
quantification se fait après l'entraînement (on parle de **PTQ**, quantification post-entraînement) : rien n'est réentraîné.

**GDN** : la normalisation propre aux modèles de Ballé. Elle contient une racine carrée qu'Aidge 0.10.1 ne sait pas
quantifier, d'où la recette **mixte** : convolutions en entiers, GDN en virgule flottante, calculée par le processeur.

**Calibration** : avant de quantifier, Aidge fait passer quelques tuiles d'images dans le modèle pour mesurer la plage de
chaque valeur. Prenez des images proches de celles du rover. L'**écrêtage** (MAX, MSE, KL) dit comment fixer cette plage ;
**par canal** règle une échelle par filtre au lieu d'une seule par couche.

**Tuile** : l'image est découpée en carrés de cette taille, traités un par un. Plus la tuile est petite, moins il faut de
mémoire de travail.

**BRAM** : la mémoire rapide du FPGA, 615 Kio sur la carte. **Mémoire de travail / BRAM** : place des calculs
intermédiaires. **Poids dans le FPGA / BRAM** : place des poids des convolutions, seuls à rester dans le FPGA. Au-delà de
100 %, cela ne tient pas.

**Δ débit** et **Δ PSNR** : écarts face à fp32, même modèle, même tuile. +2 % de débit : fichiers 2 % plus gros ;
-0,05 dB de PSNR : perte de qualité très faible. Plus proche de 0 est meilleur.

**BD-rate** : l'écart de débit à qualité égale sur toute la courbe, face à fp32. Il faut au moins 4 λ sous le même nom de
série, et des courbes régulières. **Indicative** : les modèles n'ont pas eu le même entraînement.

**Export C++** : le code d'Aidge pour le processeur. **Vérifié** : il a été compilé et exécuté, et donne les mêmes
symboles que le graphe Aidge ; en dessous de 99 % de symboles identiques, la variante passe en **ÉCHEC** et ce code ne doit
pas partir sur la carte. **Fichier LoCod** : un seul `main.c` pour la chaîne FPGA (recette mixte), où les convolutions
deviennent des accélérateurs.

La conception complète est dans `docs/conception_optimisation.md`, le guide de l'équipe dans `docs/guide_interface.md`.
"""


def list_sources():
    """! @brief Checkpoints Ballé 2017 que l'on peut optimiser.

    @return Liste de dictionnaires @c label, @c checkpoint (relatif au dépôt si possible), @c codec, @c in_ch, @c N, @c lmbda.
    """
    out = []
    for codec in all_codecs():
        if not isinstance(codec, Balle2017Codec):
            continue
        for entry, level in zip(codec.entries, codec.levels(codec.modes[0])):
            path = Path(entry["path"])
            try:
                shown = path.resolve().relative_to(Path(config.ROOT).resolve()).as_posix()
            except ValueError:
                shown = str(path)
            out.append({"label": f"{codec.label} · {level.label}", "checkpoint": shown, "codec": codec.key,
                        "in_ch": codec.in_ch, "N": codec.N, "lmbda": entry["lmbda"]})
    return out


def source_choices():
    """! @brief Choix du menu des checkpoints : couples (libellé, chemin)."""
    return [(s["label"], s["checkpoint"]) for s in list_sources()]


def sources_hint(sources):
    """! @brief Aide sous la liste des modèles : quoi faire s'il n'y en a pas, sinon comment obtenir une courbe (fonction pure).

    @param sources  Choix du menu des checkpoints.
    @return Texte Markdown.
    """
    if not sources:
        return ("**Aucun modèle Ballé 2017 entraîné n'est disponible.** Entraînez-en un dans **1. Entraîner** (ou copiez un "
                "`.pt` dans `outputs/checkpoints/`) : il apparaîtra ici à la prochaine ouverture de l'onglet.")
    return ("Un job par modèle coché. Cochez plusieurs λ du même entraînement pour obtenir une courbe, et au moins 4 pour "
            "un BD-rate.")


def calibration_hint(datasets):
    """! @brief Aide sous le menu des images de calibration (fonction pure).

    @param datasets  Choix du menu des jeux d'images.
    @return Texte Markdown.
    """
    if not datasets:
        return ("**Aucun jeu d'images dans `data/train/`.** Ajoutez-en un dans **1. Entraîner**, section « Ajouter des "
                "images » (le jeu DFKI lunaire est le plus proche du rover).")
    return "Aidge observe ces images pour régler les entiers : prenez-les proches de ce que verra le rover."


def menu_updates(checked, calibration_set):
    """! @brief Menus relus d'après l'état du serveur : un entraînement ou un jeu d'images a pu apparaître depuis le démarrage.

    @param checked          Checkpoints cochés ; ceux qui ont disparu sont décochés.
    @param calibration_set  Jeu de calibration choisi.
    @return Quadruplet : gr.update des checkpoints, aide des checkpoints, gr.update du jeu de calibration, aide de la
            calibration.
    """
    sources, datasets = source_choices(), dataset_choices()
    known = {path for _, path in sources}
    return (gr.update(choices=sources, value=[c for c in checked or [] if c in known]), sources_hint(sources),
            choice_update(datasets, calibration_set), calibration_hint(datasets))


def aidge_available():
    """! @brief Vrai si Aidge est installé dans l'environnement de l'interface, sans l'importer."""
    return importlib.util.find_spec("aidge_core") is not None


def optim_banner(store, aidge=None):
    """! @brief Bandeau de l'onglet : présence du worker et, si besoin, absence d'Aidge.

    @param store  JobStore.
    @param aidge  Aidge présent ; None : aidge_available() à l'appel.
    @return Texte Markdown.
    """
    msg = worker_banner(store, "optim")
    if not (aidge_available() if aidge is None else aidge):
        msg += f"\n\n⚠️ {AIDGE_MISSING} Les jobs tournent dans le worker, qui doit avoir Aidge lui aussi."
    return msg


def friendly_error(message):
    """! @brief Message d'erreur de l'analyse, traduit quand la cause est connue (fonction pure).

    @param message  Dernière ligne d'erreur du sous-processus.
    @return Message affichable.
    """
    if "No module named 'aidge" in message:
        return AIDGE_MISSING.replace("**", "")
    return message


def build_config(name, author, checkpoint, codec, in_ch, tile_side, recipes, clipping, per_channel, n_calibration,
                 calibration_set, n_images, export, compile_export, indicative=False, indicative_reason=""):
    """! @brief Configuration d'optimisation depuis le formulaire.

    @return OptimConfig validée.
    @throws ValueError si un champ est invalide (message affichable tel quel).
    """
    if not calibration_set:
        raise ValueError("Choisissez des images de calibration à l'étape 2.")
    if not recipes:
        raise ValueError("Cochez au moins une recette à l'étape 2 : mixte est conseillée.")
    return OptimConfig.from_dict({
        "name": name or "", "author": author or "", "source": {"codec": codec or "", "checkpoint": checkpoint or ""},
        "tile": [int(in_ch), int(tile_side), int(tile_side)], "recipes": list(recipes),
        "ptq": {"clipping": clipping, "per_channel": bool(per_channel), "n_calibration": int(n_calibration),
                "calibration_set": calibration_set},
        "evaluate": {"n_images": int(n_images), "indicative_series": bool(indicative), "indicative_reason": indicative_reason or ""},
        "export": {"generate": bool(export), "compile": bool(compile_export)}})


def _checked_sources(checkpoints, sources):
    """! @brief Sources des checkpoints cochés, dans l'ordre.

    @throws ValueError si rien n'est coché ou si un checkpoint est inconnu.
    """
    if not checkpoints:
        raise ValueError("Cochez au moins un modèle (checkpoint) à l'étape 1.")
    known = {s["checkpoint"]: s for s in (list_sources() if sources is None else sources)}
    unknown = [c for c in checkpoints if c not in known]
    if unknown:
        raise ValueError(f"Checkpoint inconnu : {unknown[0]}. Il a peut-être été supprimé : rouvrez l'onglet.")
    return [known[c] for c in checkpoints]


def analysis_config(checkpoints, tile_side, recipes, clipping, per_channel, n_calibration, calibration_set, n_images,
                    sources=None):
    """! @brief Configuration de l'analyse : le premier checkpoint coché, sans nom ni jeu obligatoires.

    @param sources  Résultat de list_sources() ; None : lu à l'appel.
    @return OptimConfig.
    @throws ValueError si rien n'est coché ou si un champ est invalide.
    """
    s = _checked_sources(checkpoints, sources)[0]
    return build_config("analyse", "", s["checkpoint"], s["codec"], s["in_ch"], tile_side, recipes or [DEFAULT_RECIPE],
                        clipping, per_channel, n_calibration, calibration_set or "analyse", n_images, False, False)


def launch_jobs(store, name, author, checkpoints, tile_side, recipes, clipping, per_channel, n_calibration, calibration_set,
                n_images, export, compile_export, indicative=False, indicative_reason="", sources=None):
    """! @brief Dépose un job par checkpoint coché : toutes les configurations sont validées avant la première soumission.

    @param store        JobStore.
    @param checkpoints  Chemins cochés.
    @param sources      Résultat de list_sources() ; None : lu à l'appel.
    @return Identifiants des jobs, dans l'ordre des λ.
    @throws ValueError si un champ est invalide, sans rien déposer.
    """
    if not (name or "").strip():
        raise ValueError("Donnez un nom à la série à l'étape 3 : il regroupe les λ en une courbe.")
    jobs = []
    for s in _checked_sources(checkpoints, sources):
        cfg = build_config(name, author, s["checkpoint"], s["codec"], s["in_ch"], tile_side, recipes, clipping, per_channel,
                           n_calibration, calibration_set, n_images, export, compile_export, indicative, indicative_reason)
        jobs.append((s["lmbda"], cfg.to_job(s["in_ch"], s["N"], s["lmbda"], n_steps(cfg))))
    return [store.submit(job, kind="optim") for _, job in sorted(jobs, key=lambda j: j[0])]


def launch_message(ids):
    """! @brief Retour après « Lancer » : les jobs déposés et où les suivre (fonction pure).

    @param ids  Identifiants des jobs déposés.
    @return Texte Markdown.
    """
    jobs = ", ".join(f"n°{i}" for i in ids)
    what = "Job déposé" if len(ids) == 1 else f"{len(ids)} jobs déposés"
    return (f"✅ **{what} : {jobs}.** Suivez-le dans **4. Suivre les optimisations**, à droite : le n°{ids[0]} y est "
            "sélectionné. Une fois terminé, ses résultats s'ajoutent à l'étape 5, plus bas.")


def parse_analysis(stdout):
    """! @brief Résultat de l'analyse : la dernière ligne de la sortie, en JSON (fonction pure).

    @param stdout  Sortie standard du sous-processus.
    @return Dictionnaire.
    @throws ValueError si la sortie n'en contient pas.
    """
    lines = [l for l in (stdout or "").splitlines() if l.strip()]
    if not lines:
        raise ValueError("L'analyse n'a rien renvoyé.")
    try:
        return json.loads(lines[-1])
    except ValueError as e:
        raise ValueError("L'analyse a renvoyé un résultat illisible.") from e


def analyze(cfg, cmd=None):
    """! @brief Lance l'analyse dans un sous-processus court.

    @param cfg  OptimConfig.
    @param cmd  Commande, à laquelle le fichier YAML est ajouté ; None : ANALYZE_CMD.
    @return Résultat de parse_analysis().
    @throws ValueError si l'analyse échoue (message affichable).
    """
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "analyse.yaml"
        cfg.to_yaml(path)
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        try:
            r = subprocess.run([*(ANALYZE_CMD if cmd is None else cmd), str(path)], capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=ANALYZE_TIMEOUT_S, cwd=str(config.ROOT), env=env)
        except subprocess.TimeoutExpired as e:
            raise ValueError(f"L'analyse dépasse {ANALYZE_TIMEOUT_S} s.") from e
    if r.returncode != 0:
        lines = [l for l in r.stderr.splitlines() if l.strip()]
        raise ValueError(friendly_error(lines[-1]) if lines else f"L'analyse a échoué (code {r.returncode}).")
    return parse_analysis(r.stdout)


def _ratio(part, whole):
    """! @brief Pourcentage arrondi, pour l'affichage."""
    return f"{part / whole * 100:.0f} %"


def analysis_markdown(result, n_checked=1):
    """! @brief Compte rendu de l'analyse, pour des lecteurs qui ne connaissent pas Aidge (fonction pure).

    @param result     Résultat de analyze().
    @param n_checked  Nombre de modèles cochés : seul le premier est analysé.
    @return Texte Markdown, avec un verdict en tête.
    """
    q = result["quantizability"]
    unknown = [o["type"] for o in q["operators"] if o["state"] != "ok"]
    fp32, mixed = result["memory_fp32"], result["memory_mixte"]
    bram = fp32["bram_bytes"]
    total = mixed["peak_bytes"] + mixed["weights_bytes"]
    lines = [f"**Modèle N={result['source']['N']}, λ={result['source']['lmbda']}, tuile {result['tile'][1]} x "
             f"{result['tile'][2]}.** Aperçu : rien n'a été enregistré ni lancé.", ""]
    if n_checked > 1:
        lines += [f"Seul le premier des {n_checked} modèles cochés est analysé.", ""]
    if total <= bram:
        lines += [f"✅ **Avec la recette mixte, poids et mémoire de travail tiennent ensemble dans la BRAM** "
                  f"({_ratio(total, bram)}).", ""]
    else:
        lines += [f"❌ **Avec la recette mixte, poids et mémoire de travail dépassent la BRAM** ({_ratio(total, bram)}) : "
                  "les poids devront vivre en mémoire externe, ou il faut un modèle plus petit (N=64) ou une tuile plus "
                  "petite.", ""]
    if unknown:
        lines += [f"- **Quantification du graphe entier impossible** : Aidge ne prend pas en charge {', '.join(unknown)}. "
                  "La recette mixte contourne ce blocage.", ""]
    else:
        lines += ["- Tous les opérateurs sont pris en charge par la quantification d'Aidge.", ""]
    lines += ["| | Mémoire de travail (pic) | Poids | Total | Face à la BRAM |", "| :- | -: | -: | -: | -: |",
              f"| fp32 | {fp32['peak_bytes'] // 1024} Kio | {fp32['weights_bytes'] // 1024} Kio | "
              f"{(fp32['peak_bytes'] + fp32['weights_bytes']) // 1024} Kio | {_ratio(fp32['peak_bytes'] + fp32['weights_bytes'], bram)} |",
              f"| mixte (poids estimés) | {mixed['peak_bytes'] // 1024} Kio | {mixed['weights_bytes'] // 1024} Kio | "
              f"{total // 1024} Kio | {_ratio(total, bram)} |", "",
              f"Avec la GDN sur le processeur, seuls les poids des convolutions restent dans le FPGA : "
              f"{mixed['conv_weights_bytes'] // 1024} Kio, soit {_ratio(mixed['conv_weights_bytes'], bram)} de la BRAM "
              f"({bram // 1024} Kio).", "",
              "**Ensuite :** donnez un nom à la série et cliquez sur **Lancer** pour produire et mesurer les variantes."]
    return "\n".join(lines)


def variants_table(reports=None):
    """! @brief Tableau complet des variantes, toutes colonnes techniques comprises.

    @param reports  Rapports ; None : lus à l'appel.
    @return DataFrame aux colonnes de aidge_in_space.optim.results.TABLE_COLUMNS.
    """
    reports = results.read_reports() if reports is None else reports
    return pd.DataFrame(results.variant_rows(reports), columns=results.TABLE_COLUMNS)


def _origin(report):
    """! @brief Origine d'un rapport : « n°12 » pour un job, « ligne de commande » pour un rapport produit par --config."""
    rid = str(report.get("id", ""))
    return f"n°{rid}" if rid.isdigit() else "ligne de commande"


def _pct(part, whole):
    """! @brief Pourcentage entier, ou None."""
    return None if part is None or not whole else round(part / whole * 100)


def _rounded(v, digits):
    """! @brief Arrondi, « - » pour une valeur absente (une colonne sans NaN garde ses entiers à l'affichage)."""
    return "-" if v is None else round(v, digits)


def summary_rows(reports):
    """! @brief Lignes du tableau synthétique des variantes : verdict, mémoire, écarts et code, en clair (fonction pure).

    @param reports  Rapports de read_reports().
    @return Liste de dictionnaires aux clés SUMMARY_COLUMNS, du plus récent au plus ancien.
    """
    rows = []
    for report in reversed(reports):
        src, bram = report["source"], report.get("bram_bytes") or config.BRAM_BYTES
        for recipe, v in report["variants"].items():
            mem, rd = v.get("memory") or {}, v.get("rd") or {}
            ref = recipe == "fp32" and v.get("status") == "ok"
            rows.append({"origine": _origin(report), "série": report["name"], "λ": lmbda_text(src["lmbda"]),
                         "modèle": f"N={src['N']}, tuile {report['tile'][1]}", "recette": recipe,
                         "verdict": results.variant_verdict(v),
                         "mémoire de travail / BRAM (%)": _rounded(_pct(mem.get("peak_bytes"), bram), 0),
                         "poids dans le FPGA / BRAM (%)": _rounded(_pct(mem.get("conv_weights_bytes"), bram), 0),
                         "Δ débit (%)": "référence" if ref else _rounded(rd.get("delta_bpp_pct"), 2),
                         "Δ PSNR (dB)": "référence" if ref else _rounded(rd.get("delta_psnr_db"), 3),
                         "export C++": results.export_text(v.get("export")), "fichier LoCod": results.locod_text(v.get("locod"))})
    return rows


def summary_table(reports=None):
    """! @brief Tableau synthétique des variantes.

    @param reports  Rapports ; None : lus à l'appel.
    @return DataFrame aux colonnes SUMMARY_COLUMNS.
    """
    reports = results.read_reports() if reports is None else reports
    return pd.DataFrame(summary_rows(reports), columns=SUMMARY_COLUMNS)


def alerts_markdown(reports=None):
    """! @brief Alertes à afficher : quantification sans effet, variantes en échec.

    @param reports  Rapports ; None : lus à l'appel.
    @return Texte Markdown, vide s'il n'y a rien à signaler.
    """
    reports = results.read_reports() if reports is None else reports
    if not reports:
        return ""
    lines = []
    if any(v.get("status") == "sans_effet" for r in reports for v in r.get("variants", {}).values()):
        lines.append(NO_EFFECT_ALERT)
    failures = results.no_effect_alerts([
        {**r, "variants": {k: v for k, v in r["variants"].items() if v.get("status") == "echec"}} for r in reports])
    if failures:
        lines.append("❌ **Variantes en échec** : leur code ne doit pas partir sur la carte.\n\n"
                     + "\n".join(f"- {m}" for m in failures))
    return "\n\n".join(lines)


def results_hint(reports):
    """! @brief Phrase d'état de l'étape 5 : aucun résultat encore, ou combien de variantes (fonction pure).

    @param reports  Rapports.
    @return Texte Markdown.
    """
    if not reports:
        return ("Aucun résultat pour l'instant : les variantes s'affichent ici dès qu'une optimisation est terminée. "
                "Celles produites en ligne de commande (`python -m aidge_in_space.optimize --config`) aussi.")
    n = sum(len(r.get("variants", {})) for r in reports)
    return (f"{n} variante(s) dans {len(reports)} rapport(s), de la plus récente à la plus ancienne. Les variantes "
            "**utilisables** apparaissent aussi dans **3. Tester** et **4. Valider**, sous un nom qui commence par « Aidge ».")


def _series_levels(reports, recipe):
    """! @brief Niveaux d'une série pour le BD-rate : pour chaque job, la variante et le FP32 du même checkpoint."""
    levels = []
    for report in reports:
        v, fp32 = report["variants"].get(recipe, {}), report["variants"].get("fp32", {})
        if v.get("status") == "ok" and v.get("rd") and fp32.get("rd"):
            levels.append({"lmbda": report["source"]["lmbda"], "variant": v["rd"], "fp32": fp32["rd"]})
    return levels


def bd_tables(reports=None):
    """! @brief BD-rate de chaque série face au FP32, et écarts point par point.

    @param reports  Rapports ; None : lus à l'appel.
    @return Couple de DataFrame : les BD-rate (colonnes BD_COLUMNS), et les écarts par niveau (POINT_COLUMNS).
    """
    from .optim.evaluate import bd_series
    reports = results.read_reports() if reports is None else reports
    groups = {}
    for report in reports:
        for recipe in report.get("variants", {}):
            if recipe != "fp32":
                groups.setdefault(results.series_key(report, recipe), (recipe, []))[1].append(report)
    rows, points = [], []
    for (name, *_), (recipe, members) in groups.items():
        levels = _series_levels(members, recipe)
        if not levels:
            continue
        s = bd_series(levels)
        indicative = any(r.get("series", {}).get("indicative") for r in members)
        rows.append({"série": name, "recette": recipe, "niveaux": s["n_levels"],
                     "BD-rate (%)": None if s["bd_rate"] is None else round(s["bd_rate"], 2), "raison": s["reason"] or "",
                     "indicative": "oui" if indicative else "non"})
        points += [{"série": name, "recette": recipe, "λ": p["lmbda"], "Δbpp (%)": round(p["delta_bpp_pct"], 2),
                    "ΔPSNR (dB)": round(p["delta_psnr_db"], 3)} for p in s["points"]]
    return pd.DataFrame(rows, columns=BD_COLUMNS), pd.DataFrame(points, columns=POINT_COLUMNS)


def bd_hint(bd):
    """! @brief Phrase sous le tableau des BD-rate : comment l'obtenir quand il manque (fonction pure).

    @param bd  DataFrame des BD-rate (bd_tables).
    @return Texte Markdown.
    """
    if bd.empty:
        return ("Pas encore de série à comparer. Lancez la même optimisation, **sous le même nom**, sur au moins 4 λ du même "
                "entraînement.")
    if bd["BD-rate (%)"].isna().all():
        return "Aucune série n'a encore de BD-rate : la colonne **raison** dit ce qui manque."
    return "BD-rate négatif : la variante demande moins de bits que fp32 à qualité égale ; positif : plus de bits."


def download_choices(reports=None):
    """! @brief Fichiers à télécharger : zip de l'export C++ et fichier C pour LoCod, de tous les rapports.

    @param reports  Rapports ; None : lus à l'appel.
    @return Liste de couples (libellé, chemin existant), du rapport le plus récent au plus ancien.
    """
    reports = results.read_reports() if reports is None else reports
    out = []
    for report in reversed(reports):
        for recipe, v in report.get("variants", {}).items():
            base = f"{_origin(report)} · {report['name']} · λ={lmbda_text(report['source']['lmbda'])} · {recipe}"
            zip_rel = (v.get("export") or {}).get("zip")
            if zip_rel and (Path(report["_dir"]) / zip_rel).is_file():
                out.append((f"{base} : export C++ (zip)", str(Path(report["_dir"]) / zip_rel)))
            c_rel = (v.get("locod") or {}).get("path")
            if c_rel and (Path(report["_dir"]) / c_rel).is_file():
                out.append((f"{base} : fichier C pour LoCod", str(Path(report["_dir"]) / c_rel)))
    return out


def files_hint(files):
    """! @brief Phrase de la zone de téléchargement (fonction pure).

    @param files  Résultat de download_choices().
    @return Texte Markdown.
    """
    if not files:
        return "Aucun code produit pour l'instant : il apparaît ici à la fin d'une optimisation qui génère le code C."
    return ("**export C++ (zip)** : le code d'Aidge, pour le processeur de la carte. **fichier C pour LoCod** : un seul "
            "`main.c`, pour la chaîne FPGA. Choisir une optimisation à l'étape 4 présélectionne son code ici.")


def current_stage(store, job):
    """! @brief Dernière étape consignée par une optimisation en cours (« mixte : export »), ou None.

    @param store  JobStore.
    @param job    Job d'optimisation.
    @return Nom de l'étape, ou None si le job ne tourne pas ou n'a encore rien consigné.
    """
    if job.status != "running":
        return None
    stages = [m["stage"] for m in store.metrics(job.id, "eval") if m.get("stage")]
    return stages[-1] if stages else None


def _tile_text(cfg):
    """! @brief Tuile d'une configuration d'optimisation, « 64 x 64 », ou « ? »."""
    tile = cfg.get("tile") or []
    return f"{tile[1]} x {tile[2]}" if len(tile) == 3 else "?"


def _finished_text(job):
    """! @brief Date de fin de la dernière tentative, « - » pour un job en attente ou en cours."""
    return "-" if job.status in ACTIVE_STATUSES else format_date(job.finished_at)


def optim_jobs_table(store):
    """! @brief Tableau des optimisations lancées, de la plus récente à la plus ancienne. Les entraînements n'y sont pas.

    @param store  JobStore.
    @return DataFrame aux colonnes JOBS_COLUMNS.
    """
    rows = []
    for job in visible_jobs(store, "optim"):
        pct, _ = job_progress(job, [], time.time())
        rows.append({"n°": job.id, "nom": job.name, "auteur": job.author, "statut": status_label(job),
                     "λ": lmbda_text(job.lmbda), "N": job.N, "tuile": _tile_text(job.config),
                     "recettes": ", ".join(job.config.get("recipes") or []), "progression (%)": pct,
                     "étape": current_stage(store, job) or "-", "créé le": format_date(job.created_at),
                     "terminé le": _finished_text(job)})
    return pd.DataFrame(rows, columns=JOBS_COLUMNS)


def optim_job_summary(job, report, stage=None):
    """! @brief Synthèse Markdown d'une optimisation : statut, étape, réglages, dates, résultat par recette, erreur (fonction pure).

    @param job     Job d'optimisation.
    @param report  Son rapport (aidge_in_space.optim.results.report_for_job), ou None.
    @param stage   Étape en cours (current_stage), ou None.
    @return Texte Markdown.
    """
    c = job.config
    source, ptq = c.get("source") or {}, c.get("ptq") or {}
    lines = [f"**n°{job.id} · {job.name}** ({job.author or 'auteur inconnu'}) : **{status_label(job)}**, "
             f"étape {job.last_step} sur {job.steps}."]
    if stage:
        lines.append(f"- Étape en cours : {stage}.")
    if job.status == "queued":
        lines.append("- En attente du worker : il traite les jobs un par un (voir le bandeau en haut de l'onglet).")
    lines += [f"- Checkpoint `{source.get('checkpoint', '?')}`, N={job.N}, λ={lmbda_text(job.lmbda)}, tuile {_tile_text(c)}, "
              f"recettes {', '.join(c.get('recipes') or []) or '?'}, calibration `{ptq.get('calibration_set', '?')}`.",
              f"- Créé le {format_date(job.created_at)}, terminé le {_finished_text(job)}.", "", "**Variantes**", ""]
    if report is None and job.status not in ACTIVE_STATUSES and job.status != "done":
        lines.append("Pas de rapport : l'optimisation n'est pas allée au bout. **Relancer** la recommence depuis le début.")
    else:
        lines.append(results.summary_markdown(report))
    if job.error:
        if "No module named 'aidge" in job.error:
            lines += ["", "**Le worker n'a pas Aidge** : installez les dépendances (`requirements.txt`) dans son "
                      "environnement, ou lancez-le avec l'image Docker, puis **Relancer**."]
        lines += ["", f"Message d'erreur :\n\n```\n{job.error}\n```"]
    return "\n".join(lines)


def optim_job_details(store, job_id, n_log_lines=50):
    """! @brief Détail d'une optimisation : synthèse, fin du journal, variantes et fichiers à télécharger de ce job.

    @param store        JobStore.
    @param job_id       Identifiant, ou None.
    @param n_log_lines  Nombre de lignes de train.log à renvoyer.
    @return Quadruplet (Markdown de synthèse, texte du journal, DataFrame synthétique des variantes de ce job, choix de
            download_choices() pour ce job) ; une invitation (NO_JOB_SELECTED, ou NO_JOB_YET s'il n'y a aucune
            optimisation) et des zones vides si @p job_id est None.
    @throws KeyError si le job est inconnu.
    @throws ValueError si le job est un entraînement (il se suit dans l'onglet 2. Suivre).
    """
    if job_id is None:
        empty = not visible_jobs(store, "optim")
        return NO_JOB_YET if empty else NO_JOB_SELECTED, "", summary_table([]), []
    job = store.get(int(job_id))
    if job.kind != "optim":
        raise ValueError(f"Le job n°{job.id} est un entraînement : suivez-le dans l'onglet 2. Suivre.")
    report = results.report_for_job(job.id)
    reports = [report] if report else []
    files = download_choices(reports)
    summary = optim_job_summary(job, report, current_stage(store, job))
    if files:
        summary += "\n\nSon code est présélectionné plus bas, dans **Télécharger le code** (étape 5)."
    return summary, log_tail(store.job_dir(job.id) / "train.log", n_log_lines), summary_table(reports), files


def done_optim_jobs(store):
    """! @brief Identifiants des optimisations terminées : leur changement signale un nouveau rapport à relire."""
    return {j.id for j in visible_jobs(store, "optim") if j.status == "done"}


def build_tab(store=None):
    """! @brief Crée l'onglet, dans le @c with gr.Tabs() de app.build().

    @param store  JobStore ; None : la base par défaut.
    """
    store = store or JobStore()
    sources, datasets = source_choices(), dataset_choices()
    jobs0 = job_choices(store, "optim")
    reports0 = results.read_reports()
    bd0, pts0 = bd_tables(reports0)
    files0 = download_choices(reports0)
    summary0, log0, job_table0, _ = optim_job_details(store, jobs0[0][1] if jobs0 else None)
    with gr.Tab("5. Optimisation Aidge", id="optimisation") as tab:
        ui_style.intro("**Préparez un modèle entraîné pour la carte.** Aidge remplace ses nombres à virgule par de petits "
                       "entiers (« quantifier »), mesure la mémoire face à celle du FPGA, la perte de qualité, et produit le "
                       "code C. Suivez les étapes 1 à 5 ; le vocabulaire est expliqué dans l'aide, en bas de l'onglet.",
                       "une variante terminée apparaît dans **3. Tester** et **4. Valider**, face à l'original.")
        banner = gr.Markdown(optim_banner(store), elem_classes="status-banner")
        with gr.Row():
            with gr.Column(scale=2, min_width=340):
                with ui_style.card("1. Modèle à optimiser", "Un modèle Ballé 2017 entraîné, à un ou plusieurs λ."):
                    o_sources = gr.CheckboxGroup(sources, value=[sources[0][1]] if sources else [], label="Modèles",
                                                 show_label=False)
                    o_sources_hint = gr.Markdown(sources_hint(sources), elem_classes="hint")
                with ui_style.card("2. Recette", "Ce qu'Aidge transforme en entiers."):
                    o_recipes = gr.CheckboxGroup(RECIPE_CHOICES, value=[DEFAULT_RECIPE], label="Recettes", show_label=False)
                    ui_style.hint("La référence fp32 est toujours calculée, même décochée : les écarts se mesurent face à elle.")
                    o_calib = gr.Dropdown(datasets, value=datasets[0][1] if datasets else None,
                                          label="Images de calibration")
                    o_calib_hint = gr.Markdown(calibration_hint(datasets), elem_classes="hint")
                    o_tile = gr.Dropdown(TILE_LABELS, value=DEFAULT_TILE, label="Taille de tuile",
                                         info="L'image est traitée par carrés de cette taille : plus petit, moins de mémoire.")
                    with gr.Accordion("Réglages avancés", open=False):
                        o_clip = gr.Dropdown(CLIPPING_CHOICES, value="MAX", label="Écrêtage de la calibration",
                                             info="Comment Aidge fixe la plage de chaque valeur.")
                        o_channel = gr.Checkbox(value=False, label="Quantifier par canal",
                                                info="Une échelle par filtre au lieu d'une par couche.")
                        o_ncal = gr.Slider(1, 512, value=32, step=1, label="Tuiles de calibration",
                                           info="32 suffit d'habitude ; moins, l'erreur augmente.")
                        o_nimg = gr.Slider(1, 24, value=24, step=1, label="Images d'évaluation (Kodak)",
                                           info="Moins d'images : plus rapide, mesure moins précise.")
                        o_export = gr.Checkbox(value=True, label="Générer le code C (export C++ et fichier pour LoCod)")
                        o_compile = gr.Checkbox(value=True, label="Compiler et vérifier le code (si g++ est installé)",
                                                info="Sans vérification, rien ne garantit que le code est juste.")
                        o_ind = gr.Checkbox(value=False, label="Signaler la série comme indicative",
                                            info="Si les modèles n'ont pas eu le même entraînement.")
                        o_ind_reason = gr.Textbox(label="Raison affichée", placeholder="ex. entraînements de 2 000 à 60 000 pas")
                with ui_style.card("3. Analyser ou lancer"):
                    o_name = gr.Textbox(label="Nom de la série", placeholder="ex. lune-mixte-n64",
                                        info="Regroupe les λ cochés en une courbe, dans Tester, Valider et le BD-rate.")
                    o_author = gr.Textbox(label="Auteur (facultatif)")
                    with gr.Row():
                        o_analyze = gr.Button("Analyser")
                        o_go = gr.Button("Lancer", variant="primary", elem_classes="main-action")
                    ui_style.hint("**Analyser** : quelques secondes, rien n'est lancé ni enregistré. **Lancer** : un job par "
                                  "modèle coché part dans la file ; comptez plusieurs minutes chacun.")
                    o_status = gr.Markdown()
            with gr.Column(scale=5, min_width=340):
                with ui_style.card("Analyse", "Le modèle tiendra-t-il dans la mémoire du FPGA ?"):
                    o_report = gr.Markdown(ANALYSIS_PLACEHOLDER)
                with ui_style.card("4. Suivre les optimisations", "Elles tournent en arrière-plan : vous pouvez fermer la "
                                   "page. Le tableau se met à jour tout seul."):
                    o_jobs = gr.Dataframe(optim_jobs_table(store), show_label=False, interactive=False, wrap=True)
                    o_pick = gr.Dropdown(jobs0, value=jobs0[0][1] if jobs0 else None, label="Optimisation à examiner")
                    o_summary = gr.Markdown(summary0)
                    o_job_table = gr.Dataframe(job_table0, label="Variantes de cette optimisation", wrap=True,
                                               interactive=False)
                    o_requeue = gr.Button("Relancer")
                    ui_style.hint("**Relancer** recommence depuis le début une optimisation échouée, annulée ou interrompue.")
                    o_msg = gr.Markdown()
                    with gr.Accordion("Journal", open=False):
                        o_log = gr.Textbox(log0, label="Fin du journal (50 dernières lignes)", lines=12, max_lines=12,
                                           interactive=False)
                    with gr.Accordion("Annuler cette optimisation", open=False):
                        with gr.Row():
                            o_confirm = gr.Checkbox(label="Je confirme l'annulation")
                            o_cancel = gr.Button("Annuler", variant="stop")

        # ------------------------------------------------ résultats de toutes les optimisations
        with ui_style.card("5. Comparer les variantes", "Toutes les optimisations terminées, y compris celles lancées en "
                           "ligne de commande. **Δ débit**, **Δ PSNR** : écart face à fp32, plus proche de 0 est meilleur. "
                           "**Mémoire / BRAM** : au-delà de 100 %, cela ne tient pas dans le FPGA."):
            o_results_hint = gr.Markdown(results_hint(reports0), elem_classes="hint")
            o_alerts = gr.Markdown(alerts_markdown(reports0))
            o_table = gr.Dataframe(summary_table(reports0), show_label=False, wrap=True, interactive=False)
            gr.Markdown("**BD-rate face à fp32** : l'écart de débit à qualité égale, sur toute la courbe d'une série.")
            o_bd = gr.Dataframe(bd0, show_label=False, interactive=False)
            o_bd_hint = gr.Markdown(bd_hint(bd0), elem_classes="hint")
            with gr.Accordion("Détails : écarts point par point et tableau complet", open=False):
                o_points = gr.Dataframe(pts0, label="Écarts point par point", interactive=False)
                o_full = gr.Dataframe(variants_table(reports0), label="Toutes les colonnes (identiques Aidge et PyTorch, "
                                      "binaire, poids)", wrap=True, interactive=False)
            o_refresh = gr.Button("Actualiser")
        with ui_style.card("Télécharger le code", "Le code C de toutes les optimisations, jobs et ligne de commande."):
            o_files = gr.Dropdown(files0, value=None, label="Fichier")
            o_files_hint = gr.Markdown(files_hint(files0), elem_classes="hint")
            o_file = gr.File(label="Téléchargement", interactive=False)
        with gr.Accordion("Aide : vocabulaire de l'optimisation (quantifier, calibration, BRAM, BD-rate…)", open=False):
            gr.Markdown(HELP_MD)
        o_choices = gr.State(None)
        o_done = gr.State(None)

        def on_analyze(checkpoints, tile, recipes, clip, channel, ncal, calib, nimg):
            """! @brief Bouton Analyser : erreurs converties en message simple."""
            try:
                cfg = analysis_config(checkpoints, tile, recipes, clip, channel, ncal, calib, nimg)
                return analysis_markdown(analyze(cfg), len(checkpoints))
            except ValueError as e:
                raise gr.Error(str(e))

        def on_go(name, author, checkpoints, recipes, tile, calib, clip, channel, ncal, nimg, export, compile_export, ind, reason):
            """! @brief Bouton Lancer : dépose les jobs et sélectionne le premier dans le suivi."""
            try:
                ids = launch_jobs(store, name, author, checkpoints, tile, recipes, clip, channel, ncal, calib, nimg, export,
                                  compile_export, ind, reason)
            except (ValueError, KeyError) as e:
                raise gr.Error(str(e))
            msg = launch_message(ids)
            gr.Info(msg.replace("**", "").replace("✅ ", ""))
            return msg, optim_jobs_table(store), gr.update(choices=job_choices(store, "optim"), value=ids[0])

        def report_updates(reports=None):
            """! @brief Zones qui dépendent des rapports : synthèse, alertes, BD-rate, détails, fichiers."""
            reports = results.read_reports() if reports is None else reports
            bd, points = bd_tables(reports)
            files = download_choices(reports)
            return (results_hint(reports), alerts_markdown(reports), summary_table(reports), bd, bd_hint(bd), points,
                    variants_table(reports), gr.update(choices=files), files_hint(files))

        def refresh():
            """! @brief Bouton Actualiser : bandeau, optimisations lancées et rapports."""
            return (optim_banner(store), optim_jobs_table(store)) + report_updates()

        def details(job_id):
            """! @brief Synthèse, journal et variantes de l'optimisation choisie."""
            try:
                summary, log, table, _ = optim_job_details(store, job_id)
            except (KeyError, ValueError):
                return "Ce job n'existe plus.", "", summary_table([])
            return summary, log, table

        def on_pick(job_id, current_file):
            """! @brief Choix d'une optimisation : son détail, et son code présélectionné dans « Télécharger le code »."""
            try:
                files = optim_job_details(store, job_id)[3]
            except (KeyError, ValueError):
                files = []
            paths = [p for _, p in files]
            if not paths or current_file in paths:
                return details(job_id) + (gr.update(), gr.update())
            return details(job_id) + (gr.update(choices=download_choices(), value=paths[0]), paths[0])

        def on_tick(job_id, last_choices, last_done):
            """! @brief Rafraîchissement périodique : bandeau, tableau, menu des jobs, détail, et rapports quand un job finit."""
            choices = job_choices(store, "optim")
            pick = choice_update(choices, job_id) if choices != last_choices else gr.update()
            done = done_optim_jobs(store)
            reports = report_updates() if done != last_done else tuple(gr.update() for _ in report_outputs)
            detail = details(job_id) if job_id is not None else tuple(gr.update() for _ in detail_outputs)
            return (optim_banner(store), optim_jobs_table(store), pick, choices, done) + reports + detail

        def on_tab(job_id):
            """! @brief Ouverture de l'onglet : optimisations lancées depuis une autre session."""
            return optim_jobs_table(store), choice_update(job_choices(store, "optim"), job_id)

        def on_cancel(job_id, ok):
            """! @brief Bouton Annuler : même sémantique que dans l'onglet Suivre (confirmation obligatoire)."""
            return job_action(cancel_job, store, job_id, ok, missing=NO_JOB_SELECTED), False, optim_jobs_table(store)

        def on_requeue(job_id):
            """! @brief Bouton Relancer : remet en file un job échoué, annulé ou interrompu, qui recommence au début."""
            return job_action(requeue_job, store, job_id, missing=NO_JOB_SELECTED), optim_jobs_table(store)

        report_outputs = [o_results_hint, o_alerts, o_table, o_bd, o_bd_hint, o_points, o_full, o_files, o_files_hint]
        detail_outputs = [o_summary, o_log, o_job_table]
        o_analyze.click(on_analyze, [o_sources, o_tile, o_recipes, o_clip, o_channel, o_ncal, o_calib, o_nimg], o_report,
                        api_name="analyze_optim")
        o_go.click(on_go, [o_name, o_author, o_sources, o_recipes, o_tile, o_calib, o_clip, o_channel, o_ncal, o_nimg, o_export,
                           o_compile, o_ind, o_ind_reason], [o_status, o_jobs, o_pick], api_name="launch_optim")
        o_refresh.click(refresh, None, [banner, o_jobs] + report_outputs)
        menus = [o_sources, o_calib]
        menu_outputs = [o_sources, o_sources_hint, o_calib, o_calib_hint]
        o_refresh.click(menu_updates, menus, menu_outputs, show_progress="hidden")
        tab.select(menu_updates, menus, menu_outputs, show_progress="hidden")
        tab.select(on_tab, o_pick, [o_jobs, o_pick], show_progress="hidden")
        o_files.input(lambda p: p, o_files, o_file)
        o_pick.change(on_pick, [o_pick, o_files], detail_outputs + [o_files, o_file], show_progress="hidden")
        o_cancel.click(on_cancel, [o_pick, o_confirm], [o_msg, o_confirm, o_jobs])
        o_requeue.click(on_requeue, o_pick, [o_msg, o_jobs])
        gr.Timer(REFRESH_S).tick(on_tick, [o_pick, o_choices, o_done],
                                 [banner, o_jobs, o_pick, o_choices, o_done] + report_outputs + detail_outputs,
                                 show_progress="hidden")
