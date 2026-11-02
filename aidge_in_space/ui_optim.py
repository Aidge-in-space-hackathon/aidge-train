"""! @brief Onglet « 5. Optimisation Aidge » de l'interface Gradio.

L'interface ne quantifie jamais elle-même : « Lancer » dépose un job, et « Analyser » passe par un sous-processus court, car
la capture du rapport d'Aidge agit sur tout le processus et un plantage C++ tuerait l'interface. Toute la logique est dans
des fonctions pures ; les callbacks de build_tab() n'en sont que de minces adaptateurs. Contrat :
docs/conception_optimisation.md (§ 9).
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd

from . import config, ui_style
from .codecs import all_codecs
from .codecs.balle2017 import Balle2017Codec
from .jobs import JobStore
from .optim import results
from .optim.config import DEFAULT_RECIPE, CLIPPINGS, RECIPES, OptimConfig, n_steps
from .ui_training import REFRESH_S, choice_update, dataset_choices, worker_banner

## Tailles de tuile proposées (côté).
TILE_CHOICES = [32, 64, 128, 256]
## Tuile par défaut : à N=64, poids et activations y tiennent ensemble dans la BRAM.
DEFAULT_TILE = 64
## Délai maximal de l'analyse, en secondes.
ANALYZE_TIMEOUT_S = 300
## Commande de l'analyse. Remplaçable dans les tests.
ANALYZE_CMD = [sys.executable, "-m", "aidge_in_space.optimize", "--analyze"]
## Colonnes du tableau des BD-rate.
BD_COLUMNS = ["série", "recette", "niveaux", "BD-rate (%)", "raison", "indicative"]
## Alerte quand la tentative INT8 n'a rien changé.
NO_EFFECT_ALERT = ("**Aidge 0.10.1 ne sait pas quantifier la GDN : le graphe n'a pas changé.** La recette mixte garde la GDN "
                   "en flottant et quantifie les convolutions.")


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


def menu_updates(checked, calibration_set):
    """! @brief Menus relus d'après l'état du serveur : un entraînement ou un jeu d'images a pu apparaître depuis le démarrage.

    @param checked          Checkpoints cochés ; ceux qui ont disparu sont décochés.
    @param calibration_set  Jeu de calibration choisi.
    @return Couple de gr.update : les checkpoints, le jeu de calibration.
    """
    sources = source_choices()
    known = {path for _, path in sources}
    return (gr.update(choices=sources, value=[c for c in checked or [] if c in known]),
            choice_update(dataset_choices(), calibration_set))


def build_config(name, author, checkpoint, codec, in_ch, tile_side, recipes, clipping, per_channel, n_calibration,
                 calibration_set, n_images, export, compile_export, indicative=False, indicative_reason=""):
    """! @brief Configuration d'optimisation depuis le formulaire.

    @return OptimConfig validée.
    @throws ValueError si un champ est invalide (message affichable tel quel).
    """
    if not calibration_set:
        raise ValueError("Choisissez un jeu de calibration.")
    return OptimConfig.from_dict({
        "name": name or "", "author": author or "", "source": {"codec": codec or "", "checkpoint": checkpoint or ""},
        "tile": [int(in_ch), int(tile_side), int(tile_side)], "recipes": list(recipes or []),
        "ptq": {"clipping": clipping, "per_channel": bool(per_channel), "n_calibration": int(n_calibration),
                "calibration_set": calibration_set},
        "evaluate": {"n_images": int(n_images), "indicative_series": bool(indicative), "indicative_reason": indicative_reason or ""},
        "export": {"generate": bool(export), "compile": bool(compile_export)}})


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
        raise ValueError("Donnez un nom à cette optimisation.")
    if not checkpoints:
        raise ValueError("Cochez au moins un checkpoint.")
    known = {s["checkpoint"]: s for s in (list_sources() if sources is None else sources)}
    jobs = []
    for checkpoint in checkpoints:
        if checkpoint not in known:
            raise ValueError(f"Checkpoint inconnu : {checkpoint}")
        s = known[checkpoint]
        cfg = build_config(name, author, checkpoint, s["codec"], s["in_ch"], tile_side, recipes, clipping, per_channel,
                           n_calibration, calibration_set, n_images, export, compile_export, indicative, indicative_reason)
        jobs.append((s["lmbda"], cfg.to_job(s["in_ch"], s["N"], s["lmbda"], n_steps(cfg))))
    return [store.submit(job, kind="optim") for _, job in sorted(jobs, key=lambda j: j[0])]


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
        raise ValueError(lines[-1] if lines else f"L'analyse a échoué (code {r.returncode}).")
    return parse_analysis(r.stdout)


def _ratio(part, whole):
    """! @brief Pourcentage arrondi, pour l'affichage."""
    return f"{part / whole * 100:.0f} %"


def analysis_markdown(result):
    """! @brief Compte rendu de l'analyse, pour des lecteurs qui ne connaissent pas Aidge (fonction pure).

    @param result  Résultat de analyze().
    @return Texte Markdown.
    """
    q = result["quantizability"]
    unknown = [o["type"] for o in q["operators"] if o["state"] != "ok"]
    fp32, mixed = result["memory_fp32"], result["memory_mixte"]
    bram = fp32["bram_bytes"]
    total = mixed["peak_bytes"] + mixed["weights_bytes"]
    lines = [f"**Modèle N={result['source']['N']}, λ={result['source']['lmbda']}, tuile {result['tile'][1]} x {result['tile'][2]}.**", ""]
    if unknown:
        lines += [f"- **Quantification du graphe entier impossible** : Aidge ne prend pas en charge {', '.join(unknown)}. "
                  "La recette mixte contourne ce blocage.", ""]
    else:
        lines += ["- Tous les opérateurs sont pris en charge par la quantification d'Aidge.", ""]
    lines += ["| | Activations (pic) | Poids | Total | Face à la BRAM |", "| :- | -: | -: | -: | -: |",
              f"| FP32 | {fp32['peak_bytes'] // 1024} Kio | {fp32['weights_bytes'] // 1024} Kio | "
              f"{(fp32['peak_bytes'] + fp32['weights_bytes']) // 1024} Kio | {_ratio(fp32['peak_bytes'] + fp32['weights_bytes'], bram)} |",
              f"| Mixte (poids estimés) | {mixed['peak_bytes'] // 1024} Kio | {mixed['weights_bytes'] // 1024} Kio | "
              f"{total // 1024} Kio | {_ratio(total, bram)} |", "",
              f"Avec la GDN sur le processeur, seuls les poids des convolutions restent dans le FPGA : "
              f"{mixed['conv_weights_bytes'] // 1024} Kio, soit {_ratio(mixed['conv_weights_bytes'], bram)} de la BRAM "
              f"({bram // 1024} Kio)."]
    return "\n".join(lines)


def variants_table(reports=None):
    """! @brief Tableau des variantes.

    @param reports  Rapports ; None : lus à l'appel.
    @return DataFrame aux colonnes de aidge_in_space.optim.results.TABLE_COLUMNS.
    """
    reports = results.read_reports() if reports is None else reports
    return pd.DataFrame(results.variant_rows(reports), columns=results.TABLE_COLUMNS)


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
    lines += [f"- {m}" for m in results.no_effect_alerts([
        {**r, "variants": {k: v for k, v in r["variants"].items() if v.get("status") == "echec"}} for r in reports])]
    return "\n\n".join(lines)


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
    @return Couple de DataFrame : les BD-rate (colonnes BD_COLUMNS), et les écarts par niveau.
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
    return pd.DataFrame(rows, columns=BD_COLUMNS), pd.DataFrame(points, columns=["série", "recette", "λ", "Δbpp (%)", "ΔPSNR (dB)"])


def download_choices(reports=None):
    """! @brief Fichiers à télécharger : zip de l'export C++ et fichier C pour LoCod.

    @param reports  Rapports ; None : lus à l'appel.
    @return Liste de couples (libellé, chemin existant).
    """
    reports = results.read_reports() if reports is None else reports
    out = []
    for report in reversed(reports):
        for recipe, v in report.get("variants", {}).items():
            base = f"n°{report['id']} · {report['name']} · {recipe}"
            zip_rel = (v.get("export") or {}).get("zip")
            if zip_rel and (Path(report["_dir"]) / zip_rel).is_file():
                out.append((f"{base} : export C++ (zip)", str(Path(report["_dir"]) / zip_rel)))
            c_rel = (v.get("locod") or {}).get("path")
            if c_rel and (Path(report["_dir"]) / c_rel).is_file():
                out.append((f"{base} : fichier C pour LoCod", str(Path(report["_dir"]) / c_rel)))
    return out


def build_tab(store=None):
    """! @brief Crée l'onglet, dans le @c with gr.Tabs() de app.build().

    @param store  JobStore ; None : la base par défaut.
    """
    store = store or JobStore()
    sources, datasets = source_choices(), dataset_choices()
    with gr.Tab("5. Optimisation Aidge", id="optimisation") as tab:
        ui_style.intro("**Préparez un modèle pour la carte.** Aidge quantifie l'encodeur, mesure sa mémoire face à la BRAM, "
                       "et produit le code C pour le processeur et le FPGA.",
                       "la variante apparaît dans **3. Tester** et **4. Valider**, face à l'original.")
        banner = gr.Markdown(worker_banner(store))
        with gr.Row():
            with gr.Column(scale=2, min_width=340):
                with ui_style.card("Modèle"):
                    o_name = gr.Textbox(label="Nom de l'optimisation", info="Les checkpoints cochés forment une courbe.")
                    o_author = gr.Textbox(label="Auteur (facultatif)")
                    o_sources = gr.CheckboxGroup(sources, value=[sources[0][1]] if sources else [], label="Checkpoints")
                with ui_style.card("Recette"):
                    o_recipes = gr.CheckboxGroup(list(RECIPES), value=["fp32", DEFAULT_RECIPE], label="Recettes",
                                                 info="mixte : convolutions en entiers, GDN en flottant. fp32 sert de référence.")
                    o_tile = gr.Dropdown(TILE_CHOICES, value=DEFAULT_TILE, label="Taille de tuile")
                    o_calib = gr.Dropdown(datasets, value=datasets[0][1] if datasets else None, label="Images de calibration")
                with gr.Row():
                    o_analyze = gr.Button("Analyser")
                    o_go = gr.Button("Lancer", variant="primary", elem_classes="main-action")
                o_status = gr.Markdown()
                with gr.Accordion("Options", open=False):
                    o_clip = gr.Dropdown(list(CLIPPINGS), value="MAX", label="Écrêtage de la calibration")
                    o_channel = gr.Checkbox(value=False, label="Quantifier par canal")
                    o_ncal = gr.Slider(1, 512, value=32, step=1, label="Tuiles de calibration")
                    o_nimg = gr.Slider(1, 24, value=24, step=1, label="Images d'évaluation")
                    o_export = gr.Checkbox(value=True, label="Générer le code C")
                    o_compile = gr.Checkbox(value=True, label="Compiler et vérifier le code (si g++ est là)")
                    o_ind = gr.Checkbox(value=False, label="Série indicative (entraînements différents)")
                    o_ind_reason = gr.Textbox(label="Raison affichée")
            with gr.Column(scale=5, min_width=340):
                with ui_style.card("Analyse"):
                    o_report = gr.Markdown("Choisissez un checkpoint puis cliquez sur **Analyser**.")
                with ui_style.card("Variantes", "**pic / BRAM** : mémoire de travail. **poids conv / BRAM** : ce qui reste dans le "
                                   "FPGA. **Δbpp**, **ΔPSNR** : écart face à l'original, plus proche de 0 est meilleur."):
                    o_alerts = gr.Markdown(alerts_markdown())
                    o_table = gr.Dataframe(variants_table(), show_label=False, wrap=True)
                    o_refresh = gr.Button("Actualiser")
                with ui_style.card("BD-rate face au FP32", "Écart de débit à qualité égale : positif, la variante demande plus de "
                                   "bits. Il faut 4 niveaux, et des courbes monotones."):
                    bd0, pts0 = bd_tables()
                    o_bd = gr.Dataframe(bd0, show_label=False)
                    o_points = gr.Dataframe(pts0, label="Écarts point par point")
                with ui_style.card("Code C à télécharger"):
                    o_files = gr.Dropdown(download_choices(), value=None, label="Fichier")
                    o_file = gr.File(label="Téléchargement")

        def form_config(checkpoints, tile, recipes, clip, channel, ncal, calib, nimg, export, compile_export):
            """! @brief Configuration d'analyse : le premier checkpoint coché."""
            if not checkpoints:
                raise ValueError("Cochez au moins un checkpoint.")
            s = next((x for x in list_sources() if x["checkpoint"] == checkpoints[0]), None)
            if s is None:
                raise ValueError(f"Checkpoint inconnu : {checkpoints[0]}")
            return build_config("analyse", "", s["checkpoint"], s["codec"], s["in_ch"], tile, recipes, clip, channel, ncal,
                                calib or "analyse", nimg, export, compile_export)

        def on_analyze(*values):
            """! @brief Bouton Analyser : erreurs converties en message simple."""
            try:
                return analysis_markdown(analyze(form_config(*values)))
            except ValueError as e:
                raise gr.Error(str(e))

        def on_go(name, author, checkpoints, recipes, tile, calib, clip, channel, ncal, nimg, export, compile_export, ind, reason):
            """! @brief Bouton Lancer : dépose les jobs."""
            try:
                ids = launch_jobs(store, name, author, checkpoints, tile, recipes, clip, channel, ncal, calib, nimg, export,
                                  compile_export, ind, reason)
            except (ValueError, KeyError) as e:
                raise gr.Error(str(e))
            return f"Job(s) déposé(s) : {', '.join(f'n°{i}' for i in ids)}. Suivez-les dans **2. Suivre**."

        def refresh():
            """! @brief Relit les rapports et le bandeau du worker."""
            reports = results.read_reports()
            bd, points = bd_tables(reports)
            return (worker_banner(store), alerts_markdown(reports), variants_table(reports), bd, points,
                    gr.update(choices=download_choices(reports)))

        analyze_inputs = [o_sources, o_tile, o_recipes, o_clip, o_channel, o_ncal, o_calib, o_nimg, o_export, o_compile]
        o_analyze.click(on_analyze, analyze_inputs, o_report, api_name="analyze_optim")
        o_go.click(on_go, [o_name, o_author, o_sources, o_recipes, o_tile, o_calib, o_clip, o_channel, o_ncal, o_nimg, o_export,
                           o_compile, o_ind, o_ind_reason], o_status, api_name="launch_optim")
        outputs = [banner, o_alerts, o_table, o_bd, o_points, o_files]
        o_refresh.click(refresh, None, outputs)
        menus = [o_sources, o_calib]
        o_refresh.click(menu_updates, menus, menus, show_progress="hidden")
        tab.select(menu_updates, menus, menus, show_progress="hidden")
        o_files.input(lambda p: p, o_files, o_file)
        gr.Timer(REFRESH_S * 6).tick(refresh, None, outputs)
