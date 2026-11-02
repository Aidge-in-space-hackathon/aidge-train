"""! @brief Enchaînement d'une optimisation : import, rapport, mémoire, calibration, puis chaque recette, et report.json.

Une recette en échec n'arrête pas les suivantes, mais sans sa référence FP32 le job échoue. L'arrêt se vérifie entre les
étapes ; il n'y a pas de reprise à mi-chemin : tout s'écrit dans un dossier temporaire, renommé seulement à la fin.
Contrat : docs/conception_optimisation.md (§ 5 et § 6).
"""
import json
import os
import shutil
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

import aidge_core
import aidge_quantization as aq

from .. import backends, config
from ..images import list_images
from ..nn import freeze
from ..training import ConfigError
from ..training.config import resolve_path
from . import evaluate, export, export_c, graph, memory, ptq, results
from .config import n_steps

## Nombre de tuiles de contrôle tirées dans le jeu d'évaluation (fidélité et vérification des exports).
N_CONTROL_TILES = 4
## Écart de graine entre la calibration et le contrôle.
CONTROL_SEED_OFFSET = 1


class Stopped(Exception):
    """! @brief Arrêt demandé entre deux étapes : le dossier temporaire est abandonné."""


class _Progress:
    """! @brief Compte les étapes, vérifie la demande d'arrêt et prévient l'appelant."""

    def __init__(self, should_stop, on_stage):
        """! @brief Prépare le suivi.

        @param should_stop  Fonction sans argument, ou None.
        @param on_stage     Fonction (numéro d'étape, nom), ou None.
        """
        ## Demande d'arrêt.
        self.should_stop = should_stop
        ## Notification d'étape.
        self.on_stage = on_stage
        ## Numéro de la dernière étape.
        self.step = 0

    def check(self):
        """! @brief Lève Stopped si l'arrêt est demandé."""
        if self.should_stop is not None and self.should_stop():
            raise Stopped()

    def stage(self, name):
        """! @brief Commence une étape.

        @param name  Nom affiché.
        @throws Stopped si l'arrêt est demandé.
        """
        self.check()
        self.step += 1
        if self.on_stage is not None:
            self.on_stage(self.step, name)


def _write_report(report, folder):
    """! @brief Écrit report.json de façon atomique.

    @param report  Rapport.
    @param folder  Dossier de résultats.
    """
    path = Path(folder) / results.REPORT_NAME
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({k: v for k, v in report.items() if not k.startswith("_")}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    os.replace(tmp, path)


def _prepare(cfg):
    """! @brief Vérifie que les données existent et charge le modèle.

    @param cfg  OptimConfig.
    @return Dictionnaire @c model, @c encoder, @c checkpoint, @c source (clé de report.json), @c eval_paths, @c calib_dir.
    @throws ConfigError si un fichier, un jeu d'images ou la tuile ne convient pas.
    """
    checkpoint = resolve_path(cfg.source.checkpoint)
    if not checkpoint.is_file():
        raise ConfigError(f"checkpoint introuvable : {cfg.source.checkpoint}")
    calib_dir = Path(config.TRAIN_DIR) / cfg.ptq.calibration_set
    if not calib_dir.is_dir():
        raise ConfigError(f"jeu de calibration introuvable : {cfg.ptq.calibration_set} (dans {config.TRAIN_DIR})")
    eval_paths = list_images(resolve_path(cfg.evaluate.eval_dir))[:cfg.evaluate.n_images]
    if not eval_paths:
        raise ConfigError(f"aucune image d'évaluation dans : {cfg.evaluate.eval_dir}")
    try:
        model, ck = graph.load_model(checkpoint)
    except Exception as e:
        raise ConfigError(f"checkpoint illisible ({cfg.source.checkpoint}) : {e}") from e
    if model.in_ch != cfg.tile[0]:
        raise ConfigError(f"tile : le checkpoint a {model.in_ch} canal(aux), la tuile {cfg.tile[0]} (reçu {cfg.tile})")
    encoder = freeze(model.g_a).eval()
    lam = ck["cfg"]["lmbda"]
    source = {"codec": cfg.source.codec, "checkpoint": cfg.source.checkpoint, "file_sha256": results.file_sha256(checkpoint),
              "fingerprint": backends.fingerprint(encoder), "in_ch": model.in_ch, "N": model.N,
              "gdn_last": bool(ck["cfg"].get("gdn_last", True)), "lmbda": lam, "step": int(ck.get("step", 0))}
    return {"model": model, "encoder": encoder, "checkpoint": checkpoint, "source": source, "eval_paths": eval_paths,
            "calib_dir": calib_dir}


def _versions():
    """! @brief Versions d'Aidge utilisées."""
    import importlib.metadata as md
    out = {}
    for key, package in (("core", "aidge-core"), ("quantization", "aidge-quantization"), ("export_cpp", "aidge-export-cpp")):
        try:
            out[key] = md.version(package)
        except md.PackageNotFoundError:
            out[key] = None
    return out


def analyze(cfg):
    """! @brief Analyse rapide, sans quantifier : import, rapport de quantifiabilité, mémoire.

    Les poids de la recette mixte sont estimés (convolutions sur 1 octet par poids et 4 par biais, GDN inchangée).
    @param cfg  OptimConfig.
    @return Dictionnaire sérialisable en JSON : @c source, @c tile, @c nodes, @c quantizability, @c memory_fp32, @c memory_mixte.
    @throws ConfigError si le checkpoint ou la tuile ne convient pas.
    """
    aidge_core.Log.set_console_level(aidge_core.Level.Fatal)
    checkpoint = resolve_path(cfg.source.checkpoint)
    if not checkpoint.is_file():
        raise ConfigError(f"checkpoint introuvable : {cfg.source.checkpoint}")
    model, ck = graph.load_model(checkpoint)
    if model.in_ch != cfg.tile[0]:
        raise ConfigError(f"tile : le checkpoint a {model.in_ch} canal(aux), la tuile {cfg.tile[0]} (reçu {cfg.tile})")
    encoder = freeze(model.g_a).eval()
    g = graph.import_encoder(encoder, cfg.tile)
    nodes = graph.count_nodes(g)
    report = graph.quantizability(g)
    check = aq.check_architecture(g)
    sample = np.zeros((1, *cfg.tile), dtype=np.float32)
    mem = memory.memory_report(g, sample)
    conv_int8 = sum(int(np.array(n.get_operator().get_output(0)).size) * (1 if "weight" in n.name() else 4)
                    for n in g.get_nodes() if n.type() == "Producer" and n.name().startswith("conv"))
    mixed = memory.summarize(mem["peak_bytes"], mem["weights_bytes"] - mem["conv_weights_bytes"] + conv_int8, conv_int8)
    return {"source": {"N": model.N, "in_ch": model.in_ch, "lmbda": ck["cfg"]["lmbda"], "step": int(ck.get("step", 0))},
            "tile": list(cfg.tile), "nodes": nodes, "quantizability": report, "check_architecture": bool(check),
            "memory_fp32": mem, "memory_mixte": mixed}


def _skip(progress, cfg):
    """! @brief Avance le compteur des étapes qu'une recette sans graphe ne fera pas."""
    progress.step += 2 + (3 if cfg.export.generate else 0)


def _run_export(cfg, variant, folder, build_graph, runner, reference, control, compiler, progress):
    """! @brief Export C++ d'une recette : génération, compilation, vérification, zip.

    @param cfg          OptimConfig.
    @param variant      Entrée de la variante, complétée en place.
    @param folder       Dossier de la variante.
    @param build_graph  Fonction sans argument qui reconstruit le graphe : l'export modifie celui qu'il reçoit.
    @param runner       Fonction tuile vers latents, par le graphe Aidge.
    @param reference    Fonction tuile vers latents, par PyTorch.
    @param control      Tuiles de contrôle.
    @param compiler     Chemin de g++, ou None.
    @param progress     _Progress.
    @return Chemin du binaire si l'export est vérifié, sinon None.
    """
    progress.stage("export C++")
    target = Path(folder) / "export"
    manifest = export.generate(build_graph(), target, control[:1])
    result = {"status": "non_compile", "path": _portable_rel(target, folder), "zip": None, "compiler": None, "text_bytes": None,
              "bss_bytes": None, "verified_against": None, "symbols_identical_pct": None,
              "symbols_identical_vs_pytorch_pct": None, "message": None}
    variant["export"] = result
    binary = None
    progress.stage("compilation")
    if not cfg.export.compile:
        result["message"] = "Compilation désactivée : export seul."
    elif compiler is None:
        result["message"] = "Pas de compilateur g++ sur cette machine : export seul, à compiler ailleurs."
    else:
        result["compiler"] = export.compiler_version(compiler)
        try:
            binary, error = export.compile_export(target, manifest, compiler, progress.should_stop)
        except export.CompileStopped as e:
            raise Stopped() from e
        if binary is None:
            result["status"], result["message"] = "echec_compilation", error
    progress.stage("vérification")
    if binary is not None:
        check = export.verify(binary, control, runner, reference)
        result.update({"verified_against": "aidge", "symbols_identical_pct": check["symbols_identical_pct"],
                       "symbols_identical_vs_pytorch_pct": check["symbols_identical_vs_pytorch_pct"]})
        result.update(export.binary_sizes(binary))
        result["status"] = "verifie" if check["ok"] else "divergent"
        if not check["ok"]:
            result["message"] = (f"Seuls {check['symbols_identical_pct']} % des symboles sont identiques au graphe Aidge "
                                 f"(seuil : {export.EXPORT_MIN_IDENTICAL_PCT:g} %).")
            variant["status"], variant["reason"] = ptq.STATUS_FAILED, result["message"]
            binary = None
    zip_path = Path(folder) / "export.zip"
    export.make_zip(target, zip_path)
    result["zip"] = _portable_rel(zip_path, folder)
    return binary


def _portable_rel(path, folder):
    """! @brief Chemin relatif au dossier de résultats (le parent de @p folder), en POSIX."""
    return Path(path).relative_to(Path(folder).parent).as_posix()


def _run_locod(variant, folder, build_graph, control, cpp_binary, cpp_folder, progress):
    """! @brief Fichier C pour LoCod : génération, compilation native et vérification couche par couche.

    @param variant      Entrée de la variante mixte, complétée en place.
    @param folder       Dossier de la variante.
    @param build_graph  Fonction sans argument qui reconstruit le graphe.
    @param control      Tuiles de contrôle.
    @param cpp_binary   Binaire vérifié de l'export C++ (référence couche par couche), ou None.
    @param cpp_folder   Dossier de l'export C++.
    @param progress     _Progress.
    """
    progress.stage("fichier C pour LoCod")
    result = {"status": "non_compile", "path": None, "accelerators": None, "pl_weights_bytes": None,
              "transfers_per_tile_bytes": None, "layers_identical_pct": None, "symbols_identical_pct": None, "message": None}
    variant["locod"] = result
    g = build_graph()
    runner = graph.Runner(g)
    runner(control[:1])
    try:
        out = export_c.export_locod(g, Path(folder) / "locod", name=variant.get("_name", "encodeur"))
    except export_c.ExportCError as e:
        result["status"], result["message"] = "echec", str(e)
        return
    result.update({"path": _portable_rel(out.path, folder), "accelerators": out.accelerators,
                   "pl_weights_bytes": out.pl_weights_bytes, "transfers_per_tile_bytes": out.transfers_per_tile_bytes})
    compiler = shutil.which("gcc")
    if compiler is None:
        result["message"] = "Pas de compilateur C sur cette machine : fichier non vérifié."
        return
    binary, error = export_c.compile_native(out.path, compiler)
    if binary is None:
        result["status"], result["message"] = "echec_compilation", error
        return
    last = out.layers[-1][0]
    finals, per_layer = [], {}
    for t in control:
        layers = export_c.run_native(binary, t[None], out)
        reference = runner(t[None])
        finals.append(evaluate.symbols_identical_pct(reference, layers[last][None]))
        if cpp_binary is not None:
            export.run_binary(cpp_binary, t[None], reference.shape)
            for name, values in export.conv_layer_outputs(cpp_binary, cpp_folder).items():
                if name.startswith("conv") and name in layers:
                    per_layer.setdefault(name, []).append(float(np.mean(layers[name] == values)) * 100)
    result["symbols_identical_pct"] = round(min(finals), 3)
    result["layers_identical_pct"] = {k: round(min(v), 3) for k, v in per_layer.items()} or None
    worst = min([result["symbols_identical_pct"], *(result["layers_identical_pct"] or {}).values()])
    if worst >= export.EXPORT_MIN_IDENTICAL_PCT:
        result["status"] = "verifie"
    else:
        result["status"] = "divergent"
        result["message"] = f"Seuls {worst:.3f} % des valeurs sont identiques (seuil : {export.EXPORT_MIN_IDENTICAL_PCT:g} %)."
        variant["status"], variant["reason"] = ptq.STATUS_FAILED, result["message"]


def _run_recipe(cfg, recipe, data, calibration, control, reference, folder, report, progress):
    """! @brief Exécute une recette et complète son entrée du rapport.

    @param cfg          OptimConfig.
    @param recipe       Recette.
    @param data         Résultat de _prepare().
    @param calibration  Tuiles de calibration.
    @param control      Tuiles de contrôle.
    @param reference    Fonction tuile vers latents, par PyTorch.
    @param folder       Dossier de résultats.
    @param report       Rapport, complété en place.
    @param progress     _Progress.
    @throws Stopped si l'arrêt est demandé.
    """
    encoder, tile = data["encoder"], tuple(cfg.tile)
    clip, per_channel = cfg.ptq.clipping, cfg.ptq.per_channel
    variant = {"status": ptq.STATUS_OK, "reason": None, "signals": None, "nodes_before": report["nodes"], "nodes_after": None,
               "memory": None, "fidelity": None, "rd": None, "export": None, "locod": None}
    report["variants"][recipe] = variant
    progress.stage(f"{recipe} : quantification")
    if recipe == "fp32":
        build = lambda: graph.import_encoder(encoder, tile)
    elif recipe == "mixte":
        build = lambda: ptq.build_mixed(encoder, tile, calibration, clip, per_channel)
    else:
        attempt = ptq.try_int8(encoder, tile, calibration, clip, per_channel)
        variant.update({"status": attempt["status"], "reason": attempt["reason"], "signals": attempt["signals"],
                        "nodes_after": attempt["nodes_after"]})
        if attempt["status"] != ptq.STATUS_OK:
            _skip(progress, cfg)
            return
        built = attempt["graph"]
        build = lambda: built
    g = build()
    variant["nodes_after"] = graph.count_nodes(g)
    run = graph.Runner(g)
    variant["memory"] = memory.memory_report(build() if recipe != "int8" else g, calibration[:1])
    progress.stage(f"{recipe} : fidélité")
    variant["fidelity"] = evaluate.fidelity([reference(t[None]) for t in control], [run(t[None]) for t in control])
    progress.stage(f"{recipe} : débit-distorsion")
    rd = evaluate.rd_average(data["model"], run, data["eval_paths"], cfg.mode, tile, progress.should_stop)
    if rd is None:
        raise Stopped()
    variant["rd"] = evaluate.add_deltas(rd, report["variants"]["fp32"]["rd"]) if recipe != "fp32" else rd
    if cfg.export.generate:
        sub = Path(folder) / recipe
        sub.mkdir(parents=True, exist_ok=True)
        binary = _run_export(cfg, variant, sub, build, run, reference, control, export.find_compiler(), progress)
        if recipe == "mixte":
            variant["_name"] = cfg.name
            _run_locod(variant, sub, build, control, binary, sub / "export", progress)
            variant.pop("_name", None)


def _check_out(final):
    """! @brief Refuse un dossier de résultats qui contient autre chose qu'un résultat : il serait remplacé à la fin du job.

    @param final  Dossier de résultats.
    @throws ConfigError si le dossier n'est ni absent, ni vide, ni un résultat précédent.
    """
    if final.is_dir() and any(final.iterdir()) and not (final / results.REPORT_NAME).is_file():
        raise ConfigError(f"{final} contient autre chose qu'un résultat d'optimisation : il serait écrasé")


def run(cfg, job_id=None, out=None, should_stop=None, on_stage=None, log_path=None):
    """! @brief Exécute toute l'optimisation.

    @param cfg          OptimConfig.
    @param job_id       Numéro du job, ou None en mode --config.
    @param out          Dossier de résultats ; None : config.OPTIM_DIR/<id>.
    @param should_stop  Fonction sans argument scrutée entre les étapes.
    @param on_stage     Fonction (numéro d'étape, nom).
    @param log_path     Journal du job, recopié en run.log à la fin.
    @return Dossier de résultats.
    @throws ConfigError si les données ne conviennent pas.
    @throws Stopped si l'arrêt est demandé.
    @throws RuntimeError si la référence FP32 échoue.
    """
    aidge_core.Log.set_console_level(aidge_core.Level.Error)
    progress = _Progress(should_stop, on_stage)
    run_id = str(job_id) if job_id is not None else f"cfg-{cfg.name or 'optim'}-{datetime.now():%Y%m%d-%H%M%S}"
    final = Path(out) if out else Path(config.OPTIM_DIR) / run_id
    _check_out(final)
    data = _prepare(cfg)
    results.remove_temporary(final.name, final.parent)     # tentative précédente tuée sans nettoyage
    tmp = final.with_name(f"{final.name}{results.TMP_MARK}{os.getpid()}")
    tmp.mkdir(parents=True)
    try:
        cfg.to_yaml(tmp / "config.yaml")
        encoder, tile = data["encoder"], tuple(cfg.tile)
        indicative = cfg.evaluate.indicative_series
        report = {"schema": results.REPORT_SCHEMA, "id": run_id, "name": cfg.name, "author": cfg.author, "status": "running",
                  "aidge": _versions(), "source": data["source"], "tile": list(tile), "bram_bytes": config.BRAM_BYTES,
                  "n_steps": n_steps(cfg), "config": cfg.to_dict(), "nodes": None, "quantizability": None, "variants": {},
                  "series": {"indicative": indicative, "reason": cfg.evaluate.indicative_reason or None}}
        progress.stage("import du graphe")
        g = graph.import_encoder(encoder, tile)
        report["nodes"] = graph.count_nodes(g)
        progress.stage("rapport de quantifiabilité")
        report["quantizability"] = graph.quantizability(g)
        progress.stage("plan mémoire")
        progress.stage("calibration")
        calibration = ptq.calibration_tiles(data["calib_dir"], cfg.ptq.n_calibration, tile, cfg.ptq.seed)
        np.savez_compressed(tmp / "calib.npz", tiles=calibration)
        control = ptq.calibration_tiles(resolve_path(cfg.evaluate.eval_dir), N_CONTROL_TILES, tile,
                                        cfg.ptq.seed + CONTROL_SEED_OFFSET)
        reference = backends.get_runner("torch", encoder, (1, *tile))
        for recipe in cfg.recipes:
            try:
                _run_recipe(cfg, recipe, data, calibration, control, reference, tmp, report, progress)
            except Stopped:
                raise
            except Exception as e:
                if recipe == "fp32":
                    raise RuntimeError(f"la référence FP32 a échoué : {e}") from e
                traceback.print_exc()
                report["variants"].setdefault(recipe, {})
                report["variants"][recipe].update({"status": ptq.STATUS_FAILED, "reason": f"{type(e).__name__} : {e}"})
            _write_report(report, tmp)
        report["status"] = "done"
        _write_report(report, tmp)
        if log_path is not None and Path(log_path).exists():
            shutil.copyfile(log_path, tmp / "run.log")
        shutil.rmtree(final, ignore_errors=True)
        os.replace(tmp, final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return final
