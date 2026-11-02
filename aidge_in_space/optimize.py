"""! @brief CLI d'optimisation Aidge : python -m aidge_in_space.optimize (--config FICHIER | --job-id N | --analyze FICHIER).

Mode @c --config : optimise depuis un fichier YAML, sans base de données. Mode @c --job-id : exécute un job de la file
(statut @c running, fixé par le worker) ; le statut final n'est jamais écrit ici, le worker le déduit du code de retour.
Mode @c --analyze : import, rapport et mémoire, en JSON sur la sortie standard. Les codes de retour sont ceux de train.
Contrat : docs/conception_optimisation.md (§ 5).
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

import yaml

from .jobs import EXIT_CANCELLED, EXIT_CONFIG, EXIT_DONE, EXIT_FAILED, JobStore
from .optim.config import OptimConfig
from .train import StopWatcher, stop_code
from .training import ConfigError


def _say(*args) -> None:
    """! @brief print() immédiat (la sortie est redirigée vers le journal du job)."""
    print(*args, flush=True)


def build_parser() -> argparse.ArgumentParser:
    """! @brief Analyseur d'arguments de la CLI."""
    p = argparse.ArgumentParser(prog="python -m aidge_in_space.optimize",
                                description="Optimise un encodeur avec Aidge : quantification, mémoire, évaluation, exports.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", metavar="FICHIER", help="configuration YAML (sans base de données)")
    src.add_argument("--job-id", type=int, metavar="N", help="job de la base (statut running)")
    src.add_argument("--analyze", metavar="FICHIER", help="import, rapport et mémoire seulement, en JSON sur stdout")
    p.add_argument("--db", metavar="CHEMIN", help="base des jobs (défaut : RUNS_DIR/runs.sqlite)")
    p.add_argument("--out", metavar="DOSSIER", help="dossier de résultats (défaut : outputs/optim/<id>)")
    p.add_argument("--watch-stdin", action="store_true", help="s'arrête à la fin de stdin")
    return p


def _load(args):
    """! @brief Lit la configuration et le job demandés par les arguments.

    @param args  Arguments analysés par build_parser().
    @return Triplet (configuration, JobStore ou None, Job ou None).
    @throws ConfigError si la configuration ou le job est invalide.
    """
    if args.job_id is None:
        cfg = OptimConfig.from_yaml(args.config)
        if not cfg.name:
            cfg.name = Path(args.config).stem
        return cfg, None, None
    store = JobStore(args.db)
    try:
        job = store.get(args.job_id)
    except KeyError:
        raise ConfigError(f"job inconnu : {args.job_id}")
    if job.status != "running":
        raise ConfigError(f"le job {job.id} n'est pas en cours (statut {job.status})")
    if job.kind != "optim":
        raise ConfigError(f"le job {job.id} n'est pas une optimisation (type {job.kind})")
    return OptimConfig.from_job(job.config), store, job


def run(args) -> int:
    """! @brief Exécute l'optimisation demandée par les arguments.

    @param args  Arguments analysés par build_parser().
    @return Code de retour (EXIT_DONE, EXIT_CANCELLED ou EXIT_STOPPED ; les autres erreurs sont traitées par main()).
    @throws ConfigError si la configuration, le job ou les données sont invalides.
    """
    from .optim import pipeline
    if args.analyze:
        cfg = OptimConfig.from_yaml(args.analyze)
        result = pipeline.analyze(cfg)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return EXIT_DONE
    cfg, store, job = _load(args)
    job_id = job.id if job else None
    watcher = StopWatcher(store, job_id, args.watch_stdin, attempts=job.attempts if job else None)
    watcher.install()
    try:
        if watcher():
            return stop_code(watcher)
        if job:
            store.reset_progress(job_id, 0)
            store.job_dir(job_id).mkdir(parents=True, exist_ok=True)
            cfg.to_yaml(store.job_dir(job_id) / "config.yaml")

        def on_stage(step, stage):
            """! @brief Enregistre l'étape dans le journal et dans la base."""
            _say(f"Étape {step} : {stage}")
            if store is not None:
                store.log_metrics(job_id, step, "eval", {"kind": "eval", "stage": stage})

        log_path = store.job_dir(job_id) / "train.log" if job else None
        out = pipeline.run(cfg, job_id=job_id, out=args.out, should_stop=watcher, on_stage=on_stage, log_path=log_path)
    except pipeline.Stopped:
        code = stop_code(watcher)
        _say("Arrêt demandé (" + ("annulation" if code == EXIT_CANCELLED else "arrêt de l'application") + ").")
        return code
    finally:
        watcher.restore()
    _say(f"Terminé : {out}")
    return EXIT_DONE


def main(argv=None) -> int:
    """! @brief Point d'entrée de la CLI.

    @param argv  Arguments (défaut : sys.argv). Une erreur d'arguments lève SystemExit(2) (argparse).
    @return EXIT_DONE (0), EXIT_FAILED (1), EXIT_CONFIG (2), EXIT_CANCELLED (3) ou EXIT_STOPPED (4).
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (ConfigError, FileNotFoundError, yaml.YAMLError) as e:
        print(f"Configuration invalide : {e}", file=sys.stderr, flush=True)
        return EXIT_CONFIG
    except Exception:
        traceback.print_exc()
        return EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
