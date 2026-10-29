"""! @brief CLI d'entraînement : python -m aidge_in_space.train (--config FICHIER.yaml | --job-id N).

Mode @c --config : entraîne depuis un fichier YAML, sans base de données. Mode @c --job-id : exécute
un job de la file (statut @c running, fixé par le worker) et écrit ses métriques dans la base ; le
statut final du job n'est jamais écrit ici, c'est le worker qui le déduit du code de retour.
Contrat : docs/conception_entrainement.md (§ 2.4).
"""
import argparse
import dataclasses
import os
import signal
import sqlite3
import sys
import threading
import time
import traceback
from pathlib import Path

import yaml

from .jobs import EXIT_CANCELLED, EXIT_CONFIG, EXIT_DONE, EXIT_FAILED, EXIT_STOPPED, JobStore
from .training import ConfigError, TrainConfig

## Période minimale, en secondes, entre deux lectures du drapeau d'annulation dans la base.
CANCEL_POLL_S = 5.0


def _say(*args) -> None:
    """! @brief print() immédiat (la sortie est redirigée vers train.log)."""
    print(*args, flush=True)


class StopWatcher:
    """! @brief Demande d'arrêt de l'entraînement : annulation dans la base, fin de stdin ou signal.

    L'instance est appelable sans argument et sert de @c should_stop à train() : elle ne lit que des
    drapeaux en mémoire, sauf la lecture de la base, faite au plus toutes les @c poll_s secondes.
    """

    def __init__(self, store=None, job_id=None, watch_stdin=False, poll_s=CANCEL_POLL_S, attempts=None):
        """! @brief Prépare la surveillance (sans installer les signaux : voir install()).

        @param store        JobStore pour lire @c cancel_requested, ou None.
        @param job_id       Numéro du job surveillé.
        @param watch_stdin  Lance un thread démon qui détecte la fin de stdin.
        @param poll_s       Période minimale des lectures de la base.
        @param attempts     Valeur de @c attempts du job lue au démarrage ; si elle change dans la base (ou si le
                            job n'est plus @c running), un autre processus a repris le job : voir _taken_over().
        """
        ## Valeur de attempts au démarrage, ou None (pas de contrôle de reprise).
        self.attempts = attempts
        ## Base et job surveillés.
        self.store, self.job_id = store, job_id
        ## Période minimale entre deux lectures de la base.
        self.poll_s = poll_s
        ## Cause de l'arrêt : "cancel", "stdin", "signal", ou None tant qu'aucun arrêt n'est demandé.
        self.reason = None
        ## Instant (time.monotonic) de la dernière lecture de la base.
        self._last_poll = None
        ## Gestionnaires de signaux remplacés, pour les restaurer.
        self._previous = {}
        if watch_stdin:
            threading.Thread(target=self._watch_stdin, daemon=True).start()

    def _watch_stdin(self) -> None:
        """! @brief Lit stdin octet par octet (descripteur 0), sans exploiter les données, jusqu'à sa fin ou une erreur."""
        try:
            while os.read(0, 1):      # os.read et non sys.stdin.buffer : un fil bloqué dans le tampon fait planter Windows à l'arrêt
                pass
        except Exception:
            pass
        if self.reason is None:
            self.reason = "stdin"

    def _on_signal(self, signum, frame) -> None:
        """! @brief Gestionnaire de SIGINT, SIGTERM et SIGBREAK : demande un arrêt propre."""
        if self.reason is None:
            self.reason = "signal"

    def install(self) -> None:
        """! @brief Installe les gestionnaires de signaux (sans effet hors du fil principal)."""
        names = ("SIGINT", "SIGTERM", "SIGBREAK")
        for name in names:
            sig = getattr(signal, name, None)
            if sig is None:
                continue
            try:
                self._previous[sig] = signal.signal(sig, self._on_signal)
            except ValueError:
                return

    def restore(self) -> None:
        """! @brief Restaure les gestionnaires de signaux d'origine."""
        for sig, handler in self._previous.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, TypeError):
                pass
        self._previous = {}

    def cancel_in_db(self) -> bool:
        """! @brief Lit le drapeau d'annulation de la base, sans limite de fréquence.

        @return Vrai si l'annulation est demandée ; faux sans base ou en cas d'erreur SQLite (avertissement).
        """
        if self.store is None or self.job_id is None:
            return False
        try:
            return bool(self.store.cancel_requested(self.job_id))
        except sqlite3.Error as e:
            _say(f"ATTENTION : lecture de l'annulation impossible, ignorée ({e})")
            return False

    def _taken_over(self, job) -> None:
        """! @brief Arrête le processus sans sauvegarder si le job a été repris par un autre processus.

        Sans effet tant que le job est @c running avec la même valeur de @c attempts. Sinon : message dans
        le journal puis @c os._exit(EXIT_STOPPED), sans sauvegarde, pour ne pas écraser le checkpoint de
        l'autre processus.
        @param job  Job lu dans la base, ou None s'il a disparu.
        """
        if self.attempts is None:
            return
        if job is not None and job.status == "running" and job.attempts == self.attempts:
            return
        _say("ATTENTION : ce job a été repris par un autre processus, arrêt sans sauvegarde.")
        os._exit(EXIT_STOPPED)

    def _poll(self) -> bool:
        """! @brief Lit le job dans la base (une seule lecture) : contrôle de reprise, puis drapeau d'annulation.

        @return Vrai si l'annulation est demandée ; faux en cas d'erreur SQLite (avertissement).
        """
        try:
            job = self.store.get(self.job_id)
        except KeyError:
            job = None
        except sqlite3.Error as e:
            _say(f"ATTENTION : lecture du job impossible, ignorée ({e})")
            return False
        self._taken_over(job)
        return bool(job is not None and job.cancel_requested)

    def __call__(self) -> bool:
        """! @brief Vrai si un arrêt est demandé (la base est lue au plus une fois toutes les @c poll_s secondes)."""
        if self.reason is not None:
            return True
        if self.store is not None and self.job_id is not None:
            now = time.monotonic()
            if self._last_poll is None or now - self._last_poll >= self.poll_s:
                self._last_poll = now
                if self._poll():
                    self.reason = "cancel"
        return self.reason is not None


def build_parser() -> argparse.ArgumentParser:
    """! @brief Analyseur d'arguments de la CLI."""
    p = argparse.ArgumentParser(prog="python -m aidge_in_space.train",
                                description="Entraîne un modèle Ballé 2017 depuis un fichier YAML ou un job de la file.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", metavar="FICHIER", help="configuration YAML (sans base de données)")
    src.add_argument("--job-id", type=int, metavar="N", help="job de la base (statut running)")
    p.add_argument("--db", metavar="CHEMIN", help="base des jobs (défaut : RUNS_DIR/runs.sqlite)")
    p.add_argument("--out", metavar="DOSSIER", help="dossier des checkpoints (défaut : outputs/checkpoints)")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--watch-stdin", action="store_true", help="s'arrête (avec sauvegarde) à la fin de stdin")
    return p


def _describe(cfg: TrainConfig, job_id=None) -> str:
    """! @brief Première ligne du journal : job, nom, λ, N, mode et auteur."""
    head = f"Job {job_id} · " if job_id is not None else "Entraînement · "
    line = f"{head}« {cfg.name} » · λ={cfg.lmbda:g} · N={cfg.N} · {cfg.mode}"
    return line + (f" · auteur : {cfg.author}" if cfg.author else "")


def _device_line(device) -> str:
    """! @brief Ligne « Appareil : ... » avec le nom du GPU le cas échéant."""
    if device.type == "cuda":
        import torch
        return f"Appareil : cuda ({torch.cuda.get_device_name(0)})"
    return "Appareil : cpu"


def _start_step(ckpt_path: Path) -> int:
    """! @brief Pas de départ : celui du checkpoint repris, 0 s'il n'existe pas ou est illisible (train() tranchera)."""
    if not Path(ckpt_path).exists():
        return 0
    try:
        from .training.loop import load_checkpoint
        return int(load_checkpoint(ckpt_path)["step"])
    except Exception:
        return 0


def stop_code(watcher) -> int:
    """! @brief Code de retour d'un arrêt : EXIT_CANCELLED si l'annulation est demandée dans la base, sinon EXIT_STOPPED.

    L'annulation l'emporte sur les autres causes (fin de stdin, signal) quand elles surviennent ensemble.
    @param watcher  StopWatcher dont l'arrêt a été constaté.
    """
    if watcher.reason == "cancel" or watcher.cancel_in_db():
        return EXIT_CANCELLED
    return EXIT_STOPPED


def run(args) -> int:
    """! @brief Exécute l'entraînement demandé par les arguments.

    @param args  Arguments analysés par build_parser().
    @return Code de retour (EXIT_DONE, EXIT_CONFIG, EXIT_CANCELLED ou EXIT_STOPPED ; les autres erreurs
            sont traitées par main()).
    @throws ConfigError si la configuration, le job ou les données sont invalides.
    """
    store = job = None
    out = Path(args.out) if args.out else None
    if args.job_id is not None:
        store = JobStore(args.db)
        try:
            job = store.get(args.job_id)
        except KeyError:
            raise ConfigError(f"job inconnu : {args.job_id}")
        if job.status != "running":
            raise ConfigError(f"le job {job.id} n'est pas en cours (statut {job.status})")
        cfg = TrainConfig.from_dict(job.config)
    else:
        cfg = TrainConfig.from_yaml(args.config)
        if not cfg.name:
            cfg = dataclasses.replace(cfg, name=Path(args.config).stem)

    from .training.loop import checkpoint_path, git_commit, select_device, train

    job_id = job.id if job else None
    if job:
        job_dir = store.job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        cfg.to_yaml(job_dir / "config.yaml")
    ckpt_path = checkpoint_path(cfg, job_id, out)
    if job:
        store.set_ckpt_path(job_id, str(ckpt_path))
    _say(_describe(cfg, job_id))
    device = select_device(args.device)
    _say(_device_line(device))

    watcher = StopWatcher(store, job_id, args.watch_stdin, attempts=job.attempts if job else None)
    watcher.install()
    try:
        if watcher():
            return stop_code(watcher)
        if job:
            store.reset_progress(job_id, _start_step(ckpt_path))

        def on_metrics(step, data):
            """! @brief Enregistre les métriques dans la base (une erreur SQLite est signalée puis ignorée)."""
            if store is None:
                return
            try:
                store.log_metrics(job_id, step, data["kind"], data)
            except sqlite3.Error as e:
                _say(f"ATTENTION : métriques du pas {step} non enregistrées dans la base ({e})")

        run_info = {"job_id": job_id, "name": cfg.name, "author": cfg.author, "commit": git_commit()}
        result = train(cfg, ckpt_path, run_info=run_info, device=device, should_stop=watcher,
                       on_metrics=on_metrics)
    finally:
        watcher.restore()
    if result.status == "done":
        return EXIT_DONE
    code = stop_code(watcher)
    cause = "annulation" if code == EXIT_CANCELLED else "arrêt de l'application"
    _say(f"Arrêt demandé ({cause}) : checkpoint sauvegardé au pas {result.step}.")
    return code


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
