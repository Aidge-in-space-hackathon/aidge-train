"""! @brief Worker de la file d'entraînement : exécute les jobs un par un, chacun dans un sous-processus.

Lancement : `python -m aidge_in_space.worker [--poll 5] [--once] [--db CHEMIN]`.

Le worker ne fait jamais d'entraînement lui-même : il lance `python -m aidge_in_space.train --job-id N
--db CHEMIN --watch-stdin`, redirige sa sortie vers `RUNS_DIR/N/train.log`, entretient les battements de
cœur du job et du worker, puis fixe le statut final à partir du code de retour (voir jobs.EXIT_*).
Fermer le stdin de l'enfant lui demande de sauvegarder et de s'arrêter (code 4) ; le système le ferme
aussi si le worker disparaît, ce qui évite les processus orphelins.
"""
from __future__ import annotations

import argparse
import os
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from . import config
from .jobs import (EXIT_CANCELLED, EXIT_DONE, EXIT_STOPPED, STATUS_LABELS, Job, JobStore)

## Commande d'entraînement ; le worker y ajoute --job-id N --db CHEMIN --watch-stdin. Remplaçable dans les tests.
TRAIN_CMD = [sys.executable, "-m", "aidge_in_space.train"]
## Période de scrutation et des battements de cœur, en secondes.
POLL_S = 5.0
## Âge au-delà duquel un job running est orphelin.
STALE_AFTER_S = 60.0
## Délai laissé à l'enfant pour sauvegarder après la fermeture de son stdin, avant l'arrêt forcé de son arbre de processus.
STOP_TIMEOUT_S = 60.0
## Délai après lequel le worker ferme lui-même le stdin d'un job dont l'annulation est demandée.
CANCEL_GRACE_S = 60.0
## Délai maximal d'attente de la fin de l'enfant après son arrêt forcé, en secondes.
KILL_WAIT_S = 10.0
## Lignes de train.log recopiées dans l'erreur d'un job en échec.
ERROR_TAIL_LINES = 20
## Octets lus en fin de train.log pour en extraire les dernières lignes.
TAIL_BYTES = 64 * 1024
## Pas maximal (secondes) entre deux vérifications de la fin de l'enfant.
SLICE_S = 0.25
## Nombre de tentatives d'écriture du statut final quand la base est inaccessible.
FINISH_RETRIES = 5
## Attente (secondes) entre deux tentatives d'écriture du statut final.
FINISH_RETRY_WAIT_S = 2.0


def tail(path: Path, n: int = ERROR_TAIL_LINES) -> str:
    """! @brief Dernières lignes d'un fichier texte.
    @param path Fichier (lecture des 64 derniers Kio, UTF-8 avec remplacement).
    @param n Nombre de lignes.
    @return Les n dernières lignes, ou "" si le fichier est absent.
    """
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - TAIL_BYTES))
            data = f.read()
    except OSError:
        return ""
    text = data.decode("utf-8", errors="replace")
    return "\n".join(text.splitlines()[-n:])


def final_status(rc: int | None, cancel_requested: bool, stopping: bool, killed: bool) -> str:
    """! @brief Statut final d'un job à partir de la fin de son sous-processus.
    @param rc Code de retour de l'enfant (None s'il est inconnu).
    @param cancel_requested Annulation demandée dans la base.
    @param stopping Arrêt du worker demandé.
    @param killed Enfant tué après dépassement du délai.
    @return "done", "cancelled", "interrupted" ou "failed".
    """
    if rc == EXIT_DONE:
        return "done"
    if rc == EXIT_CANCELLED:
        return "cancelled"
    if cancel_requested and (rc == EXIT_STOPPED or killed):
        return "cancelled"
    if rc == EXIT_STOPPED or (stopping and killed):
        return "interrupted"
    return "failed"


def _close_stdin(proc: subprocess.Popen) -> None:
    """! @brief Ferme le stdin de l'enfant (demande d'arrêt propre), sans lever d'erreur.
    @param proc Sous-processus d'entraînement.
    """
    try:
        if proc.stdin is not None:
            proc.stdin.close()
    except OSError:
        pass


def _kill_tree(proc: subprocess.Popen, wait_s: float = KILL_WAIT_S) -> None:
    """! @brief Tue l'enfant et tous ses descendants, sans lever d'erreur.
    Sous Windows, le python du venv est un lanceur dont le vrai interpréteur est un second processus :
    proc.kill() ne tuerait que le lanceur et laisserait l'entraînement orphelin, d'où taskkill /F /T.
    Sous POSIX, l'enfant est chef de sa propre session (start_new_session) : SIGKILL à son groupe.
    Repli sur proc.kill() si l'arrêt de l'arbre échoue.
    @param proc Sous-processus d'entraînement.
    @param wait_s Délai maximal d'attente de la fin de proc, en secondes.
    """
    try:
        if os.name == "nt":
            res = subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True,
                                 creationflags=subprocess.CREATE_NO_WINDOW, timeout=wait_s)
            if res.returncode != 0 and proc.poll() is None:
                proc.kill()
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        try:
            proc.kill()
        except OSError:
            pass
    try:
        proc.wait(timeout=wait_s)
    except subprocess.TimeoutExpired:
        pass


def run_job(store: JobStore, job: Job, cmd: list[str], poll: float, stop_event: threading.Event,
            name: str | None = None) -> str:
    """! @brief Exécute un job dans un sous-processus et fixe son statut final.
    @param store File d'attente.
    @param job Job pris (statut running).
    @param cmd Commande de base ; --job-id, --db et --watch-stdin sont ajoutés.
    @param poll Période des battements de cœur, en secondes.
    @param stop_event Arrêt du worker demandé quand il est positionné.
    @param name Nom du worker (battements de cœur et en-tête du journal).
    @return Le statut final calculé.
    """
    name = name or f"{socket.gethostname()}-{os.getpid()}"
    job_dir = store.job_dir(job.id)
    job_dir.mkdir(parents=True, exist_ok=True)
    log_path = job_dir / "train.log"
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    kwargs: dict = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    else:
        kwargs["start_new_session"] = True
    stopping = killed = False
    with open(log_path, "ab") as log:
        header = (f"=== tentative {job.attempts} · {datetime.now().isoformat(timespec='seconds')} "
                  f"· worker {name} ===\n")
        log.write(header.encode("utf-8"))
        log.flush()
        argv = list(cmd) + ["--job-id", str(job.id), "--db", str(store.path), "--watch-stdin"]
        try:
            proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                                    cwd=str(config.ROOT), env=env, **kwargs)
        except OSError as e:
            msg = f"impossible de lancer l'entraînement : {e}"
            log.write((msg + "\n").encode("utf-8"))
            _finish(store, job.id, "failed", msg)
            return "failed"
        stdin_closed = False
        deadline: float | None = None
        cancel_since: float | None = None
        next_beat = 0.0
        while True:
            try:
                proc.wait(timeout=min(max(poll, 0.01), SLICE_S))
                break
            except subprocess.TimeoutExpired:
                pass
            now = time.monotonic()
            if stop_event.is_set() and not stdin_closed:
                stopping = stdin_closed = True
                _close_stdin(proc)
                deadline = now + STOP_TIMEOUT_S
            if now >= next_beat:
                next_beat = now + poll
                try:
                    store.heartbeat(job.id)
                    store.worker_heartbeat(name, job.id)
                    if store.cancel_requested(job.id):
                        if cancel_since is None:
                            cancel_since = now
                        elif now - cancel_since > CANCEL_GRACE_S and not stdin_closed:
                            stdin_closed = True
                            _close_stdin(proc)
                            deadline = now + STOP_TIMEOUT_S
                except Exception as e:  # base momentanément inaccessible : l'entraînement continue
                    print(f"ATTENTION : battement de cœur du job {job.id} impossible : {e}", flush=True)
            if deadline is not None and now > deadline and not killed:
                killed = True
                _kill_tree(proc)
        _close_stdin(proc)
        rc = proc.returncode
    try:
        cancel = store.cancel_requested(job.id)
    except Exception:
        cancel = False
    status = final_status(rc, cancel, stopping or stop_event.is_set(), killed)
    error = None
    if status == "failed":
        error = tail(log_path) or f"l'entraînement s'est terminé avec le code {rc}"
    elif status == "interrupted" and killed:
        error = f"arrêt forcé après {STOP_TIMEOUT_S:g} s"
    _finish(store, job.id, status, error)
    return status


def _finish(store: JobStore, job_id: int, status: str, error: str | None) -> None:
    """! @brief Écrit le statut final ; réessaie si la base est verrouillée, avertit sans jamais lever.
    @param store File d'attente.
    @param job_id Identifiant du job.
    @param status Statut final.
    @param error Message d'erreur éventuel.
    """
    for attempt in range(1, FINISH_RETRIES + 1):
        try:
            store.finish(job_id, status, error)
            return
        except ValueError:
            try:
                current = STATUS_LABELS[store.get(job_id).status]
            except Exception:
                current = "inconnu"
            print(f"ATTENTION : le job {job_id} n'est plus en cours (statut {current}) : "
                  f"statut final {status} non écrit", flush=True)
            return
        except sqlite3.Error as e:
            if attempt < FINISH_RETRIES:
                time.sleep(FINISH_RETRY_WAIT_S)
            else:
                print(f"ATTENTION : statut final {status} du job {job_id} non écrit après {attempt} "
                      f"tentatives ({e}) ; le job reste en cours et sera repris comme interrompu",
                      flush=True)


def run_worker(store: JobStore, poll: float = POLL_S, once: bool = False, cmd: list[str] | None = None,
               stop_event: threading.Event | None = None, name: str | None = None) -> int:
    """! @brief Boucle du worker : prend et exécute les jobs un par un.
    @param store File d'attente.
    @param poll Période de scrutation et des battements de cœur, en secondes.
    @param once Traiter au plus un job puis quitter.
    @param cmd Commande d'entraînement (défaut TRAIN_CMD).
    @param stop_event Événement d'arrêt ; créé et relié aux signaux si None.
    @param name Nom du worker (défaut hôte-pid).
    @return Le nombre de jobs traités.
    """
    cmd = list(cmd) if cmd is not None else list(TRAIN_CMD)
    name = name or f"{socket.gethostname()}-{os.getpid()}"
    previous: dict = {}
    if stop_event is None:
        stop_event = threading.Event()
        names = ["SIGINT", "SIGTERM"] + (["SIGBREAK"] if os.name == "nt" else [])
        for sig_name in names:
            sig = getattr(signal, sig_name, None)
            if sig is None:
                continue
            try:
                previous[sig] = signal.signal(sig, lambda *_: stop_event.set())
            except ValueError:  # pas dans le fil principal
                pass
    done = 0
    try:
        _safe(lambda: store.recover_orphans(STALE_AFTER_S))
        while not stop_event.is_set():
            job = None
            try:
                store.worker_heartbeat(name)
                job = store.claim_next(name)
            except Exception as e:
                print(f"ATTENTION : base inaccessible : {e}", flush=True)
            if job is None:
                if once:
                    break
                _safe(lambda: store.recover_orphans(STALE_AFTER_S))
                stop_event.wait(poll)
                continue
            print(f"Job {job.id} « {job.name} » pris par {name}", flush=True)
            try:
                status = run_job(store, job, cmd, poll, stop_event, name)
                print(f"Job {job.id} : {STATUS_LABELS[status]}", flush=True)
            except Exception as e:
                print(f"ATTENTION : erreur sur le job {job.id} : {e}", flush=True)
                _safe(lambda: _finish(store, job.id, "failed", f"erreur du worker : {e}"))
            done += 1
            _safe(lambda: store.worker_heartbeat(name))
            if once:
                break
    finally:
        for sig, handler in previous.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, TypeError):
                pass
    return done


def _safe(fn) -> None:
    """! @brief Exécute fn en affichant (sans propager) toute exception.
    @param fn Fonction sans argument.
    """
    try:
        fn()
    except Exception as e:
        print(f"ATTENTION : {e}", flush=True)


def main(argv: list[str] | None = None) -> int:
    """! @brief Point d'entrée de la ligne de commande.
    @param argv Arguments (défaut sys.argv).
    @return 0.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description="Worker de la file d'entraînement.")
    ap.add_argument("--poll", type=float, default=POLL_S, help="période de scrutation, en secondes")
    ap.add_argument("--once", action="store_true", help="traiter au plus un job puis quitter")
    ap.add_argument("--db", default=None, help="base SQLite (défaut RUNS_DIR/runs.sqlite)")
    args = ap.parse_args(argv)
    store = JobStore(args.db)
    print(f"Worker démarré, base : {store.path}", flush=True)
    run_worker(store, poll=args.poll, once=args.once)
    return 0


if __name__ == "__main__":
    sys.exit(main())
