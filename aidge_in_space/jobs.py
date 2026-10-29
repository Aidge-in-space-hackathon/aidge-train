"""! @brief File d'attente SQLite des entraînements (JobStore) : jobs, métriques et battements de cœur des workers.

Bibliothèque standard uniquement. Trois acteurs se partagent la base : l'interface (soumission, annulation,
relance), le worker (prise des jobs, statut final) et le processus d'entraînement (progression, métriques).

Choix du journal `DELETE` (et non WAL) : la base peut vivre sur un dossier monté depuis Windows dans un
conteneur Docker, où la mémoire partagée exigée par le mode WAL et les verrous de fichiers ne sont pas
fiables. Pour compenser, chaque opération ouvre sa propre connexion (`timeout` et `busy_timeout` de 30 s),
les écritures sont de courtes transactions `BEGIN IMMEDIATE`, et toutes les requêtes sont paramétrées.
"""
from __future__ import annotations

import json
import os
import re
import socket
import sqlite3
import time
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from . import config

## Codes de retour de python -m aidge_in_space.train.
EXIT_DONE, EXIT_FAILED, EXIT_CONFIG, EXIT_CANCELLED, EXIT_STOPPED = 0, 1, 2, 3, 4
## Statuts internes (ASCII).
STATUSES = ("queued", "running", "done", "failed", "cancelled", "interrupted")
## Libellés affichés.
STATUS_LABELS = {"queued": "En attente", "running": "En cours", "done": "Terminé",
                 "failed": "En échec", "cancelled": "Annulé", "interrupted": "Interrompu"}
## Statuts finaux qu'un job peut recevoir de finish().
FINAL_STATUSES = ("done", "failed", "cancelled", "interrupted")
## Statuts depuis lesquels requeue() est permis.
REQUEUE_FROM = ("failed", "cancelled", "interrupted")
## Nom du fichier de base dans RUNS_DIR.
DB_NAME = "runs.sqlite"
## Version du schéma (PRAGMA user_version).
SCHEMA_VERSION = 1
## Délai d'attente (secondes et millisecondes) sur une base verrouillée par un autre processus.
BUSY_TIMEOUT_S = 30
## Longueur maximale (caractères) du message d'erreur d'un job.
MAX_ERROR_CHARS = 4000
## Types de métriques acceptés.
METRIC_KINDS = ("train", "eval")

## Schéma SQL, créé dans une transaction BEGIN IMMEDIATE si user_version vaut 0.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    status           TEXT    NOT NULL DEFAULT 'queued'
                     CHECK (status IN ('queued','running','done','failed','cancelled','interrupted')),
    name             TEXT    NOT NULL,
    author           TEXT    NOT NULL DEFAULT '',
    config_json      TEXT    NOT NULL,
    in_ch            INTEGER NOT NULL,
    n                INTEGER NOT NULL,
    lmbda            REAL    NOT NULL,
    steps            INTEGER NOT NULL,
    last_step        INTEGER NOT NULL DEFAULT 0,
    cancel_requested INTEGER NOT NULL DEFAULT 0 CHECK (cancel_requested IN (0, 1)),
    attempts         INTEGER NOT NULL DEFAULT 0,
    worker           TEXT,
    ckpt_path        TEXT,
    error            TEXT,
    created_at       REAL    NOT NULL,
    queued_at        REAL    NOT NULL,
    started_at       REAL,
    finished_at      REAL,
    heartbeat_at     REAL
);
CREATE INDEX IF NOT EXISTS jobs_status_queue ON jobs (status, queued_at, id);
CREATE TABLE IF NOT EXISTS metrics (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id     INTEGER NOT NULL REFERENCES jobs (id),
    step       INTEGER NOT NULL,
    kind       TEXT    NOT NULL CHECK (kind IN ('train', 'eval')),
    data_json  TEXT    NOT NULL,
    created_at REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS metrics_job ON metrics (job_id, kind, step);
CREATE TABLE IF NOT EXISTS workers (
    name       TEXT PRIMARY KEY,
    pid        INTEGER NOT NULL,
    host       TEXT    NOT NULL,
    job_id     INTEGER,
    started_at REAL    NOT NULL,
    seen_at    REAL    NOT NULL
);
"""


@dataclass
class Job:
    """! @brief Un job d'entraînement tel que stocké dans la base."""
    ## Identifiant (jamais réutilisé ; le checkpoint du job le porte).
    id: int
    ## Statut interne, parmi STATUSES.
    status: str
    ## Nom du run.
    name: str
    ## Auteur (peut être vide).
    author: str
    ## TrainConfig.to_dict(), relu depuis config_json.
    config: dict
    ## Nombre de canaux (1 = gris, 3 = RGB).
    in_ch: int
    ## Largeur N du modèle.
    N: int
    ## Compromis débit-distorsion λ.
    lmbda: float
    ## Nombre de pas visé.
    steps: int
    ## Dernier pas consigné (écrit par log_metrics et reset_progress).
    last_step: int
    ## Annulation demandée par l'interface.
    cancel_requested: bool
    ## Nombre de prises par un worker.
    attempts: int
    ## Nom du worker qui exécute ou a exécuté le job.
    worker: str | None
    ## Chemin du checkpoint.
    ckpt_path: str | None
    ## Message d'erreur (échec ou interruption).
    error: str | None
    ## Création, en secondes Unix (time.time()).
    created_at: float
    ## Dernière mise en file, en secondes Unix.
    queued_at: float
    ## Début de la dernière tentative, en secondes Unix.
    started_at: float | None
    ## Fin de la dernière tentative, en secondes Unix.
    finished_at: float | None
    ## Dernier battement de cœur du worker pour ce job, en secondes Unix.
    heartbeat_at: float | None


def _row_to_job(row: sqlite3.Row) -> Job:
    """! @brief Convertit une ligne de la table jobs en Job.
    @param row Ligne SQLite (accès par nom de colonne).
    @return Le Job correspondant.
    """
    return Job(
        id=row["id"], status=row["status"], name=row["name"], author=row["author"],
        config=json.loads(row["config_json"]), in_ch=row["in_ch"], N=row["n"],
        lmbda=row["lmbda"], steps=row["steps"], last_step=row["last_step"],
        cancel_requested=bool(row["cancel_requested"]), attempts=row["attempts"],
        worker=row["worker"], ckpt_path=row["ckpt_path"], error=row["error"],
        created_at=row["created_at"], queued_at=row["queued_at"], started_at=row["started_at"],
        finished_at=row["finished_at"], heartbeat_at=row["heartbeat_at"])


def _is_number(v) -> bool:
    """! @brief Indique si une valeur est un nombre réel (bool exclu).
    @param v Valeur à tester.
    @return Vrai pour un int ou un float qui n'est pas un bool.
    """
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _max_run_id(ckpt_dir: Path) -> int:
    """! @brief Plus grand N des checkpoints bls2017_*_run{N}.pt d'un dossier.
    @param ckpt_dir Dossier des checkpoints (peut ne pas exister).
    @return Le plus grand N trouvé, 0 s'il n'y en a pas.
    """
    best = 0
    try:
        names = [p.name for p in Path(ckpt_dir).glob("bls2017_*_run*.pt")]
    except OSError:
        return 0
    for name in names:
        m = re.fullmatch(r"bls2017_.*_run(\d+)\.pt", name)
        if m:
            best = max(best, int(m.group(1)))
    return best


class JobStore:
    """! @brief Accès à la base des jobs d'entraînement (une connexion par opération)."""

    def __init__(self, db_path: str | Path | None = None):
        """! @brief Ouvre (et crée si besoin) la base.
        @param db_path Chemin de la base ; None : config.RUNS_DIR / DB_NAME (lu à l'appel).
        @throws RuntimeError Si la base a un schéma plus récent que SCHEMA_VERSION.
        """
        ## Chemin de la base SQLite.
        self.path = Path(db_path) if db_path is not None else Path(config.RUNS_DIR) / DB_NAME
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ---- connexions ----------------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """! @brief Ouvre une connexion neuve (mode autocommit, accès par nom de colonne).
        @return La connexion, à fermer par l'appelant.
        """
        con = sqlite3.connect(str(self.path), timeout=BUSY_TIMEOUT_S, isolation_level=None)
        con.row_factory = sqlite3.Row
        con.execute(f"PRAGMA busy_timeout = {int(BUSY_TIMEOUT_S * 1000)}")
        return con

    @contextmanager
    def _read(self):
        """! @brief Contexte de lecture : connexion fermée à la sortie.
        @return Un générateur de connexion.
        """
        con = self._connect()
        try:
            yield con
        finally:
            con.close()

    @contextmanager
    def _write(self):
        """! @brief Contexte d'écriture : transaction BEGIN IMMEDIATE validée (ou annulée sur exception).
        @return Un générateur de connexion.
        """
        con = self._connect()
        try:
            con.execute("BEGIN IMMEDIATE")
            try:
                yield con
            except BaseException:
                if con.in_transaction:
                    con.execute("ROLLBACK")
                raise
            else:
                con.execute("COMMIT")
        finally:
            con.close()

    def _init_schema(self) -> None:
        """! @brief Crée le schéma si la base est neuve et amorce l'AUTOINCREMENT sur les checkpoints existants.
        @throws RuntimeError Si user_version dépasse SCHEMA_VERSION.
        """
        con = self._connect()
        try:
            version = con.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                raise RuntimeError(f"base {self.path} de version {version} : "
                                   f"cette version du code gère jusqu'à {SCHEMA_VERSION}")
            if version == 0:
                con.execute("PRAGMA journal_mode = DELETE")
                con.execute("BEGIN IMMEDIATE")
                try:
                    # Un autre processus a pu créer le schéma entre-temps.
                    if con.execute("PRAGMA user_version").fetchone()[0] == 0:
                        for stmt in _SCHEMA.split(";"):
                            if stmt.strip():
                                con.execute(stmt)
                        seed = _max_run_id(config.CKPT_DIR)
                        if seed > 0:
                            con.execute("INSERT INTO sqlite_sequence (name, seq) VALUES ('jobs', ?)",
                                        (seed,))
                        con.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                    con.execute("COMMIT")
                except BaseException:
                    if con.in_transaction:
                        con.execute("ROLLBACK")
                    raise
        finally:
            con.close()

    # ---- jobs ----------------------------------------------------------------------------------

    def job_dir(self, job_id: int) -> Path:
        """! @brief Dossier d'un job (config.yaml, train.log) ; non créé.
        @param job_id Identifiant du job.
        @return Chemin RUNS_DIR/<id>.
        """
        return self.path.parent / str(job_id)

    def submit(self, config: dict) -> int:
        """! @brief Insère un job en file d'attente.
        @param config TrainConfig.to_dict() ; clés name, author, in_ch, N, lmbda, steps obligatoires.
        @return L'identifiant du job.
        @throws ValueError Clé absente ou mal typée, nom vide, steps < 1, config non sérialisable en JSON.
        """
        if not isinstance(config, dict):
            raise ValueError("la configuration doit être un dictionnaire")
        for key in ("name", "author", "in_ch", "N", "lmbda", "steps"):
            if key not in config:
                raise ValueError(f"clé de configuration absente : {key!r}")
        name, author = config["name"], config["author"]
        if not isinstance(name, str) or not isinstance(author, str):
            raise ValueError("name et author doivent être des chaînes")
        name, author = name.strip(), author.strip()
        if not name:
            raise ValueError("Donnez un nom au run.")
        for key in ("in_ch", "N", "steps"):
            v = config[key]
            if isinstance(v, bool) or not isinstance(v, int):
                raise ValueError(f"{key} : entier attendu (reçu {v!r})")
        if not _is_number(config["lmbda"]):
            raise ValueError(f"lmbda : nombre attendu (reçu {config['lmbda']!r})")
        if config["steps"] < 1:
            raise ValueError(f"steps : au moins 1 pas (reçu {config['steps']})")
        try:
            blob = json.dumps(config, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            raise ValueError(f"configuration non sérialisable en JSON : {e}") from e
        now = time.time()
        with self._write() as con:
            cur = con.execute(
                "INSERT INTO jobs (status, name, author, config_json, in_ch, n, lmbda, steps, "
                "created_at, queued_at) VALUES ('queued', ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (name, author, blob, config["in_ch"], config["N"], float(config["lmbda"]),
                 config["steps"], now, now))
            return int(cur.lastrowid)

    def get(self, job_id: int) -> Job:
        """! @brief Lit un job.
        @param job_id Identifiant du job.
        @return Le Job.
        @throws KeyError Si le job est inconnu.
        """
        with self._read() as con:
            row = con.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return _row_to_job(row)

    def list_jobs(self, limit: int | None = 200, statuses: Iterable[str] | None = None) -> list[Job]:
        """! @brief Liste les jobs du plus récent au plus ancien.
        @param limit Nombre maximal de jobs ; None pour ne pas limiter.
        @param statuses Statuts à garder (parmi STATUSES) ; None pour tous.
        @return Les jobs, id décroissant.
        @throws ValueError Si un statut est inconnu.
        """
        sql = "SELECT * FROM jobs"
        params: list = []
        if statuses is not None:
            wanted = list(dict.fromkeys(statuses))
            bad = [x for x in wanted if x not in STATUSES]
            if bad:
                raise ValueError(f"statut inconnu : {bad[0]!r}")
            if not wanted:
                return []
            sql += " WHERE status IN (" + ",".join("?" * len(wanted)) + ")"
            params += wanted
        sql += " ORDER BY id DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        with self._read() as con:
            rows = con.execute(sql, params).fetchall()
        return [_row_to_job(r) for r in rows]

    def claim_next(self, worker: str) -> Job | None:
        """! @brief Prend atomiquement le plus ancien job en attente et le passe en cours.
        @param worker Nom du worker.
        @return Le job pris, ou None si la file est vide.
        """
        now = time.time()
        with self._write() as con:
            row = con.execute("SELECT id FROM jobs WHERE status = 'queued' "
                              "ORDER BY queued_at, id LIMIT 1").fetchone()
            if row is None:
                return None
            con.execute(
                "UPDATE jobs SET status = 'running', started_at = ?, heartbeat_at = ?, worker = ?, "
                "attempts = attempts + 1, cancel_requested = 0, error = NULL WHERE id = ?",
                (now, now, worker, row["id"]))
            full = con.execute("SELECT * FROM jobs WHERE id = ?", (row["id"],)).fetchone()
        return _row_to_job(full)

    def heartbeat(self, job_id: int) -> None:
        """! @brief Met à jour le battement de cœur d'un job en cours (sans effet sinon).
        @param job_id Identifiant du job.
        @throws KeyError Si le job est inconnu.
        """
        with self._write() as con:
            self._require(con, job_id)
            con.execute("UPDATE jobs SET heartbeat_at = ? WHERE id = ? AND status = 'running'",
                        (time.time(), job_id))

    def set_ckpt_path(self, job_id: int, path: str) -> None:
        """! @brief Enregistre le chemin du checkpoint d'un job.
        @param job_id Identifiant du job.
        @param path Chemin du checkpoint.
        @throws KeyError Si le job est inconnu.
        """
        with self._write() as con:
            self._require(con, job_id)
            con.execute("UPDATE jobs SET ckpt_path = ? WHERE id = ?", (str(path), job_id))

    def reset_progress(self, job_id: int, step: int) -> None:
        """! @brief Efface les métriques de pas supérieur à step et remet last_step.
        @param job_id Identifiant du job.
        @param step Pas de départ (0 ou pas du checkpoint repris).
        @throws KeyError Si le job est inconnu.
        """
        with self._write() as con:
            self._require(con, job_id)
            con.execute("DELETE FROM metrics WHERE job_id = ? AND step > ?", (job_id, int(step)))
            con.execute("UPDATE jobs SET last_step = ? WHERE id = ?", (int(step), job_id))

    def log_metrics(self, job_id: int, step: int, kind: str, data: dict) -> None:
        """! @brief Consigne une ligne de métriques et avance last_step.
        @param job_id Identifiant du job.
        @param step Pas d'entraînement.
        @param kind "train" ou "eval".
        @param data Valeurs (JSON-sérialisables).
        @throws ValueError Si kind est invalide.
        @throws KeyError Si le job est inconnu.
        """
        if kind not in METRIC_KINDS:
            raise ValueError(f"type de métrique invalide : {kind!r}")
        blob = json.dumps(data, ensure_ascii=False)
        with self._write() as con:
            self._require(con, job_id)
            con.execute("INSERT INTO metrics (job_id, step, kind, data_json, created_at) "
                        "VALUES (?, ?, ?, ?, ?)", (job_id, int(step), kind, blob, time.time()))
            con.execute("UPDATE jobs SET last_step = MAX(last_step, ?) WHERE id = ?",
                        (int(step), job_id))

    def metrics(self, job_id: int, kind: str | None = None) -> list[dict]:
        """! @brief Métriques d'un job triées par pas (la ligne la plus récente par (kind, pas)).
        @param job_id Identifiant du job.
        @param kind Filtre "train" ou "eval" ; None : les deux.
        @return Dictionnaires {"step", "kind", "created_at", **data}.
        @throws KeyError Si le job est inconnu.
        """
        with self._read() as con:
            self._require(con, job_id)
            if kind is None:
                rows = con.execute("SELECT * FROM metrics WHERE job_id = ? ORDER BY id",
                                   (job_id,)).fetchall()
            else:
                rows = con.execute("SELECT * FROM metrics WHERE job_id = ? AND kind = ? ORDER BY id",
                                   (job_id, kind)).fetchall()
        latest: dict[tuple[str, int], dict] = {}
        for r in rows:
            d = json.loads(r["data_json"])
            d.update({"step": r["step"], "kind": r["kind"], "created_at": r["created_at"]})
            latest[(r["kind"], r["step"])] = d
        return sorted(latest.values(), key=lambda d: (d["step"], d["kind"]))

    def cancel(self, job_id: int) -> str:
        """! @brief Annule un job en attente ou demande l'annulation d'un job en cours.
        @param job_id Identifiant du job.
        @return "cancelled" (job en attente) ou "cancel_requested" (job en cours).
        @throws ValueError Si le job est déjà dans un statut final.
        @throws KeyError Si le job est inconnu.
        """
        with self._write() as con:
            status = self._require(con, job_id)["status"]
            if status == "queued":
                con.execute("UPDATE jobs SET status = 'cancelled', finished_at = ? WHERE id = ?",
                            (time.time(), job_id))
                return "cancelled"
            if status == "running":
                con.execute("UPDATE jobs SET cancel_requested = 1 WHERE id = ?", (job_id,))
                return "cancel_requested"
        raise ValueError(f"le job {job_id} est déjà terminé ({STATUS_LABELS[status]}) : rien à annuler")

    def cancel_requested(self, job_id: int) -> bool:
        """! @brief Indique si l'annulation d'un job a été demandée.
        @param job_id Identifiant du job.
        @return Vrai si cancel_requested vaut 1.
        @throws KeyError Si le job est inconnu.
        """
        with self._read() as con:
            return bool(self._require(con, job_id)["cancel_requested"])

    def finish(self, job_id: int, status: str, error: str | None = None) -> None:
        """! @brief Fixe le statut final d'un job en cours.
        @param job_id Identifiant du job.
        @param status Statut final, parmi FINAL_STATUSES.
        @param error Message d'erreur éventuel (tronqué à 4 000 caractères).
        @throws ValueError Si status n'est pas final ou si le job n'est pas en cours.
        @throws KeyError Si le job est inconnu.
        """
        if status not in FINAL_STATUSES:
            raise ValueError(f"statut final invalide : {status!r}")
        with self._write() as con:
            current = self._require(con, job_id)["status"]
            if current != "running":
                raise ValueError(f"le job {job_id} n'est pas en cours (statut {STATUS_LABELS[current]})")
            con.execute(
                "UPDATE jobs SET status = ?, finished_at = ?, cancel_requested = 0, error = ? WHERE id = ?",
                (status, time.time(), None if error is None else error[:MAX_ERROR_CHARS], job_id))

    def requeue(self, job_id: int) -> None:
        """! @brief Remet en file un job en échec, annulé ou interrompu (même id : reprise du checkpoint).
        @param job_id Identifiant du job.
        @throws ValueError Si le job est en attente, en cours ou terminé.
        @throws KeyError Si le job est inconnu.
        """
        with self._write() as con:
            status = self._require(con, job_id)["status"]
            if status not in REQUEUE_FROM:
                raise ValueError(f"le job {job_id} ne peut pas être relancé (statut {STATUS_LABELS[status]})")
            con.execute(
                "UPDATE jobs SET status = 'queued', queued_at = ?, error = NULL, started_at = NULL, "
                "finished_at = NULL, heartbeat_at = NULL, worker = NULL, cancel_requested = 0 WHERE id = ?",
                (time.time(), job_id))

    def recover_orphans(self, stale_after_s: float = 60.0) -> list[int]:
        """! @brief Passe en interrompus les jobs en cours dont le battement de cœur est périmé.
        @param stale_after_s Âge maximal du battement de cœur, en secondes.
        @return Les identifiants des jobs récupérés.
        """
        now = time.time()
        with self._write() as con:
            ids = [r["id"] for r in con.execute(
                "SELECT id FROM jobs WHERE status = 'running' AND (heartbeat_at IS NULL OR heartbeat_at < ?)",
                (now - float(stale_after_s),)).fetchall()]
            for i in ids:
                con.execute(
                    "UPDATE jobs SET status = 'interrupted', finished_at = ?, cancel_requested = 0, "
                    "error = ? WHERE id = ?",
                    (now, "worker disparu (battement de cœur périmé)", i))
        return ids

    # ---- workers -------------------------------------------------------------------------------

    def worker_heartbeat(self, name: str, job_id: int | None = None) -> None:
        """! @brief Enregistre ou met à jour la présence d'un worker.
        @param name Nom du worker.
        @param job_id Job en cours, ou None.
        """
        now = time.time()
        with self._write() as con:
            con.execute(
                "INSERT INTO workers (name, pid, host, job_id, started_at, seen_at) VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(name) DO UPDATE SET pid = excluded.pid, host = excluded.host, "
                "job_id = excluded.job_id, seen_at = excluded.seen_at",
                (name, os.getpid(), socket.gethostname(), job_id, now, now))

    def worker_alive(self, max_age_s: float = 30.0) -> bool:
        """! @brief Indique si un worker a donné signe de vie récemment.
        @param max_age_s Âge maximal du dernier signe de vie, en secondes.
        @return Vrai si au moins un worker est vivant.
        """
        with self._read() as con:
            row = con.execute("SELECT 1 FROM workers WHERE seen_at >= ? LIMIT 1",
                              (time.time() - float(max_age_s),)).fetchone()
        return row is not None

    # ---- interne -------------------------------------------------------------------------------

    @staticmethod
    def _require(con: sqlite3.Connection, job_id: int) -> sqlite3.Row:
        """! @brief Lit le statut et le drapeau d'annulation d'un job, ou lève KeyError.
        @param con Connexion ouverte.
        @param job_id Identifiant du job.
        @return Ligne (id, status, cancel_requested).
        @throws KeyError Si le job est inconnu.
        """
        row = con.execute("SELECT id, status, cancel_requested FROM jobs WHERE id = ?",
                          (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return row
