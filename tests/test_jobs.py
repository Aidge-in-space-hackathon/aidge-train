"""Tests de la file d'entraînement (jobs.JobStore) et du worker : python -m pytest tests/test_jobs.py

Base SQLite dans tmp_path ; le worker lance une commande factice (un petit script Python) à la place
de l'entraînement. Contrat : docs/conception_entrainement.md (§ 2.5 et 2.6).
"""
import sys
import textwrap
import threading
import time

import pytest

from aidge_in_space import config, jobs, worker
from aidge_in_space.jobs import JobStore
from aidge_in_space.training import TrainConfig


@pytest.fixture(autouse=True)
def ckpt_vide(tmp_path, monkeypatch):
    """Dossier de checkpoints vide : les identifiants de jobs ne dépendent pas du vrai outputs/."""
    d = tmp_path / "ckpts"
    d.mkdir()
    monkeypatch.setattr(config, "CKPT_DIR", d)
    return d


@pytest.fixture
def store(tmp_path):
    return JobStore(tmp_path / "runs" / "runs.sqlite")


def cfg(name="essai", **kw):
    return TrainConfig(name=name, author="Alice", **kw).to_dict()


# --- JobStore -------------------------------------------------------------------------------

def test_soumission(store):
    jid = store.submit(cfg(lmbda=512, steps=100))
    job = store.get(jid)
    assert job.status == "queued" and job.name == "essai" and job.author == "Alice"
    assert job.lmbda == 512 and job.steps == 100 and job.N == 128 and job.in_ch == 1
    assert job.config["lmbda"] == 512 and job.attempts == 0 and job.last_step == 0
    assert job.worker is None and job.started_at is None and job.cancel_requested is False
    assert [j.id for j in store.list_jobs()] == [jid]
    assert store.job_dir(jid) == store.path.parent / str(jid)
    with pytest.raises(KeyError):
        store.get(999)


def test_soumission_invalide(store):
    with pytest.raises(ValueError):
        store.submit(cfg(name=""))
    with pytest.raises(ValueError):
        store.submit(cfg(name="   "))
    bad = cfg()
    del bad["steps"]
    with pytest.raises(ValueError):
        store.submit(bad)
    with pytest.raises(ValueError):
        store.submit(cfg(steps=0))
    with pytest.raises(ValueError):
        store.submit({**cfg(), "extra": object()})
    assert store.list_jobs() == []


def test_requetes_parametrees(store):
    jid = store.submit(cfg(name="x'); DROP TABLE jobs;--"))
    assert store.get(jid).name == "x'); DROP TABLE jobs;--"
    assert len(store.list_jobs()) == 1


def test_list_jobs_du_plus_recent_au_plus_ancien(store):
    ids = [store.submit(cfg(name=f"j{i}")) for i in range(3)]
    assert [j.id for j in store.list_jobs()] == ids[::-1]
    assert len(store.list_jobs(limit=2)) == 2


def test_ids_ne_reprennent_pas_un_ancien_checkpoint(tmp_path, ckpt_vide):
    (ckpt_vide / "bls2017_c1_N16_lambda2048_run12.pt").write_bytes(b"x")
    (ckpt_vide / "bls2017_c1_N16_lambda512_run7.pt").write_bytes(b"x")
    s = JobStore(tmp_path / "neuve" / "runs.sqlite")
    assert s.submit(cfg()) == 13


def test_base_d_une_version_future_refusee(tmp_path):
    import sqlite3
    path = tmp_path / "future.sqlite"
    con = sqlite3.connect(path)
    con.execute(f"PRAGMA user_version = {jobs.SCHEMA_VERSION + 1}")
    con.close()
    with pytest.raises(RuntimeError):
        JobStore(path)


def test_prise_dans_l_ordre_et_etat(store):
    a, b = store.submit(cfg("a")), store.submit(cfg("b"))
    job = store.claim_next("w1")
    assert job.id == a and job.status == "running" and job.worker == "w1" and job.attempts == 1
    assert job.started_at and job.heartbeat_at and job.cancel_requested is False
    assert store.claim_next("w2").id == b
    assert store.claim_next("w1") is None


def test_prise_atomique_par_workers_concurrents(tmp_path):
    path = tmp_path / "c.sqlite"
    JobStore(path)
    n_jobs = 30
    for i in range(n_jobs):
        JobStore(path).submit(cfg(f"j{i}"))
    taken: dict[str, list[int]] = {"w1": [], "w2": [], "w3": []}
    start = threading.Barrier(3)

    def loop(name):
        s = JobStore(path)
        start.wait()
        while (job := s.claim_next(name)) is not None:
            taken[name].append(job.id)

    threads = [threading.Thread(target=loop, args=(n,)) for n in taken]
    [t.start() for t in threads]
    [t.join(60) for t in threads]
    allids = sum(taken.values(), [])
    assert len(allids) == n_jobs and len(set(allids)) == n_jobs        # aucun job pris deux fois
    assert all(j.status == "running" and j.attempts == 1 for j in JobStore(path).list_jobs())


def test_annulation_d_un_job_en_attente(store):
    jid = store.submit(cfg())
    assert store.cancel(jid) == "cancelled"
    job = store.get(jid)
    assert job.status == "cancelled" and job.finished_at
    assert store.claim_next("w") is None
    with pytest.raises(ValueError):
        store.cancel(jid)


def test_annulation_d_un_job_en_cours(store):
    jid = store.submit(cfg())
    store.claim_next("w")
    assert store.cancel(jid) == "cancel_requested"
    assert store.cancel(jid) == "cancel_requested"            # idempotent
    assert store.cancel_requested(jid) is True
    job = store.get(jid)
    assert job.status == "running" and job.cancel_requested is True
    store.finish(jid, "cancelled")
    job = store.get(jid)
    assert job.status == "cancelled" and job.cancel_requested is False
    with pytest.raises(KeyError):
        store.cancel(999)


def test_transitions_interdites(store):
    jid = store.submit(cfg())
    for status in ("done", "failed"):
        with pytest.raises(ValueError):
            store.finish(jid, status)                           # queued -> final
    with pytest.raises(ValueError):
        store.requeue(jid)                                      # queued -> queued
    store.claim_next("w")
    with pytest.raises(ValueError):
        store.finish(jid, "queued")                             # statut non final
    with pytest.raises(ValueError):
        store.finish(jid, "running")
    with pytest.raises(ValueError):
        store.requeue(jid)                                      # running
    store.finish(jid, "done")
    with pytest.raises(ValueError):
        store.finish(jid, "failed")                             # done -> failed
    with pytest.raises(ValueError):
        store.requeue(jid)                                      # done
    with pytest.raises(ValueError):
        store.cancel(jid)
    for call in (store.finish, ):
        with pytest.raises(KeyError):
            call(999, "done")
    with pytest.raises(KeyError):
        store.requeue(999)


def test_finish_tronque_l_erreur_et_conserve_checkpoint(store):
    jid = store.submit(cfg())
    store.claim_next("w")
    store.set_ckpt_path(jid, "a/b.pt")
    store.log_metrics(jid, 10, "train", {"loss": 1.0})
    store.finish(jid, "failed", "e" * 10_000)
    job = store.get(jid)
    assert len(job.error) == 4000 and job.ckpt_path == "a/b.pt" and job.last_step == 10


@pytest.mark.parametrize("final", ["failed", "cancelled", "interrupted"])
def test_relance_avec_le_meme_id(store, final):
    jid = store.submit(cfg())
    store.claim_next("w")
    store.set_ckpt_path(jid, "x.pt")
    store.log_metrics(jid, 50, "train", {"loss": 1.0})
    store.finish(jid, final, "boom" if final == "failed" else None)
    store.requeue(jid)
    job = store.get(jid)
    assert job.id == jid and job.status == "queued"
    assert job.error is None and job.worker is None and job.started_at is None
    assert job.finished_at is None and job.heartbeat_at is None and job.cancel_requested is False
    assert job.last_step == 50 and job.ckpt_path == "x.pt" and job.attempts == 1
    again = store.claim_next("w2")
    assert again.id == jid and again.attempts == 2


def test_metriques(store):
    jid = store.submit(cfg())
    store.claim_next("w")
    store.log_metrics(jid, 20, "train", {"loss": 2.0})
    store.log_metrics(jid, 10, "train", {"loss": 3.0})
    store.log_metrics(jid, 10, "eval", {"psnr": 20.0})
    store.log_metrics(jid, 20, "train", {"loss": 1.5})             # même (kind, step) : la plus récente
    assert store.get(jid).last_step == 20
    train = store.metrics(jid, "train")
    assert [(m["step"], m["loss"]) for m in train] == [(10, 3.0), (20, 1.5)]
    assert {m["kind"] for m in store.metrics(jid)} == {"train", "eval"}
    with pytest.raises(ValueError):
        store.log_metrics(jid, 1, "autre", {})
    store.reset_progress(jid, 10)
    assert store.get(jid).last_step == 10
    assert [m["step"] for m in store.metrics(jid, "train")] == [10]
    with pytest.raises(KeyError):
        store.log_metrics(999, 1, "train", {})


def test_detection_des_orphelins(store):
    stale, alive, queued = store.submit(cfg("a")), store.submit(cfg("b")), store.submit(cfg("c"))
    store.claim_next("w1")
    store.claim_next("w2")
    time.sleep(0.3)
    store.heartbeat(alive)
    assert store.recover_orphans(60.0) == []                      # battements récents : rien
    assert store.recover_orphans(0.2) == [stale]                  # seul le battement périmé
    job = store.get(stale)
    assert job.status == "interrupted" and "battement" in job.error
    assert store.get(alive).status == "running" and store.get(queued).status == "queued"
    store.requeue(stale)
    assert store.get(stale).status == "queued"


def test_battement_sans_effet_hors_running(store):
    jid = store.submit(cfg())
    store.heartbeat(jid)
    assert store.get(jid).heartbeat_at is None
    with pytest.raises(KeyError):
        store.heartbeat(999)


def test_worker_alive(store):
    assert store.worker_alive() is False
    store.worker_heartbeat("w1")
    assert store.worker_alive() is True
    time.sleep(0.3)
    assert store.worker_alive(max_age_s=0.1) is False
    store.worker_heartbeat("w1", None)
    assert store.worker_alive(max_age_s=0.1) is True


# --- Worker ---------------------------------------------------------------------------------

FAKE_TRAIN = textwrap.dedent('''
    import os, sys, time
    mode = os.environ.get("FAKE_MODE", "0")
    print("arguments :", " ".join(sys.argv[1:]), flush=True)
    assert "--watch-stdin" in sys.argv and "--job-id" in sys.argv
    if mode == "wait":        # attend la fin de stdin, comme train --watch-stdin, puis sort avec 4
        while os.read(0, 1):
            pass
        sys.exit(4)
    if mode == "hang":        # ignore stdin : doit être tué
        time.sleep(60)
    if mode == "1":
        print("Traceback (most recent call last):", flush=True)
        print("ValueError: kaboum", file=sys.stderr, flush=True)
    sys.exit(int(mode))
''')


@pytest.fixture
def fake_cmd(tmp_path):
    script = tmp_path / "fake_train.py"
    script.write_text(FAKE_TRAIN, encoding="utf-8")
    return [sys.executable, str(script)]


@pytest.mark.parametrize("rc,status", [(0, "done"), (3, "cancelled"), (4, "interrupted"), (1, "failed"), (2, "failed")])
def test_worker_once_code_de_retour_vers_statut(store, fake_cmd, monkeypatch, rc, status):
    monkeypatch.setenv("FAKE_MODE", str(rc))
    jid = store.submit(cfg())
    assert worker.run_worker(store, poll=0.1, once=True, cmd=fake_cmd, stop_event=threading.Event(), name="wt") == 1
    job = store.get(jid)
    assert job.status == status and job.worker == "wt"
    log = (store.job_dir(jid) / "train.log").read_text(encoding="utf-8")
    assert "tentative 1" in log and f"--job-id {jid}" in log and "--watch-stdin" in log
    if status == "failed":
        assert rc == 2 or ("ValueError: kaboum" in job.error and "Traceback" in job.error)
        assert job.error                                     # erreur extraite du journal
    else:
        assert job.error is None
    assert store.worker_alive()


def test_worker_sans_job_et_once(store, fake_cmd):
    assert worker.run_worker(store, poll=0.1, once=True, cmd=fake_cmd, stop_event=threading.Event()) == 0


def test_worker_commande_introuvable_donne_failed(store):
    jid = store.submit(cfg())
    worker.run_worker(store, poll=0.1, once=True, cmd=["commande_inexistante_xyz"], stop_event=threading.Event())
    job = store.get(jid)
    assert job.status == "failed" and job.error


def _run_in_thread(store, cmd, stop, **kw):
    t = threading.Thread(target=lambda: worker.run_worker(store, poll=0.1, once=True, cmd=cmd, stop_event=stop, **kw))
    t.start()
    return t


def _wait(pred, timeout=20):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.05)
    return False


def test_worker_annulation_d_un_job_en_cours(store, fake_cmd, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "wait")
    monkeypatch.setattr(worker, "CANCEL_GRACE_S", 0.3)           # le worker ferme lui-même le stdin
    jid = store.submit(cfg())
    t = _run_in_thread(store, fake_cmd, threading.Event())
    assert _wait(lambda: store.get(jid).status == "running")
    assert store.cancel(jid) == "cancel_requested"
    t.join(30)
    assert not t.is_alive()
    job = store.get(jid)
    assert job.status == "cancelled" and job.error is None


def test_worker_arret_demande_donne_interrupted(store, fake_cmd, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "wait")
    stop = threading.Event()
    jid = store.submit(cfg())
    t = _run_in_thread(store, fake_cmd, stop)
    assert _wait(lambda: store.get(jid).status == "running")
    stop.set()
    t.join(30)
    assert not t.is_alive()
    assert store.get(jid).status == "interrupted"
    store.requeue(jid)                                           # relance manuelle possible


def test_worker_tue_l_enfant_apres_le_delai(store, fake_cmd, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "hang")
    monkeypatch.setattr(worker, "STOP_TIMEOUT_S", 0.5)
    stop = threading.Event()
    jid = store.submit(cfg())
    t = _run_in_thread(store, fake_cmd, stop)
    assert _wait(lambda: store.get(jid).status == "running")
    stop.set()
    t.join(30)
    assert not t.is_alive()
    job = store.get(jid)
    assert job.status == "interrupted" and "arrêt forcé" in job.error


def test_worker_recupere_les_orphelins_au_demarrage(store, fake_cmd, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "0")
    monkeypatch.setattr(worker, "STALE_AFTER_S", 0.2)
    orphan = store.submit(cfg("orphelin"))
    store.claim_next("mort")                                     # worker disparu sans battement
    time.sleep(0.4)
    worker.run_worker(store, poll=0.1, once=True, cmd=fake_cmd, stop_event=threading.Event())
    assert store.get(orphan).status == "interrupted"


def test_worker_ne_plante_pas_si_le_job_n_est_plus_en_cours(store, fake_cmd, monkeypatch, capsys):
    monkeypatch.setenv("FAKE_MODE", "wait")
    stop = threading.Event()
    jid = store.submit(cfg())
    t = _run_in_thread(store, fake_cmd, stop)
    assert _wait(lambda: store.get(jid).status == "running")
    time.sleep(0.3)
    assert store.recover_orphans(0.0) == [jid]                   # un autre worker l'a déclaré orphelin
    stop.set()
    t.join(30)
    assert not t.is_alive()
    assert store.get(jid).status == "interrupted"
    assert "n'est plus en cours" in capsys.readouterr().out


@pytest.mark.parametrize("args,expected", [
    ((0, False, False, False), "done"),
    ((3, False, False, False), "cancelled"),
    ((4, True, False, False), "cancelled"),
    ((None, True, False, True), "cancelled"),
    ((4, False, False, False), "interrupted"),
    ((None, False, True, True), "interrupted"),
    ((1, False, False, False), "failed"),
    ((2, False, False, False), "failed"),
    ((-9, False, False, False), "failed"),
])
def test_final_status(args, expected):
    assert worker.final_status(*args) == expected


def test_tail(tmp_path):
    p = tmp_path / "t.log"
    assert worker.tail(p) == ""
    p.write_text("\n".join(f"ligne {i}" for i in range(100)) + "\n", encoding="utf-8")
    out = worker.tail(p, 3).splitlines()
    assert out == ["ligne 97", "ligne 98", "ligne 99"]


def test_integration_worker_et_vrai_entrainement(tmp_path, store):
    """Bout en bout minuscule : worker --once -> sous-processus train -> checkpoint et métriques."""
    import numpy as np
    from PIL import Image
    train_dir = tmp_path / "imgs"
    train_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(2):
        Image.fromarray(rng.integers(0, 256, (96, 96), dtype=np.uint8)).save(train_dir / f"i{i}.png")
    out = tmp_path / "out"
    c = TrainConfig(name="e2e", author="T", train_dir=str(train_dir), eval_dir=str(train_dir), N=16, crop=64,
                    batch_size=2, n_patches=4, steps=6, log_every=2, eval_every=6, save_every=3,
                    n_eval_images=1).to_dict()
    jid = store.submit(c)
    cmd = [sys.executable, "-m", "aidge_in_space.train", "--out", str(out), "--device", "cpu"]
    worker.run_worker(store, poll=0.2, once=True, cmd=cmd, stop_event=threading.Event(), name="e2e")
    job = store.get(jid)
    assert job.status == "done", job.error
    assert job.last_step == 6 and job.ckpt_path and (out / f"bls2017_c1_N16_lambda2048_run{jid}.pt").exists()
    assert (store.job_dir(jid) / "train.log").stat().st_size > 0
    assert {m["kind"] for m in store.metrics(jid)} == {"train", "eval"}


# --- worker._finish : base verrouillée ---------------------------------------------------------

def _finish_qui_echoue(store, monkeypatch, fois):
    """Fait échouer store.finish (database is locked) `fois` fois (None : toujours), puis délègue."""
    import sqlite3
    reel = store.finish
    etat = {"appels": 0}

    def finish(job_id, status, error=None):
        etat["appels"] += 1
        if fois is None or etat["appels"] <= fois:
            raise sqlite3.OperationalError("database is locked")
        return reel(job_id, status, error)

    monkeypatch.setattr(store, "finish", finish)
    monkeypatch.setattr(worker, "FINISH_RETRY_WAIT_S", 0.01)
    return etat


def test_worker_finish_verrou_transitoire_le_job_finit_done(store, fake_cmd, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "0")
    etat = _finish_qui_echoue(store, monkeypatch, 1)
    jid = store.submit(cfg())
    assert worker.run_worker(store, poll=0.1, once=True, cmd=fake_cmd, stop_event=threading.Event()) == 1
    assert store.get(jid).status == "done" and etat["appels"] == 2


def test_worker_finish_verrou_permanent_le_worker_survit_et_le_job_reste_running(store, fake_cmd, monkeypatch, capsys):
    monkeypatch.setenv("FAKE_MODE", "0")
    etat = _finish_qui_echoue(store, monkeypatch, None)
    jid = store.submit(cfg())
    assert worker.run_worker(store, poll=0.1, once=True, cmd=fake_cmd, stop_event=threading.Event()) == 1
    assert store.get(jid).status == "running"
    assert etat["appels"] == worker.FINISH_RETRIES
    assert "non écrit" in capsys.readouterr().out
    assert store.recover_orphans(0.0) == [jid]            # sera repris comme interrompu


# --- list_jobs : limit et statuts --------------------------------------------------------------

def test_list_jobs_sans_limite_et_avec_statuts(store):
    ids = [store.submit(cfg(f"j{i}")) for i in range(5)]
    store.claim_next("w")                                  # ids[0] running
    store.cancel(ids[1])                                   # ids[1] cancelled
    assert [j.id for j in store.list_jobs(limit=None)] == ids[::-1]
    assert [j.id for j in store.list_jobs(limit=2)] == ids[::-1][:2]
    assert [j.id for j in store.list_jobs(statuses=["queued"])] == ids[:1:-1]
    assert {j.id for j in store.list_jobs(statuses=["running", "cancelled"])} == {ids[0], ids[1]}
    assert [j.id for j in store.list_jobs(limit=1, statuses=("queued",))] == [ids[4]]
    assert store.list_jobs(statuses=[]) == []
    assert store.list_jobs(statuses=["done"]) == []
    with pytest.raises(ValueError):
        store.list_jobs(statuses=["inconnu"])
    with pytest.raises(ValueError):
        store.list_jobs(statuses=["queued", "nimporte"])


# --- arrêt forcé de l'arbre de processus (_kill_tree) ----------------------------------------

import os   # noqa: E402
import subprocess   # noqa: E402

# L'enfant lance un petit-fils sourd (stdin ignoré, sommeil) qui écrit son propre PID, puis dort lui-même.
ENFANT_SOURD = textwrap.dedent('''
    import os, subprocess, sys, time
    pidfile = os.environ["FAKE_PIDFILE"]
    code = ("import os, sys, time; open(sys.argv[1] + '.tmp', 'w').write(str(os.getpid())); "
            "os.replace(sys.argv[1] + '.tmp', sys.argv[1]); time.sleep(120)")
    subprocess.Popen([sys.executable, "-c", code, pidfile + ".petit"], stdin=subprocess.DEVNULL)
    with open(pidfile + ".enfant.tmp", "w") as f:
        f.write(str(os.getpid()))
    os.replace(pidfile + ".enfant.tmp", pidfile + ".enfant")
    time.sleep(120)
''')


def _pid_vivant(pid):
    """Vrai si le processus existe encore (un zombie compte comme mort).

    Sans init dans le conteneur (docker run sans --init), pytest est le PID 1 et ne récupère pas le
    petit-fils orphelin tué : il reste zombie et os.kill(pid, 0) réussit encore.
    """
    if os.name == "nt":
        res = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True)
        return str(pid) in res.stdout.split()
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii", errors="replace") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return True


def _forcer_fin(*pids):
    """Nettoyage de secours : tue les processus restants."""
    for pid in pids:
        if pid and _pid_vivant(pid):
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
            else:
                os.kill(pid, 9)


def _lire_pid(path, timeout=30):
    assert _wait(path.exists, timeout), f"{path} jamais écrit"
    return int(path.read_text())


@pytest.fixture
def enfant_sourd(tmp_path, monkeypatch):
    """Commande factice sourde + fichier de PID ; les processus restants sont tués à la fin."""
    script = tmp_path / "enfant_sourd.py"
    script.write_text(ENFANT_SOURD, encoding="utf-8")
    pidfile = tmp_path / "pid"
    monkeypatch.setenv("FAKE_PIDFILE", str(pidfile))
    pids = {}
    yield [sys.executable, str(script)], pidfile, pids
    _forcer_fin(*pids.values())


def _pids(pidfile, pids):
    pids["enfant"] = _lire_pid(pidfile.with_name("pid.enfant"))
    pids["petit"] = _lire_pid(pidfile.with_name("pid.petit"))
    return pids


def _attendre_morts(*pids, timeout=15):
    assert _wait(lambda: not any(_pid_vivant(p) for p in pids), timeout), \
        f"processus encore vivants : {[p for p in pids if _pid_vivant(p)]}"


def _popen_sourd(cmd):
    kw = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
          if os.name == "nt" else {"start_new_session": True})
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kw)


def test_kill_tree_tue_l_enfant_et_le_petit_fils(enfant_sourd):
    cmd, pidfile, pids = enfant_sourd
    proc = _popen_sourd(cmd)
    try:
        _pids(pidfile, pids)
        assert _pid_vivant(pids["petit"])
        worker._kill_tree(proc, wait_s=10)
        assert proc.poll() is not None
        _attendre_morts(pids["enfant"], pids["petit"])
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(10)
        proc.stdin.close()


def test_kill_tree_sur_un_processus_deja_termine(tmp_path):
    proc = _popen_sourd([sys.executable, "-c", "pass"])
    proc.wait(30)
    worker._kill_tree(proc, wait_s=2)          # ne lève rien
    worker._kill_tree(proc, wait_s=2)          # idem une seconde fois
    assert proc.poll() == 0
    proc.stdin.close()


def _job_pris(store):
    jid = store.submit(cfg())
    job = store.claim_next("wt")
    assert job.id == jid
    return job


def test_run_job_arret_force_tue_tout_l_arbre(store, enfant_sourd, monkeypatch):
    cmd, pidfile, pids = enfant_sourd
    monkeypatch.setattr(worker, "STOP_TIMEOUT_S", 1.0)
    stop = threading.Event()
    job = _job_pris(store)
    res = {}
    t = threading.Thread(target=lambda: res.update(s=worker.run_job(store, job, cmd, 0.1, stop, name="wt")))
    t.start()
    try:
        _pids(pidfile, pids)
        stop.set()
        t.join(60)
        assert not t.is_alive()
        assert res["s"] == "interrupted"
        done = store.get(job.id)
        assert done.status == "interrupted" and "arrêt forcé" in done.error
        _attendre_morts(pids["enfant"], pids["petit"])
    finally:
        stop.set()
        t.join(30)


def test_run_job_annulation_sourde_tue_tout_l_arbre(store, enfant_sourd, monkeypatch):
    cmd, pidfile, pids = enfant_sourd
    monkeypatch.setattr(worker, "STOP_TIMEOUT_S", 1.0)
    monkeypatch.setattr(worker, "CANCEL_GRACE_S", 0.5)
    stop = threading.Event()
    job = _job_pris(store)
    res = {}
    t = threading.Thread(target=lambda: res.update(s=worker.run_job(store, job, cmd, 0.1, stop, name="wt")))
    t.start()
    try:
        _pids(pidfile, pids)
        assert store.cancel(job.id) == "cancel_requested"
        t.join(60)
        assert not t.is_alive()
        assert res["s"] == "cancelled" and store.get(job.id).status == "cancelled"
        _attendre_morts(pids["enfant"], pids["petit"])
    finally:
        stop.set()
        t.join(30)
