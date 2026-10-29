"""Tests de l'entraînement (paquet training, CLI train) : python -m pytest tests/test_training.py

Tout tourne sur CPU, dans des dossiers temporaires, avec des images générées (aucun réseau,
ni vrai outputs/ ni vrai data/). Le contrat est dans docs/conception_entrainement.md (§ 2.2 à 2.4, 2.9).
"""
import os

import numpy as np
import pytest
import torch
from PIL import Image

from aidge_in_space import jobs, train as train_cli
from aidge_in_space.codecs import balle2017
from aidge_in_space.training import ConfigError, TrainConfig
from aidge_in_space.training import loop
from aidge_in_space.training.config import PRESETS, schedule_for

CPU = torch.device("cpu")


def make_images(folder, n=3, size=96, seed=0):
    """Écrit n images PNG en niveaux de gris texturées dans folder."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    for i in range(n):
        base = xx * 1.5 + yy * 0.8 + 40 * i
        img = np.clip(base + rng.normal(0, 15, (size, size)), 0, 255).astype(np.uint8)
        Image.fromarray(img).save(folder / f"img{i}.png")
    return folder


@pytest.fixture
def dirs(tmp_path):
    """Jeux d'entraînement et d'évaluation générés."""
    return make_images(tmp_path / "train", 3), make_images(tmp_path / "eval", 2, seed=5)


def tiny(dirs, **kw):
    """Configuration minuscule (N=16, crop 64, 4 pas)."""
    base = dict(name="essai", author="Alice", train_dir=str(dirs[0]), eval_dir=str(dirs[1]),
                N=16, crop=64, batch_size=2, n_patches=8, steps=4, log_every=2, eval_every=4,
                save_every=4, n_eval_images=1, lmbda=2048)
    base.update(kw)
    return TrainConfig(**base)


def run_tiny(cfg, ckpt, **kw):
    """Entraîne sur CPU."""
    return loop.train(cfg, ckpt, device=CPU, run_info=kw.pop("run_info", None), **kw)


# --- TrainConfig ----------------------------------------------------------------------------

def test_defaults_et_roundtrip(tmp_path):
    cfg = TrainConfig()
    assert cfg.name == "" and cfg.N == 128 and cfg.crop == 256 and cfg.steps == 50_000
    d = cfg.to_dict()
    assert isinstance(d["lmbda"], int) and d["lmbda"] == 2048
    assert TrainConfig.from_dict(d) == cfg
    p = tmp_path / "c.yaml"
    cfg.to_yaml(p)
    assert TrainConfig.from_yaml(p) == cfg
    assert "name" not in cfg.to_cfg_dict() and "author" not in cfg.to_cfg_dict()
    assert TrainConfig(in_ch=3).mode == "rgb" and cfg.mode == "gris"


@pytest.mark.parametrize("bad", [
    {"inconnue": 1}, {"crop": 50}, {"crop": 8}, {"crop": "abc"}, {"crop": True}, {"N": 4}, {"N": 1000},
    {"in_ch": 2}, {"lmbda": 0}, {"lmbda": float("inf")}, {"lr": 0}, {"lr": 1.0}, {"steps": 0},
    {"batch_size": 0}, {"lr_decay_at": 0}, {"lr_decay_at": 1.5}, {"n_eval_images": 0},
    {"gdn_last": 1}, {"name": "x" * 61}, {"name": "a\x00b"}, {"train_dir": ""},
    {"n_patches": 200_000, "crop": 1024}, {"seed": -1}, {"log_every": 0},
])
def test_config_invalide(bad):
    with pytest.raises(ValueError):
        TrainConfig.from_dict(bad)
    with pytest.raises(ConfigError):
        TrainConfig.from_dict(bad)


def test_config_normalise_et_yaml_invalide(tmp_path):
    assert TrainConfig(name="  lune  ").name == "lune"
    assert TrainConfig.from_dict({"crop": 64.0}).crop == 64
    with pytest.raises(FileNotFoundError):
        TrainConfig.from_yaml(tmp_path / "absent.yaml")
    (tmp_path / "liste.yaml").write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        TrainConfig.from_yaml(tmp_path / "liste.yaml")


def test_schedule_et_presets():
    assert schedule_for(300) == dict(log_every=10, eval_every=150, save_every=100, n_eval_images=2)
    assert schedule_for(50000) == dict(log_every=200, eval_every=2500, save_every=5000, n_eval_images=4)
    for preset in PRESETS.values():
        TrainConfig.from_dict({**preset, "name": "p"})


# --- Entraînement ---------------------------------------------------------------------------

def test_train_checkpoint_nom_et_format(dirs, tmp_path):
    cfg = tiny(dirs)
    ckpt = loop.checkpoint_path(cfg, 7, tmp_path / "ck")
    assert ckpt.name == "bls2017_c1_N16_lambda2048_run7.pt"
    assert loop.checkpoint_path(cfg, None, tmp_path).name == "bls2017_c1_N16_lambda2048.pt"
    assert loop.checkpoint_path(tiny(dirs, lmbda=0.5), 1, tmp_path).name == "bls2017_c1_N16_lambda0.5_run1.pt"
    calls = []
    res = run_tiny(cfg, ckpt, run_info={"job_id": 7, "name": "essai", "author": "Alice", "commit": "abc"},
                   on_metrics=lambda s, d: calls.append((s, d["kind"])))
    assert res.status == "done" and res.step == 4 and res.ckpt_path == ckpt
    assert res.final_eval["source"] == "eval"
    assert ckpt.exists() and not list(ckpt.parent.glob("*.tmp-*"))
    ck = torch.load(ckpt, map_location="cpu", weights_only=True)
    assert {"state_dict", "cfg", "history", "step", "optimizer", "run"} <= set(ck)
    assert ck["step"] == 4
    assert ck["cfg"]["N"] == 16 and ck["cfg"]["lmbda"] == 2048 and "name" not in ck["cfg"]
    assert ck["run"]["job_id"] == 7 and ck["run"]["name"] == "essai" and ck["run"]["author"] == "Alice"
    assert ck["run"]["date"]
    h = ck["history"]
    assert set(h) == {"step", "loss", "bpp", "psnr", "eval_step", "eval_psnr", "eval_bpp"}
    assert len(h["step"]) == len(h["loss"]) == len(h["bpp"]) == len(h["psnr"]) == 2
    assert h["eval_step"] == [4] and len(h["eval_psnr"]) == len(h["eval_bpp"]) == 1
    assert (2, "train") in calls and (4, "eval") in calls


def test_train_decouvert_par_le_banc(dirs, tmp_path, monkeypatch):
    ck_dir = tmp_path / "ck"
    for lmbda, jid in ((2048, 1), (512, 2)):
        cfg = tiny(dirs, lmbda=lmbda, steps=2, log_every=1, eval_every=2, save_every=2)
        run_tiny(cfg, loop.checkpoint_path(cfg, jid, ck_dir),
                 run_info={"job_id": jid, "name": "Lune LRO (été)", "author": "Bob", "commit": ""})
    other = tiny(dirs, name="autre", steps=2, log_every=1, eval_every=2, save_every=2)
    run_tiny(other, loop.checkpoint_path(other, 3, ck_dir),
             run_info={"job_id": 3, "name": "autre", "author": "", "commit": ""})
    monkeypatch.setattr(balle2017, "CKPT_DIR", ck_dir)
    found = balle2017.codecs()
    assert len(found) == 2
    lune = next(c for c in found if c.series == "Lune LRO (été)")
    assert lune.key == "balle2017-c1-n16-lune-lro-ete" and lune.label.endswith("Lune LRO (été)")
    levels = lune.levels("gris")
    assert [lv.label.split(" ")[0] for lv in levels] == ["λ=512", "λ=2048"]      # tri par λ croissant
    assert "job 2" in levels[0].label and "Lune LRO (été)" in levels[0].label
    assert len({c.key for c in found}) == 2
    img = np.clip(np.add.outer(np.arange(64), np.arange(64)) * 2, 0, 255).astype(np.uint8)
    res = lune.run(img, levels[0], "gris", real_rate=True)
    assert res.recon.shape == img.shape and res.bits > 0


def test_checkpoint_sans_run_garde_cle_et_libelle(dirs, tmp_path, monkeypatch):
    ck_dir = tmp_path / "ck"
    cfg = tiny(dirs, steps=2, log_every=1, eval_every=2, save_every=2)
    path = loop.checkpoint_path(cfg, None, ck_dir)
    run_tiny(cfg, path)
    ck = torch.load(path, map_location="cpu", weights_only=True)
    del ck["run"]                                   # checkpoint du notebook
    torch.save(ck, path)
    monkeypatch.setattr(balle2017, "CKPT_DIR", ck_dir)
    (codec,) = balle2017.codecs()
    assert codec.series is None
    assert codec.key == "balle2017-c1-n16"
    assert codec.label == "Ballé 2017 · N=16 · gris"
    assert codec.levels("gris")[0].label == "λ=2048 (2 pas)"


def test_reprise_prolonge_historique(dirs, tmp_path):
    info = {"job_id": 5, "name": "essai", "author": "Alice", "commit": ""}
    cfg4 = tiny(dirs, steps=4)
    ckpt = loop.checkpoint_path(cfg4, 5, tmp_path / "ck")
    run_tiny(cfg4, ckpt, run_info=info)
    before = torch.load(ckpt, map_location="cpu", weights_only=True)
    # déjà complet : rien à faire, fichier non réécrit
    mtime = ckpt.stat().st_mtime_ns
    res = run_tiny(cfg4, ckpt, run_info=info)
    assert res.status == "done" and res.step == 4 and ckpt.stat().st_mtime_ns == mtime
    # reprise jusqu'à 8
    steps_seen = []
    cfg8 = tiny(dirs, steps=8)
    res = run_tiny(cfg8, ckpt, run_info=info, on_metrics=lambda s, d: steps_seen.append(s))
    after = torch.load(ckpt, map_location="cpu", weights_only=True)
    assert res.step == 8 and after["step"] == 8
    assert min(steps_seen) > 4                        # aucune métrique rejouée avant le pas repris
    assert after["history"]["step"][:2] == before["history"]["step"]
    assert after["history"]["step"] == [2, 4, 6, 8]
    assert after["history"]["eval_step"] == [4, 8]


def test_reprise_refuse_checkpoint_incompatible(dirs, tmp_path):
    cfg = tiny(dirs, steps=2, eval_every=2, save_every=2)
    ckpt = loop.checkpoint_path(cfg, 5, tmp_path / "ck")
    run_tiny(cfg, ckpt, run_info={"job_id": 5, "name": "essai", "author": "", "commit": ""})
    with pytest.raises(ConfigError):                  # autre job
        run_tiny(tiny(dirs, steps=4), ckpt, run_info={"job_id": 6, "name": "essai", "author": "", "commit": ""})
    with pytest.raises(ConfigError):                  # autre N
        run_tiny(tiny(dirs, steps=4, N=24), ckpt, run_info={"job_id": 5, "name": "essai", "author": "", "commit": ""})


def test_arret_par_should_stop_sauvegarde(dirs, tmp_path):
    cfg = tiny(dirs, steps=50, eval_every=50, save_every=50)
    ckpt = loop.checkpoint_path(cfg, 9, tmp_path / "ck")
    n = {"k": 0}

    def stop():
        n["k"] += 1
        return n["k"] >= 3                            # appelé à la fin de chaque pas

    res = run_tiny(cfg, ckpt, should_stop=stop, run_info={"job_id": 9, "name": "essai", "author": "", "commit": ""})
    assert res.status == "cancelled" and res.step == 3
    assert torch.load(ckpt, map_location="cpu", weights_only=True)["step"] == 3


def test_exception_on_metrics_ninterrompt_pas(dirs, tmp_path):
    cfg = tiny(dirs, steps=2, log_every=1, eval_every=2, save_every=2)

    def boom(step, data):
        raise RuntimeError("base verrouillée")

    res = run_tiny(cfg, loop.checkpoint_path(cfg, None, tmp_path), on_metrics=boom)
    assert res.status == "done" and res.step == 2


def test_jeu_entrainement_vide_est_une_erreur_de_config(dirs, tmp_path):
    vide = tmp_path / "vide"
    vide.mkdir()
    cfg = tiny(dirs, train_dir=str(vide))
    with pytest.raises(ConfigError):
        run_tiny(cfg, tmp_path / "x.pt")


def test_evaluation_repli_sur_entrainement(dirs, tmp_path):
    vide = tmp_path / "pas_d_eval"
    vide.mkdir()
    cfg = tiny(dirs, eval_dir=str(vide), steps=2, log_every=1, eval_every=2, save_every=2)
    res = run_tiny(cfg, tmp_path / "a.pt")
    assert res.final_eval["source"] == "train"


# --- CLI ------------------------------------------------------------------------------------

def write_yaml(cfg, path):
    cfg.to_yaml(path)
    return str(path)


def test_cli_config_termine_code_0_et_nom_du_fichier(dirs, tmp_path, capsys):
    cfg = tiny(dirs, name="", steps=2, log_every=1, eval_every=2, save_every=2)
    yml = write_yaml(cfg, tmp_path / "mon_essai.yaml")
    out = tmp_path / "out"
    assert train_cli.main(["--config", yml, "--out", str(out), "--device", "cpu"]) == jobs.EXIT_DONE == 0
    ckpt = out / "bls2017_c1_N16_lambda2048.pt"
    assert ckpt.exists()
    ck = torch.load(ckpt, map_location="cpu", weights_only=True)
    assert ck["run"]["name"] == "mon_essai" and ck["run"]["job_id"] is None
    # relance : checkpoint déjà complet -> 0
    assert train_cli.main(["--config", yml, "--out", str(out), "--device", "cpu"]) == 0


def test_cli_config_invalide_code_2(dirs, tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("crop: 50\n", encoding="utf-8")
    assert train_cli.main(["--config", str(bad), "--out", str(tmp_path / "o"), "--device", "cpu"]) == jobs.EXIT_CONFIG == 2
    assert "Configuration invalide" in capsys.readouterr().err
    assert train_cli.main(["--config", str(tmp_path / "absent.yaml"), "--device", "cpu"]) == 2
    empty = tmp_path / "empty"
    empty.mkdir()
    cfg = tiny(dirs, train_dir=str(empty), steps=2)
    assert train_cli.main(["--config", write_yaml(cfg, tmp_path / "e.yaml"), "--out", str(tmp_path / "o"),
                           "--device", "cpu"]) == 2


def test_cli_arguments_incorrects():
    with pytest.raises(SystemExit) as e:
        train_cli.main([])
    assert e.value.code == 2
    with pytest.raises(SystemExit) as e:
        train_cli.main(["--config", "a.yaml", "--job-id", "1"])
    assert e.value.code == 2


def test_cli_job_inconnu_ou_non_en_cours_code_2(dirs, tmp_path):
    db = tmp_path / "runs.sqlite"
    store = jobs.JobStore(db)
    assert train_cli.main(["--job-id", "99", "--db", str(db), "--device", "cpu"]) == 2
    jid = store.submit(tiny(dirs).to_dict())             # statut queued : refusé
    assert train_cli.main(["--job-id", str(jid), "--db", str(db), "--device", "cpu"]) == 2


def test_cli_job_termine_et_metriques(dirs, tmp_path):
    db = tmp_path / "runs.sqlite"
    store = jobs.JobStore(db)
    jid = store.submit(tiny(dirs, steps=4).to_dict())
    store.claim_next("w")
    out = tmp_path / "out"
    assert train_cli.main(["--job-id", str(jid), "--db", str(db), "--out", str(out), "--device", "cpu"]) == 0
    job = store.get(jid)
    assert job.status == "running"                       # le statut est écrit par le worker, jamais par train
    assert job.last_step == 4
    assert job.ckpt_path.endswith(f"bls2017_c1_N16_lambda2048_run{jid}.pt")
    assert (store.job_dir(jid) / "config.yaml").exists()
    assert {m["kind"] for m in store.metrics(jid)} == {"train", "eval"}


def test_cli_job_annule_avant_demarrage_code_3(dirs, tmp_path):
    db = tmp_path / "runs.sqlite"
    store = jobs.JobStore(db)
    jid = store.submit(tiny(dirs).to_dict())
    store.claim_next("w")
    store.cancel(jid)
    assert train_cli.main(["--job-id", str(jid), "--db", str(db), "--out", str(tmp_path / "o"),
                           "--device", "cpu"]) == jobs.EXIT_CANCELLED == 3


# --- StopWatcher ----------------------------------------------------------------------------

class FakeJob:
    """Job minimal lu par StopWatcher : statut, tentatives et drapeau d'annulation."""

    def __init__(self, status="running", attempts=1, cancel_requested=False):
        self.status, self.attempts, self.cancel_requested = status, attempts, cancel_requested


class FakeStore:
    """Compte les lectures de la base (get et cancel_requested)."""

    def __init__(self):
        self.reads, self.flag = 0, False
        self.status, self.attempts = "running", 1

    def get(self, job_id):
        self.reads += 1
        return FakeJob(self.status, self.attempts, self.flag)

    def cancel_requested(self, job_id):
        self.reads += 1
        return self.flag


def test_stopwatcher_limite_les_lectures_de_la_base():
    store = FakeStore()
    w = train_cli.StopWatcher(store, 1, poll_s=3600, attempts=1)
    assert [w() for _ in range(50)] == [False] * 50
    assert store.reads == 1
    store.flag = True
    w._last_poll = None
    assert w() is True and w.reason == "cancel" and train_cli.stop_code(w) == jobs.EXIT_CANCELLED


def test_stopwatcher_signal_donne_code_4():
    w = train_cli.StopWatcher()
    assert w() is False
    w._on_signal(2, None)
    assert w() is True and w.reason == "signal" and train_cli.stop_code(w) == jobs.EXIT_STOPPED


def test_stopwatcher_annulation_l_emporte_sur_stdin():
    store = FakeStore()
    store.flag = True
    w = train_cli.StopWatcher(store, 1)
    w.reason = "stdin"
    assert train_cli.stop_code(w) == jobs.EXIT_CANCELLED


def test_stopwatcher_reprise_par_un_autre_processus_sort_avec_4(monkeypatch):
    sorties = []

    def fausse_sortie(code):
        sorties.append(code)
        raise SystemExit(code)

    monkeypatch.setattr(train_cli.os, "_exit", fausse_sortie)
    store = FakeStore()
    w = train_cli.StopWatcher(store, 1, poll_s=0, attempts=1)
    assert w() is False and sorties == []             # même tentative, toujours en cours
    store.attempts = 2                                 # un autre worker a repris le job
    with pytest.raises(SystemExit):
        w()
    assert sorties == [jobs.EXIT_STOPPED] == [4]
    store.attempts, store.status = 1, "cancelled"      # statut qui n'est plus running
    with pytest.raises(SystemExit):
        train_cli.StopWatcher(store, 1, poll_s=0, attempts=1)()
    assert sorties == [4, 4]


# --- training.imaging.to_8bit et banque de patchs ---------------------------------------------

from aidge_in_space.training.imaging import to_8bit      # noqa: E402


def image_16bits(size=256, top=4000, seed=0):
    """Image I;16 : dégradé horizontal de 0 à top, plus un peu de bruit (le maximum vaut top)."""
    rng = np.random.default_rng(seed)
    arr = np.tile(np.linspace(0, top, size), (size, 1)) * 0.9 + rng.uniform(0, top * 0.1, (size, size))
    arr[0, 0] = top
    return Image.fromarray(np.clip(arr, 0, top).astype(np.uint16))


@pytest.mark.parametrize("ext", ["tif", "png"])
def test_to_8bit_image_16_bits_mise_a_l_echelle(tmp_path, ext):
    path = tmp_path / f"lune.{ext}"
    image_16bits().save(path)
    img = Image.open(path)
    assert img.mode.startswith("I")
    out, converted = to_8bit(img)
    assert converted is True and out.mode == "L"
    arr = np.asarray(out)
    assert arr.max() == 255
    assert (arr == 255).mean() < 0.01                # Pillow convert("L") en écrêtant donnerait du blanc


def test_to_8bit_inchange_si_deja_8_bits():
    for mode in ("L", "RGB"):
        img = Image.new(mode, (8, 8), 7)
        out, converted = to_8bit(img)
        assert out is img and converted is False


def test_to_8bit_image_nulle_donne_du_noir():
    out, converted = to_8bit(Image.fromarray(np.zeros((8, 8), np.uint16)))
    assert converted is True and out.mode == "L" and np.asarray(out).max() == 0


def petites_images(folder, n, size=40):
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    for i in range(n):
        Image.fromarray(rng.integers(0, 256, (size, size), dtype=np.uint8)).save(folder / f"p{i:03d}.png")
    return sorted(folder.glob("*.png"))


@pytest.mark.parametrize("n_images", [50, 7, 10, 3])
def test_banque_de_patchs_exactement_n_patches(tmp_path, n_images):
    files = petites_images(tmp_path / "imgs", n_images)
    loop.seed_everything(0)
    bank = loop.build_patch_bank(files, 10, 32, 1, CPU, seed=0)
    assert bank.shape == (10, 1, 32, 32) and bank.dtype == torch.uint8


def test_banque_de_patchs_reproductible_avec_la_meme_graine(tmp_path):
    files = petites_images(tmp_path / "imgs", 50)
    loop.seed_everything(4)
    a = loop.build_patch_bank(files, 10, 32, 1, CPU, seed=4)
    loop.seed_everything(4)
    b = loop.build_patch_bank(files, 10, 32, 1, CPU, seed=4)
    assert torch.equal(a, b)


def test_banque_de_patchs_choix_des_images_depend_de_la_graine(tmp_path, monkeypatch):
    """Seul le tirage des chemins dépend de seed (le reste utilise le générateur global)."""
    files = petites_images(tmp_path / "imgs", 50)
    pris = {}
    original = loop.Image.open
    courant = {"seed": None}

    def espion(p, *a, **k):
        pris.setdefault(courant["seed"], []).append(str(p))
        return original(p, *a, **k)

    monkeypatch.setattr(loop.Image, "open", espion)
    for s in (1, 1, 2):
        courant["seed"] = s
        loop.build_patch_bank(files, 10, 32, 1, CPU, seed=s)
    assert len(pris[1]) == 20 and pris[1][:10] == pris[1][10:] and len(set(pris[1])) == 10
    assert set(pris[1]) != set(pris[2])


def test_banque_de_patchs_image_16_bits_non_saturee(tmp_path):
    d = tmp_path / "imgs"
    d.mkdir()
    image_16bits(256).save(d / "lune.tif")
    stats = {}
    loop.seed_everything(0)
    bank = loop.build_patch_bank([d / "lune.tif"], 6, 64, 1, CPU, seed=0, stats=stats)
    assert stats["converted"] == 1
    assert (bank == 255).float().mean().item() < 0.05
    assert bank.float().mean().item() > 20           # pas un noir uniforme non plus


# --- Sauvegarde périodique tolérante ----------------------------------------------------------

def patch_replace(monkeypatch, fail):
    """os.replace qui échoue (PermissionError) tant que fail() est vrai ; REPLACE_DELAY_S est nul."""
    real = os.replace
    appels = {"n": 0}

    def fausse(src, dst, *a, **k):
        if ".tmp-" in str(src) and fail():
            appels["n"] += 1
            raise PermissionError("fichier ouvert par un autre processus")
        return real(src, dst, *a, **k)

    monkeypatch.setattr(loop, "REPLACE_DELAY_S", 0)
    monkeypatch.setattr(loop.os, "replace", fausse)
    return appels


def test_sauvegarde_periodique_en_echec_ne_stoppe_pas_l_entrainement(dirs, tmp_path, monkeypatch):
    cfg = tiny(dirs, steps=6, log_every=2, eval_every=6, save_every=2)
    ckpt = loop.checkpoint_path(cfg, None, tmp_path / "ck")
    bloque = {"v": True}
    appels = patch_replace(monkeypatch, lambda: bloque["v"])

    def on_metrics(step, data):
        if data["kind"] == "eval":                    # évaluation finale, juste avant la sauvegarde finale
            bloque["v"] = False

    res = run_tiny(cfg, ckpt, on_metrics=on_metrics)
    assert appels["n"] >= 2 * (loop.REPLACE_RETRIES + 1)          # sauvegardes aux pas 2 et 4 échouées
    assert res.status == "done" and res.step == 6
    assert torch.load(ckpt, map_location="cpu", weights_only=True)["step"] == 6
    assert not list(ckpt.parent.glob("*.tmp*"))


def test_sauvegarde_finale_en_echec_leve_une_erreur(dirs, tmp_path, monkeypatch):
    cfg = tiny(dirs, steps=2, log_every=1, eval_every=2, save_every=2)
    ckpt = loop.checkpoint_path(cfg, None, tmp_path / "ck")
    patch_replace(monkeypatch, lambda: True)
    with pytest.raises(OSError):
        run_tiny(cfg, ckpt)
    assert not ckpt.exists() and not list(ckpt.parent.glob("*.tmp*"))


# --- Vrais sous-processus : StopWatcher et --watch-stdin ---------------------------------------

import subprocess      # noqa: E402
import sys             # noqa: E402
import time            # noqa: E402

from aidge_in_space import config      # noqa: E402

LONG = dict(steps=200_000, log_every=10, eval_every=200_000, save_every=200_000)


def lance_train(args, log_path, **kw):
    """Lance python -m aidge_in_space.train ; sortie (stdout et stderr) dans log_path."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    log = open(log_path, "wb")
    proc = subprocess.Popen([sys.executable, "-m", "aidge_in_space.train", *args, "--device", "cpu"],
                            cwd=str(config.ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, **kw)
    proc.log_file = log
    return proc


def attend(condition, timeout, what):
    """Attend que condition() soit vraie ; échoue avec un message sinon."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if condition():
            return
        time.sleep(0.2)
    pytest.fail(f"délai dépassé ({timeout} s) : {what}")


def termine(proc, timeout=60):
    """Attend la fin du sous-processus (le tue s'il reste vivant) et ferme son journal."""
    try:
        return proc.wait(timeout=timeout)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        proc.log_file.close()


def lire(path):
    return path.read_bytes().decode("utf-8", errors="replace") if path.exists() else ""


def test_sous_processus_job_repris_par_un_autre_sort_avec_4_sans_sauvegarder(dirs, tmp_path):
    import sqlite3
    db, out = tmp_path / "runs.sqlite", tmp_path / "out"
    store = jobs.JobStore(db)
    jid = store.submit(tiny(dirs, **LONG).to_dict())
    store.claim_next("w1")
    log = tmp_path / "t.log"
    proc = lance_train(["--job-id", str(jid), "--db", str(db), "--out", str(out)], log)
    try:
        attend(lambda: store.get(jid).last_step > 0 or proc.poll() is not None, 120, "premières métriques")
        assert proc.poll() is None, lire(log)
        con = sqlite3.connect(str(db), timeout=30)
        con.execute("UPDATE jobs SET attempts = attempts + 1 WHERE id = ?", (jid,))   # un autre worker reprend
        con.commit()
        con.close()
        t0 = time.time()
        rc = termine(proc, timeout=60)
        duree = time.time() - t0
    finally:
        if proc.poll() is None:
            proc.kill()
    assert rc == jobs.EXIT_STOPPED == 4, lire(log)
    assert "repris par un autre processus" in lire(log)
    assert duree < 30
    assert not list(out.glob("*.pt")) and not list(out.glob("*.tmp*"))     # aucune sauvegarde


def test_sous_processus_watch_stdin_ouvert_jusqu_a_la_fin_code_0(dirs, tmp_path):
    cfg = tiny(dirs, steps=10, log_every=5, eval_every=10, save_every=10)
    yml = tmp_path / "c.yaml"
    cfg.to_yaml(yml)
    out = tmp_path / "out"
    proc = lance_train(["--config", str(yml), "--out", str(out), "--watch-stdin"], tmp_path / "t.log",
                       stdin=subprocess.PIPE)
    try:
        rc = termine(proc, timeout=120)               # stdin reste ouvert jusqu'à la fin
    finally:
        proc.stdin.close()
    assert rc == 0, lire(tmp_path / "t.log")
    assert (out / "bls2017_c1_N16_lambda2048.pt").exists()


def test_sous_processus_watch_stdin_ferme_en_cours_code_4_avec_checkpoint(dirs, tmp_path):
    cfg = tiny(dirs, **LONG)
    yml = tmp_path / "c.yaml"
    cfg.to_yaml(yml)
    out = tmp_path / "out"
    log = tmp_path / "t.log"
    proc = lance_train(["--config", str(yml), "--out", str(out), "--watch-stdin"], log, stdin=subprocess.PIPE)
    try:
        attend(lambda: "it/s" in lire(log) or proc.poll() is not None, 120, "entraînement démarré")
        assert proc.poll() is None, lire(log)
        proc.stdin.close()
        rc = termine(proc, timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert rc == jobs.EXIT_STOPPED == 4, lire(log)
    ck = torch.load(out / "bls2017_c1_N16_lambda2048.pt", map_location="cpu", weights_only=True)
    assert 0 < ck["step"] < LONG["steps"]
