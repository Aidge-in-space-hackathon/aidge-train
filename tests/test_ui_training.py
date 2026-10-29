"""Tests de l'interface d'entraînement (fonctions pures de ui_training, app.build) : python -m pytest tests/test_ui_training.py

Aucun Gradio lancé : on teste les fonctions pures et la construction de l'interface. Contrat :
docs/conception_entrainement.md (§ 2.7 et 2.8). Dossiers temporaires uniquement.
"""
import io
import time
import zipfile

import matplotlib
import numpy as np
import pytest
from PIL import Image

matplotlib.use("Agg")
ui = pytest.importorskip("aidge_in_space.ui_training")

from aidge_in_space import config   # noqa: E402
from aidge_in_space.jobs import JobStore   # noqa: E402
from aidge_in_space.training import TrainConfig   # noqa: E402


def png_bytes(size=32, seed=0):
    """Octets d'un PNG gris aléatoire."""
    arr = np.random.default_rng(seed).integers(0, 256, (size, size), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


def write_png(path, size=32, seed=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png_bytes(size, seed))
    return path


def make_zip(path, members):
    """members : {nom: octets} ou {nom: ZipInfo+octets via tuple}."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in members.items():
            if isinstance(data, tuple):
                z.writestr(*data)
            else:
                z.writestr(name, data)
    return path


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(config, "CKPT_DIR", tmp_path / "ckpts")
    return JobStore(tmp_path / "runs" / "runs.sqlite")


@pytest.fixture
def train_dir(tmp_path):
    return tmp_path / "train"


# --- assainissement des noms ----------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("Lune LRO (été)", "Lune_LRO_ete"), ("../../etc", "etc"), ("con", "_con"), ("COM1.txt", "_COM1.txt"),
    ("a   b", "a_b"), ("..cache..", "cache"), ("photo.PNG", "photo.PNG"), ("nul", "_nul"),
])
def test_safe_name(raw, expected):
    assert ui.safe_name(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "../..", "日本語", "!!!"])
def test_safe_name_vide_refuse(raw):
    with pytest.raises(ValueError):
        ui.safe_name(raw)


def test_safe_name_longueur():
    assert len(ui.safe_name("a" * 200)) <= 60
    assert not ui.safe_name("ab" + "." * 100 + "c", max_len=3).endswith(".")


def test_max_upload_mb(monkeypatch):
    monkeypatch.delenv("AIDGE_MAX_UPLOAD_MB", raising=False)
    assert ui.max_upload_mb() == ui.DEFAULT_MAX_UPLOAD_MB == 500
    monkeypatch.setenv("AIDGE_MAX_UPLOAD_MB", "12")
    assert ui.max_upload_mb() == 12
    for bad in ("abc", "0", "-3"):
        monkeypatch.setenv("AIDGE_MAX_UPLOAD_MB", bad)
        with pytest.raises(ValueError):
            ui.max_upload_mb()


# --- extraction des archives ----------------------------------------------------------------

@pytest.mark.parametrize("evil", ["../evil.png", "a/../../evil.png", "/abs.png", "\\abs.png",
                                  "C:\\x.png", "C:/x.png", "..\\evil.png", "a\\..\\..\\evil.png"])
def test_zip_slip_refuse_sans_rien_ecrire(tmp_path, evil):
    z = make_zip(tmp_path / "z.zip", {"ok.png": png_bytes(), evil: png_bytes(seed=1)})
    dest = tmp_path / "dest"
    dest.mkdir()
    with pytest.raises(ValueError, match="archive refusée"):
        ui.extract_zip(z, dest, 10_000_000)
    assert list(dest.iterdir()) == []                      # tout est contrôlé avant toute écriture
    assert not (tmp_path / "evil.png").exists() and not (tmp_path.parent / "evil.png").exists()


def test_zip_lien_symbolique_refuse(tmp_path):
    info = zipfile.ZipInfo("lien.png")
    info.external_attr = 0o120777 << 16
    z = make_zip(tmp_path / "z.zip", {"lien.png": (info, "/etc/passwd")})
    dest = tmp_path / "dest"
    dest.mkdir()
    with pytest.raises(ValueError, match="archive refusée"):
        ui.extract_zip(z, dest, 10_000_000)
    assert list(dest.iterdir()) == []


def test_zip_bombe_taille_decompressee(tmp_path):
    z = make_zip(tmp_path / "bombe.zip", {"gros.png": b"\0" * 5_000_000})     # se compresse en quelques Ko
    assert z.stat().st_size < 100_000
    dest = tmp_path / "dest"
    dest.mkdir()
    with pytest.raises(ValueError, match="archive refusée"):
        ui.extract_zip(z, dest, 1_000_000)
    assert list(dest.iterdir()) == []


def test_zip_trop_de_fichiers(tmp_path):
    z = make_zip(tmp_path / "z.zip", {f"{i}.png": b"x" for i in range(10)})
    dest = tmp_path / "dest"
    dest.mkdir()
    with pytest.raises(ValueError, match="archive refusée"):
        ui.extract_zip(z, dest, 10_000_000, max_files=5)


def test_zip_illisible(tmp_path):
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"ce n'est pas un zip")
    with pytest.raises(ValueError, match="illisible"):
        ui.extract_zip(bad, tmp_path, 1_000_000)


def test_zip_aplatit_ignore_et_gere_les_collisions(tmp_path):
    z = make_zip(tmp_path / "z.zip", {
        "dossier/a.png": png_bytes(seed=1), "autre/a.png": png_bytes(seed=2), "notes.txt": b"hello",
        "__MACOSX/._a.png": b"junk", ".cache.png": png_bytes(), "Photo Été.JPG": png_bytes(seed=3),
        "sous/": b"",
    })
    dest = tmp_path / "dest"
    dest.mkdir()
    written, messages = ui.extract_zip(z, dest, 10_000_000)
    names = sorted(p.name for p in dest.iterdir())
    assert names == ["Photo_Ete.jpg", "a.png", "a_2.png"]
    assert sorted(p.name for p in written) == names
    assert all(p.parent == dest for p in written)
    assert messages                                        # fichiers ignorés signalés


# --- création des jeux d'images -------------------------------------------------------------

def test_create_dataset_images_et_zip(tmp_path, train_dir):
    a = write_png(tmp_path / "in" / "a.png", 300)
    z = make_zip(tmp_path / "in" / "lot.zip", {"x/b.png": png_bytes(300, 1), "c.jpg": b"pas une image"})
    notes = tmp_path / "in" / "notes.txt"
    notes.write_text("x")
    folder, messages = ui.create_dataset("Lune LRO (été)", [a, z, notes], train_dir=train_dir, max_upload_mb=5)
    assert folder == train_dir / "Lune_LRO_ete"
    assert sorted(p.name for p in folder.iterdir()) == ["a.png", "b.png"]    # c.jpg illisible supprimé, .txt ignoré
    assert messages
    assert [d["name"] for d in ui.list_datasets(train_dir)] == ["Lune_LRO_ete"]
    info = ui.list_datasets(train_dir)[0]
    assert info["n_images"] == 2 and len(info["samples"]) == 2 and info["path"] == folder
    assert ui.dataset_choices(train_dir) == [("Lune_LRO_ete (2 images)", "Lune_LRO_ete")]


def test_create_dataset_complete_sans_ecraser(tmp_path, train_dir):
    a = write_png(tmp_path / "in" / "a.png", 300, seed=1)
    ui.create_dataset("jeu", [a], train_dir=train_dir, max_upload_mb=5)
    other = write_png(tmp_path / "in2" / "a.png", 300, seed=2)
    before = (train_dir / "jeu" / "a.png").read_bytes()
    folder, _ = ui.create_dataset("jeu", [other], train_dir=train_dir, max_upload_mb=5)
    assert (train_dir / "jeu" / "a.png").read_bytes() == before
    assert len(list(folder.iterdir())) == 2


def test_create_dataset_extensions_refusees_et_dossier_supprime(tmp_path, train_dir):
    bad = [tmp_path / "x.exe", tmp_path / "y.txt", tmp_path / "z.svg"]
    for p in bad:
        p.write_bytes(b"x")
    with pytest.raises(ValueError):
        ui.create_dataset("rien", bad, train_dir=train_dir, max_upload_mb=5)
    assert not (train_dir / "rien").exists()


def test_create_dataset_image_illisible_refusee(tmp_path, train_dir):
    fake = tmp_path / "faux.png"
    fake.write_bytes(b"pas une image du tout")
    with pytest.raises(ValueError):
        ui.create_dataset("faux", [fake], train_dir=train_dir, max_upload_mb=5)
    assert not (train_dir / "faux").exists()


def test_create_dataset_erreurs_diverses(tmp_path, train_dir):
    a = write_png(tmp_path / "a.png")
    with pytest.raises(ValueError):
        ui.create_dataset("x", [], train_dir=train_dir)
    with pytest.raises(ValueError):
        ui.create_dataset("x", None, train_dir=train_dir)
    with pytest.raises(ValueError):
        ui.create_dataset("../..", [a], train_dir=train_dir)
    big = tmp_path / "big.png"
    big.write_bytes(b"\0" * (2 * 1024 * 1024))
    with pytest.raises(ValueError):
        ui.create_dataset("gros", [big], train_dir=train_dir, max_upload_mb=1)
    zip_slip = make_zip(tmp_path / "s.zip", {"../x.png": png_bytes()})
    with pytest.raises(ValueError, match="archive refusée"):
        ui.create_dataset("slip", [zip_slip], train_dir=train_dir, max_upload_mb=5)
    assert not (train_dir / "slip").exists()
    assert not (tmp_path / "x.png").exists()


def test_list_datasets_ignore_vides_et_fichiers(train_dir):
    (train_dir / "vide").mkdir(parents=True)
    (train_dir / "fichier.txt").write_text("x")
    write_png(train_dir / "b" / "1.png")
    write_png(train_dir / "a" / "1.png")
    assert [d["name"] for d in ui.list_datasets(train_dir)] == ["a", "b"]
    assert ui.list_datasets(train_dir / "absent") == []


# --- estimation BRAM ------------------------------------------------------------------------

def test_estimate_bram():
    small = ui.estimate("gris", 32, True)
    big = ui.estimate("gris", 128, True)
    assert small["fits_bram"] is True and big["fits_bram"] is False
    assert 50_000 < small["params"] < 65_000 and 800_000 < big["params"] < 950_000
    assert small["int8_bytes"] == small["params"] and small["bram_bytes"] == ui.BRAM_BYTES
    assert small["fits_bram"] == (small["int8_bytes"] <= small["bram_bytes"])
    assert small["mac_per_pixel"] > 0 and isinstance(small["markdown"], str) and small["markdown"]
    assert ui.estimate("rgb", 32, True)["params"] > small["params"]
    with pytest.raises(ValueError):
        ui.estimate("sepia", 32, True)
    with pytest.raises(ValueError):
        ui.estimate("gris", 4, True)
    with pytest.raises(ValueError):
        ui.estimate("gris", 600, True)


# --- préréglages et lancement ---------------------------------------------------------------

def test_preset_values():
    v = ui.preset_values("Test rapide (quelques minutes)")
    assert {"N", "gdn_last", "lambdas", "steps", "batch_size", "lr", "crop", "n_patches", "seed"} <= set(v)
    assert v["lambdas"] == [2048]
    std = ui.preset_values("Standard (plusieurs heures, GPU conseillé)")
    assert std["lambdas"] == [512, 1024, 2048, 4096] and std["steps"] > v["steps"]
    assert ui.preset_values("Personnalisé") == {}
    with pytest.raises(ValueError):
        ui.preset_values("inconnu")


def launch(store, train_dir, **over):
    write_png(train_dir / "lune" / "1.png", 64)
    args = dict(name="lune-lro", author="Alice", dataset="lune", mode="gris", N=32, gdn_last=True,
                lambdas=[4096, 512, 512], steps=100, batch_size=4, lr=1e-4, crop=64, n_patches=16, seed=1,
                train_dir=train_dir)
    args.update(over)
    return ui.launch_jobs(store, **args)


def test_lancement_un_job_par_lambda(store, train_dir, monkeypatch):
    monkeypatch.setattr(config, "ROOT", train_dir.parent)           # chemin portable relatif à ROOT
    ids = launch(store, train_dir)
    assert len(ids) == 2
    jobs_ = [store.get(i) for i in ids]
    assert [j.lmbda for j in jobs_] == [512, 4096]                 # dédupliqués, ordre croissant
    assert all(j.status == "queued" and j.name == "lune-lro" and j.author == "Alice" for j in jobs_)
    c = jobs_[0].config
    assert c["N"] == 32 and c["crop"] == 64 and c["steps"] == 100 and c["in_ch"] == 1
    assert c["train_dir"] == "train/lune"                          # relatif à ROOT, notation POSIX
    assert c["log_every"] == min(200, max(10, 100 // 30))          # schedule_for(100)
    TrainConfig.from_dict(c)                                       # config relisible telle quelle


def test_lancement_mode_rgb(store, train_dir):
    (jid,) = launch(store, train_dir, mode="rgb", lambdas=[1024])
    assert store.get(jid).config["in_ch"] == 3


@pytest.mark.parametrize("over", [
    {"name": ""}, {"name": "   "}, {"lambdas": []}, {"dataset": "absent"}, {"crop": 50}, {"N": 4},
    {"steps": 0},
])
def test_lancement_refuse_et_tout_ou_rien(store, train_dir, over):
    with pytest.raises(ValueError):
        launch(store, train_dir, **over)
    assert store.list_jobs() == []                                 # aucune soumission partielle


def test_lancement_jeu_vide(store, train_dir):
    (train_dir / "vide").mkdir(parents=True)
    with pytest.raises(ValueError):
        launch(store, train_dir, dataset="vide")
    assert store.list_jobs() == []


def test_lancement_lambda_invalide_ne_soumet_rien(store, train_dir):
    with pytest.raises(ValueError):
        launch(store, train_dir, lambdas=[512, 0])
    assert store.list_jobs() == []


# --- suivi ----------------------------------------------------------------------------------

def test_format_duration():
    assert ui.format_duration(None) == "-"
    assert ui.format_duration(0) == "< 1 min" and ui.format_duration(59) == "< 1 min"
    assert ui.format_duration(60) == "1 min" and ui.format_duration(3599) == "59 min"
    assert ui.format_duration(3600) == "1 h 00 min" and ui.format_duration(7500) == "2 h 05 min"


def running_job(store, steps=100, **kw):
    jid = store.submit(TrainConfig(name="essai", author="A", steps=steps, **kw).to_dict())
    store.claim_next("w")
    return jid


def test_job_progress(store):
    jid = running_job(store, steps=200)
    store.log_metrics(jid, 50, "train", {"eta_s": 100.0})
    job = store.get(jid)
    m = store.metrics(jid, "train")
    pct, remaining = ui.job_progress(job, m, m[0]["created_at"] + 30)
    assert pct == 25.0 and remaining == pytest.approx(70.0)       # eta_s - (now - date de la métrique)
    pct, remaining = ui.job_progress(job, [], time.time())
    assert pct == 25.0 and remaining is None                       # pas encore de métrique
    pct, remaining = ui.job_progress(job, m, m[0]["created_at"] + 10_000)
    assert remaining == 0                                          # jamais négatif
    store.finish(jid, "done")
    pct, remaining = ui.job_progress(store.get(jid), m, time.time())
    assert pct == 100 and remaining is None
    queued = store.submit(TrainConfig(name="q").to_dict())
    assert ui.job_progress(store.get(queued), [], time.time()) == (0.0, None)


def test_jobs_table(store):
    a = running_job(store, steps=100)
    store.log_metrics(a, 30, "train", {"eta_s": 600.0})
    b = store.submit(TrainConfig(name="second", lmbda=512).to_dict())
    df = ui.jobs_table(store)
    assert list(df.columns) == ["n°", "nom", "auteur", "statut", "λ", "N", "mode", "progression (%)",
                                "temps restant", "créé le"]
    assert list(df["n°"]) == [b, a]                                # du plus récent au plus ancien
    assert list(df["statut"]) == ["En attente", "En cours"]
    assert df["mode"].iloc[0] == "gris" and df["progression (%)"].iloc[1] == 30
    assert time.strftime("%Y-%m-%d") in df["créé le"].iloc[0]
    store.cancel(a)
    assert ui.jobs_table(store)["statut"].iloc[1] == "Annulation demandée"
    assert list(ui.jobs_table(JobStore(store.path.parent / "vide.sqlite")).columns) == list(df.columns)


def test_job_choices(store):
    a = running_job(store)
    (label, value), = ui.job_choices(store)
    assert value == a and label == f"n°{a} · essai · λ=2048 · En cours"


def test_job_details(store, tmp_path):
    assert ui.job_details(store, None) == (None, "", "Sélectionnez un job.")
    with pytest.raises(KeyError):
        ui.job_details(store, 999)
    jid = running_job(store)
    fig, log, md = ui.job_details(store, jid)
    assert log == "(journal vide)" and "essai" in md
    for step in (10, 20):
        store.log_metrics(jid, step, "train", {"loss": 5.0 / step, "bpp": 0.5, "psnr": 20.0 + step, "it_s": 3.0})
    store.log_metrics(jid, 20, "eval", {"bpp": 0.4, "psnr": 25.0, "source": "kodak", "n_images": 4})
    (store.job_dir(jid)).mkdir(parents=True, exist_ok=True)
    (store.job_dir(jid) / "train.log").write_text("\n".join(f"ligne {i}" for i in range(200)), encoding="utf-8")
    fig, log, md = ui.job_details(store, jid, n_log_lines=5)
    from matplotlib.figure import Figure
    assert isinstance(fig, Figure) and len(fig.axes) == 3
    assert log.splitlines() == [f"ligne {i}" for i in range(195, 200)]


def test_annuler_et_relancer(store):
    jid = store.submit(TrainConfig(name="x").to_dict())
    with pytest.raises(ValueError, match="confirmation"):
        ui.cancel_job(store, jid, False)
    assert store.get(jid).status == "queued"
    assert ui.cancel_job(store, jid, True)
    assert store.get(jid).status == "cancelled"
    with pytest.raises(ValueError):
        ui.cancel_job(store, jid, True)                            # déjà terminé
    assert ui.requeue_job(store, jid)
    assert store.get(jid).status == "queued"
    with pytest.raises(ValueError):
        ui.requeue_job(store, jid)
    with pytest.raises(KeyError):
        ui.cancel_job(store, 999, True)


def test_duplicate_config(store, train_dir):
    (jid,) = launch(store, train_dir, lambdas=[1024], name="lune", author="Bob")
    d = ui.duplicate_config(store, jid)
    assert d["name"] == "lune (copie)" and d["author"] == "Bob" and d["dataset"] == "lune"
    assert d["mode"] == "gris" and d["preset"] == "Personnalisé" and d["lambdas"] == [1024]
    assert d["N"] == 32 and d["steps"] == 100 and d["crop"] == 64
    with pytest.raises(KeyError):
        ui.duplicate_config(store, 999)


def test_worker_banner(store):
    assert "lancer_plateforme.bat" in ui.worker_banner(store) or "docker compose up worker" in ui.worker_banner(store)
    store.worker_heartbeat("w")
    assert "Worker actif" in ui.worker_banner(store)


def test_new_done_jobs(store):
    a = running_job(store)
    assert ui.new_done_jobs(store, None) == (set(), False)         # premier tick : jamais vrai
    seen, new = ui.new_done_jobs(store, set())
    assert not new
    store.finish(a, "done")
    seen, new = ui.new_done_jobs(store, seen)
    assert seen == {a} and new is True
    assert ui.new_done_jobs(store, seen) == ({a}, False)
    assert ui.new_done_jobs(store, None) == ({a}, False)


# --- application ----------------------------------------------------------------------------

def test_app_build_contient_les_onglets(store, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TRAIN_DIR", tmp_path / "train_vide")      # jamais le vrai data/train
    import gradio as gr
    from aidge_in_space import app
    demo = app.build()
    assert isinstance(demo, gr.Blocks)
    labels = {getattr(b, "label", None) for b in demo.blocks.values() if isinstance(b, gr.Tab)}
    assert {"Données", "Entraînement", "Suivi"} <= labels
    ids = {getattr(b, "id", None) for b in demo.blocks.values() if isinstance(b, gr.Tab)}
    assert {"donnees", "entrainement", "suivi"} <= ids


def test_app_parse_users():
    from aidge_in_space import app
    assert app.parse_users(None) == [] and app.parse_users("") == []
    assert app.parse_users(" a : b:c , d:e ") == [("a", "b:c"), ("d", "e")]
    for bad in ("a", "a:", ":b", "a:b,,c:d", "a:b,c"):
        with pytest.raises(ValueError) as e:
            app.parse_users(bad)
        assert "AIDGE_UI_USERS mal formé" in str(e.value)
    with pytest.raises(ValueError) as e:
        app.parse_users("alice:secret,bob")
    assert "secret" not in str(e.value) and "n°2" in str(e.value)


@pytest.mark.parametrize("host,expected", [
    ("127.0.0.1", True), ("localhost", True), ("::1", True), ("127.0.0.5", True),
    ("0.0.0.0", False), ("::", False), ("", False), ("192.168.1.2", False), ("exemple.org", False),
])
def test_app_is_loopback(host, expected):
    from aidge_in_space import app
    assert app.is_loopback(host) is expected


def test_app_check_exposure():
    from aidge_in_space import app
    app.check_exposure("127.0.0.1", [], None, False)
    app.check_exposure("0.0.0.0", [("a", "b")], None, False)
    app.check_exposure("0.0.0.0", [], "127.0.0.1", True)            # Docker : l'adresse publiée compte
    with pytest.raises(ValueError, match="Exposition réseau refusée"):
        app.check_exposure("0.0.0.0", [], None, False)
    with pytest.raises(ValueError, match="AIDGE_UI_USERS"):
        app.check_exposure("0.0.0.0", [], "0.0.0.0", True)
    app.check_exposure("0.0.0.0", [("a", "b")], "0.0.0.0", True)
    with pytest.raises(ValueError):
        app.check_exposure("192.168.1.2", [], "127.0.0.1", False)    # hors Docker, bind ignoré


# --- _Names, conversion 16 bits, tout ou rien, concurrence ----------------------------------------

def test_names_dix_mille_noms_identiques_sans_ecrasement(tmp_path):
    (tmp_path / "a.png").write_bytes(b"x")
    (tmp_path / "A_2.PNG").write_bytes(b"x")            # la casse est ignorée (Windows)
    (tmp_path / "a_5.png").write_bytes(b"x")
    names = ui._Names(tmp_path)
    t0 = time.perf_counter()
    got = [names.free("a", ".png") for _ in range(10_000)]
    assert time.perf_counter() - t0 < 2
    low = [p.name.lower() for p in got]
    assert len(set(low)) == 10_000
    assert not {"a.png", "a_2.png", "a_5.png"} & set(low)       # aucun nom existant rendu
    assert all(p.parent == tmp_path for p in got)


def tiff_16(path, size=300, top=4000, seed=0):
    rng = np.random.default_rng(seed)
    arr = np.tile(np.linspace(0, top, size), (size, 1)) * 0.9 + rng.uniform(0, top * 0.1, (size, size))
    arr[0, 0] = top
    Image.fromarray(np.clip(arr, 0, top).astype(np.uint16)).save(path)
    return path


def test_create_dataset_tiff_16_bits_reecrit_en_8_bits_un_seul_message(tmp_path, train_dir):
    (tmp_path / "in").mkdir()
    a = tiff_16(tmp_path / "in" / "a.tif")
    b = tiff_16(tmp_path / "in" / "b.tif", seed=1)
    assert Image.open(a).mode.startswith("I")
    folder, messages = ui.create_dataset("lune", [a, b], train_dir=train_dir, max_upload_mb=50)
    assert len([m for m in messages if "16 bits" in m]) == 1       # un message pour l'ensemble
    for f in folder.iterdir():
        with Image.open(f) as im:
            assert im.mode == "L"
            arr = np.asarray(im)
        assert arr.max() == 255 and (arr == 255).mean() < 0.01


def test_create_dataset_tout_ou_rien_sur_un_jeu_existant(tmp_path, train_dir):
    ui.create_dataset("jeu", [write_png(tmp_path / "in" / "a.png", 300)], train_dir=train_dir, max_upload_mb=5)
    avant = {p.name: p.read_bytes() for p in (train_dir / "jeu").iterdir()}
    bon = make_zip(tmp_path / "bon.zip", {"n.png": png_bytes(300, 7)})
    mauvais = make_zip(tmp_path / "mauvais.zip", {"../evil.png": png_bytes()})
    with pytest.raises(ValueError, match="archive refusée"):
        ui.create_dataset("jeu", [bon, mauvais], train_dir=train_dir, max_upload_mb=5)
    assert {p.name: p.read_bytes() for p in (train_dir / "jeu").iterdir()} == avant
    assert [p.name for p in train_dir.iterdir() if p.name.startswith(".envoi-")] == []
    assert sorted(p.name for p in train_dir.iterdir()) == ["jeu"]


def test_create_dataset_envoi_sans_image_valide_sur_un_jeu_existant(tmp_path, train_dir):
    ui.create_dataset("jeu", [write_png(tmp_path / "in" / "a.png", 300)], train_dir=train_dir, max_upload_mb=5)
    avant = {p.name: p.read_bytes() for p in (train_dir / "jeu").iterdir()}
    faux = tmp_path / "faux.png"
    faux.write_bytes(b"pas une image")
    with pytest.raises(ValueError, match="Aucune image valide"):
        ui.create_dataset("jeu", [faux], train_dir=train_dir, max_upload_mb=5)
    assert {p.name: p.read_bytes() for p in (train_dir / "jeu").iterdir()} == avant
    assert [p.name for p in train_dir.iterdir() if p.name.startswith(".envoi-")] == []


def test_list_datasets_ignore_les_dossiers_caches(train_dir):
    write_png(train_dir / "visible" / "1.png")
    write_png(train_dir / ".envoi-abc123" / "1.png")
    write_png(train_dir / ".cache" / "1.png")
    assert [d["name"] for d in ui.list_datasets(train_dir)] == ["visible"]
    assert [v for _, v in ui.dataset_choices(train_dir)] == ["visible"]


def test_create_dataset_envois_concurrents_valide_et_invalide(tmp_path, train_dir):
    import threading
    valides = [write_png(tmp_path / "ok" / f"v{i}.png", 300, seed=i) for i in range(4)]
    attendus = {p.name: p.read_bytes() for p in valides}
    faux = tmp_path / "ko" / "faux.png"
    faux.parent.mkdir()
    faux.write_bytes(b"pas une image")
    depart = threading.Barrier(2)
    resultats = {}

    def envoi(cle, fichiers):
        depart.wait()
        try:
            resultats[cle] = ui.create_dataset("commun", fichiers, train_dir=train_dir, max_upload_mb=5)
        except ValueError as e:
            resultats[cle] = e

    threads = [threading.Thread(target=envoi, args=("ok", valides)), threading.Thread(target=envoi, args=("ko", [faux]))]
    [t.start() for t in threads]
    [t.join(60) for t in threads]
    assert not any(t.is_alive() for t in threads)
    assert isinstance(resultats["ko"], ValueError)
    folder, _ = resultats["ok"]
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == attendus          # fichiers du valide intacts
    assert sorted(p.name for p in train_dir.iterdir()) == ["commun"]               # aucun .envoi-* restant


# --- suivi avec beaucoup de jobs --------------------------------------------------------------

def beaucoup_de_jobs(store, n=250):
    import sqlite3
    ids = [store.submit(TrainConfig(name=f"j{i}").to_dict()) for i in range(n)]
    con = sqlite3.connect(str(store.path))
    con.execute("UPDATE jobs SET status = 'done'")
    con.execute("UPDATE jobs SET status = 'failed' WHERE id = ?", (ids[0],))
    con.commit()
    con.close()
    return ids


def test_suivi_le_job_en_echec_ancien_reste_visible(store):
    ids = beaucoup_de_jobs(store)
    visibles = ui.visible_jobs(store)
    assert ids[0] in [j.id for j in visibles]
    assert [j.id for j in visibles] == sorted((j.id for j in visibles), reverse=True)
    assert len(visibles) == ui.RECENT_JOBS + 1
    assert ids[0] in list(ui.jobs_table(store)["n°"])
    assert ids[0] in [v for _, v in ui.job_choices(store)]
    assert len(ui.job_choices(store)) == len(visibles) == len(ui.jobs_table(store))


# --- menus rafraîchis par session ------------------------------------------------------------

@pytest.fixture
def codecs_non_charges(monkeypatch):
    """Remet l'état partagé du registre de codecs à « jamais chargé » (restauré à la fin)."""
    monkeypatch.setattr(ui, "_codecs_loaded", None)


CHOIX = [("A", "a"), ("B", "b"), ("C", "c")]


def test_choice_update_garde_la_valeur_courante():
    u = ui.choice_update(CHOIX, "b")
    assert u["choices"] == CHOIX and u["value"] == "b"


@pytest.mark.parametrize("current", ["zzz", None])
def test_choice_update_valeur_disparue_prend_la_premiere(current):
    assert ui.choice_update(CHOIX, current)["value"] == "a"


def test_choice_update_liste_vide_donne_none():
    u = ui.choice_update([], "a")
    assert u["choices"] == [] and u["value"] is None


def test_codecs_stale():
    assert ui.codecs_stale({1}, None) is True
    assert ui.codecs_stale(set(), None) is True
    assert ui.codecs_stale({1, 2}, frozenset({1, 2})) is False
    assert ui.codecs_stale({1, 2}, frozenset({1})) is True
    assert ui.codecs_stale(set(), frozenset({1})) is True


def test_mark_codecs_loaded(codecs_non_charges):
    ui.mark_codecs_loaded([3, 1, 3])
    assert ui._codecs_loaded == frozenset({1, 3})
    assert not ui.codecs_stale({1, 3}, ui._codecs_loaded)


def test_codec_updates_un_seul_rechargement_par_changement(codecs_non_charges):
    appels = {"reload": 0, "menus": 0}

    def reload_fn(m):
        appels["reload"] += 1
        return ("R", m)

    def menus_fn(m):
        appels["menus"] += 1
        return ("M", m)

    assert ui.codec_updates({1}, "gris", reload_fn, menus_fn, 2) == ("R", "gris")
    assert appels == {"reload": 1, "menus": 0}
    assert ui.codec_updates({1}, "rgb", reload_fn, menus_fn, 2) == ("M", "rgb")      # même ensemble : relecture
    assert appels == {"reload": 1, "menus": 1}
    assert ui.codec_updates({1, 2}, "gris", reload_fn, menus_fn, 2) == ("R", "gris")  # nouveau job terminé
    assert appels == {"reload": 2, "menus": 1}


def test_codec_updates_sans_menus_fn_renvoie_des_updates_vides(codecs_non_charges):
    ui.mark_codecs_loaded({1})
    out = ui.codec_updates({1}, "gris", lambda m: pytest.fail("rechargement inattendu"), None, 4)
    assert isinstance(out, tuple) and len(out) == 4
    import gradio as gr
    assert all(o == gr.update() for o in out)


def _fonction(demo, api_name):
    fns = [f for f in demo.fns.values() if f.api_name == api_name]
    assert len(fns) == 1, f"{api_name} : {len(fns)} fonctions enregistrées"
    return fns[0].fn


def test_app_build_enregistre_load_et_select(store, tmp_path, monkeypatch, codecs_non_charges):
    monkeypatch.setattr(config, "TRAIN_DIR", tmp_path / "train_vide")
    from aidge_in_space import app
    demo = app.build()
    api = demo.get_api_info()
    assert "/on_load" in api["named_endpoints"] and "/on_train_tab" in api["named_endpoints"]
    assert _fonction(demo, "on_load") is not None


def test_app_codec_menus_meme_forme_que_reload_codecs(store, tmp_path, monkeypatch):
    from aidge_in_space import app
    from aidge_in_space.codecs import balle2017, discover
    vide = tmp_path / "ck_vide"
    vide.mkdir()
    monkeypatch.setattr(config, "CKPT_DIR", vide)
    monkeypatch.setattr(balle2017, "CKPT_DIR", vide)
    try:
        discover()
        for mode in ("gris", "rgb"):
            assert len(app.codec_menus(mode)) == len(app.reload_codecs(mode)) == 7
    finally:
        monkeypatch.undo()
        discover()


def test_session_neuve_voit_jeu_et_checkpoint_crees_apres_build(store, tmp_path, monkeypatch, codecs_non_charges):
    """Bout en bout sans réseau : build, puis création d'un jeu et d'une série terminée, puis callback load."""
    from aidge_in_space import app
    from aidge_in_space.codecs import balle2017, discover
    from aidge_in_space.training import loop
    train = tmp_path / "train"
    ck = store.path.parent.parent / "ckpts"
    ck.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "TRAIN_DIR", train)
    monkeypatch.setattr(config, "CKPT_DIR", ck)
    monkeypatch.setattr(balle2017, "CKPT_DIR", ck)
    try:
        discover()
        demo = app.build()
        on_load = _fonction(demo, "on_load")
        # avant : ni jeu ni série
        out = on_load("gris", None, None, None)
        assert out[0]["choices"] == [] and out[0]["value"] is None
        assert not any("Plage" in str(c) for c in out[8]["choices"])
        # après build : un jeu et un job terminé avec son checkpoint
        for i in range(3):
            write_png(train / "lune" / f"i{i}.png", size=96, seed=i)
        cfg = TrainConfig(name="Plage", author="Z", train_dir=str(train / "lune"), eval_dir=str(train / "lune"),
                          N=16, crop=64, batch_size=2, n_patches=8, steps=2, log_every=1, eval_every=2,
                          save_every=2, n_eval_images=1, lmbda=2048)
        jid = store.submit(cfg.to_dict())
        job = store.claim_next("wt")
        import torch
        loop.train(cfg, loop.checkpoint_path(cfg, job.id, ck), device=torch.device("cpu"),
                   run_info={"job_id": job.id, "name": "Plage", "author": "Z", "commit": ""})
        store.finish(jid, "done")
        out = on_load("gris", None, None, None)
        assert [v for _, v in out[0]["choices"]] == ["lune"] and out[0]["value"] == "lune"
        assert out[1]["value"] == "lune"                                    # menu de la visionneuse de la session
        assert any(v == jid for _, v in out[4]["choices"])                   # job visible dans le Suivi
        assert out[7] == {jid}
        libelles = [str(c) for upd in out[8:14] for c in (upd.get("choices") or [])]
        assert any("Plage" in c for c in libelles), libelles                 # série dans les menus de codecs
        assert "Plage" in out[14]                                            # tableau Markdown des codecs
        # une seconde session, même ensemble : relecture sans nouvelle découverte
        monkeypatch.setattr(app, "reload_codecs", lambda m: pytest.fail("rechargement inattendu"), raising=True)
        out2 = on_load("gris", None, None, None)
        assert any("Plage" in str(c) for upd in out2[8:14] for c in (upd.get("choices") or []))
    finally:
        monkeypatch.undo()
        discover()
