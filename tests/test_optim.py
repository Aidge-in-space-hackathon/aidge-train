"""Tests de l'optimisation Aidge : contrat de docs/conception_optimisation.md.

Sur CPU, sans réseau, dans des dossiers temporaires, avec un petit modèle généré. Les tests marqués slow quantifient avec
Aidge ou compilent un export : python -m pytest tests/test_optim.py -m "not slow" les exclut.
"""
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import threading

import numpy as np
import pytest
import torch
from PIL import Image

from aidge_in_space import backends, codecs, config, ui_optim, worker
from aidge_in_space.jobs import EXIT_CONFIG, EXIT_DONE, JobStore
from aidge_in_space.nn import Balle2017, freeze
from aidge_in_space.optim import evaluate, results, tiling
from aidge_in_space.optim.config import ConfigError, OptimConfig, n_steps

## Un petit modèle : N réduit, tuile 64 x 64.
SMALL_N = 8
TILE = (3, 64, 64)


def base_config(**kw):
    """Configuration minimale valide, avec des champs à remplacer."""
    d = {"name": "essai", "source": {"checkpoint": "x.pt"}, "tile": list(TILE),
         "ptq": {"calibration_set": "jeu"}, "evaluate": {"n_images": 1}, "export": {"generate": False}}
    d.update(kw)
    return d


def perturbed_model(N=SMALL_N):
    """Petit modèle aux paramètres tous différents : un modèle neuf a des GDN identiques, que l'export ONNX fusionne."""
    torch.manual_seed(N)
    model = Balle2017(3, N, True).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(1 + 1e-3 * torch.randn_like(p))
    return model


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Dossiers temporaires à la place d'outputs et de data, un checkpoint et des images générées."""
    monkeypatch.setattr(config, "OPTIM_DIR", tmp_path / "optim")
    monkeypatch.setattr(config, "TRAIN_DIR", tmp_path / "train")
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(backends, "ONNX_DIR", tmp_path / "onnx")
    rng = np.random.default_rng(0)
    for folder, n in ((tmp_path / "train" / "jeu", 4), (tmp_path / "eval", 2)):
        folder.mkdir(parents=True)
        for i in range(n):
            base = rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)
            Image.fromarray(base).resize((192, 128), Image.BICUBIC).save(folder / f"{i}.png")
    model = perturbed_model()
    ckpt = tmp_path / "bls2017_c3_N8_lambda1024_run1.pt"
    torch.save({"state_dict": model.state_dict(), "cfg": {"in_ch": 3, "N": SMALL_N, "gdn_last": True, "lmbda": 1024}, "step": 10}, ckpt)
    return {"root": tmp_path, "checkpoint": ckpt, "eval": tmp_path / "eval", "model": model}


def run_config(workspace, **kw):
    """Configuration qui pointe sur les données du dossier temporaire."""
    return OptimConfig.from_dict(base_config(source={"checkpoint": str(workspace["checkpoint"])},
                                             ptq={"calibration_set": "jeu", "n_calibration": 4},
                                             evaluate={"eval_dir": str(workspace["eval"]), "n_images": 1}, **kw))


def fake_report(job_id="1", name="serie", lmbda=1024, variants=None, checkpoint="x.pt", sha="s"):
    """Rapport minimal conforme, pour les fonctions pures."""
    rd = {"images": 1, "rate_kind": "réel", "bpp": 0.5, "psnr": 30.0, "ms_ssim": 14.0, "delta_bpp_pct": None, "delta_psnr_db": None,
          "delta_ms_ssim_db": None}
    return {"schema": results.REPORT_SCHEMA, "id": job_id, "name": name, "status": "done", "tile": list(TILE), "bram_bytes": 630000,
            "source": {"checkpoint": checkpoint, "file_sha256": sha, "in_ch": 3, "N": SMALL_N, "gdn_last": True, "lmbda": lmbda, "step": 10},
            "config": {"ptq": {"clipping": "MAX", "per_channel": False}},
            "series": {"indicative": False, "reason": None},
            "variants": variants if variants is not None else {"fp32": {"status": "ok", "rd": rd, "reason": None},
                                                               "mixte": {"status": "ok", "rd": dict(rd, bpp=0.51), "reason": None}}}


# ------------------------------------------------------------------ configuration

def test_config_valide_et_aller_retour_yaml(tmp_path):
    cfg = OptimConfig.from_dict(base_config())
    path = tmp_path / "c.yaml"
    cfg.to_yaml(path)
    assert OptimConfig.from_yaml(path).to_dict() == cfg.to_dict()


def test_config_ajoute_toujours_fp32():
    assert OptimConfig.from_dict(base_config(recipes=["mixte"])).recipes == ["fp32", "mixte"]


@pytest.mark.parametrize("change", [
    {"inconnu": 1},
    {"tile": [3, 64, 60]},
    {"tile": [2, 64, 64]},
    {"tile": [3, 8, 8]},
    {"recipes": []},
    {"recipes": ["int4"]},
    {"source": {"checkpoint": ""}},
    {"source": {"checkpoint": "x", "autre": 1}},
    {"ptq": {"calibration_set": "jeu", "bits": 16}},
    {"ptq": {"calibration_set": "jeu", "clipping": "AA"}},
    {"ptq": {"calibration_set": "jeu", "n_calibration": 0}},
    {"ptq": {"calibration_set": ""}},
    {"ptq": {"calibration_set": "jeu", "float_ops": ["Conv2D"]}},
    {"evaluate": {"n_images": 25}},
    {"export": {"generate": "oui"}},
    {"name": "x" * 61},
])
def test_config_refuse(change):
    with pytest.raises(ConfigError):
        OptimConfig.from_dict(base_config(**change))


def test_config_sections_obligatoires():
    with pytest.raises(ConfigError):
        OptimConfig.from_dict({"name": "x"})


def test_config_du_job_se_relit_sans_les_colonnes_de_la_base():
    cfg = OptimConfig.from_dict(base_config())
    job = cfg.to_job(3, 64, 1024, n_steps(cfg))
    assert (job["in_ch"], job["N"], job["lmbda"]) == (3, 64, 1024)
    assert OptimConfig.from_job(job).to_dict() == cfg.to_dict()


def test_module_de_config_sans_torch_ni_aidge():
    code = ("import sys; import aidge_in_space.optim.config, aidge_in_space.optim.results; "
            "bad = [m for m in ('torch', 'aidge_core', 'aidge_quantization') if m in sys.modules]; print(bad); sys.exit(bool(bad))")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(config.ROOT))
    assert r.returncode == 0, r.stdout + r.stderr


# ------------------------------------------------------------------ fonctions pures

def test_detection_sans_effet_un_seul_signal_suffit():
    from aidge_in_space.optim.ptq import no_effect_signals
    a, b = {"Conv": 1}, {"Conv": 1, "Quantizer": 2}
    assert no_effect_signals(True, a, b, False)["no_effect"] is False
    assert no_effect_signals(False, a, b, False)["no_effect"] is True
    assert no_effect_signals(True, a, a, False)["no_effect"] is True
    assert no_effect_signals(True, a, b, True)["no_effect"] is True


def test_fidelite_donne_les_symboles_non_nuls():
    ref = np.array([0.0, 0.0, 0.0, 2.0, 3.0])
    out = evaluate.fidelity([ref], [np.array([0.0, 0.0, 0.0, 2.0, 5.0])])
    assert out["symbols_identical_pct"] == 80.0
    assert out["nonzero_share_pct"] == 40.0
    assert out["nonzero_identical_pct"] == 50.0
    assert out["max_rounded_error"] == 2


def test_fidelite_sans_symbole_non_nul():
    assert evaluate.fidelity([np.zeros(3)], [np.zeros(3)])["nonzero_identical_pct"] is None


def _levels(fp32=(0.2, 0.4, 0.6, 0.8), psnr=(28, 30, 32, 34), factor=1.03):
    return [{"lmbda": i + 1, "fp32": {"bpp": b, "psnr": p}, "variant": {"bpp": b * factor, "psnr": p}}
            for i, (b, p) in enumerate(zip(fp32, psnr))]


def test_bd_rate_quatre_niveaux_monotones():
    out = evaluate.bd_series(_levels())
    assert out["bd_rate"] == pytest.approx(3.0, abs=1e-6)
    assert out["reason"] is None and len(out["points"]) == 4


def test_bd_rate_moins_de_quatre_niveaux_garde_les_ecarts():
    out = evaluate.bd_series(_levels()[:3])
    assert out["bd_rate"] is None and "3 niveaux sur 4" in out["reason"]
    assert out["points"][0]["delta_bpp_pct"] == pytest.approx(3.0)


@pytest.mark.parametrize("fp32,psnr", [((0.2, 0.5, 0.4, 0.8), (28, 30, 32, 34)), ((0.2, 0.4, 0.6, 0.8), (28, 31, 30, 34))])
def test_bd_rate_refuse_une_courbe_non_monotone(fp32, psnr):
    out = evaluate.bd_series(_levels(fp32, psnr))
    assert out["bd_rate"] is None and "non monotone" in out["reason"]
    assert len(out["points"]) == 4


def test_ecarts_face_au_fp32():
    rd = evaluate.add_deltas({"bpp": 1.1, "psnr": 29.0, "ms_ssim": None}, {"bpp": 1.0, "psnr": 30.0, "ms_ssim": 14.0})
    assert rd["delta_bpp_pct"] == pytest.approx(10.0)
    assert rd["delta_psnr_db"] == pytest.approx(-1.0)


def test_decoupage_en_tuiles():
    x = torch.rand(1, 3, 70, 100)
    xp = tiling.pad_to_tiles(x, 64, 64)
    assert xp.shape[-2:] == (128, 128)
    assert torch.equal(xp[..., :70, :100], x)
    down = lambda t: t[:, :, ::16, ::16]
    y = tiling.encode_tiles(lambda a: down(torch.from_numpy(a)).numpy(), xp, 64, 64)
    assert y.shape == (1, 3, 8, 8)
    assert torch.equal(tiling.decode_tiles(lambda t: t, y, 4, 4), y)


def test_analyse_du_rapport_de_quantifiabilite():
    pytest.importorskip("aidge_quantization")
    from aidge_in_space.optim.graph import parse_quantizability
    out = parse_quantizability("- [ok] Conv2D x3\n- [?] Sqrt x3\n")
    assert out["quantizable"] is False
    assert out["operators"][1] == {"type": "Sqrt", "count": 3, "state": "inconnu"}
    assert parse_quantizability("- [ok] Conv2D x3")["quantizable"] is True
    assert parse_quantizability("rien")["quantizable"] is False


def test_comparaison_memoire_a_la_bram():
    pytest.importorskip("aidge_quantization")
    from aidge_in_space.optim.memory import summarize
    # N=64, tuile 64, mixte : valeurs de la conception
    m = summarize(196608, 271040, 221120)
    assert m["bram_bytes"] == config.BRAM_BYTES == 630_000
    assert m["total_fits_bram"] and m["fits_bram"] and m["conv_weights_fit_bram"]
    assert m["total_bram_ratio"] == pytest.approx(0.742, abs=1e-3)
    assert not summarize(393216, 1049984, 851840)["total_fits_bram"]


# ------------------------------------------------------------------ jobs et worker

V1_SCHEMA = """CREATE TABLE jobs (id INTEGER PRIMARY KEY AUTOINCREMENT, status TEXT NOT NULL DEFAULT 'queued', name TEXT NOT NULL,
author TEXT NOT NULL DEFAULT '', config_json TEXT NOT NULL, in_ch INTEGER NOT NULL, n INTEGER NOT NULL, lmbda REAL NOT NULL,
steps INTEGER NOT NULL, last_step INTEGER NOT NULL DEFAULT 0, cancel_requested INTEGER NOT NULL DEFAULT 0, attempts INTEGER NOT NULL DEFAULT 0,
worker TEXT, ckpt_path TEXT, error TEXT, created_at REAL NOT NULL, queued_at REAL NOT NULL, started_at REAL, finished_at REAL, heartbeat_at REAL);
CREATE TABLE metrics (id INTEGER PRIMARY KEY AUTOINCREMENT, job_id INTEGER NOT NULL, step INTEGER NOT NULL, kind TEXT NOT NULL,
data_json TEXT NOT NULL, created_at REAL NOT NULL);
CREATE TABLE workers (name TEXT PRIMARY KEY, pid INTEGER NOT NULL, host TEXT NOT NULL, job_id INTEGER, started_at REAL NOT NULL, seen_at REAL NOT NULL);
PRAGMA user_version = 1;
"""


def _v1_database(path):
    con = sqlite3.connect(path)
    con.executescript(V1_SCHEMA)
    con.execute("INSERT INTO jobs (name, config_json, in_ch, n, lmbda, steps, created_at, queued_at) VALUES ('ancien', '{}', 1, 8, 1, 1, 0, 0)")
    con.commit()
    con.close()


def _job_config(name="essai"):
    cfg = OptimConfig.from_dict(base_config(name=name))
    return cfg.to_job(3, SMALL_N, 1024, n_steps(cfg))


def test_migration_de_la_table_jobs(tmp_path):
    db = tmp_path / "runs.sqlite"
    _v1_database(db)
    store = JobStore(db)
    assert store.get(1).kind == "train" and store.get(1).name == "ancien"
    assert sqlite3.connect(db).execute("PRAGMA user_version").fetchone()[0] == 2
    new = store.submit(_job_config(), kind="optim")
    assert store.get(new).kind == "optim"
    assert JobStore(db).get(1).kind == "train"     # seconde ouverture : rien à migrer


def test_migrations_concurrentes(tmp_path):
    db = tmp_path / "runs.sqlite"
    _v1_database(db)
    errors = []

    def open_store():
        try:
            JobStore(db)
        except Exception as e:
            errors.append(e)
    threads = [threading.Thread(target=open_store) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
    assert JobStore(db).get(1).kind == "train"


def test_base_plus_recente_refusee(tmp_path):
    db = tmp_path / "runs.sqlite"
    JobStore(db)
    con = sqlite3.connect(db)
    con.execute("PRAGMA user_version = 99")
    con.commit()
    con.close()
    with pytest.raises(RuntimeError):
        JobStore(db)


def test_base_neuve_a_le_type_de_job(tmp_path):
    store = JobStore(tmp_path / "runs.sqlite")
    job_id = store.submit({"name": "e", "author": "", "in_ch": 1, "N": 8, "lmbda": 1, "steps": 1})
    assert store.get(job_id).kind == "train"
    with pytest.raises(ValueError):
        store.submit({"name": "e", "author": "", "in_ch": 1, "N": 8, "lmbda": 1, "steps": 1}, kind="autre")


def test_le_worker_choisit_la_commande_selon_le_type(tmp_path):
    store = JobStore(tmp_path / "runs.sqlite")
    train = store.get(store.submit({"name": "e", "author": "", "in_ch": 1, "N": 8, "lmbda": 1, "steps": 1}))
    optim = store.get(store.submit(_job_config(), kind="optim"))
    assert worker.job_command(train) == worker.TRAIN_CMD
    assert worker.job_command(optim) == worker.OPTIM_CMD
    assert worker.job_command(optim, ["x"]) == ["x"]


def test_le_worker_execute_un_job_d_optimisation(tmp_path, monkeypatch):
    """Commande factice : le worker lance la commande du type et fixe le statut d'après le code de retour."""
    store = JobStore(tmp_path / "runs.sqlite")
    job_id = store.submit(_job_config(), kind="optim")
    marker = tmp_path / "lance.txt"
    code = f"open({str(marker)!r}, 'w').write('optim')"
    monkeypatch.setattr(worker, "OPTIM_CMD", [sys.executable, "-c", code])
    monkeypatch.setattr(worker, "TRAIN_CMD", [sys.executable, "-c", "raise SystemExit(1)"])
    assert worker.run_worker(store, poll=0.05, once=True, stop_event=threading.Event()) == 1
    assert marker.read_text() == "optim"
    assert store.get(job_id).status == "done"


def test_la_cli_refuse_un_job_d_entrainement(tmp_path):
    from aidge_in_space import optimize
    db = tmp_path / "runs.sqlite"
    store = JobStore(db)
    job_id = store.submit({"name": "e", "author": "", "in_ch": 1, "N": 8, "lmbda": 1, "steps": 1})
    store.claim_next("w")
    assert optimize.main(["--job-id", str(job_id), "--db", str(db)]) == EXIT_CONFIG


def test_la_cli_refuse_une_config_invalide(tmp_path):
    from aidge_in_space import optimize
    bad = tmp_path / "c.yaml"
    bad.write_text("name: x\ninconnu: 1\n", encoding="utf-8")
    assert optimize.main(["--config", str(bad)]) == EXIT_CONFIG
    assert optimize.main(["--config", str(tmp_path / "absent.yaml")]) == EXIT_CONFIG


# ------------------------------------------------------------------ résultats, plugin, interface

def _write_report(root, report, name=None):
    folder = root / (name or str(report["id"]))
    folder.mkdir(parents=True)
    (folder / results.REPORT_NAME).write_text(json.dumps(report), encoding="utf-8")
    return folder


def test_les_rapports_inacheves_ou_illisibles_sont_ignores(tmp_path, capsys):
    _write_report(tmp_path, fake_report("1"))
    _write_report(tmp_path, fake_report("2"), name=f"2{results.TMP_MARK}123")
    (tmp_path / "3").mkdir()
    (tmp_path / "3" / results.REPORT_NAME).write_text("{pas du json", encoding="utf-8")
    assert [r["id"] for r in results.read_reports(tmp_path)] == ["1"]
    assert "ignoré" in capsys.readouterr().out


def test_le_plugin_decouvre_les_variantes_utilisables(tmp_path, monkeypatch):
    from aidge_in_space.codecs import aidge_variants
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"poids")
    sha = results.file_sha256(ckpt)
    monkeypatch.setattr(config, "OPTIM_DIR", tmp_path / "optim")
    variants = {"fp32": {"status": "ok"}, "int8": {"status": "sans_effet"}, "mixte": {"status": "ok"}}
    _write_report(tmp_path / "optim", fake_report("1", variants=variants, checkpoint=str(ckpt), sha=sha, lmbda=2048))
    _write_report(tmp_path / "optim", fake_report("2", variants=variants, checkpoint=str(ckpt), sha=sha, lmbda=1024))
    found = {c.recipe: c for c in aidge_variants.codecs()}
    assert set(found) == {"fp32", "mixte"}     # une recette sans effet n'est pas un codec
    codec = found["mixte"]
    assert codec.key == f"aidge-serie-c3-n{SMALL_N}-mixte-t64"
    assert codec.backends == ("aidge",)
    assert [l.label.split(" ")[0] for l in codec.levels("rgb")] == ["λ=1024", "λ=2048"]     # du plus bas au plus haut débit
    codecs.discover()      # le registre accepte le plugin
    assert codecs.get(codec.key).label == codec.label
    monkeypatch.undo()
    codecs.discover()


def test_le_plugin_refuse_un_checkpoint_modifie(tmp_path, monkeypatch):
    pytest.importorskip("aidge_core")
    from aidge_in_space.codecs import aidge_variants
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"poids")
    monkeypatch.setattr(config, "OPTIM_DIR", tmp_path / "optim")
    _write_report(tmp_path / "optim", fake_report("1", variants={"mixte": {"status": "ok"}}, checkpoint=str(ckpt), sha=results.file_sha256(ckpt)))
    codec = aidge_variants.codecs()[0]
    assert codec.available()[0] is True
    ckpt.write_bytes(b"autres poids")
    ok, why = codec.available()
    assert ok is False and "modifié" in why
    assert codec.levels("rgb") == []


def test_tableau_des_variantes_et_alerte_sans_effet():
    variants = {"fp32": {"status": "ok", "memory": {"peak_bytes": 393216, "weights_bytes": 1024, "conv_weights_bytes": 512},
                         "rd": {"delta_bpp_pct": None, "delta_psnr_db": None}},
                "int8": {"status": "sans_effet", "reason": "Sqrt"},
                "mixte": {"status": "echec", "reason": "Seuls 50 % des symboles", "export": {"status": "divergent", "symbols_identical_pct": 50.0}}}
    reports = [fake_report(variants=variants)]
    table = ui_optim.variants_table(reports)
    assert list(table.columns) == results.TABLE_COLUMNS and len(table) == 3
    assert table.iloc[0]["pic / BRAM"] == "62 %"
    assert table.iloc[1]["statut"] == "SANS EFFET"
    assert table.iloc[2]["identiques Aidge (%)"] == 50.0
    text = ui_optim.alerts_markdown(reports)
    assert "ne sait pas quantifier la GDN" in text and "Seuls 50 %" in text
    assert ui_optim.alerts_markdown([]) == ""


def test_bd_rate_de_l_interface_et_serie_indicative():
    reports = []
    for lam, bpp, psnr in ((1, 0.2, 28), (2, 0.4, 30), (3, 0.6, 32), (4, 0.8, 34)):
        rd = {"bpp": bpp, "psnr": psnr}
        r = fake_report(str(lam), lmbda=lam, variants={"fp32": {"status": "ok", "rd": rd},
                                                       "mixte": {"status": "ok", "rd": {"bpp": bpp * 1.03, "psnr": psnr}}})
        r["series"]["indicative"] = True
        reports.append(r)
    bd, points = ui_optim.bd_tables(reports)
    assert bd.iloc[0]["BD-rate (%)"] == pytest.approx(3.0)
    assert bd.iloc[0]["indicative"] == "oui"
    assert len(points) == 4
    bd2, _ = ui_optim.bd_tables(reports[:2])
    assert bd2.iloc[0]["BD-rate (%)"] is None or np.isnan(bd2.iloc[0]["BD-rate (%)"])
    assert "2 niveaux sur 4" in bd2.iloc[0]["raison"]


SOURCES = [{"label": "a", "checkpoint": "a.pt", "codec": "c", "in_ch": 3, "N": 64, "lmbda": 2048},
           {"label": "b", "checkpoint": "b.pt", "codec": "c", "in_ch": 3, "N": 64, "lmbda": 1024}]


def _launch(store, name="essai", checkpoints=("a.pt", "b.pt"), tile=64, recipes=("mixte",), calib="jeu"):
    return ui_optim.launch_jobs(store, name, "", list(checkpoints), tile, list(recipes), "MAX", False, 8, calib, 2, True, True, sources=SOURCES)


def test_lancer_depose_un_job_par_checkpoint_du_plus_bas_debit(tmp_path):
    store = JobStore(tmp_path / "runs.sqlite")
    ids = _launch(store)
    jobs = [store.get(i) for i in ids]
    assert [j.lmbda for j in jobs] == [1024, 2048]
    assert all(j.kind == "optim" and j.N == 64 and j.steps > 1 for j in jobs)
    assert OptimConfig.from_job(jobs[0].config).recipes == ["fp32", "mixte"]


@pytest.mark.parametrize("kw,message", [({"name": " "}, "nom"), ({"checkpoints": ()}, "checkpoint"),
                                        ({"checkpoints": ("a.pt", "inconnu.pt")}, "inconnu"), ({"calib": ""}, "calibration"),
                                        ({"tile": 60}, "tile")])
def test_lancer_tout_ou_rien(tmp_path, kw, message):
    store = JobStore(tmp_path / "runs.sqlite")
    with pytest.raises(ValueError, match=message):
        _launch(store, **kw)
    assert store.list_jobs() == []


def test_mixte_est_la_recette_par_defaut():
    from aidge_in_space.optim.config import DEFAULT_RECIPE
    assert DEFAULT_RECIPE == "mixte"


def test_compte_rendu_de_l_analyse():
    mem = lambda p, w, c: {"peak_bytes": p, "weights_bytes": w, "conv_weights_bytes": c, "bram_bytes": 630000}
    text = ui_optim.analysis_markdown({
        "source": {"N": 64, "lmbda": 1024}, "tile": [3, 64, 64],
        "quantizability": {"quantizable": False, "operators": [{"type": "Sqrt", "count": 3, "state": "inconnu"}]},
        "memory_fp32": mem(196608, 932096, 882176), "memory_mixte": mem(196608, 271040, 221120)})
    assert "Sqrt" in text and "74 %" in text and "35 %" in text


def test_analyse_lit_la_derniere_ligne_et_rapporte_les_erreurs(tmp_path):
    assert ui_optim.parse_analysis('bruit\n{"a": 1}\n') == {"a": 1}
    with pytest.raises(ValueError):
        ui_optim.parse_analysis("")
    with pytest.raises(ValueError):
        ui_optim.parse_analysis("pas du json")
    cfg = OptimConfig.from_dict(base_config())
    fail = [sys.executable, "-c", "import sys; print('la raison', file=sys.stderr); sys.exit(2)"]
    with pytest.raises(ValueError, match="la raison"):
        ui_optim.analyze(cfg, cmd=fail)
    ok = [sys.executable, "-c", "print('{\"b\": 2}')"]
    assert ui_optim.analyze(cfg, cmd=ok) == {"b": 2}


def test_job_d_optimisation_dans_l_onglet_suivre(tmp_path, monkeypatch):
    from aidge_in_space import ui_training
    monkeypatch.setattr(config, "OPTIM_DIR", tmp_path / "optim")
    store = JobStore(tmp_path / "runs.sqlite")
    job_id = store.submit(_job_config(), kind="optim")
    table = ui_training.jobs_table(store)
    assert table.iloc[0]["type"] == "Aidge"
    assert "Aidge" in ui_training.job_choices(store)[0][0]
    figure, _, summary = ui_training.job_details(store, job_id)
    assert figure is None and "Optimisation Aidge" in summary and "Rapport pas encore disponible" in summary
    with pytest.raises(ValueError):
        ui_training.duplicate_config(store, job_id)
    _write_report(tmp_path / "optim", fake_report(str(job_id), variants={"int8": {"status": "sans_effet", "reason": "Sqrt"}}))
    assert "SANS EFFET" in ui_training.job_details(store, job_id)[2]


def test_interface_construite_avec_l_onglet():
    from aidge_in_space import app
    assert app.build() is not None


# ------------------------------------------------------------------ générateur C : fonctions sans graphe

def test_virgule_fixe():
    pytest.importorskip("aidge_quantization")
    from aidge_in_space.optim.export_c import ExportCError, fixed_point
    # Valeurs du spike
    assert fixed_point(9.045691695e-04) == (1989168640, 41)
    for scale in (1e-6, 0.3, 0.5, 0.999, 1.5):
        m, n = fixed_point(scale)
        assert m < 2 ** 31 and n >= 1
        assert m / 2 ** n == pytest.approx(scale, rel=1e-8)
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ExportCError):
            fixed_point(bad)


# ------------------------------------------------------------------ Aidge : tests lents

@pytest.fixture
def small_setup(workspace):
    """Petit encodeur, tuiles de calibration et de contrôle."""
    pytest.importorskip("aidge_quantization")
    from aidge_in_space.optim import ptq
    enc = freeze(workspace["model"].g_a).eval()
    calib = ptq.calibration_tiles(config.TRAIN_DIR / "jeu", 4, TILE, 0)
    control = ptq.calibration_tiles(workspace["eval"], 2, TILE, 1)
    return {**workspace, "encoder": enc, "calibration": calib, "control": control}


@pytest.mark.slow
def test_quantification_sans_effet_detectee(small_setup):
    from aidge_in_space.optim import ptq
    out = ptq.try_int8(small_setup["encoder"], TILE, small_setup["calibration"])
    assert out["status"] == ptq.STATUS_NO_EFFECT and out["graph"] is None
    assert out["signals"]["no_effect"] and not out["signals"]["check_architecture"]
    assert "Sqrt" in out["reason"]


@pytest.mark.slow
def test_quantifiabilite_signale_sqrt(small_setup):
    from aidge_in_space.optim import graph
    q = graph.quantizability(graph.import_encoder(small_setup["encoder"], TILE))
    assert not q["quantizable"]
    assert {"type": "Sqrt", "count": 3, "state": "inconnu"} in q["operators"]


@pytest.mark.slow
def test_recette_mixte_entiere_et_deterministe(small_setup):
    from aidge_in_space.optim import graph, ptq
    enc, calib, control = small_setup["encoder"], small_setup["calibration"], small_setup["control"]
    g = ptq.build_mixed(enc, TILE, calib)
    nodes = graph.count_nodes(g)
    assert nodes["Quantizer"] == 6 and nodes["Sqrt"] == 3 and "Dequantizer" not in nodes
    dtypes = {n.name(): str(n.get_operator().get_output(0).dtype) for n in g.get_nodes() if n.type() == "Producer"}
    assert sum("int8" in d for d in dtypes.values()) == 3     # les poids des trois convolutions
    a = graph.Runner(g)(control[:1])
    b = graph.Runner(ptq.build_mixed(enc, TILE, calib))(control[:1])
    assert np.array_equal(a, b)


@pytest.mark.slow
def test_plan_memoire_du_vrai_plan(small_setup):
    from aidge_in_space.optim import graph, memory, ptq
    enc, calib = small_setup["encoder"], small_setup["calibration"]
    fp32 = memory.memory_report(graph.import_encoder(enc, TILE), calib[:1])
    mixed = memory.memory_report(ptq.build_mixed(enc, TILE, calib), calib[:1])
    assert fp32["peak_bytes"] > 0 and mixed["weights_bytes"] < fp32["weights_bytes"]
    assert mixed["conv_weights_bytes"] * 4 == pytest.approx(fp32["conv_weights_bytes"], rel=0.05)


@pytest.mark.slow
def test_generateur_c_identique_a_aidge(small_setup):
    if shutil.which("gcc") is None:
        pytest.skip("pas de compilateur C")
    from aidge_in_space.optim import export_c, graph, ptq
    enc, calib, control = small_setup["encoder"], small_setup["calibration"], small_setup["control"]
    g = ptq.build_mixed(enc, TILE, calib)
    run = graph.Runner(g)
    run(calib[:1])
    out = export_c.export_locod(g, small_setup["root"] / "locod", name="essai")
    source = out.path.read_text(encoding="utf-8")
    assert out.accelerators == 3
    assert source.count("FPGA(") == 3, "locod.sh repère les accélérateurs par recherche de texte"
    local = [l for l in source.splitlines() if l.startswith('#include "')]
    assert local == ['#include "locod.h"'], "fichier unique : pas d'autre inclusion locale"
    fpga_part = source[:source.index("#ifndef LOCOD_FPGA", source.index("void conv"))]
    assert "float" not in fpga_part.split("void conv1", 1)[1], "pas de flottant dans les accélérateurs"
    binary, error = export_c.compile_native(out.path, "gcc")
    assert binary is not None, error
    for t in control:
        layers = export_c.run_native(binary, t[None], out)
        np.testing.assert_array_equal(layers[out.layers[-1][0]][None], run(t[None]))


@pytest.mark.slow
def test_generateur_c_refuse_un_operateur_inconnu(small_setup, monkeypatch):
    from aidge_in_space.optim import export_c, ptq
    g = ptq.build_mixed(small_setup["encoder"], TILE, small_setup["calibration"])
    monkeypatch.delitem(export_c.EMITTERS, "Sqrt")
    with pytest.raises(export_c.ExportCError, match="Sqrt"):
        export_c.export_locod(g, small_setup["root"] / "x")


@pytest.mark.slow
def test_generateur_c_refuse_un_graphe_flottant_entier(small_setup):
    from aidge_in_space.optim import export_c, graph
    g = graph.import_encoder(small_setup["encoder"], TILE)
    with pytest.raises(export_c.ExportCError):
        export_c.export_locod(g, small_setup["root"] / "x")


@pytest.mark.slow
def test_pipeline_sans_compilateur_export_seul(workspace, monkeypatch):
    """De bout en bout, sans compiler : FP32, INT8 sans effet, mixte ; rapport, zip, puis découverte par le plugin."""
    pytest.importorskip("aidge_quantization")
    from aidge_in_space.codecs import aidge_variants
    from aidge_in_space.optim import export, pipeline
    monkeypatch.setattr(export, "find_compiler", lambda: None)
    monkeypatch.setattr(pipeline.shutil, "which", lambda name: None)
    cfg = run_config(workspace, name="bout-en-bout", recipes=["fp32", "int8", "mixte"], export={"generate": True, "compile": True})
    stages = []
    out = pipeline.run(cfg, job_id=7, on_stage=lambda k, s: stages.append((k, s)))
    assert out == config.OPTIM_DIR / "7"
    assert stages[-1][0] == n_steps(cfg), "le compteur d'étapes doit finir sur le total prévu"
    report = json.loads((out / results.REPORT_NAME).read_text(encoding="utf-8"))
    assert report["status"] == "done" and report["source"]["N"] == SMALL_N
    v = report["variants"]
    assert v["int8"]["status"] == "sans_effet" and v["int8"]["signals"]["no_effect"]
    assert v["fp32"]["status"] == v["mixte"]["status"] == "ok"
    for recipe in ("fp32", "mixte"):
        assert v[recipe]["export"]["status"] == "non_compile"
        assert "compilateur" in v[recipe]["export"]["message"]
        assert (out / v[recipe]["export"]["zip"]).is_file()
    assert v["mixte"]["rd"]["delta_bpp_pct"] is not None and v["fp32"]["rd"]["delta_bpp_pct"] is None
    assert v["mixte"]["locod"]["status"] == "non_compile" and (out / v["mixte"]["locod"]["path"]).is_file()
    assert (out / "calib.npz").is_file() and (out / "config.yaml").is_file()
    assert not (config.ROOT / "exported_graph.mmd").exists(), "l'export ne doit rien laisser à la racine du dépôt"
    assert not list(config.OPTIM_DIR.glob(f"*{results.TMP_MARK}*"))
    # La variante est utilisable comme codec, et le débit réel se décode exactement
    codec = next(c for c in aidge_variants.codecs() if c.recipe == "mixte")
    assert codec.available()[0], codec.available()[1]
    img = np.asarray(Image.open(next(workspace["eval"].glob("*.png"))).convert("RGB"))
    res = codec.run(img, codec.levels("rgb")[0], "rgb")
    assert res.info["décodage exact"] is True and res.recon.shape == img.shape


@pytest.mark.slow
def test_pipeline_arret_demande_ne_laisse_rien(workspace):
    pytest.importorskip("aidge_quantization")
    from aidge_in_space.optim import pipeline
    calls = []

    def stop_after_a_while():
        calls.append(1)
        return len(calls) > 3
    with pytest.raises(pipeline.Stopped):
        pipeline.run(run_config(workspace), job_id=8, should_stop=stop_after_a_while)
    assert not config.OPTIM_DIR.exists() or not list(config.OPTIM_DIR.iterdir())


def test_pipeline_donnees_absentes(workspace):
    pytest.importorskip("aidge_quantization")
    from aidge_in_space.optim import pipeline
    with pytest.raises(ConfigError, match="checkpoint"):
        pipeline.run(OptimConfig.from_dict(base_config(source={"checkpoint": str(workspace["root"] / "absent.pt")})))
    with pytest.raises(ConfigError, match="calibration"):
        pipeline.run(OptimConfig.from_dict(base_config(source={"checkpoint": str(workspace["checkpoint"])},
                                                       ptq={"calibration_set": "absent"})))
    bad_tile = run_config(workspace, tile=[1, 64, 64])
    with pytest.raises(ConfigError, match="tile"):
        pipeline.run(bad_tile)


@pytest.mark.slow
def test_export_cpp_compile_et_verifie(small_setup):
    """Export C++ réel : binaire identique au graphe Aidge, et seuil bloquant si on le fausse."""
    from aidge_in_space.optim import export, graph, ptq
    compiler = export.find_compiler()
    if compiler is None:
        pytest.skip("pas de g++")
    enc, calib, control = small_setup["encoder"], small_setup["calibration"], small_setup["control"]
    folder = small_setup["root"] / "cpp"
    manifest = export.generate(ptq.build_mixed(enc, TILE, calib), folder, control[:1])
    binary, error = export.compile_export(folder, manifest, compiler)
    assert binary is not None, error
    run = graph.Runner(ptq.build_mixed(enc, TILE, calib))
    reference = backends.get_runner("torch", enc, (1, *TILE))
    ok = export.verify(binary, control, run, reference)
    assert ok["ok"] and ok["symbols_identical_pct"] == 100.0
    # Latents arbitrairement décalés, loin de zéro : la vérification doit échouer
    wrong = export.verify(binary, control, lambda t: run(t) + 7.0, reference)
    assert not wrong["ok"] and wrong["symbols_identical_pct"] < export.EXPORT_MIN_IDENTICAL_PCT
    layers = export.conv_layer_outputs(binary, folder)
    assert set(layers) == {"conv1", "conv2", "conv3", "gdn1", "gdn2", "gdn3"}
    archive = export.make_zip(folder, small_setup["root"] / "cpp.zip")
    import zipfile
    names = zipfile.ZipFile(archive).namelist()
    assert "dnn/include/forward.hpp" in names and not any(n.startswith(export.VERIF_DIR) for n in names)


def test_compilation_interrompue(tmp_path):
    """Une demande d'arrêt pendant la compilation tue le compilateur."""
    pytest.importorskip("aidge_quantization")
    from aidge_in_space.optim import export
    (tmp_path / "a.cpp").write_text("int main() { return 0; }\n", encoding="utf-8")

    class Manifest:
        inputs = [{"dtype": "float", "dims": [1, 1]}]
        outputs = [{"dtype": "float", "dims": [1, 1]}]
    sleeper = tmp_path / "lent.py"
    sleeper.write_text("import time; time.sleep(60)\n", encoding="utf-8")
    wrapper = tmp_path / ("cc.bat" if os.name == "nt" else "cc.sh")
    wrapper.write_text(f"#!/bin/sh\nexec {sys.executable} {sleeper}\n", encoding="utf-8")
    wrapper.chmod(0o755)
    with pytest.raises(export.CompileStopped):
        export.compile_export(tmp_path, Manifest, str(wrapper), should_stop=lambda: True)


def test_cli_analyse_et_codes_de_retour_constants():
    assert (EXIT_DONE, EXIT_CONFIG) == (0, 2)
