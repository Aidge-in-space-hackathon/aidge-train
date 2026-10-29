"""Tests du banc d'essai : python -m pytest tests

Chaque codec enregistré est testé automatiquement. Les modèles pré-entraînés ne sont testés
que si leurs poids sont déjà dans outputs/weights (pas de téléchargement pendant les tests).
"""
import ast
from pathlib import Path

import numpy as np
import pytest
import torch

from aidge_in_space import backends
from aidge_in_space.codecs import LearnedCodec, all_codecs, bmshj2018
from aidge_in_space.config import WEIGHTS_DIR
from aidge_in_space.entropy import decode_symbols, encode_symbols, make_tables
from aidge_in_space import bench
from aidge_in_space.metrics import MS_SSIM_MIN_SIDE, bd_rate, ms_ssim, psnr
from aidge_in_space.nn import GDN, FactorizedPrior2018, freeze


def make_image(mode, H=96, W=128):
    # dégradé + texture : compressible mais pas trivial
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:H, 0:W]
    base = (xx * 255 / W * 0.6 + yy * 255 / H * 0.4)
    img = np.clip(base + rng.normal(0, 12, (H, W)), 0, 255).astype(np.uint8)
    return img if mode == "gris" else np.stack([img, np.roll(img, 7, 1), 255 - img], axis=-1)


def _weights_ready(codec, level):
    if isinstance(codec, bmshj2018.Pretrained2018):
        return (WEIGHTS_DIR / bmshj2018.URLS[codec.key][level.value]).exists()
    return True


def _cases():
    for codec in all_codecs():
        if not codec.available()[0]:
            continue
        for mode in codec.modes:
            yield pytest.param(codec, mode, id=f"{codec.key}-{mode}")


@pytest.mark.parametrize("codec,mode", list(_cases()))
def test_codec_roundtrip(codec, mode):
    levels = codec.levels(mode)
    assert levels, "un codec doit proposer au moins un niveau"
    level = levels[0]
    if not _weights_ready(codec, level):
        pytest.skip("poids pré-entraînés non téléchargés")
    img = make_image(mode)
    res = codec.run(img, level, mode, real_rate=True)
    assert res.recon.shape == img.shape and res.recon.dtype == np.uint8
    assert res.bits > 0 and res.rate_kind == "réel"
    assert psnr(img, res.recon) > 10
    if isinstance(codec, LearnedCodec):
        assert res.info["décodage exact"]
        est = codec.run(img, level, mode, real_rate=False)
        # débit réel = estimé + en-tête (13 octets au plus) + arrondi des fréquences
        assert abs(res.bits - est.bits) < 0.05 * est.bits + 8 * 16


@pytest.mark.parametrize("codec,mode", list(_cases()))
def test_levels_increase_rate(codec, mode):
    levels = codec.levels(mode)
    if len(levels) < 2 or not _weights_ready(codec, levels[0]) or not _weights_ready(codec, levels[-1]):
        pytest.skip("un seul niveau ou poids absents")
    img = make_image(mode)
    lo = codec.run(img, levels[0], mode, real_rate=False).bits
    hi = codec.run(img, levels[-1], mode, real_rate=False).bits
    assert lo < hi, "les niveaux doivent aller du plus bas au plus haut débit"


@pytest.mark.parametrize("backend", backends.available_backends())
def test_backends_match_torch(backend):
    torch.manual_seed(0)
    enc = freeze(FactorizedPrior2018(16, 24).g_a)
    x = np.random.default_rng(0).random((1, 3, 64, 64), dtype=np.float32)
    ref = backends.get_runner("torch", enc, x.shape)(x)
    out = backends.get_runner(backend, enc, x.shape)(x)
    assert out.shape == ref.shape
    assert np.abs(out - ref).max() < 1e-3


def test_runner_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(backends, "MAX_RUNNERS", 2)
    backends.clear_runners()
    torch.manual_seed(0)
    enc = freeze(FactorizedPrior2018(8, 8).g_a)
    for size in (16, 32, 48):
        backends.get_runner("torch", enc, (1, 3, size, size))
    assert list(k[2][-1] for k in backends._runners) == [32, 48]
    backends.clear_runners()


def test_failed_onnx_export_leaves_no_file(monkeypatch, tmp_path):
    monkeypatch.setattr(backends, "ONNX_DIR", tmp_path)

    def broken_export(module, args, f, **kw):
        open(f, "wb").write(b"tronque")
        raise RuntimeError("export interrompu")
    monkeypatch.setattr(torch.onnx, "export", broken_export)
    with pytest.raises(RuntimeError):
        backends.export_onnx(freeze(FactorizedPrior2018(8, 8).g_a), (1, 3, 16, 16))
    assert list(tmp_path.iterdir()) == []


def test_campaign_names_never_overwrite(monkeypatch, tmp_path):
    monkeypatch.setattr(bench, "BENCH_DIR", tmp_path)
    assert bench.campaign_name("essai 1") == "essai_1"
    (tmp_path / "essai_1.csv").write_text("x")
    assert bench.campaign_name("essai 1") == "essai_1_2"


def test_frozen_gdn_matches():
    torch.manual_seed(0)
    g = GDN(8, symmetric=True)
    with torch.no_grad():
        g.gamma.add_(0.05 * torch.rand_like(g.gamma))
    x = torch.randn(1, 8, 5, 5)
    assert torch.allclose(g(x), freeze(g)(x), atol=1e-6)


def test_arithmetic_coder_roundtrip_with_escapes():
    rng = np.random.default_rng(0)
    pmf = np.array([0.1, 0.6, 0.3])
    tables = make_tables([pmf, pmf[::-1]], [-1, 0])
    values = rng.choice([-1, 0, 1], size=2000, p=pmf)
    values[::97] = 500                                    # hors support : échappement
    idx = np.zeros_like(values)
    idx[1::2] = 1
    data, n_esc = encode_symbols(values, idx, tables)
    assert n_esc > 0
    assert np.array_equal(decode_symbols(data, idx, tables), values)


def test_every_definition_has_a_doxygen_docstring():
    # Doxygen (HIDE_UNDOC_MEMBERS = YES) ne signale pas les oublis : on les vérifie ici
    root = Path(backends.__file__).parent
    missing = []

    def visit(node, path, prefix=""):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.ClassDef)):
                doc = ast.get_docstring(child, clean=False)
                if not doc or not doc.startswith("!"):
                    missing.append(f"{path.relative_to(root)}:{child.lineno} {prefix}{child.name}")
                if isinstance(child, ast.ClassDef):          # méthodes, pas les fonctions imbriquées
                    visit(child, path, f"{child.name}.")

    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if not (ast.get_docstring(tree, clean=False) or "").startswith("!"):
            missing.append(f"{path.relative_to(root)} (module)")
        visit(tree, path)
    assert not missing, "docstrings \"\"\"! ... \"\"\" manquantes :\n" + "\n".join(missing)


def test_metrics():
    img = make_image("rgb", 192, 192)
    assert psnr(img, img) == float("inf")
    assert ms_ssim(img, img) == pytest.approx(1.0)
    assert np.isnan(ms_ssim(img[:100, :100], img[:100, :100]))
    big = make_image("gris", MS_SSIM_MIN_SIDE, MS_SSIM_MIN_SIDE + 3)
    assert ms_ssim(big, big) == pytest.approx(1.0)          # plus petite taille acceptée
    small = big[:-1]
    assert np.isnan(ms_ssim(small, small))                   # un pixel de moins : NaN, pas d'exception
    r = np.array([0.1, 0.2, 0.4, 0.8, 1.6])
    q = np.array([25.0, 28.0, 31.0, 34.0, 37.0])
    assert bd_rate(r, q, r, q) == pytest.approx(0.0, abs=1e-9)
    assert bd_rate(r, q, r * 0.8, q) == pytest.approx(-20.0, abs=1e-6)
