"""Tests du codeur arithmétique : noyaux numba contre le codeur Python de référence.

python -m pytest tests/test_entropy.py -q
"""
import importlib.util
import sys

import numpy as np
import pytest
import torch

from aidge_in_space import coding, entropy
from aidge_in_space.codecs.base import LearnedCodec, Level
from aidge_in_space.entropy import (ZIGZAG_LIMIT, decode_symbols, decode_symbols_python, encode_symbols,
                                    encode_symbols_python, make_tables)
from aidge_in_space.nn import FactorizedPrior2018

needs_numba = pytest.mark.skipif(not entropy.NUMBA, reason="numba non installé")


def random_tables(rng, n_tables, max_size, precision):
    # tailles de support variées, dont des tables à un seul symbole (plus l'échappement)
    pmfs, los = [], []
    for k in range(n_tables):
        size = 1 if k == 0 else int(rng.integers(2, max_size + 1))
        pmfs.append(rng.dirichlet(np.full(size, 0.3)) if size > 1 else np.array([1.0]))
        los.append(int(rng.integers(-300, 300)))
    return make_tables(pmfs, los, precision=precision)


def random_symbols(rng, tables, n, esc_rate):
    idx = rng.integers(0, len(tables.lo), n)
    lo = np.asarray(tables.lo)[idx]
    width = np.array([len(c) - 2 for c in tables.cum])[idx]
    values = lo + (rng.random(n) * width).astype(np.int64)
    esc = rng.random(n) < esc_rate                         # échappements forcés, jusqu'à ±2^19
    values[esc] = rng.integers(-ZIGZAG_LIMIT, ZIGZAG_LIMIT, int(esc.sum()))
    values[:4] = [-ZIGZAG_LIMIT, ZIGZAG_LIMIT - 1, -ZIGZAG_LIMIT + 1, ZIGZAG_LIMIT - 2]
    return values, idx


@needs_numba
@pytest.mark.parametrize("seed,n_tables,max_size,precision,esc_rate", [
    (0, 1, 3, 16, 0.0),
    (1, 8, 40, 16, 0.02),
    (2, 64, 600, 16, 0.05),
    (3, 20, 200, 12, 0.3),
    (4, 5, 1000, 24, 0.01),
    (5, 192, 511, 16, 0.001),
])
def test_numba_stream_identical_to_python(seed, n_tables, max_size, precision, esc_rate):
    rng = np.random.default_rng(seed)
    tables = random_tables(rng, n_tables, max_size, precision)
    values, idx = random_symbols(rng, tables, 6000, esc_rate)
    ref, ref_esc = encode_symbols_python(values, idx, tables)
    data, n_esc = encode_symbols(values, idx, tables)
    assert data == ref and n_esc == ref_esc
    assert np.array_equal(decode_symbols(data, idx, tables), decode_symbols_python(ref, idx, tables))
    assert np.array_equal(decode_symbols(data, idx, tables), values)


@needs_numba
def test_numba_identical_on_model_tables():
    # tables réelles : gaussiennes de l'hyperprior, indices int64 et valeurs int32
    tables = coding.gaussian_tables()
    rng = np.random.default_rng(7)
    idx = rng.integers(0, len(tables.lo), 20000)
    values = np.round(rng.normal(0, 1 + idx / 4)).astype(np.int32)
    values[::1009] = 4000                                  # hors support
    data, n_esc = encode_symbols(values, idx, tables)
    assert (data, n_esc) == encode_symbols_python(values, idx, tables)
    assert np.array_equal(decode_symbols(data, idx, tables), values)


@pytest.mark.parametrize("values", [[], [0], [5, -5, 0, 1], list(range(-40, 40))])
def test_roundtrip_small_inputs(values):
    tables = make_tables([[0.2, 0.6, 0.2], [0.5, 0.5]], [-1, 0])
    values = np.array(values, dtype=np.int64)
    idx = np.arange(len(values)) % 2
    data, _ = encode_symbols(values, idx, tables)
    assert data == encode_symbols_python(values, idx, tables)[0]
    for stream in (data, bytearray(data), memoryview(data)):
        assert np.array_equal(decode_symbols(stream, idx, tables), values)


@pytest.mark.parametrize("bad", [ZIGZAG_LIMIT, -ZIGZAG_LIMIT - 1, 2 ** 40, -2 ** 62])
def test_escape_out_of_range_raises_like_python(bad):
    tables = make_tables([[0.5, 0.5]], [0])
    values, idx = np.array([0, 1, bad, 0]), np.zeros(4, dtype=np.int64)
    with pytest.raises(ValueError) as ref:
        encode_symbols_python(values, idx, tables)
    with pytest.raises(ValueError) as err:
        encode_symbols(values, idx, tables)
    assert str(err.value) == str(ref.value)


@needs_numba
def test_numba_rejects_table_index_out_of_range():
    tables = make_tables([[0.5, 0.5]], [0])
    with pytest.raises(IndexError):
        encode_symbols(np.zeros(3, dtype=np.int64), np.array([0, 1, 0]), tables)
    with pytest.raises(IndexError):
        decode_symbols(b"\x00", np.array([-1]), tables)


def test_large_total_falls_back_to_python(monkeypatch):
    # 2^31 dépasse NUMBA_MAX_TOTAL (63 bits) : le codeur Python prend le relais
    tables = make_tables([[0.1, 0.8, 0.1]], [-1], precision=31)
    calls = []
    monkeypatch.setattr(entropy, "encode_symbols_python",
                        lambda *a: calls.append(1) or encode_symbols_python(*a))
    values, idx = np.array([-1, 0, 1, 9]), np.zeros(4, dtype=np.int64)
    data, n_esc = encode_symbols(values, idx, tables)
    assert calls and n_esc == 1
    assert np.array_equal(decode_symbols(data, idx, tables), values)


def test_fallback_when_numba_disabled(monkeypatch):
    rng = np.random.default_rng(11)
    tables = random_tables(rng, 10, 50, 16)
    values, idx = random_symbols(rng, tables, 3000, 0.02)
    expected = encode_symbols(values, idx, tables)
    monkeypatch.setattr(entropy, "NUMBA", False)
    monkeypatch.setattr(entropy, "_WARM", False)
    for kernel in ("_encode_kernel", "_decode_kernel"):
        monkeypatch.setattr(entropy, kernel, lambda *a: pytest.fail("noyau numba appelé sans numba"))
    assert encode_symbols(values, idx, tables) == expected
    assert np.array_equal(decode_symbols(expected[0], idx, tables), values)
    entropy.warmup()                                       # sans numba : ne fait rien
    assert not entropy._WARM


def test_module_imports_without_numba(monkeypatch):
    # import protégé : une copie du module chargée alors que numba est introuvable
    monkeypatch.setitem(sys.modules, "numba", None)
    spec = importlib.util.spec_from_file_location("entropy_sans_numba", entropy.__file__)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert not mod.NUMBA
    tables = mod.make_tables([[0.25, 0.5, 0.25]], [-1])
    values, idx = np.array([0, 1, -1, 30, -2]), np.zeros(5, dtype=np.int64)
    data, n_esc = mod.encode_symbols(values, idx, tables)
    ref_tables = make_tables([[0.25, 0.5, 0.25]], [-1])
    assert (data, n_esc) == encode_symbols(values, idx, ref_tables)
    assert np.array_equal(mod.decode_symbols(data, idx, tables), values)


class TinyCodec(LearnedCodec):
    # petit codec appris aux poids aléatoires, pour tester LearnedCodec.run() sans checkpoint
    key = "tiny-test"

    def make_model(self, level):
        torch.manual_seed(0)
        return FactorizedPrior2018(8, 8)

    def load_weights(self, model, level):
        pass


@needs_numba
def test_jit_compiled_before_timed_section(monkeypatch):
    order = []
    real_warmup, real_encode = coding.warmup, coding.encode_symbols
    monkeypatch.setattr(entropy, "_WARM", False)
    monkeypatch.setattr(coding, "warmup", lambda: order.append("chauffe") or real_warmup())
    monkeypatch.setattr(coding, "encode_symbols", lambda *a: order.append("codage") or real_encode(*a))
    img = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    res = TinyCodec().run(img, Level("x", "x"), "rgb", real_rate=True)
    assert order[0] == "chauffe" and "codage" in order
    assert entropy._WARM
    # une seule signature par noyau : les vrais appels réutilisent celle compilée par la chauffe
    assert len(entropy._encode_kernel.signatures) == 1
    assert len(entropy._decode_kernel.signatures) == 1
    assert res.info["décodage exact"]
    assert 0 <= res.timings["codage entropique"] <= res.timings["encodage"]


def test_estimated_rate_has_no_entropy_timing():
    img = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    res = TinyCodec().run(img, Level("x", "x"), "rgb", real_rate=False)
    assert "codage entropique" not in res.timings
