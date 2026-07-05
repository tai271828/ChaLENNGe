"""Registry models: build, physics properties, budgets, serialization (doc 04 §1)."""

import keras
import numpy as np
import pytest

from lbm_ml.lattice.stencil import LB_stencil
from lbm_ml.lattice.symmetry import LBrot90
from lbm_ml.model.network import MODEL_REGISTRY

_C, _W, _CS2, _ = LB_stencil()

# Representative slice of the registry: one per family × reconstruction style.
EQUIVARIANT = [
    "d4equivariant_softmax_cons",
    "resnet_softmax_cons",
    "lenn_18_18_18_softmax_cons",
    "lenn_resnet_softmax_cons",
    "lenn_31_31_31_softmax_bounded",
]
NON_EQUIVARIANT = ["plain_2_softmax"]


@pytest.fixture(scope="module")
def models():
    return {name: MODEL_REGISTRY[name]() for name in EQUIVARIANT + NON_EQUIVARIANT}


def test_build_and_shape(models, f_batch):
    for name, model in models.items():
        out = np.asarray(model(keras.ops.convert_to_tensor(f_batch)))
        assert out.shape == f_batch.shape, name
        assert np.all(np.isfinite(out)), name


def test_conservation(models, f_batch):
    """_cons / _bounded models conserve mass and momentum exactly."""
    for name in EQUIVARIANT:
        out = np.asarray(models[name](keras.ops.convert_to_tensor(f_batch)))
        assert np.abs(out.sum(-1) - f_batch.sum(-1)).max() < 1e-12, name
        for k in (0, 1):
            got = (out * _C[:, k]).sum(-1)
            want = (f_batch * _C[:, k]).sum(-1)
            assert np.abs(got - want).max() < 1e-12, name


def test_d4_equivariance(models, f_batch):
    """Equivariant families commute with a 90° rotation; plain_2 must not."""
    t = keras.ops.convert_to_tensor(f_batch)
    for name in EQUIVARIANT:
        m = models[name]
        lhs = np.asarray(m(LBrot90(t)))
        rhs = np.asarray(LBrot90(keras.ops.convert_to_tensor(np.asarray(m(t)))))
        assert np.abs(lhs - rhs).max() < 1e-10, name
    m = models["plain_2_softmax"]
    lhs = np.asarray(m(LBrot90(t)))
    rhs = np.asarray(LBrot90(keras.ops.convert_to_tensor(np.asarray(m(t)))))
    assert np.abs(lhs - rhs).max() > 1e-6  # by design: the negative control


def test_budget_matched_twins():
    """Doc 02 matrix pairs must stay parameter-matched (00-README rule 3)."""

    def params(name):
        return MODEL_REGISTRY[name]().count_params()

    lenn, lenn_res = params("lenn_31_31_31_softmax_cons"), params(
        "lenn_resnet_18_18_18_softmax_cons"
    )
    assert abs(lenn - lenn_res) / lenn_res < 0.01
    gavg, gavg_res = params("d4equivariant_10K_wide_softmax_cons"), params(
        "resnet_softmax_cons"
    )
    assert abs(gavg - gavg_res) / gavg_res < 0.05


def test_serialization_round_trip(models, f_batch, tmp_path):
    """Custom layers (LENN, bounded blend) survive save/load bit-exactly."""
    m = models["lenn_31_31_31_softmax_bounded"]
    path = tmp_path / "m.keras"
    m.save(path)
    m2 = keras.models.load_model(path)
    a = np.asarray(m(keras.ops.convert_to_tensor(f_batch)))
    b = np.asarray(m2(keras.ops.convert_to_tensor(f_batch)))
    assert np.array_equal(a, b)
