"""Structural checks for D4 group ops and reconstruction layers (doc 04 §1)."""

import keras
import numpy as np
import pytest

from lbm_ml.lattice.stencil import LB_stencil
from lbm_ml.lattice.symmetry import (
    BoundedBlendReconstruction,
    LBmirror,
    LBrot90,
    MultiplicativeAlgReconstruction,
    PositivitySafeAlgReconstruction,
    SymmetricAlgReconstruction,
    compute_d2q9_bias_orbit_indices,
    compute_d2q9_orbit_indices,
)

_C, _W, _CS2, _COMPUTE_FEQ = LB_stencil()


def _moments(f):
    f = np.asarray(f)
    return f.sum(-1), (f * _C[:, 0]).sum(-1), (f * _C[:, 1]).sum(-1)


def test_orbit_counts():
    assert int(compute_d2q9_orbit_indices().max()) + 1 == 15
    bias = compute_d2q9_bias_orbit_indices()
    assert int(bias.max()) + 1 == 3
    # rest / axis-aligned / diagonal orbits
    assert len({bias[0]}) == 1 and len(set(bias[1:5])) == 1 and len(set(bias[5:9])) == 1


def test_group_algebra(f_batch):
    t = keras.ops.convert_to_tensor(f_batch)
    assert np.allclose(np.asarray(LBrot90(LBrot90(LBrot90(LBrot90(t))))), f_batch)
    assert np.allclose(np.asarray(LBmirror(LBmirror(t))), f_batch)


@pytest.mark.parametrize(
    "layer_cls",
    [
        SymmetricAlgReconstruction,
        PositivitySafeAlgReconstruction,
        BoundedBlendReconstruction,
    ],
)
def test_reconstruction_conserves(layer_cls, f_batch, f_pred):
    out = np.asarray(layer_cls()(f_batch, f_pred))
    for got, want in zip(_moments(out), _moments(f_batch)):
        assert np.abs(got - want).max() < 1e-12


def test_multiplicative_reconstruction_contract(f_batch, rng):
    """Multiplicative layer: positivity always; exact momentum when the full
    correction is positivity-safe (mild prediction, alpha = 1). Under harsh
    predictions alpha < 1 trades conservation for positivity by design."""
    mild = f_batch * (1.0 + 0.01 * rng.normal(size=f_batch.shape))
    mild = np.abs(mild) / np.abs(mild).sum(-1, keepdims=True)
    out = np.asarray(MultiplicativeAlgReconstruction()(f_batch, mild))
    assert out.min() >= -1e-15
    _, px, py = _moments(f_batch)
    _, qx, qy = _moments(out)
    assert np.abs(qx - px).max() < 1e-12
    assert np.abs(qy - py).max() < 1e-12


def test_positivity_safe_keeps_positive(f_batch, f_pred):
    out = np.asarray(PositivitySafeAlgReconstruction()(f_batch, f_pred))
    assert out.min() >= -1e-15


def test_bounded_blend_equilibrium_matches_stencil(f_batch):
    layer = BoundedBlendReconstruction()
    feq_layer = np.asarray(layer._equilibrium(keras.ops.convert_to_tensor(f_batch)))
    rho, px, py = _moments(f_batch)
    ux, uy = px / rho, py / rho
    n = f_batch.shape[0]
    feq_ref = _COMPUTE_FEQ(
        np.zeros((n, 1, 9)), rho[:, None], ux[:, None], uy[:, None], _C, _W
    )[:, 0, :]
    assert np.abs(feq_layer - feq_ref).max() < 1e-14


def test_bounded_blend_limits(f_batch, f_pred):
    # g -> 1 recovers the plain symmetric projection of the raw prediction.
    raw = np.asarray(SymmetricAlgReconstruction()(f_batch, f_pred))
    lim = np.asarray(BoundedBlendReconstruction(theta_init=30.0)(f_batch, f_pred))
    assert np.abs(lim - raw).max() < 1e-10
    # Smaller g damps the output monotonically toward equilibrium.
    feq = np.asarray(
        BoundedBlendReconstruction()._equilibrium(keras.ops.convert_to_tensor(f_batch))
    )
    dists = [
        np.linalg.norm(
            np.asarray(BoundedBlendReconstruction(theta_init=th)(f_batch, f_pred)) - feq
        )
        for th in (-10.0, 0.0, 10.0)
    ]
    assert dists[0] < dists[1] < dists[2]
