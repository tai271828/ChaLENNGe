"""Driver-level units: wake metrics, geometry, provenance, data generation."""

import json

import numpy as np

from apply_nn_karman import compute_strouhal, karman_geometry
from lbm_ml.data.generation import generate_samples
from lbm_ml.lattice.stencil import LB_stencil
from lbm_ml.provenance import sha256, write_manifest

_C, _W, _CS2, _ = LB_stencil()


def test_karman_geometry_tau():
    Nx, Ny, obstacle, tau, nu = karman_geometry(res=250, u_inlet=0.12, re=150.0)
    assert (Nx, Ny) == (550, 102)
    assert abs(tau - 0.5576) < 1e-4
    assert obstacle.sum() > 0


def test_strouhal_recovers_synthetic_sine():
    D, U, f_true = 24, 0.12, 0.0014  # the BGK control's shedding frequency
    t = np.arange(20000)
    sig = 0.1 * np.sin(2 * np.pi * f_true * t) + 1e-4 * np.random.default_rng(0).normal(
        size=t.size
    )
    st, f_shed, amp = compute_strouhal(sig, warmup_frac=0.5, D=D, U=U)
    assert abs(st - f_true * D / U) < 0.02  # within one FFT bin
    assert abs(amp - 0.1) / 0.1 < 0.05


def test_strouhal_short_signal_returns_none():
    assert compute_strouhal(np.zeros(50), warmup_frac=0.5, D=24, U=0.12) is None


def test_generate_samples_bgk_conserves():
    f_eq, f_pre, f_post = generate_samples(500, sigma_min=1e-4, sigma_max=5e-4)
    for k in (0, 1):
        pre = (f_pre * _C[:, k]).sum(-1)
        post = (f_post * _C[:, k]).sum(-1)
        assert np.abs(pre - post).max() < 1e-12
    assert np.abs(f_pre.sum(-1) - f_post.sum(-1)).max() < 1e-12
    assert f_eq.min() > 0


def test_write_manifest(tmp_path):
    payload_file = tmp_path / "blob.bin"
    payload_file.write_bytes(b"chalennge")
    path = write_manifest(tmp_path, {"seed": 7, "blob": sha256(payload_file)})
    data = json.loads(path.read_text())
    assert data["seed"] == 7
    assert len(data["blob"]) == 64
    assert data["git_commit"] is None or len(data["git_commit"]) == 40
    assert "created_utc" in data and "command" in data
