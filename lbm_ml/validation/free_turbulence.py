"""Freely-decaying turbulence validation for learned LBM collision operators.

This is the 2D (D2Q9) counterpart of the "3D freely decaying turbulent flow"
a-posteriori test in

    Ortali, Gabbana et al., "Enhancing Lattice Kinetic Schemes for Fluid
    Dynamics with Lattice-Equivariant Neural Networks", AIAA J. 63(2) (2025),
    Sec. IV.C, Fig. 8.

The paper's test is 3D; here we reproduce its *methodology* inside the existing
2D framework of this repository (D2Q9 stencil, D4 symmetry, the model registry
in :mod:`lbm_ml.model.network`).  The recipe is:

  1. **Transient (forcing on).**  Starting from rest, run the *ground-truth* BGK
     collision with a stationary body force (2D analogue of the paper's ABC-type
     force, Eq. 51) until a developed, multi-scale flow emerges.  This produces a
     single initial condition ``f0`` shared by every model.

  2. **Free decay (forcing off).**  From the same ``f0`` we roll out each
     collision operator — the ground-truth BGK and every trained ML model — with
     the force switched off, and record the total kinetic energy ``E(t)`` and the
     velocity field at each step.

  3. **Metrics.**  Following Fig. 8 / Eqs. (49)-(50):
       * energy decay ``E(t)`` (should decay ~exponentially),
       * its logarithmic derivative ``d log E / dt`` (constant for exponential
         decay; the ground-truth BGK curve is the reference),
       * the a-posteriori error, an MSE on the velocity integrated in space and
         averaged over the first ``T`` timesteps (Eq. 50).

Why turbulence is the hard test (paper, Sec. IV.C): "due to the chaotic nature
of turbulence, the networks need to model effectively a much wider set of
pre-post collision pairs than in the laminar cases."  A non-equivariant MLP
typically diverges within ~100 steps, whereas the equivariant models
(GAVG/LENN) stay stable with a decay rate oscillating around the BGK truth.

Physical consistency note
-------------------------
An ML collision operator only reproduces BGK at the relaxation time ``tau`` it
was trained on.  Choose ``TurbulenceConfig.tau`` to match the model's training
data (the shipped synthetic dataset uses ``tau = 1``; a genuinely turbulent
decay needs a near-0.5 ``tau``, e.g. the paper's transient value ``0.51``, and a
model trained at that ``tau``).  The transient and the decay use the *same*
``tau`` so the built initial condition is physically consistent with the flow
the model is asked to sustain.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from lbm_ml.lattice.stencil import LB_stencil

logger = logging.getLogger(__name__)

Q = 9
_c, _w, _cs2, _compute_feq = LB_stencil()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TurbulenceConfig:
    """Parameters for the freely-decaying turbulence validation.

    Attributes
    ----------
    nx, ny
        Periodic lattice size.  The paper uses ``L = 32`` (so 32x32 here).
    tau
        BGK relaxation time, used for *both* the transient and the decay so the
        initial condition is consistent with the flow the model must sustain.
        Viscosity is ``nu = (tau - 0.5) * cs2``.  Small ``tau`` -> high Reynolds
        number -> genuinely turbulent decay (paper transient value: 0.51).
    force_amp
        Amplitude ``A`` of the stationary transient force (paper: 5e-6 at
        tau=0.51).  Steady velocity scales like ``A / (nu k^2)``.
    force_mode
        Integer wavenumber ``m`` of the forcing; physical ``k = 2*pi*m/L``.
        ``m = 1`` matches the paper's single-mode force; larger ``m`` injects
        energy at smaller scales.
    n_transient
        Number of forced BGK steps used to develop the flow before decay.
    n_decay
        Number of free-decay steps to roll out and compare (paper averages the
        a-posteriori error over the first ``T = 200``).
    seed_perturbation
        Amplitude of a small random, divergence-free velocity kick added at the
        start of the transient.  Breaks the exact symmetry of the single-mode
        force so nonlinear interactions can populate a broad range of scales.
        Set to 0 to start exactly from the symmetric forced state.
    seed
        RNG seed for the initial perturbation (reproducibility).
    error_window
        ``(t0, t1)`` step range over which the log-derivative is averaged and the
        relative decay-rate error is reported (paper uses ``[50, 200]``).
    """

    nx: int = 32
    ny: int = 32
    tau: float = 0.51
    force_amp: float = 5e-6
    force_mode: int = 1
    n_transient: int = 20000
    n_decay: int = 200
    seed_perturbation: float = 1e-4
    seed: int = 0
    error_window: tuple[int, int] = (50, 200)

    @property
    def nu(self) -> float:
        return (self.tau - 0.5) * _cs2


# ---------------------------------------------------------------------------
# Core LBM operations (periodic D2Q9)
# ---------------------------------------------------------------------------


def stream(f: np.ndarray) -> np.ndarray:
    """Periodic streaming: shift each population along its lattice velocity."""
    out = np.empty_like(f)
    for ip in range(Q):
        out[:, :, ip] = np.roll(np.roll(f[:, :, ip], _c[ip, 0], axis=0), _c[ip, 1], axis=1)
    return out


def macroscopic(f: np.ndarray, force: np.ndarray | None = None):
    """Density and velocity from populations.

    With Guo forcing the momentum carries a half-force correction
    ``rho u = sum_i f_i c_i + F/2`` so the scheme is second-order accurate.
    """
    rho = np.sum(f, axis=2)
    mom_x = np.einsum("ijk,k", f, _c[:, 0])
    mom_y = np.einsum("ijk,k", f, _c[:, 1])
    if force is not None:
        mom_x = mom_x + 0.5 * force[:, :, 0]
        mom_y = mom_y + 0.5 * force[:, :, 1]
    ux = mom_x / rho
    uy = mom_y / rho
    return rho, ux, uy


def _guo_source(rho, ux, uy, force, tau) -> np.ndarray:
    """Guo (2002) forcing source term ``S_i`` for a body force ``F``.

    S_i = (1 - 1/(2 tau)) w_i [ (c_i - u)/cs2 + (c_i . u) c_i / cs2^2 ] . F
    """
    prefac = 1.0 - 0.5 / tau
    Fx = force[:, :, 0]
    Fy = force[:, :, 1]
    S = np.empty((rho.shape[0], rho.shape[1], Q))
    inv_cs2 = 1.0 / _cs2
    inv_cs4 = inv_cs2 * inv_cs2
    for i in range(Q):
        cx, cy = _c[i, 0], _c[i, 1]
        cu = cx * ux + cy * uy
        term_x = (cx - ux) * inv_cs2 + cu * cx * inv_cs4
        term_y = (cy - uy) * inv_cs2 + cu * cy * inv_cs4
        S[:, :, i] = prefac * _w[i] * (term_x * Fx + term_y * Fy)
    return S


def bgk_collide(f: np.ndarray, tau: float, force: np.ndarray | None = None) -> np.ndarray:
    """Ground-truth BGK collision, optionally with a Guo body force."""
    rho, ux, uy = macroscopic(f, force)
    feq = _compute_feq(np.empty_like(f), rho, ux, uy, _c, _w)
    fpost = f - (f - feq) / tau
    if force is not None:
        fpost = fpost + _guo_source(rho, ux, uy, force, tau)
    return fpost


def ml_collide(f: np.ndarray, model) -> np.ndarray:
    """ML collision using the network's mass-normalise / denormalise convention.

    Mirrors the collision step in ``run_all.simulate``: divide each node's
    population vector by its mass, predict, then rescale.  This makes the
    learned map density-agnostic and (with a softmax / reconstruction head)
    keeps mass conserved.
    """
    nx, ny, _ = f.shape
    fpre = f.reshape(nx * ny, Q)
    norm = np.sum(fpre, axis=1, keepdims=True)
    fpost = model.predict(fpre / norm, verbose=0)
    return (norm * fpost).reshape(nx, ny, Q)


# ---------------------------------------------------------------------------
# Forcing field (2D analogue of the paper's Eq. 51)
# ---------------------------------------------------------------------------


def make_force_field(nx: int, ny: int, amp: float, mode: int) -> np.ndarray:
    """Stationary, zero-mean body force ``F(x, y)`` of shape ``(nx, ny, 2)``.

    2D analogue of the paper's cyclic ABC-type force (Eq. 51):
        F_x = A sin(k y),   F_y = A sin(k x),   k = 2*pi*mode/L.
    Both components are single-mode sinusoids with zero spatial mean, so the
    net momentum injection is zero and only shear/vorticity is forced.
    """
    ix, iy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    kx = 2.0 * np.pi * mode / nx
    ky = 2.0 * np.pi * mode / ny
    force = np.zeros((nx, ny, 2))
    force[:, :, 0] = amp * np.sin(ky * iy)
    force[:, :, 1] = amp * np.sin(kx * ix)
    return force


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def total_energy(f: np.ndarray) -> float:
    """Mean kinetic energy per node, ``<E> = mean(0.5 |u|^2)``."""
    _, ux, uy = macroscopic(f)
    return float(np.mean(0.5 * (ux**2 + uy**2)))


def log_derivative(energy: np.ndarray, dt: int = 1) -> np.ndarray:
    """``d log(E) / dt`` via central differences; constant for exp. decay."""
    with np.errstate(divide="ignore"):
        log_e = np.log(np.maximum(energy, 1e-300))
    return np.gradient(log_e, dt)


def aposteriori_error(u_model: np.ndarray, u_truth: np.ndarray, t_max: int | None = None) -> float:
    """A-posteriori velocity error, Eq. (50).

    Per step: ``sum_x |u_model - u_truth|^2 / sum_x |u_truth|^2`` (space
    integral as a normalised MSE), then averaged over the first ``t_max`` steps.

    Parameters
    ----------
    u_model, u_truth
        Arrays of shape ``(T, nx, ny, 2)`` — the model and ground-truth velocity
        fields over time.
    t_max
        Number of leading timesteps to average over (paper: 200).  ``None`` uses
        all available steps.
    """
    T = u_model.shape[0] if t_max is None else min(t_max, u_model.shape[0])
    num = np.sum((u_model[:T] - u_truth[:T]) ** 2, axis=(1, 2, 3))
    den = np.sum(u_truth[:T] ** 2, axis=(1, 2, 3))
    den = np.where(den == 0.0, np.inf, den)  # guard the (near) rest state
    return float(np.mean(num / den))


# ---------------------------------------------------------------------------
# Rollouts
# ---------------------------------------------------------------------------


def build_turbulent_ic(cfg: TurbulenceConfig) -> np.ndarray:
    """Develop a turbulent-like initial condition with forced ground-truth BGK.

    Returns the post-transient populations ``f0`` of shape ``(nx, ny, 9)``.
    The force is applied only here; the returned state is what every model
    subsequently decays from with the force switched off.
    """
    nx, ny = cfg.nx, cfg.ny
    force = make_force_field(nx, ny, cfg.force_amp, cfg.force_mode)

    # Start from rest + a small divergence-free perturbation to break symmetry.
    rho = np.ones((nx, ny))
    ux = np.zeros((nx, ny))
    uy = np.zeros((nx, ny))
    if cfg.seed_perturbation > 0:
        rng = np.random.default_rng(cfg.seed)
        # Stream function -> divergence-free velocity (u = curl(psi z_hat)).
        psi = rng.standard_normal((nx, ny))
        dpsi_dy = (np.roll(psi, -1, axis=1) - np.roll(psi, 1, axis=1)) * 0.5
        dpsi_dx = (np.roll(psi, -1, axis=0) - np.roll(psi, 1, axis=0)) * 0.5
        ux = dpsi_dy
        uy = -dpsi_dx
        scale = cfg.seed_perturbation / (np.max(np.abs(ux)) + 1e-30)
        ux *= scale
        uy *= scale

    f = _compute_feq(np.zeros((nx, ny, Q)), rho, ux, uy, _c, _w)

    logger.info(
        "Transient: %d forced BGK steps (tau=%.3f, nu=%.3e, force_amp=%.1e, mode=%d)",
        cfg.n_transient,
        cfg.tau,
        cfg.nu,
        cfg.force_amp,
        cfg.force_mode,
    )
    for _ in tqdm(range(cfg.n_transient), desc="Transient (forced BGK)", unit="it"):
        f = bgk_collide(f, cfg.tau, force=force)
        f = stream(f)

    u_rms = float(np.sqrt(2.0 * total_energy(f)))
    k_phys = 2.0 * np.pi * cfg.force_mode / cfg.nx
    reynolds = u_rms / max(cfg.nu * k_phys, 1e-30)
    logger.info("  developed flow: u_rms=%.3e, Re~%.1f", u_rms, reynolds)
    return f


def run_free_decay(f0: np.ndarray, collide_fn, cfg: TurbulenceConfig, desc: str = "decay"):
    """Freely decay ``f0`` (force off) with a given collision operator.

    ``collide_fn(f) -> f_post`` is any collision (e.g. ``lambda f: bgk_collide(
    f, tau)`` or ``lambda f: ml_collide(f, model)``).

    Returns
    -------
    energy : (n_decay + 1,) array of mean kinetic energy per step.
    velocity : (n_decay + 1, nx, ny, 2) array of the velocity field per step.
    diverged : bool — True if a non-finite state was produced (rollout aborted).
    """
    nx, ny = cfg.nx, cfg.ny
    n = cfg.n_decay
    energy = np.full(n + 1, np.nan)
    velocity = np.full((n + 1, nx, ny, 2), np.nan)

    def record(step, f):
        _, ux, uy = macroscopic(f)
        energy[step] = float(np.mean(0.5 * (ux**2 + uy**2)))
        velocity[step, :, :, 0] = ux
        velocity[step, :, :, 1] = uy

    f = np.copy(f0)
    record(0, f)
    diverged = False
    for t in tqdm(range(1, n + 1), desc=desc, unit="it"):
        f = collide_fn(f)
        f = stream(f)
        if not np.all(np.isfinite(f)):
            logger.warning("  %s diverged at step %d (non-finite populations)", desc, t)
            diverged = True
            break
        record(t, f)
    return energy, velocity, diverged


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


@dataclass
class ModelResult:
    """Per-operator decay result and derived metrics."""

    name: str
    energy: np.ndarray
    velocity: np.ndarray
    diverged: bool
    aposteriori: float = float("nan")
    mean_log_deriv: float = float("nan")
    rate_rel_error: float = float("nan")


def run_validation(cfg: TurbulenceConfig, models: dict[str, object]) -> dict[str, ModelResult]:
    """Run the full freely-decaying turbulence comparison.

    Parameters
    ----------
    cfg
        Test configuration.
    models
        Mapping ``{label: keras_model}`` of trained ML collision operators to
        evaluate alongside the ground-truth BGK.

    Returns
    -------
    dict mapping label -> :class:`ModelResult`.  The ground-truth BGK result is
    always present under the key ``"bgk"`` and is used as the reference for the
    a-posteriori error and the decay-rate relative error.
    """
    f0 = build_turbulent_ic(cfg)

    results: dict[str, ModelResult] = {}

    # Ground-truth BGK first — it is the reference for every error metric.
    e, v, d = run_free_decay(f0, lambda f: bgk_collide(f, cfg.tau), cfg, desc="decay [bgk]")
    results["bgk"] = ModelResult("bgk", e, v, d)

    for label, model in models.items():
        e, v, d = run_free_decay(f0, lambda f, m=model: ml_collide(f, m), cfg, desc=f"decay [{label}]")
        results[label] = ModelResult(label, e, v, d)

    # Derived metrics, referenced to BGK.
    ref = results["bgk"]
    t0, t1 = cfg.error_window
    ld_ref = log_derivative(ref.energy)
    ref.mean_log_deriv = float(np.nanmean(ld_ref[t0:t1]))
    ref.rate_rel_error = 0.0
    ref.aposteriori = 0.0

    for label, r in results.items():
        if label == "bgk":
            continue
        # A-posteriori error only over the steps the model actually completed.
        valid = min(np.count_nonzero(np.isfinite(r.energy)), t1)
        r.aposteriori = aposteriori_error(r.velocity, ref.velocity, t_max=valid)
        ld = log_derivative(r.energy)
        r.mean_log_deriv = float(np.nanmean(ld[t0:t1]))
        if np.isfinite(r.mean_log_deriv) and abs(ref.mean_log_deriv) > 0:
            r.rate_rel_error = abs(r.mean_log_deriv - ref.mean_log_deriv) / abs(ref.mean_log_deriv)

    return results
