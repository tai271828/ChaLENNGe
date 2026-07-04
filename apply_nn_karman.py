#!/usr/bin/env python3
"""Apply a trained collision operator to a Kármán vortex street (step-4 "apply").

This is the ChaLENNGe-native port of the ``apply-nn.py`` step-4 stage referenced
by ``phase01/run-exp01-snellius.sh``: it drives a full KVS simulation using the
neural network as the collision operator and, optionally, evaluates the network
against recorded BGK ``fpre``/``fpost`` snapshots.

Unlike the original apply script, this one reuses this repo's ``lbm_ml`` package,
so **LENN / LENN+ResNet** models load out of the box — importing ``lbm_ml``
self-registers the ``LENNLayer`` and the ``*AlgReconstruction`` layers via
``@keras.saving.register_keras_serializable``, which the external apply script
lacked.

What it produces:
  * ``nn_velocity_field.gif`` — the NN-driven KVS wake over time (``--animate``),
    plus optional early PNG snapshots (``--snap-every``).
  * ``eval_metrics.{csv,png}`` + ``per_direction_error.png`` — a-priori RMSRE /
    MAE / max-abs error of the NN against BGK ground truth, per saved snapshot
    (only when ``--data-dir`` holds ``fpre_*.npy`` / ``fpost_*.npy`` pairs).
  * ``wake_metrics.csv`` + ``wake_summary.json`` — a-posteriori wake dynamics of
    the rollout (with ``--animate``): probe velocities, kinetic energy, min f,
    max |u| per step; Strouhal number and stability horizon in the summary.
  * ``manifest.json`` — provenance record (model checksum, physics, seed, git
    commit, command line) so every result directory is reproducible.

``--bgk-only`` replaces the NN with the classical BGK collision — the
ground-truth control for Strouhal / energy / stability comparisons.

Geometry / physics match ``lbm_karman-ng.py`` defaults (res=250 → 550×102,
Re=150, U_inlet=0.12 → τ=0.5576), which is also the τ the Kármán-trained models
were trained at (see the free-turbulence notes).

Examples
--------
Animation + evaluation with a LENN model (the step-4 shape)::

    python apply_nn_karman.py --animate \
        --model-path /path/to/lenn_.../model.keras \
        --data-dir   /path/to/karman/every_100 \
        --out-dir    eval_results_lenn

Quick local check — short animation, no evaluation::

    python apply_nn_karman.py --animate --anim-steps 400 --update-steps 100 \
        --model-path /path/to/model.keras --out-dir /tmp/kvs_check
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from matplotlib.animation import FuncAnimation, PillowWriter

# Importing lbm_ml registers the custom layers (LENN + reconstruction + D4) that
# saved models may reference, so keras.models.load_model can deserialize them.
import lbm_ml.model.network  # noqa: F401
import lbm_ml.lattice.symmetry  # noqa: F401
from lbm_ml.lattice.stencil import LB_stencil
from lbm_ml.model.losses import rmsre

# D2Q9 stencil (channel order matches the network's expected ordering).
_c, _w, _cs2, _compute_feq = LB_stencil()
# Opposite-direction index for bounce-back, consistent with _c ordering:
# [rest, E, N, W, S, NE, NW, SW, SE] -> [rest, W, S, E, N, SW, SE, NE, NW].
_OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])


def equilibrium(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """Second-order D2Q9 equilibrium f_eq(rho, u), shape (Nx, Ny, 9)."""
    feq = np.zeros((*rho.shape, 9))
    return _compute_feq(feq, rho, ux, uy, _c, _w)


def normalize(f: np.ndarray):
    """Divide each sample by its total density (the training-pipeline convention)."""
    norm = np.sum(f, axis=1, keepdims=True)
    return f / norm, norm


# ---------------------------------------------------------------------------
# Kármán geometry
# ---------------------------------------------------------------------------


def karman_geometry(res: int, u_inlet: float, re: float):
    """Return (Nx, Ny, obstacle mask, tau, nu) for the standard KVS setup."""
    Nx = int(round(2.2 * res))
    Ny = int(round(0.41 * res))
    cx = int(round(0.2 * res))
    cy = int(round(0.2 * res))
    r = int(round(0.05 * res))
    D = 2 * r
    nu = u_inlet * D / re
    tau = 3.0 * nu + 0.5
    xg, yg = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing="ij")
    obstacle = (xg - cx) ** 2 + (yg - cy) ** 2 <= r**2
    return Nx, Ny, obstacle, tau, nu


# ---------------------------------------------------------------------------
# Wake-dynamics metrics + provenance
# ---------------------------------------------------------------------------


def compute_strouhal(uy_series: np.ndarray, warmup_frac: float, D: float, U: float):
    """Strouhal number from the wake-probe transverse velocity u_y(t).

    Discards the first ``warmup_frac`` of the series (shedding onset), removes
    the mean, applies a Hann window, and picks the dominant rFFT frequency
    (bin 0 excluded).  Time is in lattice steps, so ``St = f_shed * D / U``
    with D and U in lattice units.

    Returns (St, f_shed, amplitude), or None if the usable signal is too short
    for a meaningful spectrum (< 64 samples).
    """
    sig = np.asarray(uy_series, dtype=float)
    sig = sig[int(len(sig) * warmup_frac) :]
    if len(sig) < 64:
        return None
    sig = sig - sig.mean()
    window = np.hanning(len(sig))
    spec = np.abs(np.fft.rfft(sig * window))
    freqs = np.fft.rfftfreq(len(sig), d=1.0)
    k = int(np.argmax(spec[1:])) + 1  # skip the DC bin
    f_shed = float(freqs[k])
    amplitude = float(2.0 * spec[k] / window.sum())
    return f_shed * D / U, f_shed, amplitude


def _git_commit() -> str | None:
    """Current repo commit hash, or None if git is unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(args, Nx: int, Ny: int, tau: float, nu: float) -> None:
    """Record everything needed to reproduce this run in <out-dir>/manifest.json."""
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "git_commit": _git_commit(),
        "model_path": args.model_path,
        "model_sha256": _sha256(args.model_path) if args.model_path else None,
        "data_dir": args.data_dir,
        "seed": args.seed,
        "bgk_only": args.bgk_only,
        "physics": {
            "res": args.res,
            "re": args.re,
            "u_inlet": args.u_inlet,
            "tau": tau,
            "nu": nu,
            "grid": [Nx, Ny],
        },
        "run": {
            "anim_steps": args.anim_steps,
            "update_steps": args.update_steps,
            "batch_size": args.batch_size,
            "warmup_frac": args.warmup_frac,
            "probe_x": args.probe_x,
            "probe_y": args.probe_y,
        },
    }
    path = os.path.join(args.out_dir, "manifest.json")
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Manifest -> {path}")


# ---------------------------------------------------------------------------
# Kármán rollout (NN or BGK collision) + wake metrics + animation
# ---------------------------------------------------------------------------


def make_animation(model, args) -> None:
    """Run a KVS rollout (NN or BGK collision), record wake metrics, save a GIF.

    ``model=None`` runs the classical BGK collision (``--bgk-only``) — the
    ground-truth control.  Per-step diagnostics go to ``wake_metrics.csv``;
    Strouhal number and stability horizon to ``wake_summary.json``.
    """
    label = "BGK" if model is None else "NN predicted"
    print(f"\nRunning {label} Kármán rollout ...")
    Nx, Ny, obstacle, tau, nu = karman_geometry(args.res, args.u_inlet, args.re)
    U = args.u_inlet
    print(f"  Grid: {Nx} x {Ny},  tau={tau:.4f},  nu={nu:.6f},  Re={args.re}")

    # Wake probe: default 8 radii downstream of the cylinder, on its centreline.
    cx, cy = int(round(0.2 * args.res)), int(round(0.2 * args.res))
    r = int(round(0.05 * args.res))
    D = 2 * r
    probe_x = args.probe_x if args.probe_x is not None else min(cx + 8 * r, Nx - 2)
    probe_y = args.probe_y if args.probe_y is not None else cy
    print(f"  Wake probe at ({probe_x}, {probe_y}); D={D}, U={U}")
    fluid = ~obstacle

    _, yg = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing="ij")
    rho = np.ones((Nx, Ny))
    ux = np.full((Nx, Ny), U)
    uy = 0.001 * U * np.sin(2.0 * np.pi * yg / Ny)  # tiny seed to trigger shedding
    ux[obstacle] = 0.0
    uy[obstacle] = 0.0
    ux[:, 0] = ux[:, -1] = 0.0
    uy[:, 0] = uy[:, -1] = 0.0
    f = equilibrium(rho, ux, uy)

    frames_ux, frames_uy, frame_steps = [], [], []
    n_steps, update_every, snap_every = (
        args.anim_steps,
        args.update_steps,
        args.snap_every,
    )

    # Per-step wake diagnostics (all cheap scalars).
    diag = {
        "step": [],
        "ux_probe": [],
        "uy_probe": [],
        "energy": [],
        "max_speed": [],
        "min_f": [],
    }
    horizon = (
        None  # first step failing the stability criteria; None == stable throughout
    )
    first_negative = None  # first step with min f_i < -1e-6 (diagnostic only)

    U_max = U * 2.0
    Xg, Yg = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing="ij")
    snap_dir = os.path.join(args.out_dir, "karman_snapshots")
    if snap_every > 0:
        os.makedirs(snap_dir, exist_ok=True)

    def save_snapshot(step_i, ux_i, uy_i):
        speed = np.sqrt(ux_i**2 + uy_i**2)
        speed[obstacle] = np.nan
        fig_s, ax_s = plt.subplots(figsize=(10, 4), dpi=100)
        ax_s.imshow(
            speed.T,
            origin="lower",
            cmap="jet",
            vmin=0,
            vmax=U_max,
            aspect="auto",
            extent=[0, Nx, 0, Ny],
        )
        up, vp = ux_i.copy(), uy_i.copy()
        up[obstacle] = vp[obstacle] = 0.0
        ax_s.streamplot(Xg.T, Yg.T, up.T, vp.T, density=0.5, color="w", linewidth=0.6)
        ax_s.set_title(f"{label} velocity — step {step_i}", fontsize=12)
        ax_s.set_xlabel("x")
        ax_s.set_ylabel("y")
        path = os.path.join(snap_dir, f"step_{step_i:06d}.png")
        fig_s.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig_s)
        print(f"  PNG snapshot -> {path}")

    for step in range(1, n_steps + 1):
        rho = np.sum(f, axis=2)
        ux = np.sum(f * _c[:, 0], axis=2) / rho
        uy = np.sum(f * _c[:, 1], axis=2) / rho

        # -- Wake diagnostics + stability check (on the pre-collision state) --
        speed = np.sqrt(ux[fluid] ** 2 + uy[fluid] ** 2)
        max_speed = float(np.max(speed))
        min_f = float(np.min(f))
        diag["step"].append(step)
        diag["ux_probe"].append(float(ux[probe_x, probe_y]))
        diag["uy_probe"].append(float(uy[probe_x, probe_y]))
        diag["energy"].append(float(0.5 * np.mean(speed**2)))
        diag["max_speed"].append(max_speed)
        diag["min_f"].append(min_f)
        # Mild negative populations are normal for BGK at low tau — record the
        # first occurrence as a diagnostic, but do not treat it as divergence.
        if first_negative is None and min_f < -1e-6:
            first_negative = step
        if not np.all(np.isfinite(f)) or max_speed > 0.4:
            horizon = step
            print(
                f"  [DIVERGED] step {step}: min_f={min_f:.3e}, max|u|={max_speed:.3f}"
            )
            break

        # -- Collision: BGK control or NN (mass-normalise / denormalise) --
        if model is None:
            f_out = f + (equilibrium(rho, ux, uy) - f) / tau
        else:
            fpre = f.reshape(-1, 9)
            norm = np.sum(fpre, axis=1, keepdims=True)
            f_out = (
                model.predict(fpre / norm, batch_size=args.batch_size, verbose=0) * norm
            ).reshape(Nx, Ny, 9)

        # -- Obstacle bounce-back --
        for i in range(9):
            f_out[obstacle, i] = f[obstacle, _OPP[i]]

        # -- Streaming --
        for i in range(9):
            f[:, :, i] = np.roll(
                np.roll(f_out[:, :, i], _c[i, 0], axis=0), _c[i, 1], axis=1
            )

        # -- Wall bounce-back (top/bottom) --
        f[:, 0, 2] = f_out[:, 0, 4]
        f[:, 0, 5] = f_out[:, 0, 7]
        f[:, 0, 6] = f_out[:, 0, 8]
        f[:, -1, 4] = f_out[:, -1, 2]
        f[:, -1, 7] = f_out[:, -1, 5]
        f[:, -1, 8] = f_out[:, -1, 6]

        # -- Outlet BC (Zou-He pressure) --
        rho_out = 1.0
        iy = slice(1, -1)
        ux_out = (
            -1.0
            + (
                f[-1, iy, 0]
                + f[-1, iy, 2]
                + f[-1, iy, 4]
                + 2.0 * (f[-1, iy, 1] + f[-1, iy, 5] + f[-1, iy, 8])
            )
            / rho_out
        )
        ux_out = np.clip(ux_out, 0.0, 0.5)
        f[-1, iy, 3] = f[-1, iy, 1] - (2.0 / 3.0) * rho_out * ux_out
        f[-1, iy, 7] = (
            f[-1, iy, 5]
            + 0.5 * (f[-1, iy, 2] - f[-1, iy, 4])
            - (1.0 / 6.0) * rho_out * ux_out
        )
        f[-1, iy, 6] = (
            f[-1, iy, 8]
            - 0.5 * (f[-1, iy, 2] - f[-1, iy, 4])
            - (1.0 / 6.0) * rho_out * ux_out
        )
        for yc in (0, Ny - 1):
            f[-1, yc, 3] = f[-2, yc, 3]
            f[-1, yc, 6] = f[-2, yc, 6]
            f[-1, yc, 7] = f[-2, yc, 7]

        # -- Inlet BC (Zou-He velocity) --
        rho_in = (
            f[0, :, 0]
            + f[0, :, 2]
            + f[0, :, 4]
            + 2.0 * (f[0, :, 3] + f[0, :, 6] + f[0, :, 7])
        ) / (1.0 - U)
        f[0, :, 1] = f[0, :, 3] + (2.0 / 3.0) * rho_in * U
        f[0, :, 5] = (
            f[0, :, 7] - 0.5 * (f[0, :, 2] - f[0, :, 4]) + (1.0 / 6.0) * rho_in * U
        )
        f[0, :, 8] = (
            f[0, :, 6] + 0.5 * (f[0, :, 2] - f[0, :, 4]) + (1.0 / 6.0) * rho_in * U
        )

        if step % update_every == 0:
            frames_ux.append(ux.copy())
            frames_uy.append(uy.copy())
            frame_steps.append(step)
            print(f"  Frame saved at step {step}/{n_steps}")
        if snap_every > 0 and step % snap_every == 0:
            save_snapshot(step, ux, uy)

    # -- Wake metrics: per-step CSV + Strouhal / stability summary --
    csv_path = os.path.join(args.out_dir, "wake_metrics.csv")
    cols = ["step", "ux_probe", "uy_probe", "energy", "max_speed", "min_f"]
    np.savetxt(
        csv_path,
        np.column_stack([np.asarray(diag[c]) for c in cols]),
        delimiter=",",
        header=",".join(cols),
        comments="",
    )
    print(f"\nWake metrics -> {csv_path}")

    st_result = compute_strouhal(np.asarray(diag["uy_probe"]), args.warmup_frac, D, U)
    summary = {
        "operator": "bgk" if model is None else "nn",
        "steps_requested": n_steps,
        "steps_completed": len(diag["step"]),
        "stability_horizon": horizon,  # None == stable for the whole run
        "first_negative_step": first_negative,  # min f_i < -1e-6 (diagnostic)
        "probe": [probe_x, probe_y],
        "warmup_frac": args.warmup_frac,
        "strouhal": st_result[0] if st_result else None,
        "shedding_freq": st_result[1] if st_result else None,
        "shedding_amplitude": st_result[2] if st_result else None,
        "final_energy": diag["energy"][-1] if diag["energy"] else None,
        "max_speed_overall": max(diag["max_speed"]) if diag["max_speed"] else None,
        "min_f_overall": min(diag["min_f"]) if diag["min_f"] else None,
    }
    summary_path = os.path.join(args.out_dir, "wake_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Wake summary -> {summary_path}")
    print(
        "  stability horizon :",
        "stable (full run)" if horizon is None else f"step {horizon}",
    )
    if st_result:
        print(
            f"  Strouhal number   : {st_result[0]:.4f}  (f_shed={st_result[1]:.5f}, amp={st_result[2]:.2e})"
        )
    else:
        print(
            "  Strouhal number   : n/a (signal too short — raise --anim-steps or lower --warmup-frac)"
        )

    if not frames_ux:
        print(
            "  No frames collected (increase --anim-steps or lower --update-steps); skipping GIF."
        )
        return

    fig, ax = plt.subplots(figsize=(10, 4), dpi=100)

    def update(i):
        ax.cla()
        speed = np.sqrt(frames_ux[i] ** 2 + frames_uy[i] ** 2)
        speed[obstacle] = np.nan
        ax.imshow(
            speed.T,
            origin="lower",
            cmap="jet",
            vmin=0,
            vmax=U_max,
            aspect="auto",
            extent=[0, Nx, 0, Ny],
        )
        up, vp = frames_ux[i].copy(), frames_uy[i].copy()
        up[obstacle] = vp[obstacle] = 0.0
        ax.streamplot(Xg.T, Yg.T, up.T, vp.T, density=0.5, color="w", linewidth=0.6)
        ax.set_title(f"{label} velocity — step {frame_steps[i]}", fontsize=12)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    anim = FuncAnimation(
        fig, update, frames=len(frames_ux), interval=1000 // max(args.gif_fps, 1)
    )
    gif_path = os.path.join(args.out_dir, "nn_velocity_field.gif")
    anim.save(gif_path, writer=PillowWriter(fps=args.gif_fps))
    plt.close(fig)
    print(f"  GIF saved to: {gif_path}")


# ---------------------------------------------------------------------------
# A-priori evaluation against recorded BGK snapshots
# ---------------------------------------------------------------------------


def evaluate_snapshots(model, args) -> None:
    """Score the NN against fpre/fpost pairs and write CSV + plots."""
    fpre_files = sorted(glob.glob(os.path.join(args.data_dir, "fpre_*.npy")))
    if args.max_snapshots:
        fpre_files = fpre_files[:: max(1, len(fpre_files) // args.max_snapshots)][
            : args.max_snapshots
        ]
    print(f"\nEvaluating on {len(fpre_files)} snapshot(s) from '{args.data_dir}'")

    steps, rmsre_s, mae_s, max_s = [], [], [], []
    eps = 1e-15
    for fpre_path in fpre_files:
        step_str = os.path.basename(fpre_path).replace("fpre_", "").replace(".npy", "")
        fpost_path = os.path.join(args.data_dir, f"fpost_{step_str}.npy")
        if not os.path.exists(fpost_path):
            print(f"  [SKIP] no matching fpost for {os.path.basename(fpre_path)}")
            continue
        Q = 9
        fpre_norm, _ = normalize(np.load(fpre_path).reshape(-1, Q))
        fpost_norm, _ = normalize(np.load(fpost_path).reshape(-1, Q))
        pred = model.predict(fpre_norm, batch_size=args.batch_size, verbose=0)
        rmsre_val = float(
            np.sqrt(np.mean(((fpost_norm - pred) / (fpost_norm + eps)) ** 2))
        )
        mae_val = float(np.mean(np.abs(fpost_norm - pred)))
        max_val = float(np.max(np.abs(fpost_norm - pred)))
        steps.append(int(step_str))
        rmsre_s.append(rmsre_val)
        mae_s.append(mae_val)
        max_s.append(max_val)
        print(
            f"  Step {int(step_str):>6d} | RMSRE={rmsre_val:.4e}  MAE={mae_val:.4e}  MaxErr={max_val:.4e}"
        )

    if not steps:
        print("  No valid snapshot pairs found; skipping metric plots.")
        return

    steps = np.array(steps)
    rmsre_s, mae_s, max_s = np.array(rmsre_s), np.array(mae_s), np.array(max_s)

    csv_path = os.path.join(args.out_dir, "eval_metrics.csv")
    np.savetxt(
        csv_path,
        np.column_stack([steps, rmsre_s, mae_s, max_s]),
        delimiter=",",
        header="step,rmsre,mae,max_abs_error",
        comments="",
    )
    print(f"\nMetrics -> {csv_path}")

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].semilogy(steps, rmsre_s, "o-", color="steelblue", lw=1.5, ms=4)
    axes[0].set_ylabel("RMSRE")
    axes[0].set_title("NN evaluation on Kármán vortex data")
    axes[1].semilogy(steps, mae_s, "o-", color="darkorange", lw=1.5, ms=4)
    axes[1].set_ylabel("MAE")
    axes[2].semilogy(steps, max_s, "o-", color="crimson", lw=1.5, ms=4)
    axes[2].set_ylabel("Max Abs Error")
    axes[2].set_xlabel("Simulation step")
    for a in axes:
        a.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = os.path.join(args.out_dir, "eval_metrics.png")
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot    -> {plot_path}")

    # Per-direction error for best/worst snapshots.
    best, worst = int(np.argmin(rmsre_s)), int(np.argmax(rmsre_s))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, idx, label in zip(axes, (best, worst), ("Best", "Worst")):
        s = f"{steps[idx]:06d}"
        fpre_norm, _ = normalize(
            np.load(os.path.join(args.data_dir, f"fpre_{s}.npy")).reshape(-1, 9)
        )
        fpost_norm, _ = normalize(
            np.load(os.path.join(args.data_dir, f"fpost_{s}.npy")).reshape(-1, 9)
        )
        pred = model.predict(fpre_norm, batch_size=args.batch_size, verbose=0)
        mean_per_dir = np.abs((fpost_norm - pred) / (fpost_norm + eps)).mean(axis=0)
        ax.bar(range(9), mean_per_dir, color="steelblue", edgecolor="k", linewidth=0.5)
        ax.set_xticks(range(9))
        ax.set_xticklabels([f"f{i}" for i in range(9)])
        ax.set_ylabel("Mean relative error")
        ax.set_title(f"{label} snapshot (step {steps[idx]})\nRMSRE={rmsre_s[idx]:.4e}")
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    dist_path = os.path.join(args.out_dir, "per_direction_error.png")
    fig.savefig(dist_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Per-dir -> {dist_path}")

    print("\n-- Summary --")
    print(f"  snapshots evaluated : {len(steps)}")
    print(f"  mean RMSRE          : {rmsre_s.mean():.4e}")
    print(
        f"  best / worst RMSRE  : {rmsre_s.min():.4e} (step {steps[best]}) / {rmsre_s.max():.4e} (step {steps[worst]})"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--model-path",
        default=None,
        help="Path to the saved Keras model (.keras). Required unless --bgk-only.",
    )
    p.add_argument(
        "--bgk-only",
        action="store_true",
        help="Run the classical BGK collision instead of a NN (control run).",
    )
    p.add_argument(
        "--data-dir",
        default=None,
        help="Dir with fpre_*.npy / fpost_*.npy for a-priori evaluation.",
    )
    p.add_argument(
        "--out-dir",
        default="eval_results_karman",
        help="Output directory for GIF / plots / CSV.",
    )
    p.add_argument(
        "--batch-size", type=int, default=4096, help="Batch size for model.predict()."
    )
    p.add_argument(
        "--seed", type=int, default=0, help="NumPy RNG seed (recorded in the manifest)."
    )

    p.add_argument(
        "--animate", action="store_true", help="Produce the NN-driven KVS velocity GIF."
    )
    p.add_argument(
        "--gif-fps", type=int, default=5, help="Frames per second for the GIF."
    )
    p.add_argument(
        "--update-steps",
        type=int,
        default=50,
        help="Collect a GIF frame every N steps.",
    )
    p.add_argument(
        "--anim-steps", type=int, default=5000, help="Total NN-driven simulation steps."
    )
    p.add_argument(
        "--snap-every",
        type=int,
        default=0,
        help="Save a PNG snapshot every N steps (0 disables).",
    )

    p.add_argument(
        "--res",
        type=int,
        default=250,
        help="Geometry resolution (res=250 -> 550x102, tau=0.5576).",
    )
    p.add_argument("--u-inlet", type=float, default=0.12, help="Inlet velocity U.")
    p.add_argument(
        "--re",
        type=float,
        default=150.0,
        help="Reynolds number (sets tau via nu=U*D/Re).",
    )
    p.add_argument(
        "--max-snapshots",
        type=int,
        default=None,
        help="Cap evaluation snapshots (evenly sub-sampled).",
    )

    p.add_argument(
        "--probe-x",
        type=int,
        default=None,
        help="Wake probe x (default: cylinder_x + 8*radius).",
    )
    p.add_argument(
        "--probe-y",
        type=int,
        default=None,
        help="Wake probe y (default: cylinder centreline).",
    )
    p.add_argument(
        "--warmup-frac",
        type=float,
        default=0.5,
        help="Fraction of the probe signal discarded before the Strouhal FFT (shedding onset).",
    )

    args = p.parse_args()
    if not args.bgk_only and not args.model_path:
        p.error("--model-path is required unless --bgk-only is given")
    return args


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)

    Nx, Ny, _, tau, nu = karman_geometry(args.res, args.u_inlet, args.re)
    write_manifest(args, Nx, Ny, tau, nu)

    K.set_floatx("float64")
    if args.bgk_only:
        model = None
        print("BGK-only control run (no model loaded).")
    else:
        print(f"Loading model: {args.model_path}")
        model = keras.models.load_model(
            args.model_path, custom_objects={"rmsre": rmsre}
        )
        model.summary()

    if (
        model is not None
        and args.data_dir
        and glob.glob(os.path.join(args.data_dir, "fpre_*.npy"))
    ):
        evaluate_snapshots(model, args)
    elif args.data_dir:
        reason = (
            "BGK-only run" if model is None else f"no fpre_*.npy in '{args.data_dir}'"
        )
        print(f"\nSkipping a-priori evaluation ({reason}).")

    if args.animate:
        make_animation(model, args)

    if not args.animate and not args.data_dir:
        print("\nNothing to do: pass --animate and/or --data-dir. See --help.")


if __name__ == "__main__":
    main()
