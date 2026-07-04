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

Two things it produces (mirroring step 4):
  * ``nn_velocity_field.gif`` — the NN-driven KVS wake over time (``--animate``),
    plus optional early PNG snapshots (``--snap-every``).
  * ``eval_metrics.{csv,png}`` + ``per_direction_error.png`` — a-priori RMSRE /
    MAE / max-abs error of the NN against BGK ground truth, per saved snapshot
    (only when ``--data-dir`` holds ``fpre_*.npy`` / ``fpost_*.npy`` pairs).

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
import os

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
# NN-driven Kármán simulation + animation
# ---------------------------------------------------------------------------


def make_animation(model, args) -> None:
    """Run an NN-collision KVS simulation and save a GIF (and optional PNGs)."""
    print("\nRunning NN-driven Kármán simulation for animation ...")
    Nx, Ny, obstacle, tau, nu = karman_geometry(args.res, args.u_inlet, args.re)
    U = args.u_inlet
    print(f"  Grid: {Nx} x {Ny},  tau={tau:.4f},  nu={nu:.6f},  Re={args.re}")

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
    n_steps, update_every, snap_every = args.anim_steps, args.update_steps, args.snap_every

    U_max = U * 2.0
    Xg, Yg = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing="ij")
    snap_dir = os.path.join(args.out_dir, "karman_snapshots")
    if snap_every > 0:
        os.makedirs(snap_dir, exist_ok=True)

    def save_snapshot(step_i, ux_i, uy_i):
        speed = np.sqrt(ux_i**2 + uy_i**2)
        speed[obstacle] = np.nan
        fig_s, ax_s = plt.subplots(figsize=(10, 4), dpi=100)
        ax_s.imshow(speed.T, origin="lower", cmap="jet", vmin=0, vmax=U_max, aspect="auto", extent=[0, Nx, 0, Ny])
        up, vp = ux_i.copy(), uy_i.copy()
        up[obstacle] = vp[obstacle] = 0.0
        ax_s.streamplot(Xg.T, Yg.T, up.T, vp.T, density=0.5, color="w", linewidth=0.6)
        ax_s.set_title(f"NN predicted velocity — step {step_i}", fontsize=12)
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

        # -- NN collision (mass-normalise / denormalise) --
        fpre = f.reshape(-1, 9)
        norm = np.sum(fpre, axis=1, keepdims=True)
        f_out = (model.predict(fpre / norm, batch_size=args.batch_size, verbose=0) * norm).reshape(Nx, Ny, 9)

        # -- Obstacle bounce-back --
        for i in range(9):
            f_out[obstacle, i] = f[obstacle, _OPP[i]]

        # -- Streaming --
        for i in range(9):
            f[:, :, i] = np.roll(np.roll(f_out[:, :, i], _c[i, 0], axis=0), _c[i, 1], axis=1)

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
            + (f[-1, iy, 0] + f[-1, iy, 2] + f[-1, iy, 4] + 2.0 * (f[-1, iy, 1] + f[-1, iy, 5] + f[-1, iy, 8]))
            / rho_out
        )
        ux_out = np.clip(ux_out, 0.0, 0.5)
        f[-1, iy, 3] = f[-1, iy, 1] - (2.0 / 3.0) * rho_out * ux_out
        f[-1, iy, 7] = f[-1, iy, 5] + 0.5 * (f[-1, iy, 2] - f[-1, iy, 4]) - (1.0 / 6.0) * rho_out * ux_out
        f[-1, iy, 6] = f[-1, iy, 8] - 0.5 * (f[-1, iy, 2] - f[-1, iy, 4]) - (1.0 / 6.0) * rho_out * ux_out
        for yc in (0, Ny - 1):
            f[-1, yc, 3] = f[-2, yc, 3]
            f[-1, yc, 6] = f[-2, yc, 6]
            f[-1, yc, 7] = f[-2, yc, 7]

        # -- Inlet BC (Zou-He velocity) --
        rho_in = (f[0, :, 0] + f[0, :, 2] + f[0, :, 4] + 2.0 * (f[0, :, 3] + f[0, :, 6] + f[0, :, 7])) / (1.0 - U)
        f[0, :, 1] = f[0, :, 3] + (2.0 / 3.0) * rho_in * U
        f[0, :, 5] = f[0, :, 7] - 0.5 * (f[0, :, 2] - f[0, :, 4]) + (1.0 / 6.0) * rho_in * U
        f[0, :, 8] = f[0, :, 6] + 0.5 * (f[0, :, 2] - f[0, :, 4]) + (1.0 / 6.0) * rho_in * U

        if step % update_every == 0:
            frames_ux.append(ux.copy())
            frames_uy.append(uy.copy())
            frame_steps.append(step)
            print(f"  Frame saved at step {step}/{n_steps}")
        if snap_every > 0 and step % snap_every == 0:
            save_snapshot(step, ux, uy)

    if not frames_ux:
        print("  No frames collected (increase --anim-steps or lower --update-steps); skipping GIF.")
        return

    fig, ax = plt.subplots(figsize=(10, 4), dpi=100)

    def update(i):
        ax.cla()
        speed = np.sqrt(frames_ux[i] ** 2 + frames_uy[i] ** 2)
        speed[obstacle] = np.nan
        ax.imshow(speed.T, origin="lower", cmap="jet", vmin=0, vmax=U_max, aspect="auto", extent=[0, Nx, 0, Ny])
        up, vp = frames_ux[i].copy(), frames_uy[i].copy()
        up[obstacle] = vp[obstacle] = 0.0
        ax.streamplot(Xg.T, Yg.T, up.T, vp.T, density=0.5, color="w", linewidth=0.6)
        ax.set_title(f"NN predicted velocity — step {frame_steps[i]}", fontsize=12)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    anim = FuncAnimation(fig, update, frames=len(frames_ux), interval=1000 // max(args.gif_fps, 1))
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
        fpre_files = fpre_files[:: max(1, len(fpre_files) // args.max_snapshots)][: args.max_snapshots]
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
        rmsre_val = float(np.sqrt(np.mean(((fpost_norm - pred) / (fpost_norm + eps)) ** 2)))
        mae_val = float(np.mean(np.abs(fpost_norm - pred)))
        max_val = float(np.max(np.abs(fpost_norm - pred)))
        steps.append(int(step_str))
        rmsre_s.append(rmsre_val)
        mae_s.append(mae_val)
        max_s.append(max_val)
        print(f"  Step {int(step_str):>6d} | RMSRE={rmsre_val:.4e}  MAE={mae_val:.4e}  MaxErr={max_val:.4e}")

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
        fpre_norm, _ = normalize(np.load(os.path.join(args.data_dir, f"fpre_{s}.npy")).reshape(-1, 9))
        fpost_norm, _ = normalize(np.load(os.path.join(args.data_dir, f"fpost_{s}.npy")).reshape(-1, 9))
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
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", required=True, help="Path to the saved Keras model (.keras).")
    p.add_argument("--data-dir", default=None, help="Dir with fpre_*.npy / fpost_*.npy for a-priori evaluation.")
    p.add_argument("--out-dir", default="eval_results_karman", help="Output directory for GIF / plots / CSV.")
    p.add_argument("--batch-size", type=int, default=4096, help="Batch size for model.predict().")

    p.add_argument("--animate", action="store_true", help="Produce the NN-driven KVS velocity GIF.")
    p.add_argument("--gif-fps", type=int, default=5, help="Frames per second for the GIF.")
    p.add_argument("--update-steps", type=int, default=50, help="Collect a GIF frame every N steps.")
    p.add_argument("--anim-steps", type=int, default=5000, help="Total NN-driven simulation steps.")
    p.add_argument("--snap-every", type=int, default=0, help="Save a PNG snapshot every N steps (0 disables).")

    p.add_argument("--res", type=int, default=250, help="Geometry resolution (res=250 -> 550x102, tau=0.5576).")
    p.add_argument("--u-inlet", type=float, default=0.12, help="Inlet velocity U.")
    p.add_argument("--re", type=float, default=150.0, help="Reynolds number (sets tau via nu=U*D/Re).")
    p.add_argument("--max-snapshots", type=int, default=None, help="Cap evaluation snapshots (evenly sub-sampled).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    K.set_floatx("float64")
    print(f"Loading model: {args.model_path}")
    model = keras.models.load_model(args.model_path, custom_objects={"rmsre": rmsre})
    model.summary()

    if args.data_dir and glob.glob(os.path.join(args.data_dir, "fpre_*.npy")):
        evaluate_snapshots(model, args)
    elif args.data_dir:
        print(f"\nNo fpre_*.npy in '{args.data_dir}' — skipping evaluation.")

    if args.animate:
        make_animation(model, args)

    if not args.animate and not args.data_dir:
        print("\nNothing to do: pass --animate and/or --data-dir. See --help.")


if __name__ == "__main__":
    main()
