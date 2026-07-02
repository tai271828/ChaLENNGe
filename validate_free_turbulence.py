#!/usr/bin/env python3
"""Freely-decaying turbulence a-posteriori validation (2D D2Q9).

Reproduces the methodology of the 3D freely-decaying turbulence test in
Ortali/Gabbana et al. (2025), Sec. IV.C / Fig. 8, inside this repo's 2D
framework:

    1. build a developed, turbulent-like initial condition with forced
       ground-truth BGK (force on),
    2. freely decay it (force off) with the ground-truth BGK *and* each trained
       ML collision operator from the same initial state,
    3. compare energy decay E(t), its log-derivative d log E / dt, and the
       a-posteriori velocity error (Eq. 50).

Examples
--------
Validate the latest run of one model against ground-truth BGK::

    python validate_free_turbulence.py --model d4equivariant

Validate several trained models side by side (each --run points at a run dir
that contains a model.keras)::

    python validate_free_turbulence.py \
        --run artifacts-run-all-tensorflow/d4equivariant_20260101-000000:GAVG \
        --run artifacts-run-all-tensorflow/plain_2_20260101-000000:MLP \
        --tau 0.51 --n-transient 20000 --n-decay 200

Quick smoke test with no model (ground-truth BGK only, tiny grid)::

    python validate_free_turbulence.py --no-model --nx 16 --ny 16 \
        --n-transient 50 --n-decay 20
"""

from __future__ import annotations

import argparse
import datetime
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lbm_ml.model.losses import rmsre
from lbm_ml.model.network import MODEL_REGISTRY  # noqa: F401  (kept for --model discovery)
from lbm_ml.validation.free_turbulence import (
    TurbulenceConfig,
    log_derivative,
    run_validation,
)

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts-run-all-tensorflow"


def setup_logging(verbose: bool = True) -> None:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(message)s", force=True)


def _latest_run_dir(model_name: str) -> Path:
    matches = sorted(ARTIFACTS_DIR.glob(f"{model_name}_*"), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No runs found for model '{model_name}' in {ARTIFACTS_DIR}")
    return matches[-1]


def _load_model(run_dir: Path):
    """Load a trained keras model from a run directory (expects model.keras)."""
    import keras

    model_path = run_dir / "model.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"No model.keras in {run_dir}")
    return keras.models.load_model(str(model_path), custom_objects={"rmsre": rmsre})


def _collect_models(args) -> dict[str, object]:
    """Resolve the --model / --run flags into a {label: keras_model} mapping."""
    import keras

    keras.backend.set_floatx("float64")

    models: dict[str, object] = {}
    if args.model:
        run_dir = Path(args.run_dir) if args.run_dir else _latest_run_dir(args.model)
        logger.info("Loading model '%s' from %s", args.model, run_dir)
        models[args.model] = _load_model(run_dir)

    for spec in args.run or []:
        # "path" or "path:label" (label must not look like a path segment).
        if ":" in spec and "/" not in spec.rsplit(":", 1)[1]:
            path_str, label = spec.rsplit(":", 1)
        else:
            path_str, label = spec, None
        run_dir = Path(path_str)
        label = label or run_dir.name
        logger.info("Loading model '%s' from %s", label, run_dir)
        models[label] = _load_model(run_dir)

    return models


# ---------------------------------------------------------------------------
# Plotting (Fig. 8 b-d analogues)
# ---------------------------------------------------------------------------


def _plot_energy_decay(results, cfg, out_path: Path) -> None:
    """E(t) on a log axis for every operator (Fig. 8b)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    t = np.arange(cfg.n_decay + 1)
    for name, r in results.items():
        style = dict(linewidth=2.5, color="k") if name == "bgk" else dict(linewidth=1.5)
        label = "BGK (truth)" if name == "bgk" else name
        ax.semilogy(t, r.energy, label=label, **style)
    ax.set_xlabel(r"$t~\rm{[L.U.]}$", fontsize=14)
    ax.set_ylabel(r"$\langle E \rangle = \langle \frac{1}{2}|u|^2 \rangle$", fontsize=14)
    ax.set_title("Free-decay kinetic energy", fontsize=14)
    ax.legend(frameon=False)
    ax.tick_params(direction="in", top=True, right=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Energy decay plot -> %s", out_path)


def _plot_log_derivative(results, cfg, out_path: Path) -> None:
    """d log E / dt (Fig. 8c); a flat line == clean exponential decay."""
    fig, ax = plt.subplots(figsize=(7, 5))
    t = np.arange(cfg.n_decay + 1)
    t0, t1 = cfg.error_window
    for name, r in results.items():
        style = dict(linewidth=2.5, color="k") if name == "bgk" else dict(linewidth=1.5)
        label = "BGK (truth)" if name == "bgk" else name
        ax.plot(t, log_derivative(r.energy), label=label, **style)
    ax.axvspan(t0, t1, color="0.9", zorder=0, label=f"avg window [{t0},{t1}]")
    ax.set_xlabel(r"$t~\rm{[L.U.]}$", fontsize=14)
    ax.set_ylabel(r"$\partial \log E / \partial t$", fontsize=14)
    ax.set_title("Logarithmic decay rate", fontsize=14)
    ax.legend(frameon=False)
    ax.tick_params(direction="in", top=True, right=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Log-derivative plot -> %s", out_path)


def _plot_velocity_fields(results, cfg, out_path: Path, snap_step: int) -> None:
    """Velocity-magnitude snapshot per operator at ``snap_step`` (Fig. 8a)."""
    names = list(results)
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4), squeeze=False)
    step = min(snap_step, cfg.n_decay)
    for ax, name in zip(axes[0], names):
        r = results[name]
        v = r.velocity[step]
        umag = np.sqrt(v[:, :, 0] ** 2 + v[:, :, 1] ** 2) if np.all(np.isfinite(v)) else np.zeros((cfg.nx, cfg.ny))
        im = ax.imshow(umag.T, origin="lower", cmap="viridis")
        title = "BGK (truth)" if name == "bgk" else name
        if r.diverged:
            title += " [diverged]"
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"|u| at decay step {step}", fontsize=13)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Velocity-field snapshot -> %s", out_path)


def _animate_velocity_fields(results, cfg, out_path: Path, fps: int = 10, streamlines: bool = False) -> None:
    """Animate |u|(x, y, t) over the free-decay steps, one panel per operator.

    A **fixed** colour scale (shared across panels and frames) is used on
    purpose: as the flow decays the field literally fades, so the animation
    shows the real energy decay rather than a per-frame-renormalised picture.
    Diverged operators go blank once their populations become non-finite.
    Saved as an animated GIF (Pillow writer, no external dependency).
    """
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    names = list(results)
    n = len(names)

    # Pre-compute |u| for every operator/frame and a single global colour scale.
    mag = {name: np.sqrt(r.velocity[..., 0] ** 2 + r.velocity[..., 1] ** 2) for name, r in results.items()}
    finite_max = [np.nanmax(m[np.isfinite(m)]) for m in mag.values() if np.any(np.isfinite(m))]
    vmax = max(finite_max) if finite_max else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    norm = Normalize(vmin=0.0, vmax=vmax)

    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6), squeeze=False)
    axes = axes[0]
    fig.colorbar(ScalarMappable(norm=norm, cmap="viridis"), ax=list(axes), shrink=0.8, label="|u|")

    X, Y = np.meshgrid(np.arange(cfg.nx), np.arange(cfg.ny))
    n_frames = cfg.n_decay + 1

    def draw(t: int):
        for ax, name in zip(axes, names):
            ax.clear()
            v = results[name].velocity[t]
            finite = np.all(np.isfinite(v))
            m = mag[name][t]
            ax.imshow(np.nan_to_num(m).T, origin="lower", cmap="viridis", norm=norm)
            if streamlines and finite:
                ax.streamplot(X, Y, v[:, :, 0].T, v[:, :, 1].T, density=0.6, color="w", linewidth=0.6, arrowsize=0.6)
            title = "BGK (truth)" if name == "bgk" else name
            if not finite:
                title += " [diverged]"
            ax.set_title(title, fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"|u| — decay step {t}/{cfg.n_decay}", fontsize=13)

    anim = FuncAnimation(fig, draw, frames=n_frames, interval=1000 / max(fps, 1))
    anim.save(str(out_path), writer=PillowWriter(fps=fps))
    plt.close(fig)
    logger.info("  Velocity animation -> %s", out_path)


def _write_summary(results, cfg, out_path: Path) -> None:
    lines = [
        "Freely-decaying turbulence validation (2D D2Q9)",
        "=" * 48,
        f"grid={cfg.nx}x{cfg.ny}  tau={cfg.tau}  nu={cfg.nu:.4e}",
        f"force_amp={cfg.force_amp:.2e}  mode={cfg.force_mode}",
        f"n_transient={cfg.n_transient}  n_decay={cfg.n_decay}  window={cfg.error_window}",
        "",
        f"{'operator':<22}{'a-post err':>14}{'log-rate':>14}{'rate rel.err':>14}{'diverged':>10}",
        "-" * 74,
    ]
    for name, r in results.items():
        lines.append(
            f"{name:<22}{r.aposteriori:>14.4e}{r.mean_log_deriv:>14.4e}"
            f"{r.rate_rel_error:>14.4e}{str(r.diverged):>10}"
        )
    text = "\n".join(lines)
    out_path.write_text(text + "\n")
    logger.info("\n%s", text)
    logger.info("  Summary -> %s", out_path)


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=None, choices=list(MODEL_REGISTRY), help="Validate the latest run of this model.")
    p.add_argument("--run-dir", default=None, help="Explicit run dir for --model (default: latest run).")
    p.add_argument(
        "--run",
        action="append",
        default=None,
        help="Add a trained model by run dir, optionally 'path:label'. Repeatable.",
    )
    p.add_argument("--no-model", action="store_true", help="Run ground-truth BGK only (smoke test).")

    p.add_argument("--nx", type=int, default=32)
    p.add_argument("--ny", type=int, default=32)
    p.add_argument("--tau", type=float, default=0.51, help="Relaxation time; match the model's training tau.")
    p.add_argument("--force-amp", type=float, default=5e-6, help="Transient force amplitude A (paper: 5e-6).")
    p.add_argument("--force-mode", type=int, default=1, help="Forcing wavenumber mode m (k=2*pi*m/L).")
    p.add_argument("--n-transient", type=int, default=20000, help="Forced BGK steps to develop the flow.")
    p.add_argument("--n-decay", type=int, default=200, help="Free-decay steps to compare (paper T=200).")
    p.add_argument("--seed-perturbation", type=float, default=1e-4, help="Amplitude of divergence-free IC kick.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--window",
        type=int,
        nargs=2,
        default=(50, 200),
        metavar=("T0", "T1"),
        help="Step range for averaging the log decay rate (paper: 50 200).",
    )
    p.add_argument("--snap-step", type=int, default=200, help="Decay step for the velocity-field snapshot.")

    p.add_argument("--no-animate", action="store_true", help="Skip the |u|(x,y,t) decay animation (GIF).")
    p.add_argument("--fps", type=int, default=10, help="Frames per second for the decay animation.")
    p.add_argument("--streamlines", action="store_true", help="Overlay velocity streamlines on each animation frame.")

    p.add_argument("--out-dir", default=None, help="Output directory (default: timestamped under artifacts).")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    setup_logging(verbose=not args.quiet)

    cfg = TurbulenceConfig(
        nx=args.nx,
        ny=args.ny,
        tau=args.tau,
        force_amp=args.force_amp,
        force_mode=args.force_mode,
        n_transient=args.n_transient,
        n_decay=args.n_decay,
        seed_perturbation=args.seed_perturbation,
        seed=args.seed,
        error_window=(int(args.window[0]), int(args.window[1])),
    )

    models = {} if args.no_model else _collect_models(args)
    if not models and not args.no_model:
        logger.warning("No models selected (use --model / --run, or --no-model). Running BGK only.")

    results = run_validation(cfg, models)

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = ARTIFACTS_DIR / f"free_turbulence_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output dir: %s", out_dir)

    _plot_energy_decay(results, cfg, out_dir / "energy_decay.png")
    _plot_log_derivative(results, cfg, out_dir / "log_derivative.png")
    _plot_velocity_fields(results, cfg, out_dir / "velocity_fields.png", args.snap_step)
    if not args.no_animate:
        _animate_velocity_fields(
            results, cfg, out_dir / "velocity_evolution.gif", fps=args.fps, streamlines=args.streamlines
        )
    _write_summary(results, cfg, out_dir / "summary.txt")

    # Save raw curves for later re-plotting / aggregation.
    np.savez(
        out_dir / "curves.npz",
        **{f"energy_{name}": r.energy for name, r in results.items()},
    )
    logger.info("  Curves -> %s", out_dir / "curves.npz")


if __name__ == "__main__":
    main()
