#!/usr/bin/env python3
"""Visualize the wake-dynamics output of apply_nn_karman.py in one figure.

Reads ``wake_metrics.csv`` + ``wake_summary.json`` (+ ``manifest.json`` for
D and U) from a result directory and writes ``wake_overview.png`` with four
panels: probe u_y(t), its amplitude spectrum in Strouhal units, the kinetic
energy trace, and max|u| against the divergence threshold.

Usage::

    python -m eval_helpers.plot_wake [result_dir]   # default: eval_results_bgk_control
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE = "#2a78d6"  # data series
INK = "#52514e"  # annotations / reference lines
GRID = dict(alpha=0.25, linewidth=0.6)


def _style(ax):
    ax.grid(True, **GRID)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main() -> None:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "eval_results_bgk_control")
    d = np.loadtxt(out_dir / "wake_metrics.csv", delimiter=",", skiprows=1)
    step, ux_p, uy_p, energy, max_speed, min_f = d.T
    summary = json.loads((out_dir / "wake_summary.json").read_text())
    manifest = json.loads((out_dir / "manifest.json").read_text())

    U = manifest["physics"]["u_inlet"]
    D = 2 * int(round(0.05 * manifest["physics"]["res"]))
    warmup = summary["warmup_frac"]
    n0 = int(len(uy_p) * warmup)

    # Spectrum of the post-warmup probe signal, x-axis in Strouhal units.
    sig = uy_p[n0:] - uy_p[n0:].mean()
    win = np.hanning(len(sig))
    spec = 2.0 * np.abs(np.fft.rfft(sig * win)) / win.sum()
    st_axis = np.fft.rfftfreq(len(sig)) * D / U

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), dpi=150)
    (ax_sig, ax_spec), (ax_e, ax_u) = axes

    ax_sig.axvspan(step[0], step[n0], color=INK, alpha=0.08, lw=0)
    ax_sig.plot(step, uy_p, color=BLUE, lw=1.0)
    ax_sig.set_title(
        f"Probe $u_y(t)$ at {tuple(summary['probe'])} — shaded = warm-up", fontsize=10
    )
    ax_sig.set_xlabel("step")
    ax_sig.set_ylabel("$u_y$ (lattice units)")

    ax_spec.plot(st_axis, spec, color=BLUE, lw=1.2)
    ax_spec.set_xlim(0, 1.0)
    if summary["strouhal"]:
        st = summary["strouhal"]
        ax_spec.axvline(st, color=INK, lw=0.8, ls="--")
        ax_spec.annotate(
            f"St = {st:.3f}",
            (st, spec.max()),
            xytext=(6, -2),
            textcoords="offset points",
            fontsize=9,
            color=INK,
        )
    ax_spec.set_title("Amplitude spectrum of probe $u_y$", fontsize=10)
    ax_spec.set_xlabel("Strouhal number  $f\\,D/U$")
    ax_spec.set_ylabel("amplitude")

    ax_e.plot(step, energy, color=BLUE, lw=1.2)
    ax_e.set_title(
        "Mean kinetic energy $\\langle \\frac{1}{2}|u|^2 \\rangle$ (fluid nodes)",
        fontsize=10,
    )
    ax_e.set_xlabel("step")
    ax_e.set_ylabel("E")

    ax_u.plot(step, max_speed, color=BLUE, lw=1.2)
    ax_u.axhline(0.4, color=INK, lw=0.8, ls="--")
    ax_u.annotate(
        "divergence threshold 0.4",
        (step[0], 0.4),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=9,
        color=INK,
    )
    ax_u.set_ylim(0, 0.45)
    ax_u.set_title("max $|u|$ over fluid nodes", fontsize=10)
    ax_u.set_xlabel("step")
    ax_u.set_ylabel("max $|u|$")

    for ax in axes.flat:
        _style(ax)

    horizon = summary["stability_horizon"]
    fig.suptitle(
        f"{summary['operator'].upper()} Kármán rollout — {summary['steps_completed']} steps, "
        f"St = {summary['strouhal']:.3f}, "
        f"{'stable (full run)' if horizon is None else f'DIVERGED at step {horizon}'}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = out_dir / "wake_overview.png"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Overview -> {out_path}")


if __name__ == "__main__":
    main()
