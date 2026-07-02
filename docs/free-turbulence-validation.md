# Free-turbulence validation (2D)

This is the a-posteriori **freely-decaying turbulence** test of Ortali/Gabbana
et al. (2025), *AIAA J.* **63**(2), Sec. IV.C / Fig. 8, adapted to this repo's
2D D2Q9 framework. The paper's test is 3D (D3Q27); the *methodology* is
dimension-agnostic, so here it runs inside the existing 2D stencil, D4 symmetry,
and model registry.

## Why this test matters

The Taylor–Green vortex (`run_all.py simulate`) is a *laminar* stability check.
Turbulence is harder: *"due to the chaotic nature of turbulence, the networks
need to model effectively a much wider set of pre–post collision pairs than in
the laminar cases."* A non-equivariant MLP typically **diverges within ~100
steps**, while the equivariant models (GAVG/LENN) stay stable with a decay rate
oscillating around the ground-truth BGK value. Because turbulence is chaotic,
the ML trajectory is not expected to track BGK exactly — only to stay physically
consistent (stable, correct decay rate).

## The recipe

1. **Transient (forcing on).** From rest + a small divergence-free kick, run the
   *ground-truth* BGK collision with a stationary body force until a developed,
   multi-scale flow emerges. This yields one initial condition `f0` shared by
   every operator. Forcing uses the second-order-accurate Guo (2002) scheme; the
   force is a 2D analogue of the paper's ABC-type force (Eq. 51):
   `Fx = A sin(k y)`, `Fy = A sin(k x)`, `k = 2πm/L`.
2. **Free decay (forcing off).** From the same `f0`, roll out each operator — the
   ground-truth BGK and every trained ML model — with the force switched off.
3. **Metrics** (Fig. 8b–d, Eqs. 49–50):
   - **`E(t)`** — mean kinetic energy `⟨½|u|²⟩`, should decay ~exponentially.
   - **`∂ log E / ∂t`** — a flat line == clean exponential decay; the BGK curve
     is the reference.
   - **a-posteriori error** — normalised velocity MSE integrated in space and
     averaged over the first `T` steps (Eq. 50), referenced to BGK.

## Physical consistency: pick the right `tau`

An ML collision operator only reproduces BGK at the relaxation time `tau` it was
**trained on**. The transient and the decay use the *same* `tau`, so `f0` is
consistent with the flow the model must sustain.

- The shipped synthetic dataset (`generate_dataset`) uses `tau = 1` → viscous,
  effectively laminar decay.
- A genuinely turbulent decay needs a near-0.5 `tau` (paper transient: `0.51`)
  **and** a model trained at that `tau` (e.g. a KVS-dataset model).

Steady transient velocity scales like `A / (ν k²)` with `ν = (tau − ½)·cs²`, so
low `tau` both raises the Reynolds number and, for a fixed target velocity,
allows the paper's tiny `A = 5e-6`. The driver logs `u_rms` and an estimated
Reynolds number after the transient — use them to sanity-check the flow.

## Usage

```bash
# Validate the latest run of one model against ground-truth BGK
python validate_free_turbulence.py --model d4equivariant --tau 0.51

# Compare several trained models (each --run points at a dir with model.keras)
python validate_free_turbulence.py \
    --run artifacts-run-all-tensorflow/d4equivariant_.../:GAVG \
    --run artifacts-run-all-tensorflow/plain_2_.../:MLP \
    --tau 0.51 --n-transient 20000 --n-decay 200

# Quick smoke test — ground-truth BGK only, tiny grid, few steps
python validate_free_turbulence.py --no-model --nx 16 --ny 16 \
    --n-transient 300 --n-decay 40 --window 5 40
```

Key flags (see `--help` for all): `--nx/--ny` (paper `L=32`), `--tau`,
`--force-amp` (`A`, paper `5e-6`), `--force-mode` (`m`), `--n-transient`,
`--n-decay` (paper `T=200`), `--window T0 T1` (log-rate averaging interval,
paper `[50,200]`), `--seed-perturbation`, `--seed`.

## Outputs

A timestamped directory under `artifacts-run-all-tensorflow/` (or `--out-dir`)
containing:

- `energy_decay.png` — `E(t)` per operator (Fig. 8b analogue).
- `log_derivative.png` — `∂ log E / ∂t` with the averaging window shaded (8c).
- `velocity_fields.png` — `|u|` snapshot per operator, flagging diverged runs (8a).
- `velocity_evolution.gif` — animation of `|u|(x, y, t)` over the decay, one
  panel per operator (Fig. 8a in motion). A **fixed** colour scale is shared
  across panels and frames so the fade honestly shows the energy decay rather
  than a per-frame renormalisation; diverged operators go blank. Control with
  `--fps`, `--streamlines` (overlay velocity streamlines), or `--no-animate`.
- `summary.txt` — per-operator a-posteriori error, mean log-rate, and relative
  decay-rate error vs BGK.
- `curves.npz` — raw energy curves for later re-plotting / aggregation.

### Reading the animation honestly

Because a valid ML operator tracks BGK, the panels look near-identical and, in
a low-Reynolds / short run, the field barely fades — both are *correct*
outcomes, not a bug. The animation becomes visually interesting when (a) the run
is long / the decay is fast enough to see the fade, or (b) a model *fails*: a
non-equivariant MLP typically breaks lattice symmetry and then blanks out
(`[diverged]`) within ~100 steps, which is exactly the contrast the test is
designed to expose.

## Code map

- `lbm_ml/validation/free_turbulence.py` — physics + metrics
  (`build_turbulent_ic`, `bgk_collide`, `ml_collide`, `run_free_decay`,
  `total_energy`, `log_derivative`, `aposteriori_error`, `run_validation`).
- `validate_free_turbulence.py` — CLI, model loading, and plotting.
