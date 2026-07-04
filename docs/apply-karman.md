# Applying a learned collision operator to a Kármán vortex street

`apply_nn_karman.py` is the ChaLENNGe-native port of the **step-4 "apply"** stage
in `phase01/run-exp01-snellius.sh` (originally the external `apply-nn.py`). It
drives a full Kármán-vortex-street (KVS) simulation with the neural network as
the collision operator and, optionally, scores the network against recorded BGK
`fpre`/`fpost` snapshots.

Why a repo-native port: the external `apply-nn.py` only registered the GAVG
custom layers, so it could not load **LENN / LENN+ResNet** models. This version
imports `lbm_ml`, whose `LENNLayer` and `*AlgReconstruction` layers self-register
via `@keras.saving.register_keras_serializable`, so those models load out of the
box.

## What it produces

- `nn_velocity_field.gif` — the NN-driven KVS wake over time (`--animate`), with
  optional early PNG snapshots (`--snap-every N`).
- `eval_metrics.csv` / `eval_metrics.png` — a-priori RMSRE / MAE / max-abs error
  of the NN vs BGK ground truth, per saved snapshot (needs `--data-dir`).
- `per_direction_error.png` — mean relative error per population `f0..f8` for the
  best/worst snapshots.
- `wake_metrics.csv` — per-step rollout diagnostics (with `--animate`): probe
  `u_x`/`u_y`, mean kinetic energy over fluid nodes, `max|u|`, `min f_i`.
- `wake_summary.json` — Strouhal number and **stability horizon** (see below).
- `manifest.json` — provenance record written on every invocation: command line,
  git commit, model path + SHA256, seed, physics (τ, grid, Re), run settings.

## Wake-dynamics metrics (a-posteriori)

These measure the rollout itself rather than single-step fidelity — they are
the primary evidence for architecture comparisons (see
`context-plan-guidance/02-resnet-karman-vv-plan.md`).

- **Probe + Strouhal.** `u_y(t)` is recorded at a wake probe (default:
  8 radii downstream of the cylinder on its centreline; override with
  `--probe-x/--probe-y`). After discarding the first `--warmup-frac` (default
  0.5) of the signal, the dominant FFT frequency gives
  `St = f_shed · D / U_inlet`. **Expect St ≈ 0.28 here, not the unconfined
  ≈0.18**: this geometry (2.2×0.41 domain, D=0.1 at (0.2, 0.2)) is the
  confined Schäfer–Turek benchmark with 23.5 % blockage, which raises the
  shedding frequency (benchmark St ≈ 0.30 at Re=100). Measured BGK control at
  Re=150, 20 000 steps: St = 0.28, probe amplitude ≈ 0.10
  (`eval_results_bgk_control/`). The authoritative reference is always a
  `--bgk-only` run of the *same* configuration, which cancels discretization
  bias. The signal needs a developed wake — use `--anim-steps` ≥ 20000; short
  runs report a meaningless peak.
- **Stability horizon.** First step where the state goes bad: non-finite `f`
  or `max|u| > 0.4` (lattice units). The run stops there;
  `stability_horizon: null` in the summary means stable for the full run.
  Population negativity is **not** a stop condition — mildly negative `f_i`
  are normal for BGK near τ=0.5 (observed: min_f ≈ −2e-4 in the Re=150 control
  during shedding onset). Instead `first_negative_step` records when
  `min f_i` first drops below −1e-6, useful for positivity comparisons across
  models.
- **Energy trace.** `E(t) = ⟨½|u|²⟩` over fluid nodes, per step — catches slow
  drift that the divergence check misses.
- **BGK control.** `--bgk-only` runs the classical BGK collision instead of a
  model (no `--model-path` needed) and produces the same metrics/GIF.

## Physics / geometry

Matches `lbm_karman-ng.py` defaults: `res=250` → 550×102 grid, cylinder at
(0.2·res, 0.2·res) radius 0.05·res, `U_inlet=0.12`, `Re=150` ⇒ `nu=0.0192`,
**`tau=0.5576`**. That `tau` is exactly what the Kármán-trained models learned
(verify empirically as in the free-turbulence notes), so the applied operator is
physically consistent with the flow. Boundary conditions: obstacle + top/bottom
wall bounce-back, Zou-He velocity inlet, Zou-He pressure outlet. Streaming is
periodic via `np.roll`; the BCs overwrite the wrapped edges each step.

## Usage

```bash
# Animation + evaluation with a LENN model (the step-4 shape)
python apply_nn_karman.py --animate \
    --model-path /path/to/lenn_.../model.keras \
    --data-dir   /path/to/karman/every_100 \
    --out-dir    eval_results_lenn

# Quick local check — short animation, few eval snapshots
python apply_nn_karman.py --animate --anim-steps 400 --update-steps 100 \
    --model-path /path/to/model.keras \
    --data-dir /path/to/karman/every_100 --max-snapshots 5 \
    --out-dir /tmp/kvs_check

# BGK ground-truth control (Strouhal / energy / stability reference)
python apply_nn_karman.py --bgk-only --animate --anim-steps 20000 \
    --update-steps 2000 --out-dir eval_results_bgk_control
```

Key flags: `--anim-steps` (total NN-driven steps; vortex shedding needs several
thousand), `--update-steps` (frame cadence), `--gif-fps`, `--snap-every`,
`--batch-size` (predict batch; 4096 is a good default), `--res/--u-inlet/--re`
(geometry/physics), `--max-snapshots` (cap/sub-sample evaluation snapshots),
`--bgk-only` (control run), `--probe-x/--probe-y/--warmup-frac` (wake metrics),
`--seed` (recorded in the manifest).

## Notes

- The a-priori RMSRE grows with simulation step because the wake becomes richer
  (a wider distribution of collision states) further downstream in time — the
  early, near-uniform inflow is trivial to reproduce, the developed wake is not.
- A developed shedding wake (the "we caught the butterfly" animation) needs
  `--anim-steps` in the tens of thousands; run that on Snellius, not the laptop.
