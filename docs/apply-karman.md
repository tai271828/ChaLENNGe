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
```

Key flags: `--anim-steps` (total NN-driven steps; vortex shedding needs several
thousand), `--update-steps` (frame cadence), `--gif-fps`, `--snap-every`,
`--batch-size` (predict batch; 4096 is a good default), `--res/--u-inlet/--re`
(geometry/physics), `--max-snapshots` (cap/sub-sample evaluation snapshots).

## Notes

- The a-priori RMSRE grows with simulation step because the wake becomes richer
  (a wider distribution of collision states) further downstream in time — the
  early, near-uniform inflow is trivial to reproduce, the developed wake is not.
- A developed shedding wake (the "we caught the butterfly" animation) needs
  `--anim-steps` in the tens of thousands; run that on Snellius, not the laptop.
