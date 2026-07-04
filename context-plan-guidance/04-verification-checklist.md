# Verification checklist (recurring)

Run the relevant subset whenever a model architecture, reconstruction layer, or
physics routine changes; run the full list before any result is reported as a
finding. Commands assume repo root and `uv run`.

## 1. Structural properties (fast, laptop)

- [ ] **Parameter count** — comparisons are budget-matched
  (`00-README.md` rule 3):
  ```bash
  uv run python -m eval_helpers.count_model_params
  ```
- [ ] **D4 equivariance** — the operator commutes with all 8 group actions.
  For equivariant families (`d4equivariant*`, `resnet*`, `lenn*`), max
  deviation should be at float precision (≲1e-6 for float32); `plain_*` is
  *expected* to fail (that's its role):
  ```bash
  uv run python -m eval_helpers.equivariance_inspect
  ```
- [ ] **Conservation** — mass and momentum of `f_out` match `f_in` per sample
  (reconstruction variants: exact; `_softmax`-only: mass exact via
  normalize/denormalize, check momentum drift):
  ```bash
  uv run python -m eval_helpers.constraints_check
  ```
- [ ] **Positivity** — with `_safe`/`_softmax` variants, `min f_i ≥ 0` on a
  stress batch (near-vacuum populations, high non-equilibrium).
- [ ] **Serialization round-trip** — `model.keras` saves and loads through
  `eval_helpers/rebuild.py` (custom layers self-register; a load failure means
  a missing `@keras.saving.register_keras_serializable`).

## 2. A-priori accuracy

- [ ] RMSRE on held-out `(f_pre, f_post)` pairs from the training distribution
  (`eval_helpers/fetch_rmsre.py` for TensorBoard runs).
- [ ] Per-direction error plot (`per_direction_error.png` from
  `apply_nn_karman.py`) — asymmetry across the `f1..f8` orbits on an
  equivariant model indicates a symmetry bug, not noise.

## 3. A-posteriori (rollout) behavior

- [ ] **Laminar sanity** — Taylor–Green via `run_all.py` simulate stage:
  decays smoothly, no NaN, energy monotone (viscous regime).
- [ ] **Free-decay turbulence** — τ-consistent checkpoint
  (`docs/free-turbulence-validation.md`): equivariant models stay stable for
  the full decay; decay-rate error vs BGK reported from `summary.txt`.
  Smoke config:
  ```bash
  uv run python validate_free_turbulence.py --no-model --nx 16 --ny 16 \
      --n-transient 300 --n-decay 40 --window 5 40
  ```
- [ ] **KVS wake** — `apply_nn_karman.py` with the work-package-02 metrics:
  stability horizon = full run; St within tolerance of the BGK control;
  E(t) tracks BGK without secular drift.

## 4. Numerical hygiene

- [ ] **Manifest present** in every result directory (`00-README.md` rule 1).
- [ ] **τ consistency** between training data and rollout (`00-README.md`
  rule 2). For KVS: τ=0.5576.
- [ ] **Seeds**: ≥3 for any comparative claim; report mean and min/max, not
  just the best seed.
- [ ] **black** run on all touched Python files; `docs/*.md` updated if
  behavior they describe changed (`CLAUDE.md` requirements).

## Known expected failures (do not "fix" these)

- `plain_*` models breaking symmetry and diverging in turbulent free decay
  within ~100 steps — this is the P2 negative-control result.
- Blank panels / `[diverged]` labels in `velocity_evolution.gif` for failed
  operators — intended visualization behavior.
- A-priori RMSRE growing with KVS step index — the developed wake is a harder
  distribution than the early near-uniform inflow (see `docs/apply-karman.md`).
