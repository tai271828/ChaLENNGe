# Work package: ResNet–Kármán hypothesis (H1) — controlled V&V plan

**Hypothesis H1:** residual (ResNet-style) connections in the equivariant
collision networks improve Kármán-vortex-street (KVS) simulation: (a) longer
numerically stable rollouts, (b) more accurate wake dynamics, versus the same
networks without skip connections.

**Status of the codebase:** the architectures already exist — do NOT implement
new models. `MODEL_REGISTRY` in `lbm_ml/model/network.py` has the pairs
(`d4equivariant` ↔ `resnet`) and (`lenn_18_18_18` ↔ `lenn_resnet_18_18_18`),
each with reconstruction-variant suffixes. What is missing is **measurement**:
`apply_nn_karman.py` currently reports only a-priori errors (RMSRE/MAE/max-abs
per snapshot). H1 is about rollout dynamics, so step 1 adds wake metrics.

**Prior signal (why this needs care):** existing runs `eval_results_lenn_18x3/`
vs `eval_results_lenn_resnet/` (provenance unrecorded) show lenn_resnet with
~2× lower a-priori RMSRE at step 100 but ~6× higher max-abs error at step
30 000. Early accuracy and late robustness may trade off — the experiment must
measure both.

## Step 1 — implement wake-dynamics metrics (the only code change)

Extend `apply_nn_karman.py`; keep the existing outputs unchanged and follow
`black` + docstring conventions. Update `docs/apply-karman.md` when done.

1. **Velocity probe + Strouhal number.**
   - Record `u_y(t)` at a probe in the wake, default `x = cylinder_x + 8·radius`,
     `y = cylinder_y`, every step after a warm-up (default: discard the first
     half of the run). Expose `--probe-x/--probe-y/--warmup-frac`.
   - Strouhal: dominant frequency `f_shed` of the detrended `u_y(t)` via FFT
     (window the signal; report the peak and its amplitude), then
     `St = f_shed · D / U_inlet` with `D = 2·radius` (lattice units — both
     f and D in lattice units, U_inlet=0.12).
   - Reference: **St ≈ 0.28, not the unconfined ≈0.18** — the geometry is the
     confined Schäfer–Turek benchmark (23.5 % blockage; benchmark St ≈ 0.30 at
     Re=100). Verified BGK control at Re=150, 20 000 steps: St = 0.28
     (`eval_results_bgk_control/`). **The primary reference is the BGK run of
     the same script** (see Step 2 control), not literature, to cancel
     discretization bias.
2. **Stability horizon.** First step where NaN/Inf appears in `f` or
   `max|u| > 0.4` (lattice velocity sanity bound). Report the step index, or
   the total step count if never triggered. Also log `min f_i` and `max|u|`
   time series to the CSV. **Do not stop on negative populations** — mildly
   negative `f_i` (≈ −2e-4) occur in the pure-BGK control at τ=0.5576 and are
   physically expected near τ=0.5; record `first_negative_step`
   (`min f_i < −1e-6`) as a separate diagnostic instead.
3. **Kinetic-energy trace.** `E(t) = ⟨½|u|²⟩` over the domain each
   `--update-steps` — cheap, catches slow drift/blow-up that max|u| misses.
4. **Manifest.** Write `manifest.json` (fields per `00-README.md` rule 1) into
   `--out-dir`. Add `--seed` where randomness exists.
5. **(Optional, stretch) drag/lift** via momentum exchange on the cylinder
   bounce-back links (Bedrunka 2025 Eqs. 37–39): `C_D = 2F_x/(ρU²D)`,
   `C_L = 2F_y/(ρU²D)`. Reference at Re=150: C_D ≈ 1.3, C_L amplitude ≈ ±0.5.
   Only implement if steps 1–4 are done and validated.

**Smoke test (laptop):**
```bash
uv run python apply_nn_karman.py --animate --anim-steps 2000 --update-steps 100 \
    --model-path <checkpoint>/model.keras --out-dir /tmp/kvs_smoke
```
Acceptance for step 1 — **PASSED 2026-07-04**: pure-BGK control (`--bgk-only`)
of 20 000 steps measured St = 0.28 (confined-geometry expectation
[0.25, 0.32]) with stability horizon = full run; artifacts + manifest in
`eval_results_bgk_control/`. Note `first_negative_step = 1875` and
`min_f_overall ≈ −0.1` for pure BGK — the baseline any positivity comparison
must beat.

## Step 2 — experiment matrix (budgets verified 2026-07-05)

All models trained on the same KVS dataset (`every_100`, τ=0.5576), same
budget (epochs/patience/lr/batch as the existing
`…samp334_bs32_ep12000_pat2000_lr1e-3` runs), **seeds 0, 1, 2** via
`run_all.py --seed` (seeds python/numpy/TF; recorded in the run's
`manifest.json`).

**Budget audit result:** the naive pairs are invalid — `lenn_18_18_18`
(10,425 params) vs `lenn_resnet_18_18_18` (30,081) is a **2.9×** mismatch
that confounds "ResNet" with "bigger"; `d4equivariant` (5,900) vs `resnet`
(10,900) is 1.85×. The matrix below uses budget-matched twins (verify anytime
with `uv run python -m eval_helpers.count_model_params`):

| # | Model (registry name) | Free params | Role |
|---|----------------------|------------:|------|
| 1 | `lenn_31_31_31_softmax_cons` | 30,042 | LENN baseline (twin of row 2, Δ0.13%) |
| 2 | `lenn_resnet_18_18_18_softmax_cons` | 30,081 | LENN + residual (H1 treatment) |
| 3 | `d4equivariant_10K_wide_softmax_cons` | 10,764 | GAVG baseline (twin of row 4, Δ1.3%) |
| 4 | `resnet_softmax_cons` | 10,900 | GAVG + residual (H1 treatment) |
| 5 | `plain_2_softmax` | 3,400 | negative control (should degrade/diverge) |
| 6 | pure BGK (`apply_nn_karman.py --bgk-only`) | — | ground-truth control (`eval_results_bgk_control/`) |
| 7 | `lenn_31_31_31_softmax_bounded` | 30,045 | Stage A bounded head (doc 03; +3 params vs row 1) |

Use the `_softmax_cons` reconstruction family throughout (paper method, keeps
positivity + conservation comparable across rows); row 7 tests the Stage A
bound against row 1. The old `lenn_18_18_18` / `lenn_resnet` checkpoints are
**not** reusable for this matrix (unseeded, unmatched, no manifest).

Training (per row × seed) on Snellius, from the project root:

```bash
for SEED in 0 1 2; do
  sbatch --export=ALL,MODEL=lenn_31_31_31_softmax_cons,SEED=$SEED,\
DATA_DIR=/path/to/karman/every_100,SAMPLES_PER_STEP=334,\
BATCH_SIZE=32,N_EPOCHS=12000,PATIENCE=2000,LR=1e-3 \
    jobs/run-all-tensorflow.sh
done   # repeat with MODEL= rows 2, 3, 4, 5, 7
```

Each run writes `manifest.json` (model, params, seed, data, hyperparams, git
commit) into its run dir under the per-job artifacts tree. Evaluation (per
checkpoint):

```bash
uv run python apply_nn_karman.py --animate --anim-steps 30000 --update-steps 100 \
    --model-path <ckpt>/model.keras --data-dir <karman>/every_100 \
    --out-dir results/h1/<row>_<seed>
```

## Step 3 — secondary test: free-decay turbulence

Same checkpoints, τ=0.5576 decay (rule 2 of `00-README.md`):
```bash
uv run python validate_free_turbulence.py \
    --run results_ckpts/<row1>:LENN --run results_ckpts/<row2>:LENN+Res \
    --tau 0.5576 --n-transient 20000 --n-decay 200
```
Metric: a-posteriori error and decay-rate error vs BGK from `summary.txt`.

## Step 4 — decision criteria

> **SIGNED OFF (P6, 2026-07-05)** by the project owner, before any matrix
> data exists (ex-ante rule, doc 05 §2.1). Two explicit choices:
> (a) the decision rule is **KVS-only** — the free-turbulence decay test
> (step 3) is reported as a sanity check but does not decide H1;
> (b) criterion 3 is a **guard, not a requirement** — "precise" is judged by
> the wake dynamics (criterion 2), and single-step RMSRE only needs to avoid
> a >2× late-window regression. Do not amend these after results arrive.

Declare H1 **supported** iff, with 3 seeds per row (report mean ± min/max):
1. Stability horizon (resnet row) ≥ its non-resnet twin on every seed, and
2. |St − St_BGK|/St_BGK (resnet) < non-resnet twin on ≥2/3 seeds, and
3. late-window a-priori RMSRE (mean over steps 20 000–30 000) does not regress
   by more than 2× (guards against the max-abs blow-up seen in the prior runs).

Declare **refuted** if the non-resnet twin wins criteria 1–2 with the same
margins. Anything else: **inconclusive** — report per-criterion tables and stop
(do not add seeds without discussing compute budget).

Deliverable: `results/h1/REPORT.md` with the matrix, tables per criterion,
manifest links, and one paragraph interpreting the early-accuracy vs
late-robustness trade-off. Keep all `curves`/CSV artifacts.
