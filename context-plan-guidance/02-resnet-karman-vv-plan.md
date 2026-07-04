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
   - Reference: at Re=150 experimental/DNS St ≈ 0.18 (Williamson 1996 range
     0.179–0.185). **The primary reference is the BGK run of the same script**
     (see Step 2 control), not literature, to cancel discretization bias.
2. **Stability horizon.** First step where any of: NaN/Inf in `f`; any
   `f_i < −1e-6`; `max|u| > 0.4` (lattice velocity sanity bound). Report the
   step index, or the total step count if never triggered. Also log
   `min f_i` and `max|u|` time series to the CSV.
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
Acceptance for step 1: on a pure-BGK control run (add `--bgk-only` flag if not
present) of ≥20 000 steps, the measured St lands in [0.17, 0.20] and the
stability horizon equals the full run length.

## Step 2 — experiment matrix

All models trained on the same KVS dataset (`every_100`, τ=0.5576), same
budget (epochs/patience/lr/batch as the existing
`…samp334_bs32_ep12000_pat2000_lr1e-3` runs), **3 seeds each**. Match parameter
counts within ~10% (check with `uv run python -m eval_helpers.count_model_params`;
adjust width/channels of the non-resnet twin if needed and record the numbers).

| # | Model (registry name) | Role |
|---|----------------------|------|
| 1 | `lenn_18_18_18_softmax_cons` | LENN baseline |
| 2 | `lenn_resnet_18_18_18_softmax_cons` | LENN + residual (H1 treatment) |
| 3 | `d4equivariant_softmax_cons` | GAVG baseline |
| 4 | `resnet_softmax_cons` | GAVG + residual (H1 treatment) |
| 5 | `plain_2_softmax` | negative control (should degrade/diverge) |
| 6 | pure BGK (no NN) | ground-truth control for St, E(t), horizon |

Use the `_softmax_cons` reconstruction family throughout (paper method, keeps
positivity + conservation comparable across rows). If checkpoints for some rows
already exist in the external data folder, reuse them for seed 1 **only if**
their training config matches; record this in the manifest.

Training (per row × seed) on Snellius: `run_all.py` with `--data-dir` pointing
at the KVS `every_100` snapshots — see `jobs/run-all-tensorflow.sh` and
`scripts/job-gpu-*.sh` for the submission pattern. Evaluation (per checkpoint):

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
