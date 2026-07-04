# Implementation guidance — index

These docs are written for an implementing agent (or human) with **no prior
context** on this project. Read this file first, then the doc for your assigned
work package. Each doc states its goal, exact commands, and acceptance criteria —
do not start coding before you can restate the acceptance criteria.

## Documents

| Doc | Work package | Priority |
|-----|--------------|----------|
| `01-paper-comparison-and-factcheck.md` | Background: what each paper does, what is already reproduced, what to verify numerically | read-first |
| `02-resnet-karman-vv-plan.md` | Hypothesis H1: controlled ResNet-vs-plain experiment on the Kármán vortex street. Includes the one new code change needed (wake metrics). | **high** |
| `03-bedrunka-nco-adoption.md` | Borrow Bedrunka-2025 ideas (bounded relaxation head, a-posteriori loss). Staged; do not start before 02 is done. | medium |
| `04-verification-checklist.md` | Repo-wide V&V protocol to run before/after any model or physics change | high (recurring) |

## Repo conventions (from `CLAUDE.md` — binding)

- Python env: system-wide `uv` manages `.venv`. Run everything as
  `uv run python <script>` from the repo root (or activate `.venv`).
- Format all Python with `black` before committing.
- Concise but meaningful comments; docstrings on functions/classes.
- Update the markdown docs in `docs/` when you change behavior they describe.

## Repo map (the parts that matter here)

- `lbm_ml/model/network.py` — all architectures + `MODEL_REGISTRY` (name → factory).
  Families: `d4equivariant*` (GAVG MLP, Corbetta 2023), `resnet*` (GAVG + residual
  inner net), `plain_*` (no physics constraints, negative control), `lenn*` /
  `lenn_resnet*` (Ortali 2025). Suffixes select the conservation layer:
  none = AlgReconstruction (legacy), `_softmax` = softmax only,
  `_softmax_cons` = softmax + symmetric reconstruction (paper method),
  `_safe` = positivity-safe, `_multcons` = multiplicative.
- `lbm_ml/lattice/symmetry.py` — D4 group ops + reconstruction layers.
- `lbm_ml/training.py`, `run_all.py`, `continue_training.py` — a-priori training
  on `(f_pre, f_post)` pairs (synthetic or recorded simulation snapshots).
- `apply_nn_karman.py` — a-posteriori KVS driver (docs: `docs/apply-karman.md`).
  Grid 550×102 at `res=250`, Re=150, U_inlet=0.12, **τ=0.5576** — models must be
  trained at this τ to be physically consistent.
- `validate_free_turbulence.py` — a-posteriori free-decay turbulence test
  (docs: `docs/free-turbulence-validation.md`).
- `eval_helpers/` — `equivariance_inspect.py`, `constraints_check.py`,
  `count_model_params.py`, `fetch_rmsre.py`, `rebuild.py`.
- Papers: `context-private/*.pdf` (P1 Corbetta 2023, P2 Ortali 2025,
  P3 Bedrunka 2025 — see doc 01 for the mapping).

## External resources

- Trained checkpoints:
  `/home/tai/work-my-projects/workspace-master.course.block05-ML4PhA/data`
  (contains a directory-layout README; e.g.
  `lenn_resnet_karman_every_100_samp334_bs32_ep12000_pat2000_lr1e-3/model.keras`).
- Heavy runs go to the Snellius cluster via `scripts/job-*.sh`;
  `scripts/sync-local-snellius.sh` syncs the repo. On the laptop, run only
  smoke-scale configurations (small grids / few steps — each doc gives one).

## Non-negotiable rules for every experiment

1. **Write a manifest.** Every result directory must contain `manifest.json`
   with: model registry name, checkpoint path + SHA256, dataset path, τ, grid,
   seeds, git commit, command line. No manifest ⇒ the result does not count.
2. **τ consistency.** An ML operator only reproduces BGK at the τ it was
   trained on. Never mix a τ=1 synthetic-data model with a τ≈0.5576 KVS run.
3. **Matched budgets.** Architecture comparisons use matched parameter counts
   (verify with `eval_helpers/count_model_params.py`) and identical training
   data, epochs, and early-stopping settings.
4. **Seeds.** ≥3 training seeds per configuration before claiming a difference.
