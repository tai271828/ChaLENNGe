# ChaLENNGe — learning Lattice Boltzmann collision operators

Neural networks as the collision operator of a D2Q9 Lattice Boltzmann solver:
train a network on pre/post-collision population pairs, then run it *inside*
the solver and validate it on real flows (Taylor–Green, Kármán vortex street,
freely-decaying turbulence).

Built on and compared against three papers:

- **Corbetta et al. 2023** (*Eur. Phys. J. E* 46:10) — D4 group-averaged (GAVG)
  MLP collision surrogate; the foundation of this codebase.
- **Ortali, Gabbana et al. 2025** (*AIAA J.* 63(2)) — lattice-equivariant
  neural networks (LENN); reproduced here, including the free-turbulence test.
- **Bedrunka et al. 2025** (*PRE* 112, 055308) — neural collision operator with
  a built-in stability bound; its bounded-relaxation idea is adopted as the
  `*_softmax_bounded` model variants.

## Install

Uses [uv](https://docs.astral.sh/uv/) — see `INSTALL.md`. Short version:
`uv venv && uv sync`.

## Quickstart

```bash
# Train a model (synthetic BGK data) and simulate a Taylor-Green vortex
uv run run_all.py --model lenn_18_18_18_softmax_cons --seed 0

# Drive a Kármán vortex street with a trained model; wake metrics included
uv run apply_nn_karman.py --animate --anim-steps 20000 \
    --model-path <run_dir>/model.keras --out-dir results/my_eval

# Pure-BGK control run (ground truth for Strouhal / stability comparisons)
uv run apply_nn_karman.py --bgk-only --animate --anim-steps 20000 \
    --out-dir eval_results_bgk_control

# Freely-decaying turbulence validation (tau must match training tau!)
uv run validate_free_turbulence.py --run <run_dir>:MyModel --tau 0.5576

# One-figure summary of any wake-metrics result directory
uv run python -m eval_helpers.plot_wake <result_dir>

# Tests (fast, structural)
uv run pytest
```

Models are selected by registry name — see `MODEL_REGISTRY` in
`lbm_ml/model/network.py` for all families (`d4equivariant*`, `resnet*`,
`plain_*`, `lenn*`, `lenn_resnet*`) and reconstruction variants
(`_softmax_cons`, `_safe`, `_multcons`, `_bounded`).

## Repository map

| Path | Purpose |
|------|---------|
| `lbm_ml/` | Library: lattice/symmetry, model architectures, training, data, validation, provenance |
| `run_all.py` | Dataset generation → training → Taylor–Green simulation pipeline |
| `apply_nn_karman.py` | Kármán-vortex-street rollout + a-priori eval + wake metrics (Strouhal, stability horizon) |
| `validate_free_turbulence.py` | Free-decay turbulence a-posteriori test (Ortali 2025 Sec. IV.C, 2D) |
| `continue_training.py` | Resume/fine-tune a checkpoint |
| `eval_helpers/` | Parameter counts, equivariance/conservation checks, plots |
| `tests/` | Structural test suite (`uv run pytest`, also run in CI) |
| `docs/` | Topic docs: `apply-karman.md`, `free-turbulence-validation.md`, `equivariance-background.md` |
| `context-plan-guidance/` | Research plan, V&V protocol, and design rationale for the ongoing paper comparison |
| `jobs/`, `scripts/` | Slurm job scripts for the Snellius cluster |

## Reproducibility conventions

Every training/evaluation run writes a `manifest.json` (command, git commit,
model SHA256, seed, physics settings) into its output directory. Architecture
comparisons require matched parameter budgets and ≥3 seeds — see
`context-plan-guidance/00-README.md` for the binding rules.
