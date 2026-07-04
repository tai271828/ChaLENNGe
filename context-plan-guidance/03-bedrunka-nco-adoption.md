# Work package: borrowing from Bedrunka 2025 (NCO) — staged plan

> **DECISION (P3, 2026-07-05): borrow first, reproduce later-if-ever.**
> Context: doc 01 §Fact-check — the NCO solves a different problem class, so a
> head-to-head reproduction (Stage C) answers no near-term question, while the
> bounded-stability idea is cheap to test. Decision: implement Stage A now;
> Stage B after work package 02 reports; Stage C stays gated on both.
> Consequence: Stage A is **implemented** — `BoundedBlendReconstruction` in
> `lbm_ml/lattice/symmetry.py`, wired as `Reconstruction.BOUNDED` with six
> `*_softmax_bounded` registry entries in `lbm_ml/model/network.py`.
> Structural verification passed at float64 precision (equilibrium vs stencil
> 1e-16, conservation 2e-16, D4 equivariance 6e-17, g→1 recovers the plain
> symmetric projection, save/load round-trip exact, +3 params); a training
> smoke test confirmed gradients reach θ. The *behavioral* acceptance below
> (stability horizon on trained checkpoints) still requires the work-package-02
> training runs.

Prerequisite reading: `01-paper-comparison-and-factcheck.md` §Fact-check.
Bedrunka's Neural Collision Operator (NCO) is **not** a drop-in replacement for
this repo's collision surrogates — it learns MRT relaxation rates, not the
collision map. What we borrow are three transferable ideas, ordered by
effort/payoff. **Do not start any stage before work package 02 is complete** —
its wake metrics are the evaluation harness for every stage here.

Effort estimates assume familiarity with this repo (after reading
`00-README.md`) and Keras 3.

## Stage A — bounded-stability output head (small, ~1–2 days)

**Idea (P3 Eq. 18):** architecturally bound the operator so it cannot
under-relax. In P3, learned rates are sigmoid-squashed into (0.5, 1.0].

**2D analogue for this repo:** add a new reconstruction/post-processing option
that blends the NN output with the (analytically known) equilibrium:

```
f_out = f_eq(ρ, u of f_pre) + g(x) ⊙ (f_nn − f_eq)
```

with `g = 0.5 + 0.5·sigmoid(·)` either a learned scalar/per-orbit parameter or
a fixed hyperparameter. `g ∈ (0.5, 1]` mirrors the NCO bound: `g=1` returns the
raw NN prediction, `g→0.5` damps non-equilibrium content (over-relaxation
regime). The blend alone conserves mass/momentum only to the extent `f_nn`
does, so the implementation composes it with the minimum-norm symmetric
projection (subclassing `SymmetricAlgReconstruction`), making conservation
exact regardless of `g` — the NCO analogy holds fully: conserved moments
untouched, non-equilibrium content bounded.

Implementation notes:
- New layer in `lbm_ml/lattice/symmetry.py` next to the existing
  `*AlgReconstruction` layers (register with
  `@keras.saving.register_keras_serializable(package="lbm")`).
- `f_eq` from D2Q9 weights — reuse/factor the equilibrium already coded in
  `lbm_ml/data/simulation.py::_equilibrium_from_populations`.
- New `Reconstruction` enum member + registry entries
  (e.g. `lenn_18_18_18_softmax_bounded`) in `lbm_ml/model/network.py`.
- Verify D4 equivariance of the new layer with
  `eval_helpers/equivariance_inspect.py` and conservation with
  `eval_helpers/constraints_check.py`.

**Acceptance:** on the work-package-02 harness, the bounded variant's stability
horizon ≥ unbounded twin on all seeds, with St error not worse than 1.5×. This
is fact-check task F3 of doc 01.

## Stage B — a-posteriori fine-tuning (medium, ~1–2 weeks)

**Idea (P3 Sec. II.D):** optimize rollout behavior directly by backpropagating
through unrolled simulation steps, instead of (or after) single-step pair
fitting.

**Minimal version for this repo** (avoids adopting a new framework): a custom
Keras/TF training loop that
1. takes a short BGK-generated trajectory window (e.g. 20–50 steps on a small
   grid, 64×64 periodic, τ=0.5576);
2. rolls the model forward with differentiable streaming (`tf.roll`) and the
   model as collision, from the window's initial `f`;
3. loss = MSE on macroscopic fields (ρ, u) vs the BGK trajectory, or the 2D
   energy-spectrum discrepancy (P3 Eq. 29 analogue) — start with (ρ, u), add
   the spectrum term only if plain field-matching under-dissipates;
4. **fine-tunes** an a-priori-trained checkpoint (do not train from scratch —
   P3 needed 80 GB VRAM for full training; fine-tuning short windows on small
   grids is laptop/single-GPU feasible).

Watch out for: memory ∝ window length (use gradient checkpointing or keep
windows ≤50 steps); boundary conditions are non-differentiable in
`apply_nn_karman.py` — stay on periodic domains for training and evaluate on
KVS afterwards.

**Acceptance:** fine-tuned checkpoint beats its parent on ≥2 of the 3
work-package-02 criteria without regressing the third.

## Stage C — 2D NCO head-to-head (large, ~3–6 weeks; optional)

Only if stages A–B show promise and the head-to-head is still scientifically
needed (fact-check F4). Implement the actual P3 operator on D2Q9:

- Moment basis via 2D Hermite polynomials (P3 Eq. 12 restricted to D=2; 9
  moments: ρ, u_x, u_y, 3 second-order, then higher-order groups).
- Ω(f) = −M⁻¹ S_θ (Mf − m^eq); conserved rates = 1, shear rates from ν(τ),
  higher-order rates from a small invariant net (group-average over D4 on
  transformed moment inputs, P3 Eq. 15) with the sigmoid bound (Eq. 18).
- Verify equivariance numerically as P3 does (Eq. 14): transformed-initial-state
  run vs transformed-final-state run, element-wise agreement to machine
  precision.
- Train with the Stage-B loop (spectrum loss); evaluate on the work-package-02
  harness plus free-decay turbulence.

**Decision point before starting:** consider using Lettuce
(https://github.com/lettucecfd/lettuce, PyTorch) directly for stage C instead
of reimplementing in TF — P3's setup exists there. Trade-off: two frameworks in
one repo vs weeks of reimplementation. Discuss with the project owner first.
