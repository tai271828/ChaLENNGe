# Three-paper comparison and fact-check

Goal: give an implementer the exact conceptual map of the three reference papers,
what this repo already reproduces, and which claims still need numerical
fact-checking. Papers are in `context-private/`.

## The papers

- **P1 — Corbetta et al. 2023** (*Eur. Phys. J. E* 46:10, "Toward learning
  lattice Boltzmann collision operators").
- **P2 — Ortali, Gabbana et al. 2025** (*AIAA J.* 63(2), "Enhancing Lattice
  Kinetic Schemes for Fluid Dynamics with Lattice-Equivariant Neural Networks").
- **P3 — Bedrunka et al. 2025** (*Phys. Rev. E* 112, 055308,
  "Machine-learning-enhanced collision operator for the LBM based on invariant
  networks").

## Comparison matrix

| Aspect | P1 (Corbetta) | P2 (Ortali/LENN) | P3 (Bedrunka/NCO) |
|---|---|---|---|
| What the NN learns | Full collision map `f_post = NN(f_pre)` per node | Same task, equivariant-by-construction layers | **Relaxation rates of higher-order moments** inside a classical MRT operator (Eq. 7) |
| Stencil | D2Q9 | D2Q9 (also 3D in paper) | D3Q27 (moment space via Hermite polynomials, Eq. 12) |
| Symmetry handling | Group averaging (GAVG) over D4: run net on all 8 transforms, average | Weight-tying inside each layer (15 free weights per channel pair) | Group averaging over the 48-element octahedral group on *moment inputs* (Eq. 15) → invariant scalar outputs |
| Conservation | Reconstruction layer after the net | Reconstruction layer (symmetric, minimum-norm) | **Exact by construction** — conserved moments relaxed at rate 1, untouched by the NN |
| Stability guarantee | None (learned) | None (learned) | **Architectural**: sigmoid squashes learned rates into (0.5, 1.0] (Eq. 18); 1.0 = regularized-LBM limit |
| Training | A-priori, single-step `(f_pre, f_post)` pairs | A-priori, same | **A-posteriori**: unrolled differentiable LBM (Lettuce/PyTorch), loss = energy-spectrum MSE for 4≤κ≤10 + weighted κ_max term (Eq. 29); alternative: dissipation-rate loss on TGV3D (Sec. IV) |
| Network size | MLP ~50-wide, 2–3 hidden layers | LENN channels e.g. (1,8,8,10) | Tiny MLP: 2×20 nodes, ~1064 params (Sec. III adds a third 20-node layer in Sec. IV) |
| Target regime | Reproduce BGK (resolved) | Reproduce BGK; stability under turbulence | **Under-resolved turbulence** (implicit-LES-like closure): 32³ matching 256³ DNS spectra |
| Headline results | Learned operator runs Taylor-Green stably | Equivariant nets survive free-decay turbulence where plain MLP diverges in ~100 steps | Stable under-resolved TGV3D beating KBC/REG/BGK on dissipation; 3D cylinder C_D=1.36, C_L=0.68 at Re=200 matching literature; 2nd-order convergence preserved |

## What this repo already reproduces

- P1: `d4equivariant*`, `plain_2` (NN-Naive baseline) in `MODEL_REGISTRY`.
- P2: `lenn*` models (`LENNLayer` in `lbm_ml/model/network.py`), free-turbulence
  validation ported to 2D (`lbm_ml/validation/free_turbulence.py`).
- P3: **nothing** — and note the repo's a-priori Keras/TF training pipeline
  cannot express P3's a-posteriori training without new infrastructure
  (see `03-bedrunka-nco-adoption.md`).

## Fact-check of the draft hypothesis about P3

Draft claim: *"P3 has better neural network architecture, methodology, and
results than P1/P2."* Verdict after full read: **category error, partially true
on methodology.**

1. **Architecture:** P3's network is far *smaller and simpler* than P1/P2's; it
   is not a better collision-map surrogate because it never learns the collision
   map. Its strength is the **parametrization** (moment-space MRT with learned,
   sigmoid-bounded rates) which buys exact conservation and guaranteed
   stability — the two properties P1/P2 struggle to enforce.
2. **Methodology:** a-posteriori (rollout) training with spectrum/dissipation
   losses is genuinely stronger *for rollout behavior* than a-priori pair
   fitting, and is the most valuable thing to borrow. But it needs a
   differentiable solver (P3 uses Lettuce; 80 GB A100 for backprop through
   hundreds of steps — P3 Sec. II.D).
3. **Results:** strong, but on the under-resolved-closure task. P3 never
   evaluates single-step operator fidelity (RMSRE vs BGK pairs), so no direct
   numerical comparison with P1/P2 exists in the papers themselves.

## What still needs numerical fact-checking (tasks)

- **F1.** Confirm the P2 claim reproduced in this repo: plain MLP diverges in
  ~100 free-decay steps while equivariant models stay stable. Command in
  `docs/free-turbulence-validation.md`; needs a τ-consistent model (rule 2 of
  `00-README.md`).
- **F2.** Quantify P1-vs-P2 (GAVG vs LENN) at matched parameter counts on the
  KVS a-priori metric (`apply_nn_karman.py --data-dir …`). Existing runs
  (`eval_results_lenn_18x3`, `eval_results_lenn_resnet`) lack manifests —
  rerun under the manifest rule.
- **F3.** The only P3 claim testable *without* implementing the NCO: that
  bounded relaxation ⇒ stability. Indirect 2D analogue via doc 03, stage B.
- **F4 (optional, large).** Implement a 2D NCO (D2Q9 moment basis, learned
  higher-order rates) for a true head-to-head — only if doc 03 stages A–B show
  promise.
