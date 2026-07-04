# Task 02 — Architecture comparison & ResNet–Kármán hypothesis (polished)

Polished version of `task02-fable-architecture-01-draft.md`. Grounded in the actual
state of this repository (commit `a5b6137`, branch `tai-freeturbulence`) and a full
read of the three reference papers in `context-private/`.

## 1. Goal

Validate and verify (V&V) the neural LBM collision operators in this repository
against three reference papers, then test one concrete hypothesis:

> **Hypothesis H1:** adding residual (ResNet-style) connections to the existing
> equivariant collision-operator networks improves the simulation of a chaotic
> flow — the 2D Kármán vortex street (KVS) — measured as (a) longer numerically
> stable rollouts and (b) more accurate wake dynamics, compared to the same
> networks without skip connections.

### Reference papers (all in `context-private/`)

| # | Paper | Role in this repo |
|---|-------|-------------------|
| P1 | Corbetta et al. 2023, *Eur. Phys. J. E* 46:10 — "Toward learning lattice Boltzmann collision operators" | Foundation. GAVG D4-equivariant MLP learning `f_post = NN(f_pre)`. Reproduced as `d4equivariant*` / `plain_*` registry models. |
| P2 | Ortali, Gabbana et al. 2025, *AIAA J.* 63(2) — "Enhancing Lattice Kinetic Schemes … Lattice-Equivariant Neural Networks" | Extension. LENN layers (15 free weights per channel pair on D2Q9). Reproduced as `lenn*` registry models; its free-turbulence test is ported in `lbm_ml/validation/free_turbulence.py`. |
| P3 | Bedrunka et al. 2025, *Phys. Rev. E* 112, 055308 — "Machine-learning-enhanced collision operator … based on invariant networks" | Comparison target. **NOT the same problem class** — see §3. |

## 2. Current state (what already exists — do not re-implement)

- **ResNet variants are already implemented.** `lbm_ml/model/network.py` provides
  `resnet_sequential_model` (GAVG inner net with two-layer residual blocks) and
  `create_lenn_resnet_model` (LENN residual blocks), registered as `resnet`,
  `lenn_resnet`, `lenn_resnet_11_11_11`, `lenn_resnet_18_18_18` plus
  `_softmax` / `_softmax_cons` / `_safe` / `_multcons` reconstruction variants.
- **KVS a-posteriori driver exists:** `apply_nn_karman.py` (docs:
  `docs/apply-karman.md`). Runs the NN as collision operator on a 550×102 grid,
  Re=150, τ=0.5576; outputs a wake GIF and *a-priori* RMSRE/MAE/max-error CSV.
- **Free-turbulence validation exists:** `validate_free_turbulence.py` +
  `lbm_ml/validation/free_turbulence.py` (docs:
  `docs/free-turbulence-validation.md`). 2D adaptation of P2 Sec. IV.C.
- **Trained checkpoints** live at
  `/home/tai/work-my-projects/workspace-master.course.block05-ML4PhA/data`
  (e.g. `lenn_resnet_karman_every_100_samp334_bs32_ep12000_pat2000_lr1e-3`);
  Snellius job scripts in `scripts/`.
- **Preliminary evidence already in-repo** (`eval_results_lenn_18x3/` vs
  `eval_results_lenn_resnet/`, provenance not recorded): lenn_resnet has ~2×
  *lower* a-priori RMSRE early (3.6e-4 vs 6.0e-4 at step 100) but ~6× *higher*
  max-abs error late (0.43 vs 0.07 at step 30 000). H1 is therefore neither
  confirmed nor refuted — it needs a controlled experiment.

**Consequence:** the work is *validation & verification design*, not architecture
implementation. The only new code H1 strictly needs is wake-dynamics metrics
(Strouhal number, stability horizon) in `apply_nn_karman.py`.

## 3. Fact-check of the draft's P3 hypothesis

Draft claim: *"Bedrunka 2025 has better neural network architecture, methodology,
and results than the other two papers."* After reading P3 in full:

- **Architecture — misleading comparison.** P3's network is a *tiny* MLP
  (2×20 nodes, ~1064 parameters, group-averaged over the 48-element octahedral
  group). It does **not** learn the collision map `f_pre → f_post`. It predicts
  **relaxation rates of higher-order moments** inside a classical MRT operator:
  `Ω(f) = −M⁻¹ S_θ (Mf − m^eq)` with conserved moments fixed at rate 1, shear
  rates fixed by viscosity, and only the higher-order rates learned, squashed by
  a sigmoid into `(0.5, 1.0]` (Eq. 18). Conservation is exact by construction and
  stability is architecturally bounded — properties P1/P2 must *learn* or enforce
  via reconstruction layers.
- **Methodology — genuinely different, arguably stronger for turbulence.** P3
  trains *a-posteriori*: unrolled differentiable LBM simulations (Lettuce,
  PyTorch) with a loss on the **energy spectrum** of forced isotropic turbulence
  (Eq. 29), plus an alternative dissipation-rate loss (Sec. IV). P1/P2 (and this
  repo) train *a-priori* on single-step `(f_pre, f_post)` pairs. A-posteriori
  training directly optimizes rollout behavior — the thing H1 cares about.
- **Results — strong, but for a different task.** P3 targets *under-resolved*
  turbulence (implicit-LES-like closure): stable 32³ simulations matching 256³
  DNS spectra, TGV3D dissipation better than KBC/REG/BGK, 3D cylinder drag/lift
  matching literature. It is not evidence that its network is a better
  *collision-operator surrogate* — it never attempts that task.

**Verdict:** the hypothesis as stated is a category error. The correct framing:
P3 offers **methodology worth borrowing** (bounded relaxation parametrization,
a-posteriori spectrum/dissipation losses, wake-dynamics benchmarks like drag/lift
and Strouhal), not a drop-in superior architecture. Fair head-to-head comparison
would require implementing a 2D NCO — a separate, optional work package.

## 4. Deliverables (executed by this task)

Implementation-ready guidance docs for future (cheaper) models, in
`context-plan-guidance/`:

1. `00-README.md` — index, repo conventions, how to use these docs.
2. `01-paper-comparison-and-factcheck.md` — three-paper matrix + §3 fact-check
   with what remains to be verified numerically.
3. `02-resnet-karman-vv-plan.md` — the controlled experiment for H1: model
   matrix, training protocol, new metrics to implement, acceptance criteria,
   exact commands.
4. `03-bedrunka-nco-adoption.md` — staged plan to borrow P3 ideas (bounded
   moment-rate head, a-posteriori losses), with effort estimates.
5. `04-verification-checklist.md` — repo-wide V&V protocol (equivariance,
   conservation, convergence, free turbulence, KVS) using existing
   `eval_helpers/` tools.

## 5. Known gaps & open issues (for prioritization — see final report)

- **G1 (high):** `apply_nn_karman.py` measures only a-priori error; H1 needs
  wake-dynamics metrics (Strouhal number, stability horizon; optionally
  drag/lift via momentum exchange, P3 Eqs. 37–39).
- **G2 (high):** `eval_results_*` directories lack provenance manifests
  (checkpoint, τ, dataset, seed) — results are not reproducible as recorded.
- **G3 (high):** draft's P3 framing incorrect (§3) — decide "borrow" vs
  "reproduce" before spending compute.
- **G4 (medium):** H1 comparisons need matched parameter counts and ≥3 seeds;
  current lenn (18,18,18) vs lenn_resnet runs are not budget-matched.
- **G5 (medium):** H1's mechanism claim ("ResNet catches chaos better") is
  not itself testable; the plan reframes it as measurable rollout criteria.
- **G6 (medium):** free-turbulence validation must use a model trained at the
  same τ as the decay run (documented in `docs/free-turbulence-validation.md`);
  KVS-trained models (τ=0.5576) are the right candidates.
- **G7 (low):** no automated tests/CI; `eval_helpers/` checks are manual.
- **G8 (low):** root `README.md` is effectively empty.
