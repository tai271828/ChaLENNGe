# Equivariance background — what you need to read this codebase

This repo trains neural-network surrogates for the Lattice Boltzmann (D2Q9) BGK
collision operator. A physically correct collision operator must respect the
**D4 symmetry of the square lattice** (4 rotations + 4 reflections) and the
**conservation laws** (mass + x/y momentum). Those two requirements are baked
into the model architectures, so understanding the code means understanding
*group-equivariant deep learning*.

This document has two layers:

1. **Code-grounded prerequisites** — concepts you meet directly in the source,
   with file/line anchors.
2. **Group-equivariant deep-learning theory** — the body of theory that
   explains *why* the code is correct, not just *what* it does.

Suggested first-pass reading order:
**§1 → §2 → §3 (group averaging) → §6 (conservation) → §4–5 (LENN / theory)**.

The two papers this repo implements:
- Corbetta et al. 2023, *Eur. Phys. J. E* **46**:10 — the D4 group-averaging
  ("GAVG") baseline and the "NN Naive" plain baseline.
- Ortali et al. 2025, *AIAA J.* **63**(2), 716–731,
  [doi:10.2514/1.J064453](https://doi.org/10.2514/1.J064453) — the
  lattice-equivariant network (LENN). See
  [`lbm_ml/model/network.py:234`](../lbm_ml/model/network.py).

---

## Part 1 — Code-grounded prerequisites

### 1. Foundational group theory (the bare minimum)

| Concept | What it means here | Code anchor |
|---|---|---|
| Group, generators | The relevant group is **D4** (8 elements). Every element is built from two generators: `R` (90° rotation), `S` (mirror). | [`symmetry.py:18-19`](../lbm_ml/lattice/symmetry.py) |
| Group action as permutations | D4 acts on the 9 populations by **permuting** them — rest stays, the 4 axis-aligned cycle, the 4 diagonals cycle. | `LBrot90` [`symmetry.py:71`](../lbm_ml/lattice/symmetry.py), `LBmirror` [`symmetry.py:85`](../lbm_ml/lattice/symmetry.py) |
| Orbits | The sets of elements the group maps into each other. Found by BFS over the generators. | `compute_d2q9_orbit_indices` [`symmetry.py:5`](../lbm_ml/lattice/symmetry.py) |

### 2. Equivariance vs. invariance (the central distinction)

- **Equivariance:** transform the input → output transforms the same way,
  `f(g·x) = g·f(x)`.
- **Invariance:** transform the input → output unchanged, `f(g·x) = f(x)`.

The collision operator must be **equivariant** (rotate the input populations →
the post-collision output rotates identically). Stated explicitly at
[`symmetry.py:108-121`](../lbm_ml/lattice/symmetry.py).

### 3. Strategy A — Group averaging / "lift-pool" (`d4equivariant`, `resnet`)

Make *any* network equivariant by: **lift** the input into all 8 transformed
copies → run each through a **shared-weight** sub-network → **un-transform**
each output back → **average**. This is the whole of `_wrap_d4`
([`network.py:74-99`](../lbm_ml/model/network.py)):

- `D4Symmetry` = lift ([`symmetry.py:124`](../lbm_ml/lattice/symmetry.py))
- shared sub-network applied 8× ([`network.py:92`](../lbm_ml/model/network.py))
- `D4AntiSymmetry` = un-transform with the *inverse* permutations
  ([`symmetry.py:149`](../lbm_ml/lattice/symmetry.py))
- `layers.Average()` = pool ([`network.py:96`](../lbm_ml/model/network.py))

Weight sharing across the 8 branches is what makes the average a true
equivariant map (not just an ensemble). **Cost:** ~8× inference, since the
sub-network is evaluated 8 times.

### 4. Strategy B — Intrinsically equivariant layers (`lenn`, `lenn_resnet`)

- **Equivariance as a weight constraint:** a linear layer `A` is equivariant
  iff it **commutes with the representation**, `AP = PA` for every generator
  `P` ([`symmetry.py:9`](../lbm_ml/lattice/symmetry.py),
  [`network.py:226`](../lbm_ml/model/network.py)).
- **Weight tying from orbits:** solving `AP = PA` forces entries of `A` to be
  equal across orbits of index-pairs `(i,j)`. For D2Q9 this collapses a 9×9=81
  matrix to **15 free parameters**, and the bias from 9 → **3** (rest / axis /
  diagonal). See [`network.py:227-228`](../lbm_ml/model/network.py),
  `compute_d2q9_bias_orbit_indices` [`symmetry.py:42`](../lbm_ml/lattice/symmetry.py).
- **Implementation:** store the 15 free scalars (`A_tilde`,
  [`network.py:251`](../lbm_ml/model/network.py)), reconstruct the full matrix
  on the fly via gather (`keras.ops.take(..., orbit_idx)`,
  [`network.py:283`](../lbm_ml/model/network.py)), and contract with an einsum
  ([`network.py:285`](../lbm_ml/model/network.py)).
- **Channels:** LENN stacks equivariant layers with multiple channels, e.g.
  `(1,8,8,10)` ([`network.py:307`](../lbm_ml/model/network.py)) — directly
  analogous to convolutional feature channels.
- **Payoff:** equivariant *by construction*, so no 8× averaging — same
  inference cost as a plain MLP
  ([`network.py:320-321`](../lbm_ml/model/network.py)).

### 5. Domain coupling — conservation laws as a projection

Linear conservation constraints: `Σf=ρ`, `Σf·cx=ρuₓ`, `Σf·cy=ρu_y`
([`symmetry.py:185-187`](../lbm_ml/lattice/symmetry.py)). Two implementations,
one of which breaks equivariance:

- `AlgReconstruction` ([`symmetry.py:177`](../lbm_ml/lattice/symmetry.py)):
  algebraically solves for 3 *fixed* populations (indices 2,5,8). **Breaks D4
  equivariance** because it privileges specific directions
  ([`symmetry.py:256-257`](../lbm_ml/lattice/symmetry.py)).
- `SymmetricAlgReconstruction`
  ([`symmetry.py:241`](../lbm_ml/lattice/symmetry.py)): a **minimum-norm /
  pseudo-inverse projection** onto the conservation manifold using
  `C⁺ = (CCᵀ)⁻¹C`, with `CCᵀ = diag(9,6,6)` for D2Q9. Equivariant because it
  depends only on the conservation *defects* (scalars/vectors under D4), not on
  chosen indices.

---

## Part 2 — Group-equivariant deep-learning theory

The foundations above are necessary but not sufficient. To understand *why* the
code is correct you need the body of theory called **Geometric / Group-
Equivariant Deep Learning (GDL)**.

### 6. Group representation theory (the proper version)

- **Group representations** `ρ: G → GL(V)` — each group element becomes a matrix
  acting on a feature space. Here D4's representation on the 9 populations is a
  **permutation representation** (`LBrot90`/`LBmirror` are the matrices `ρ(g)`);
  knowing it's a permutation rep explains why everything reduces to index
  shuffling.
- **The regular representation** — features that transform like "one value per
  group element." Lifting to 8 copies (`D4Symmetry`) effectively moves into the
  regular representation of D4.
- **Irreducible representations (irreps) & isotypic decomposition** — explains
  why the conserved quantities transform differently: density is a **scalar
  (trivial irrep)**, momentum `(uₓ,u_y)` is a **2D vector irrep**. That is
  exactly what `SymmetricAlgReconstruction` exploits to stay equivariant.

### 7. Equivariant linear maps = intertwiners (Schur's lemma)

The single most important theorem behind LENN:

- The space of equivariant linear maps is the **intertwiner space** — matrices
  with `ρ_out(g)·A = A·ρ_in(g)` for all `g`. That **is** the `AP = PA`
  constraint.
- **Computing the intertwiner basis** gives the 15 free parameters for D2Q9. The
  orbit-BFS in `compute_d2q9_orbit_indices` is a pragmatic way of finding that
  basis.
- **Schur's lemma** is the deeper reason the parameter count collapses the way
  it does (intertwiner dimension = sum over shared irreps).

### 8. Group convolution & the G-CNN framework

- **Group-equivariant convolution (G-CNN; Cohen & Welling 2016)** generalizes
  CNN translation-equivariance to arbitrary groups. The **lift → group-conv →
  pool** pattern is canonical G-CNN, and `_wrap_d4` is a hand-rolled instance.
- **Steerable CNNs / feature fields** — the general framework where features
  carry a "type" (which irrep they transform under) and layers are steerable
  kernels. **LENN is a steerable / equivariant-linear layer**, not a
  group-averaging layer. The regular-rep-vs-steerable distinction is exactly the
  difference between `d4equivariant` (averaging, 8× cost) and `lenn` (intrinsic,
  1× cost).

### 9. Equivariance through parameter sharing

- **Theorem (Ravanbakhsh et al. 2017):** equivariance to a group is equivalent
  to a specific weight-sharing pattern — weights tied across the **orbits of the
  group acting on the index set**. That is *exactly* what `LENNLayer` does. The
  orbit-indexing trick is not D2Q9-specific; it is the general recipe for an
  equivariant linear layer of any finite group.
- **Orbit–stabilizer theorem & Burnside counting** — the combinatorics that tell
  you *how many* free parameters you get (15 and 3 here).

### 10. Symmetrization / the Reynolds operator (frame averaging)

- The averaging model rests on: **averaging a function over the group orbit
  produces an equivariant function** — the **Reynolds operator**
  `(1/|G|) Σ_g ρ(g)⁻¹ f(ρ(g)x)`. That formula *is* `_wrap_d4`. It always works
  for finite groups; cost is `|G|` forward passes. **Frame averaging** (average
  over a smaller, input-dependent frame) is the cheaper modern alternative.

### 11. Equivariant nonlinearities

- A linear equivariant layer is not enough — **the nonlinearity must also
  preserve equivariance.** For permutation/regular representations, **pointwise
  activations (ReLU, etc.) are equivariant**, which is why `LENNLayer` can apply
  a plain pointwise activation after its equivariant linear map
  ([`network.py:289`](../lbm_ml/model/network.py)). For other representation
  types (e.g. vector fields) pointwise ReLU would *break* equivariance and you'd
  need norm-based / gated nonlinearities.
- Likewise **bias must respect the symmetry** — hence the orbit-tied 3-value
  bias ([`network.py:257-265`](../lbm_ml/model/network.py)); the plain models
  drop bias partly for this reason
  ([`network.py:201-206`](../lbm_ml/model/network.py)).

### 12. The Geometric Deep Learning blueprint

- **symmetry group → equivariant layers → invariant readout / coarsening**
  (Bronstein et al. 2021, the "5G": Grids, Groups, Graphs, Geodesics, Gauges).
  This repo lives in the **Groups** corner: finite group D4 on a physics
  surrogate.

### 13. Expressivity / universality caveats

- Constraining a network to be equivariant **trades raw capacity for a correct
  inductive bias** (with universality results under conditions). This is why the
  repo keeps unconstrained baselines `plain_2/10/20`
  ([`network.py:201-206`](../lbm_ml/model/network.py)) — they are the "no
  symmetry, more capacity" controls used to measure what the equivariance bias
  buys.

---

## The single highest-leverage idea

> **Equivariant linear layer = intertwiner = weight-sharing over group orbits.**

That one equivalence unifies §7, the steerable view in §8, and §9 — and it is
exactly what turns the abstract `AP = PA` into the concrete 15-parameter
`LENNLayer`.

---

## Reading list

| Topic | Source |
|---|---|
| GDL overarching framework | Bronstein, Bruna, Cohen, Veličković, *Geometric Deep Learning* (2021) |
| Group-equivariant CNNs (lift/pool) | Cohen & Welling, *Group Equivariant Convolutional Networks* (ICML 2016) |
| Steerable / intrinsic equivariant layers | Cohen & Welling, *Steerable CNNs* (2017); Weiler & Cesa, *General E(2)-Equivariant CNNs* (2019) |
| Equivariance ⇔ parameter sharing | Ravanbakhsh, Schneider, Póczos (2017) |
| Representation theory background | any intro to finite-group representation theory (reps, irreps, Schur's lemma) |
| D4-GAVG baseline (this repo) | Corbetta et al. 2023, *Eur. Phys. J. E* **46**:10 |
| LENN (this repo) | Ortali et al. 2025, *AIAA J.* **63**(2), 716–731 |
