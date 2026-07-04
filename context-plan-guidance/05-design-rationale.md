# Design rationale — the rules behind docs 00–04

This meta-doc records the principles used to design `00-README.md` and docs
01–04, so that future additions to this folder follow the same rules instead
of inventing new conventions. Each principle names its source in software
engineering theory or numerical-software practice, then shows where it landed
in the docs.

## 1. Documentation architecture

### 1.1 One doc, one job (separation of concerns / Diátaxis)

The Diátaxis framework (Procida) distinguishes four documentation modes —
*explanation*, *how-to*, *reference*, *checklist/tutorial* — and warns against
mixing them, because each mode is read in a different situation. The folder
maps onto it deliberately:

| Doc | Diátaxis mode | Read when |
|-----|---------------|-----------|
| 00 | reference + navigation | before anything else; while working, for conventions |
| 01 | explanation (background, claims, evidence) | before choosing what to build |
| 02 | how-to (executable experiment plan) | while implementing H1 |
| 03 | how-to, staged (design proposal with decision gates) | after 02, if pursued |
| 04 | checklist (recurring procedure) | before/after every change |

A doc that explains *and* instructs *and* lists conventions forces re-reading
everything to find anything. Splitting by mode is the documentation analogue of
the single-responsibility principle.

### 1.2 Write for the cold-start reader

The stated audience is "an implementing agent (or human) with **no prior
context**" — cheaper models included. Consequences applied throughout:

- Every referenced artifact is a concrete path (`lbm_ml/model/network.py`,
  registry names, checkpoint directories), never "the model file".
- Every action is an exact command with flags, runnable by copy-paste
  ("executable documentation"). Prose that cannot be executed is marked as
  background.
- Acceptance criteria appear before instructions: 00 says *"do not start
  coding before you can restate the acceptance criteria"*. This is
  specification-first (design-by-contract): the deliverable is defined by its
  test, not by effort spent.
- Papers get stable local IDs (P1/P2/P3) defined once in a table, so
  cross-references cost the reader nothing.

### 1.3 Record decisions with their reasons (ADR style)

Architecture Decision Records (Nygard) keep *context → decision →
consequences* together so later readers can tell whether a rule still applies.
Doc 01's fact-check follows that shape (claim → evidence from the paper →
verdict → what remains testable), and doc 02 records its own amendments
in-line — e.g. the stability-horizon criterion documents *why* negativity is
not a stop condition (pure BGK reaches min_f ≈ −0.1) rather than just stating
the rule. A rule without its reason cannot be safely changed later.

## 2. Scientific-method framing

### 2.1 Falsifiability (Popper)

The original idea — "ResNet may catch chaotic systems better" — is a mechanism
narrative, not a testable claim. Doc 02 restates it as hypothesis H1 with
measurable outcomes and, critically, defines **all three verdicts ex ante**:
supported, refuted, and inconclusive, each with thresholds. Fixing the
decision rule before seeing data is the standard guard against HARKing
(hypothesizing after results are known) and moving goalposts.

### 2.2 Controlled experiments (design of experiments)

- **One factor at a time:** the matrix pairs each architecture with its
  residual twin (`lenn_18_18_18` ↔ `lenn_resnet_18_18_18`) while holding the
  reconstruction family, dataset, τ, and training budget fixed — the ResNet
  connection is the only manipulated variable.
- **Matched budgets:** parameter counts within ~10 %, verified by tool, not
  assumption. Otherwise "ResNet is better" is confounded with "bigger is
  better".
- **Controls, both directions:** pure BGK is the positive/ground-truth control
  (row 6); `plain_2` is the negative control expected to fail (row 5). A test
  that cannot distinguish a known-good from a known-bad operator measures
  nothing.
- **Seed replication:** ≥3 seeds with mean and min/max reported. Single-seed
  deep-learning comparisons are known to be noise-dominated (Henderson et al.
  2018, "Deep RL that Matters"); the rule is in 00 as non-negotiable #4.

### 2.3 Category discipline in comparisons

Doc 01's verdict on the P3 claim ("better architecture") is a *category-error*
check: P3 solves a different problem (learned relaxation rates as closure)
than P1/P2 (learned collision map), so "better" is undefined until the task is
fixed. The general rule: before comparing two systems, state the task on which
they are being compared; if they were built for different tasks, compare
*ideas to borrow*, not systems. This came from reading the primary source
(the PDF), not the abstract or the draft's impression — evidence over
authority.

## 3. Numerical-software V&V

The vocabulary follows Oberkampf & Roy (*Verification and Validation in
Scientific Computing*, 2010) and the ASME V&V standards:

- **Verification** — "solving the equations right": doc 04 §1 checks
  structural invariants (D4 equivariance to float precision, exact mass /
  momentum conservation, positivity, serialization round-trip). These are
  property tests against the *code*, independent of physics quality.
- **Validation** — "solving the right equations": doc 04 §3 and doc 02 compare
  against physical references (Strouhal number, energy decay rates, stability
  under chaotic flow). These test the *model* against reality/benchmark.

Further numerical-practice rules embedded in the docs:

- **A-priori vs a-posteriori metrics are different quantities.** Single-step
  pair error (RMSRE) does not predict rollout behavior; the ML-CFD literature
  (and this repo's own lenn vs lenn_resnet data) shows they can rank models
  oppositely. Doc 02 therefore requires both, and H1 is decided on the
  a-posteriori side.
- **The discretized reference, not the literature value, is the benchmark.**
  St is compared against the BGK control run of the *same* grid, boundary
  conditions, and script — cancelling discretization and confinement bias.
  The literature value only sanity-checks the control itself (the confined
  Schäfer–Turek geometry gives St ≈ 0.28–0.30, not the unconfined 0.18 — a
  mistake the first draft of doc 02 made and testing caught; see §5).
- **Physical-consistency invariants are stated as contracts.** τ-consistency
  (train and roll out at the same relaxation time) is rule #2 in 00 because it
  silently invalidates results when broken — the worst failure mode.
- **Expected failures are documented as expected.** Doc 04 ends with "known
  expected failures — do not fix": the plain MLP diverging *is* the P2
  control result. This is the characterization-test idea — pin intended
  behavior, including intended failure, so a helpful maintainer doesn't
  "repair" the experiment.

## 4. Reproducibility engineering

From Sandve et al. 2013 ("Ten Simple Rules for Reproducible Computational
Research") and Wilson et al. 2014/2017 ("Best/Good-Enough Practices in
Scientific Computing"):

- **Provenance is a gate, not a courtesy:** manifest.json (command, git
  commit, model SHA256, seed, physics) on every run; 00 rule #1 states "no
  manifest ⇒ the result does not count". The rule exists because the repo
  already contained two result directories whose provenance could not be
  reconstructed (`PROVENANCE-UNKNOWN.md`).
- **Determinism surfaced:** seeds are explicit arguments and recorded, never
  implicit.
- **Raw artifacts kept:** curves/CSVs are preserved so plots can be
  regenerated and re-aggregated without rerunning simulations.
- **Scale ladder:** every experiment has a laptop-scale smoke configuration
  before the cluster-scale run (the test-pyramid idea applied to simulations —
  cheap fast checks catch most defects before expensive runs).

## 5. Risk-ordered planning

- **Inventory before build (reuse-first / DRY):** the audit preceding these
  docs found the ResNet variants already implemented, reducing "combine with
  ResNet" from an implementation task to a measurement task. Rule: grep the
  registry before designing the module.
- **Cheapest decisive test first:** priorities were ordered by
  information-per-cost (wake metrics unblock everything and cost an
  afternoon; a 2D NCO reimplementation costs weeks and is gated behind two
  cheaper stages). Doc 03's A→B→C staging with explicit decision points is
  the real-options / fail-fast pattern; stage C carries a YAGNI guard
  ("only if A–B show promise") and a build-vs-buy question (reuse Lettuce
  instead of reimplementing).
- **Specs are falsifiable too (double-loop learning):** two doc-02 rules were
  overturned by the first acceptance run — the negativity stop-condition and
  the St reference window — and the docs were amended *with the evidence
  in-line*, not silently. The loop is: spec → cheapest empirical test of the
  spec itself → amend spec → then scale up. A plan document that testing
  cannot change is dogma, not engineering.

## 6. How to extend this folder

1. Pick the Diátaxis mode first; if your draft mixes explanation with
   procedure, split it.
2. State the audience assumption (cold-start) and write acceptance criteria
   before instructions.
3. Any claim about a paper: cite section/equation numbers you actually read.
4. Any experiment: name its controls, its seed count, its smoke-scale
   configuration, and its ex-ante decision rule.
5. Any rule: record its reason next to it, so it can be safely revised when
   the reason no longer holds.
6. Number the doc in reading order and add one row to the table in
   `00-README.md`.

## References

- D. Procida, *Diátaxis: a systematic approach to technical documentation*.
- M. Nygard, *Documenting Architecture Decisions* (ADR), 2011.
- W. Oberkampf, C. Roy, *Verification and Validation in Scientific
  Computing*, Cambridge UP, 2010.
- G. Sandve et al., "Ten Simple Rules for Reproducible Computational
  Research", *PLOS Comp. Biol.* 9(10), 2013.
- G. Wilson et al., "Best Practices for Scientific Computing", *PLOS Biol.*
  12(1), 2014; "Good Enough Practices…", *PLOS Comp. Biol.* 13(6), 2017.
- P. Henderson et al., "Deep Reinforcement Learning that Matters", AAAI 2018
  (seed-variance argument).
- K. Popper, *The Logic of Scientific Discovery* (falsifiability).
- M. Schäfer, S. Turek, "Benchmark Computations of Laminar Flow Around a
  Cylinder", 1996 (the confined-cylinder reference geometry).
