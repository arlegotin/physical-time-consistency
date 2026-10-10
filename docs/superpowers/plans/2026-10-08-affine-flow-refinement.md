# Affine-flow refinement implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Finish a targeted, measured test of whether consistent inexpensive motion propagation improves the strongest-baseline comparison.

**Architecture:** Add a two-stage affine acceleration decoder and an additive experiment driver. Reuse historical nominal data, selected comparators and measured references by hash, preserving all frozen source. Train/select nominally, then measure prediction, actual latency and the paired control conditions.

**Tech Stack:** Existing pinned Python3.11 CPU PyTorch, NumPy, MuJoCo, SciPy; no new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-08-affine-flow-refinement.md`

## Global constraints

- Current branch and meaningful local commits; no push/publish.
- PTCO-AFF-0.1 development seed9000, existing A/shared loss; changed architecture never inherits old confirmation claims.
- 18,182 parameters, two field-network evaluations, known tau0.08s, command/state units preserved.
- Existing frozen sources and all old evidence immutable; all additions live in new modules/configuration/reports.
- Nominal selection only; no artificial superiority threshold or resource ceiling.
- Finish through measured comparison; if a live improvement needs confirmation, register/execute that continuation rather than stop at the pilot.

## Review focus

- Lagged midpoint must start from original activation and exact held command; zero command still allows existing activation force.
- Both zero and positive intervals in one batch must be handled without hidden network/clock side effects.
- Historical source contexts may be used only for byte-identical data and unchanged model semantics.
- Selection, continuation and resumptions must use nominal labels and exact optimizer/RNG state.
- A faster neural query is not automatically a faster or better controller; all physical conditions and negative outcomes remain visible.

### Task 1: Mechanically coupled predictor

**Files:** Create `poc/affine_flow.py`, `tests/test_affine_flow.py`.

**Interfaces:** Consumes existing `Scales`, `interval_tensor`, `prediction` and exact analytic maps. Produces `AffineFlow(actuator, scales)`, `coefficients(qv)`, `predict(...)` and stable `lag_coefficients(h)`.

- [x] Write independent constant-field direct/lagged endpoint tests (nonzero initial w), physical-command Jacobian tests, exact h=0/mixed-zero and invalid-input tests, parameter/gradient/work assertions.
- [x] Run `.venv/bin/python -m pytest -q tests/test_affine_flow.py`; expected missing module failure.
- [x] Implement the specified two-stage map and stable coefficients without changing existing modules.
- [x] Run the same tests plus existing analytic/power tests; expected all pass. Commit spec/plan and tested model.

### Task 2: Frozen nominal experiment and safe training

**Files:** Create `poc/affine_experiment.py`, `tests/test_affine_experiment.py`, generated `config/affine_refinement_freeze.json`.

**Interfaces:** Consumes Task1 model, existing TrainingSession/selection, byte-identical historical A train/validation data and source hashes. Produces CLI `prepare`, `train`, `status` with per-plant resume and immutable selected checkpoint receipts.

- [x] Write tests for wrong dataset role, changed frozen input, validation-only selection and identical optimizer-boundary resumption using a tiny exact corpus. Run focused tests; expected absent interface failure.
- [x] Implement prospective freeze, analytic plumbing fit, nominal 100-epoch blocks, logged convergence/repair and status. Bind source/config/data/environment identities and preserve interrupted attempts.
- [x] Run both new test modules and the relevant existing training tests; expected pass. Commit tested driver before preparing any real fit.
- [x] Prepare and commit prospective freeze; run robot fitting serially, monitoring actual progress/resources. Expected qualified selected checkpoints or a documented nominal diagnostic, never selection by fine-time outcomes.

### Task 3: Practical comparison and finish

**Files:** Create `poc/affine_measurement.py` and `tests/test_affine_measurement.py`; create raw measurement/control artifacts, `reports/affine_refinement_findings.md` and completion/status receipts; update README and create a separate extended replay only with measured artifacts.

**Interfaces:** Consumes Task2 selected checkpoints, complete diagnostic reference arrays and pilot comparators/episode IDs. Produces 768 new shared-state rows, measured latency/training resource tables, 80 paired control episodes and raw-linked reports.

- [x] Write tests for missing coordinates, changed comparator/reference hash, dropped failed episode and recomputed-cost/report mismatch; run, expected absent measurement/aggregation failure.
- [x] Implement measurement, paired control dispatch and complete raw reaggregation using the same clocks, action paths, CEM and failure policy. Expected probe/vector/cost coverage exact, original comparators unchanged.
- [x] Run focused and full suites; expected pass. Commit tested measurement producer, then freeze selected checkpoints and control requests before new outcomes.
- [x] Execute shared-state/latency measurements and all qualified control cells. Expected complete rows/episodes or explicit repairable validity failures; retain losses.
- [ ] If development gives a useful strongest-baseline improvement, write and execute a fresh-data independent confirmation sub-contract under this plan. Otherwise document scoped lack of improvement and finish with the existing supported original-method utility result.
- [ ] Independently reaggregate raw values, request one fresh read-only whole-change review, resolve material findings, run final verification and commit shareable results. State finished versus nonterminal explicitly.
