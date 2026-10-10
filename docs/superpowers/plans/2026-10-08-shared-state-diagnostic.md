# Shared-state diagnostic implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Deliver a measured, reproducible explanation of the power head's remaining prediction gap against the strongest neural comparators.

**Architecture:** Add one diagnostic module around existing corpus/probe generation, selected-model restoration and independent plant references. Preserve existing frozen producers. A prospective freeze selects exact probes, sources and checkpoints before any new outputs are inspected; a complete immutable run supplies raw vectors and a concise Markdown report.

**Tech Stack:** Existing pinned Python 3.11, CPU float64 PyTorch, NumPy, MuJoCo, SciPy; no dependencies.

**Spec:** `docs/superpowers/specs/2026-10-08-shared-state-diagnostic.md`

## Global Constraints

- PTCO-DIAG-0.1; exploratory seed 9000, A/shared loss; no statistical look, tuning, training, additional MPC or new robot.
- First 64 cold and 64 warm development probes per plant; 5/10/40 ms horizons; held commands, fixed initial activation.
- Same eight selected checkpoints as the completed pilot; CPU float64, one thread.
- Immutable pilot/source artifacts; additive diagnostic module and reports only; current branch, local commits.
- Empirical numerical envelopes from three physics resolutions and independent continuous equations; no omitted rows or zero-filled missing metrics.

## Review Focus

- Warm probe identity and command/state selection must be fixed before evaluating models.
- New-command zero branch must retain initial activation and independent simulator reset.
- Signed vector cancellation must not be misreported as a sum of positive norms or as a causal control explanation.
- Composition must use the exact same held command and no added observations; NODE refinement is explicit.
- An incomplete or altered artifact must not produce a complete-looking diagnostic report.

### Task 1: Tested decomposition and frozen probe inputs

**Files:** Create `poc/shared_state_diagnostic.py`, `tests/test_shared_state_diagnostic.py`; create `config/shared_state_diagnostic_freeze.json`, `data/diagnostic_inputs/<digest>.npz` through prepare only.

**Interfaces:** Consumes existing `make_probe_library`, `load_corpus`, pilot selected identities, `predict` API. Produces `decompose(model_zero, model_command, truth_zero, truth_command, composed)` and `prepare(root)` with hash-bound probe coordinates and selected identities.

- [x] Write tests that catch sign errors/cancellation, fixed lagged initial activation through zero/nonzero and half-step queries, and short/altered probe inputs.
- [x] Run `.venv/bin/python -m pytest -q tests/test_shared_state_diagnostic.py`; expected failure because diagnostic module is absent.
- [x] Implement pure decomposition, model endpoint/composition queries, frozen selection and input validation.
- [x] Run the same tests; expected pass. Prepare must not evaluate model outputs. Commit the additive code/spec/plan/tests.

### Task 2: References, immutable measurement and report

**Files:** Extend only new module/tests; register `config/revisions/MX-shared-state-0001.json`.

**Interfaces:** Consumes Task 1 freeze and probes; existing Plant, independent integrate, selected model loader and run_job. Produces raw endpoint/error arrays, complete reference convergence/status arrays, summary rows and `reports/shared_state_diagnostic_result.{json,md}` bound to the immutable manifest.

- [x] Add failing tests for uncertainty propagation, signal resolution, changed artifact hash and missing model coordinates; run focused tests, expected failure.
- [x] Implement reference sweep, measurement, fail-closed raw aggregation/reaggregation and CLI `prepare/run/report/verify`. All four models use the same frozen arrays; report physical errors and paired comparisons by plant/stratum/horizon.
- [x] Run focused tests and `.venv/bin/python -m pytest -q`; expected all pass. Commit tested producer before freezing/running it.

### Task 3: Execute and deliver the diagnosis

**Files:** Generated freeze, immutable M40 exploratory evidence, reports, README link and completion receipt.

**Interfaces:** Consumes tested frozen implementation; produces 3,072 measured model rows and all qualified reference branches, concise findings, exact reproduction commands.

- [x] Execute `.venv/bin/python -m poc.shared_state_diagnostic prepare`; expected explicit 256 frozen probes, eight selected checkpoints, no model outputs opened.
- [x] Commit the freeze before `.venv/bin/python -m poc.shared_state_diagnostic run`; expected complete measured receipt, 3,072 rows, all numerical checks qualified.
- [x] Run `report` and `verify`; expected independent reaggregation matches saved values and hash closure, exact full row coverage. Read all cold/warm conditions and comparisons before stating conclusions.
- [x] Write practical findings and state the one next experiment justified by the diagnosis, or the remaining ambiguity. Commit measured report and complete this plan's ledger.
- [x] Request one fresh read-only whole-change review, resolve material findings with regression tests, preserve receipts. Deliver the results with links and the limits of one-seed development evidence.
