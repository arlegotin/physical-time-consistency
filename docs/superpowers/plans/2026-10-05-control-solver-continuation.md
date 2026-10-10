# Closed-loop NODE solver continuation implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan inline. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the actual fifty-cell solver request and propagate its empirical uncertainty into the registered control result.

**Architecture:** Add a fixed registry/serial runner using the existing episode engine and a read-only aggregation module. Extend the evaluation-only consumer to verify that receipt before clearing solver prerequisites; leave sealed producer modules unchanged.

**Tech stack:** Existing CPU Python 3.11, PyTorch, NumPy, MuJoCo and pytest; no new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-05-control-solver-continuation.md`.

## Global constraints

- Original weights, train corpus, checkpoint selection, eight signed contrasts and alpha ledger remain unchanged.
- Fifty nominal NODE parent cells; two/four RK4 substeps; forty fresh observations per episode.
- Existing 40 ms sensing/replanning/actuation, planner settings, cost anchors and root 6101 task streams.
- Serial scientific execution; retain all original, interrupted and repeated attempts.
- No look from missing/stale/partial evidence; uncertainty includes empirical solver discrepancies.
- Do not modify sealed control/training/refinement sources or `poc/control_audit_closure.py`.

## Review focus

- A changed parent command/checkpoint/physical clock must fail before execution.
- A partial resolution cohort or omitted failure must not look converged.
- A changed source/registry during resume must not reuse incompatible evidence.
- Refinement changing the learned nominal map must remain visible.
- Reporting twice must preserve numerical identity and statistical look spending.

## Task 1: Registry, episode runner and raw convergence

**Files:** Create `poc/control_solver_continuation.py`, `tests/test_control_solver_continuation.py` and the explicit MX request. Register only that request in `config/milestones.json`.

**Interfaces:** Consume archived trigger, primary freeze/batch, completed solver probes and original episode requests. Produce frozen resolution jobs and complete raw-audited episode receipts with parent ID and substep count.

- [x] Write tests for exact fifty-parent expansion into one hundred two/four-substep jobs; reject missing/duplicate/changed parents and mixed clocks.
- [x] Run them and observe the missing implementation failure.
- [x] Implement pure derivation and a frozen prepare/resume runner, real episode execution through a NODE-only wrapper, full manifests and raw costs.
- [x] Test realized-cost convergence with known signed changes, retained failures, missing resolutions and classification changes. Use genuine tiny mechanics for the producer's full manifest/clock/work boundary.
- [x] Verify focused tests and the complete suite; commit the implementation and ledger.

## Task 2: Nominal validation and evidence consumer

**Files:** Create `poc/control_solver_validation.py`, `tests/test_control_solver_validation.py`; modify only evaluation-only `poc/control_result.py` and its tests.

**Interfaces:** Consume the ten selected nominal NODE checkpoints and original allowed validation corpora. Produce complete one/two/four-substep nominal objectives and a verified solver-resolution receipt. The result consumer retains the original costs and expands nominal NODE envelopes.

- [x] Write failing tests proving incomplete/stale solver receipts cannot clear the existing prerequisite or spend a look.
- [x] Write tests for outward envelope addition and repeated-report idempotence, plus validation keeping selected tensors and target counts unchanged.
- [x] Run RED, implement minimal receipt verification and consumer support, then run GREEN and the full suite.
- [x] Commit before preparing scientific jobs; freeze the solver source/input hashes.

## Task 3: Execute the registered diagnostics and return

- [x] Complete the serial full finer-physics continuation; revalidate reusable subset jobs.
- [x] Prepare the solver/validation registry, measure a representative job and confirm actual resource availability.
- [x] Run the hundred serial solver episodes and ten nominal-only validation audits, preserving attempts.
- [x] Independently verify raw costs/manifests/coverage and report original-versus-refined behavior.
- [x] If convergence is unresolved, register the specific next doubling/solver-fit interpretation before new outcomes. Otherwise return to the unchanged eight-contrast result evaluator and original reproduction pipeline.
- [x] Record progress and meaningful commits; no terminal overall PASS without required reproduction/review.
