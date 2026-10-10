# Compute-matched NODE implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Determine whether AffineFlow has a useful advantage over cheap and RK4 NODEs at matched planning allocations.

**Architecture:** Add a physically correct two-evaluation NODE and a shared stateless controller; preserve frozen producers. Reuse verified selected models, train the new comparator fairly, freeze timing-selected budgets, then run independent paired tasks and report measured tradeoffs.

**Tech Stack:** Existing CPU PyTorch, NumPy, SciPy, MuJoCo, pytest, standard-library artifacts; native Python 3.11.

**Spec:** `docs/superpowers/specs/2026-10-09-compute-matched-node.md`

## Global Constraints

- Current branch; local meaningful commits; no push, paid services or new dependencies.
- Existing frozen sources/results unchanged; additive modules and reports only.
- Same lagged robot, 40 ms hardware clocks, six torque knots, 240 ms MPC, 1.6 s tasks.
- Eleven paired training seeds 2026100800–2026100810; nominal-only shared-loss training.
- All three models at 5/40 ms; budgets selected by timing before independent outcomes.
- Heavy mutable work on the authorized Data volume; raw losses and failed attempts retained.
- No effect floor or artificial resource cap; honest uncertainty and finite-scope claims.

## Review Focus

- Mixed zero intervals and differentiating the physical command must preserve fixed initial activation.
- Candidate failures must keep infinity/status and fallback instead of dropping difficult tasks.
- Unequal candidate budgets must consume paired RNG prefixes without altering task draws.
- An interrupted fit/control must resume committed boundaries and not duplicate favorable attempts.
- Coarse NODE or moving-state losses must remain visible when finer-step averages favor AffineFlow.

### Task 1: Competitive solver and shared fast controller

**Files:** Create `poc/quadrature_node.py`, `poc/stateless_mpc.py`, `tests/test_quadrature_node.py`, `tests/test_stateless_mpc.py`.

**Interfaces:** `QuadratureNODE(actuator,scales).predict(state,command,h_s,history=None,power_mode=None)` returns existing Prediction. `stateless_mpc.rollout_candidates` and `search` use the original argument/return contracts, explicitly history-independent models only and 40 ms actuation.

- [ ] Write independent constant/direct and linear-time/lagged leading-coefficient tests, mixed-zero/negative/nonfinite intervals, gradients, two-call counters and no state/history mutation. Run and observe missing-feature failure.
- [ ] Implement the spec's exact two-stage quadrature, using the inherited acceleration field and known activation.
- [ ] Write comparisons to original rollout/search for actual AffineFlow and both NODEs, exact analytic fixtures, fixed CEM ties, nonfinite candidates/all-failure, nonzero query time and invalid commands/history models. Observe missing scorer failure.
- [ ] Implement vectorized shared held-command scoring and deterministic CEM with original work/failure semantics. Preserve original files.
- [ ] Run `.venv/bin/python -m pytest -q tests/test_quadrature_node.py tests/test_stateless_mpc.py`; expect all pass. Commit solver/controller.

### Task 2: Frozen fair experiment and measured execution

**Files:** Create `poc/compute_study.py`, `poc/compute_study_runs.py`, `tests/test_compute_study.py`, `config/compute_study_*.json`, `reports/compute_study_*.json` (generated).

**Interfaces:** CLI `python -m poc.compute_study {prepare,train,calibrate,measure,controls,reproduce,status}`; `--seed` filters missing nominal fits only. `compute_study_runs.execute_episode(model,settings,seed,episode_id,step_ns,physics_step_ns)` returns complete compressed raw truth, candidates, actions, timings and task identity. Prospective immutable freeze and per-job receipts name exact model/input hashes.

- [ ] Test paired namespaces/prefixes, timing-only nearest budget selection, immutable stale-input rejection, missing/duplicate cell rejection and resume boundaries before implementation; observe RED.
- [ ] Implement verified old checkpoint loading, unchanged nominal training helper, CPU deterministic setup, atomic receipts/resume, external work directory, new probes/truth and independent tasks. Record allocation and source/model hashes before measurement.
- [ ] Launch the eleven serial quadrature fits with durable resumption and progress. Task 3 can implement independent reporting software while this external process runs; it must await qualified fits before calibration, measurement and control.
- [ ] Run task tests plus affected existing training/controller tests; expect pass. Commit foundation and frozen execution evidence separately as meaningful results arrive.

### Task 3: Decision, reproduction and shareable results

**Files:** Create `poc/compute_study_report.py`, `tests/test_compute_study_report.py`, `reports/compute_study_findings.md`, `reports/compute_study_demo.html`, machine tables and reproduction receipts/instructions; update reader-facing README with scoped findings.

**Interfaces:** CLI `python -m poc.compute_study_report {aggregate,demo,verify}`; aggregate validates expected six-by-seed-by-episode cells before statistics and uses look-bound evidence identity. Demo uses saved actual trajectories only.

- [ ] Test positive/negative/near-zero contrasts, missing/failed rows, Holm eight-contrast family, moving-state losses, compute mismatch disclosure, independent cost arithmetic and demo trajectory closure; observe RED.
- [ ] Finish all eleven nominal fits. Calibrate/profile searches and freeze six budgets before fresh task outcomes. Measure all prediction strata/horizons and run complete paired control coverage serially, updating durable progress.
- [ ] Implement raw-to-summary aggregation and targeted precision analysis. Add independent samples prospectively only if the spec's inference/precision rule requires them; otherwise finish with bounded claim.
- [ ] Execute fresh-process control replay/finer physics and first/last new-comparator retraining; verify canonical hashes/decisions and all raw costs. Produce concise findings, self-contained demo, exact reproduction instructions and consumed-resource table.
- [ ] Run `.venv/bin/python -m pytest -q`; expect full suite green. Commit measured results; one fresh whole-change review under executing-plans, one Important/Critical fix pass if necessary. Archive this plan's execution record, retain current branch and report conclusion with limits.
