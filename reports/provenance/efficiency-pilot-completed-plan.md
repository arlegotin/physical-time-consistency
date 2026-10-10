# Targeted Efficiency Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Produce an honest quality-versus-compute comparison against strong predictors on the existing robot with 40 and 10 ms command updates.

**Architecture:** Add a mathematically equivalent head wrapper and a small explicit exploratory runner. Reuse the qualified nominal engine, plant, planner and artifact writer without changing their frozen source. Preserve the completed study and statistical ledger.

**Tech Stack:** Existing CPython 3.11, CPU PyTorch, NumPy, MuJoCo and pytest pins; standard-library coordination; SVG plots.

**Spec:** `docs/superpowers/specs/2026-10-07-targeted-efficiency-pilot.md`.

## Global Constraints

- Current branch; explicit commits; no push, publication, paid compute or physical robot.
- `PTCO-EFF-0.1` is exploratory; seed 9000, root 7101, task IDs 0–9; no significance or equivalence claim from this pilot.
- Existing A/shared-loss fits and nominal recipes, tau 0.08 s, 40 ms sensing/replanning, six two-dimensional knots, CEM 128/4/16.
- Actual holds 40/10 ms, partitions 40/5 ms, physics 250 microseconds; never cross a command boundary in a prediction step.
- Models head, `mlp_h`, generic NODE, structured NODE, calibrated physics; 400 paired development episodes plus eight true-plant audits.
- Preserve frozen implementations and evidence; new files and prospective hash-bound requests only; serial expensive jobs; actual owner resource allocation has no artificial ceilings.

## Review Focus

- Lagged batching must keep initial activation fixed and count both neural rows; Task 1 checks gain and work.
- Wrong/mismatched power modes must retain their own formula; Task 1 checks all modes and mixed zero-time rows.
- Unknown stateful networks must fail instead of silently changing predictions when batching; Task 1 rejects dropout/batch normalization.
- A coarse partition must split at 10 ms executed-command changes; Task 2 checks actual input paths and 40 ms observation packets.
- Partial, duplicated or changed-checkpoint results must never produce a frontier; Tasks 2–3 validate hash closure and full expected-cell coverage.

---

## Task 1: Equivalent optimized head and honest work accounting

**Files:** Create `poc/efficient_power.py`, `tests/test_efficient_power.py`.

**Interfaces:** `EfficientPowerHead.from_head(head: PowerHead) -> EfficientPowerHead` clones the selected pointwise network/scalers and preserves the prediction API. `neural_work(prediction: Prediction, batch_size: int, actuator: str, power_mode: str | None = None) -> dict` reports dispatches, vector fields and neural rows without redefining existing counters.

- [x] Write `test_optimized_head_preserves_predictions_and_all_derivatives`: compare real seeded tanh networks in float64, all modes and specified intervals; include weight gradients, mixed zero rows, lag activation and four-step rollout. A factor error or activation shortcut must fail.
- [x] Write `test_direct_correct_uses_one_pass_and_lagged_batches_two_rows`: instrument real forward hooks; assert direct one B-row pass, lag one 2B-row pass, accurate reported work, zero queries no work, other direct modes two passes.
- [x] Write `test_stateful_or_training_network_is_rejected`: dropout, BatchNorm and training mode cannot silently alter batch semantics.
- [x] Run `.venv/bin/python -m pytest -q tests/test_efficient_power.py`; observe the intended RED from absent feature.
- [x] Implement the wrapper with direct cancellation and lagged concatenated features; preserve original formula in other direct modes and exact activation/no-op handling.
- [x] Run focused tests plus `tests/test_power.py`; compare outputs to tolerances 1e-12 float64 and 2e-6 float32. Commit the spec, plan and verified optimization.

## Task 2: Frozen pilot inputs and compatible execution

**Files:** Create `poc/efficiency_pilot.py`, `poc/efficiency_episode.py`, `tests/test_efficiency_pilot.py`, `config/revisions/MX-efficiency-pilot-0001.json`; register the exact extension in `config/milestones.json`. Generated requests live under `config/efficiency_pilot_jobs/`.

**Interfaces:** `prepare(root: Path) -> dict` freezes existing eligible seed-9000 selections, nominal recipes, explicit coordinates and dependencies. `prepare_episodes(root: Path, freeze: dict) -> dict` requires both qualified `mlp_h` selections and writes 408 exact requests. `run_episode(root: Path, request_path: Path) -> dict` produces a complete M40 exploratory manifest using `execute_episode`/`save_episode`. `run_registry(root: Path, registry: dict) -> dict` serially executes compatible missing jobs in fresh processes and records every attempt.

- [x] Write `test_pilot_rejects_changed_checkpoint_and_confirmation_seed` and `test_registry_has_exact_paired_cells_and_oracle_subset`: exact model/rate/partition/task coverage, same workload, correct eight oracle audits, no partial or duplicate cells.
- [x] Write `test_high_rate_coarse_partition_splits_without_extra_observations`: run the real action compiler/planner against a small deterministic fixture; check 10 ms queries and six cost anchors.
- [x] Write `test_completed_job_reuse_checks_request_and_raw_artifacts`: compatible completion reused, changed request/partial manifest rejected, interrupted attempt retained.
- [x] Run focused tests for RED, then implement additive producers with original source inventory plus the new wrapper/runner; do not patch frozen model allowlists.
- [x] Freeze and commit the twelve-part extension before new scientific outcomes. Run the original analytic learning controls and nominal fitting for `mlp_h` in the existing compatible development context, with checkpoint selection on validation only. Use original nominal continuation if needed.
- [x] Freeze both qualified selected checkpoints, verify every existing model and register all exact episode requests. Run focused pilot/control/artifact tests and commit before opening pilot control results.

## Task 3: Execute, compare, and publish local artifacts

**Files:** Create `poc/efficiency_result.py`, `tests/test_efficiency_result.py`; generate `reports/efficiency_pilot_result.{json,md}`, `reports/efficiency_pilot_quality_compute.svg`, and measured profiling/resource receipts.

**Interfaces:** `profile_models(root: Path, registry: dict) -> dict` times real frozen batch-one/128 inputs with 50 warm-ups and 500/100 measurements, original and optimized heads. `summarize(root: Path, registry: dict, completed: dict, profiles: dict) -> dict` validates all expected pairs and artifacts before producing condition rows, paired differences and descriptive cost/time frontiers. `write_report(root: Path, result: dict) -> None` preserves losses, failures, unresolved inference and actual-work labels.

- [x] Write `test_incomplete_duplicate_or_modified_episode_cannot_summarize` and `test_frontier_retains_losses_and_distinct_hardware_rates`: literal small fixtures; missing timing or failed candidates cannot become zero-filled wins.
- [x] Observe RED, implement the consumer, then run GREEN and appropriate existing cost/statistics/provenance tests.
- [x] Profile selected models, measure the first representative episode, project total disk/wall costs, and run all remaining pilot cells serially. Preserve attempted runs and compatible completed jobs.
- [x] Recompute summary costs from raw true anchors/commands, then produce all condition tables and the SVG quality/compute comparison. No new statistical look is spent for exploratory summaries.
- [x] Run the complete suite after final source changes, inspect results, commit measured artifacts, and obtain one fresh whole-change review. Correct any important findings with demonstrated regression tests.
- [x] Release the honest pilot result with a specific next comparison to confirm, or a targeted limitation requiring a separately specified follow-up. Do not imply the four-joint experiment, fresh population confirmation or real-time deployment has already run.
