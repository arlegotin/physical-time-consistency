# Short-calibration AffineFlow implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Keep execution on the current branch; use targeted tests and meaningful commits at each task boundary.

**Goal:** Determine whether torque-affine representation or AffineFlow propagation reduces calibration data or decision-scoring computation after adding strong cheap, physical and residual alternatives.

**Architecture:** Add a self-contained `PTCO-CAL-0.1` study alongside the frozen historical producers. New solver controls share the existing networks, a fresh session collector supplies nested data budgets, and separate prediction/decision/timing producers emit hash-bound artifacts for one raw-derived report. Complete a small pilot before expanding the development matrix; deadline control and larger-arm work are gated follow-ups.

**Tech Stack:** Native CPython 3.11, single-threaded CPU PyTorch 2.8.0, MuJoCo 3.14.0, NumPy 2.2.6, SciPy 1.15.3, pytest 8.4.1 and standard-library artifact handling.

**Spec:** [Short-calibration AffineFlow continuation](../specs/2026-10-12-short-calibration-affineflow-design.md). Read both documents before implementing. Numerical values and study claims come from that spec.

## Global constraints

- Work on the current branch and make meaningful local commits; no push or publication.
- Native CPython 3.11; CPU PyTorch 2.8.0, MuJoCo 3.14.0, NumPy 2.2.6, SciPy 1.15.3 and pytest 8.4.1; no new dependencies.
- Use additive modules and a new protocol `PTCO-CAL-0.1`; preserve historical source, freezes, checkpoints, reports and acceptance rules.
- Stage A/B use the existing two-joint lagged plant, common known `tau_s = 0.08`, integer physical time, and held torque commands in N*m.
- Float32 training; float64 numerical audits and scoring; single-threaded CPU with thread settings recorded.
- All methods receive the same available observations, command paths, nominal prior and data-budget partition. Train each learned solver through its own update rule.
- Generate all required inputs from the public checkout. Default mutable output is ignored `study/short-calibration`; allow an explicit `--output-root` without workstation-specific paths.
- Retain failures, unsuccessful fits and interrupted attempts. A completed job is reusable only when its input/source/model hashes match.
- Freeze recipes before their scored outcomes; no confirmation-based tuning, selected favorable seeds or post hoc tolerance changes.

No worktree or branch change is needed: the user explicitly chose the current branch. This plan implements Stage A/B only. Its final report is the concrete input for deciding whether to write Stage C's continuous-time deadline-controller spec.

## File structure and shared contracts

New code follows the repository's `poc/` modules and `tests/test_*.py` layout. Do not add these methods to the historically frozen `models.factory.CORE_MODELS` or reuse the old `compute_study.StudyStore` with its hard-coded protocol.

| New module | Responsibility |
|---|---|
| `poc/actuator_integrals.py` | Stable, differentiable shared-tau actuator coefficients |
| `poc/affine_rk4.py` | RK4 propagation of the existing f/B network |
| `poc/actuator_aware_node.py` | Proposed two-evaluation NODE propagation |
| `poc/calibration_models.py` | New study's model registry, seeded construction and small residual |
| `poc/calibration_store.py` | Protocol-specific job identities, immutable receipts and resume validation |
| `poc/calibration_data.py` | Fresh collector, session serialization, role partition and nested budgets |
| `poc/calibration_training.py` | Own-solver fits and selection without private historical files |
| `poc/calibration_predictions.py` | Moving-state prediction library, independent truth and propagation diagnostics |
| `poc/calibration_decisions.py` | Common candidate banks, predicted selection and independent realized regret |
| `poc/calibration_timing.py` | Rotated inference and complete bank-scoring timings |
| `poc/calibration_report.py` | Coverage, raw aggregation, figures, provisional screen and gate decision |
| `poc/calibration_study.py` | CLI coordination and status; no scientific calculations in the parser |

`Receipt` means a plain JSON-compatible dict with `job_key`, `path` relative to the output root, and `manifest_sha256`. A completed manifest names `protocol_id`, input/source hashes, coordinate request, all produced file hashes, summary and resource counts. Array files use existing NumPy dtypes and keys `state`, `command`, `duration_ns`, `time_ns`, `trajectory_id`; study-specific loading supports `(segments,26,6)` states and `(segments,25,2)` commands. Do not use the historical loader, which enforces 33 states and 32 commands.

Study coordinates are `(matrix, session_id, budget_s, model_id, initialization_id)`; physical fits have `initialization_id=None`. The pilot uses session 0, initialization 0; development contains sessions 0–4 and initializations 0–1, retaining the pilot cells. Initial network seed is `2026101200 + 2*session_id + initialization_id`; construction resets RNG within a fork so the appropriate representation families have identical initial tensors. Collector/evaluation RNGs use the spec's role/channel namespace separately from this network seed.

## Review focus

- Tiny or mixed zero intervals must not divide by zero, erase cubic response, or corrupt command/interval gradients: Task 1.
- The 25-start segments leave short windows; batching and resume must count every start once per epoch without overflow or reading validation/full-budget data: Tasks 3/4.
- A failed predictor, infeasible selected action, tied ranking, or empty feasible truth bank must produce an explicit outcome rather than a favorable regret: Tasks 5/6/8.
- A changed solver, asset, source, budget, output root or interrupted attempt must never masquerade as a reusable qualified fit: Tasks 3/4/9.
- A physical baseline win, nonmonotone curve, optimization-limited fit or slow complete scorer must remain visible even when an AffineFlow microbenchmark improves: Tasks 7/8.

## Task 1: Add the two mechanism solvers

**Files:** Create `poc/actuator_integrals.py`, `poc/affine_rk4.py`, `poc/actuator_aware_node.py`, `tests/test_actuator_integrals.py`, `tests/test_affine_rk4.py`, `tests/test_actuator_aware_node.py`.

**Interfaces:**

- `lag_integrals(h: torch.Tensor, tau_s: float) -> tuple[torch.Tensor, torch.Tensor]` returns Kv and Kq.
- `response_ratios(h: torch.Tensor, tau_s: float) -> tuple[torch.Tensor, torch.Tensor]` returns Kv/beta and Kq/beta, with exact zero values for h=0.
- `AffineRK4(actuator: str, scales: Scales)` subclasses `AffineFlow`; `predict(..., solver_substeps: int = 1) -> Prediction` exposes the existing predictor arguments plus RK4 substeps.
- `ActuatorAwareNODE(actuator: str, scales: Scales, tau_s: float = .08)` subclasses the structured `NeuralODE`; its `predict(...) -> Prediction` uses the spec's midpoint and response-ratio update. Reject unsupported nonuniform tau. Stage A accepts the historical tau only; do not let its ratios disagree with the inherited activation helper.

- [ ] Write independent Decimal-reference coefficient tests at `h/tau=(1e-8,1e-6,.0625,.1,.5,1,2,4)` with `rel=1e-11`; assert finite float32 outputs, zeros, invalid tau/interval rejection, branch continuity and finite interval gradients.
- [ ] Write constant dense-matrix/nonzero-drift/nonzero-activation endpoint tests for both models, direct/lagged leading command-response orders, immutable/mixed-zero queries and work counters (`4*substeps` versus `2`). Assert `AffineRK4` and `AffineFlow` start with identical coefficient tensors; assert all three NODEs start with identical field tensors. Include a nonlinear mechanical fixture where the new NODE is *not* exact, to constrain its claim.
- [ ] Run `.venv/bin/python -m pytest -q tests/test_actuator_integrals.py tests/test_affine_rk4.py tests/test_actuator_aware_node.py`; expect missing-module/feature failures before implementation.
- [ ] Implement the coefficient series and two solvers, using the spec's independently derived equations. Keep output normalization, exact activation, existing status semantics and differentiability. Do not alter any old source file.
- [ ] Run those tests plus `tests/test_affine_flow.py tests/test_quadrature_node.py tests/test_models.py`; expect all pass. Commit: `feat: add torque-affine RK4 and actuator-aware NODE controls`.

## Task 2: Build the complete comparator registry

**Files:** Create `poc/calibration_models.py`, `tests/test_calibration_models.py`.

**Interfaces:**

- `MODEL_IDS: tuple[str, ...]` contains the seven fitted IDs from spec section 5.
- `build_model(model_id: str, scales: Scales, seed: int) -> torch.nn.Module` constructs a fresh lagged predictor; `physics_nominal` is accepted as an extra nonfit reference.
- `PhysicsResidual(scales: Scales)` exposes `acceleration(qv: Tensor, effort: Tensor) -> Tensor` and `predict(...) -> Prediction`, with frozen nominal parameters, 32/32 residual and one 40 ms RK4 training step. For all learned methods the study default is one predictor step per observed interval.
- `model_metadata(model: torch.nn.Module) -> dict` records prior parameters, network architecture, parameter counts, solver and field-work rules.

- [ ] Test all IDs return valid six-state predictions, unregistered IDs fail, family seeds pair tensors, and constructing a model does not change the caller's RNG. Assert the residual contains 1,410 trainable parameters and frozen nominal `[.10,.02,.02,.05,.05]`; ensure no trainable physical parameter enters the residual optimizer.
- [ ] Test a zeroed residual agrees with nominal physics under the same RK4 resolution, dense activation affects acceleration, and backward passes reach every residual output. Assert one residual RK4 step records four network calls; inherited rigid-reference zero-call accounting is forbidden. Test metadata cannot describe a four-call predictor as two-call or hide physical-reference solver work.
- [ ] Run `.venv/bin/python -m pytest -q tests/test_calibration_models.py`; observe RED.
- [ ] Implement the registry and residual using `PhysicsFit` acceleration and shared activation semantics without reading plant parameters. `physics_fit` uses its existing qualified numerical integrator; residual RK4 uses its own fixed training map and is scored through that map.
- [ ] Run `tests/test_calibration_models.py tests/test_physics_fit.py tests/test_physics_training.py` via pytest; expect pass. Commit: `feat: add short-calibration comparator recipes`.

## Task 3: Generate isolated, budget-accounted calibration data

**Files:** Create `poc/calibration_store.py`, `poc/calibration_data.py`, initial `poc/calibration_study.py`, `tests/test_calibration_store.py`, `tests/test_calibration_data.py`, `tests/test_calibration_cli.py`.

**Interfaces:**

- `CalibrationStore(output_root: Path, bindings: dict[str,str])` supplies `produce(stage: str, request: dict, producer: Callable[[Path],dict]) -> Receipt` and `read(receipt: Receipt) -> tuple[Path,dict]`.
- `prepare(root: Path, output_root: Path, matrix: str = 'pilot') -> dict` freezes protocol, hashes and coordinates in the output store; development expansion creates a new matrix request linked to the unchanged pilot recipe.
- `collect_session(session_id: int, *, root: Path, output_root: Path) -> Receipt` produces a 600-segment session with trajectory-boundary resume.
- `load_budget(session_receipt: Receipt, budget_s: int, store: CalibrationStore) -> tuple[dict,dict,Scales,dict]` returns training arrays, validation arrays, common train-only scales and data-accounting metadata.
- CLI `prepare`, `collect`, `status` supports `--output-root`, `--matrix`, `--session`; no old authorization/local-context file is read.

- [ ] Test 30/120/600 total seconds split into 24/6, 96/24 and 480/120 train/validation seconds, complete segments, no shared trajectory IDs between roles, nested identity and no four-step window crossing a boundary.
- [ ] Test exact collector determinism, 40 ms command/observation ticks, initial-state/excitation laws and nontrivial moving-state velocities. Mutate unused later-budget/validation labels and assert the 30-second training scales do not change. Use a short configurable *unit-test fixture* session; the real frozen session size remains 600.
- [ ] Test immutable receipts, protocol/source/asset/root separation, tampered files, duplicate coordinates, interruption before a completed segment, retained failure events and successful repeat calls returning identical receipts.
- [ ] Run `.venv/bin/python -m pytest -q tests/test_calibration_store.py tests/test_calibration_data.py tests/test_calibration_cli.py`; observe RED.
- [ ] Implement the fresh namespaces, bounded collector and protocol-specific loader/store. Reuse atomic/hash/lock utilities from `poc.io`, not historical protocol or role loaders. A changed matrix retains pilot jobs by binding them to identical recipe/session/budget inputs, with matrix membership stored separately from model-fit identity.
- [ ] Run those tests plus `tests/test_env.py tests/test_io_gates.py`; expect pass. Commit: `feat: add fresh calibration sessions and immutable study store`.

## Task 4: Train every learned solver on its own transition

**Files:** Create `poc/calibration_training.py`, `tests/test_calibration_training.py`; extend `poc/calibration_study.py` and its CLI tests.

**Interfaces:**

- `fit_model(store: CalibrationStore, session_receipt: Receipt, model_id: str, budget_s: int, initialization_id: int | None) -> Receipt` records all allowed rates, best valid selection, failures and optimization limits.
- `load_selected(store: CalibrationStore, fit_receipt: Receipt) -> torch.nn.Module` verifies data, normalization, recipe, canonical tensors and trained solver before returning a float64 evaluation model.
- `nominal_reference(store: CalibrationStore, session_receipt: Receipt, budget_s: int) -> Receipt` records unfitted nominal construction and common train-only scales. `load_selected` accepts this reference receipt with `selection_kind='unfitted_reference'`; coverage/reporting must distinguish it from fit receipts.
- `CalibrationTrainingSession(model: nn.Module, arrays: dict, scales: Scales, seed: int, *, learning_rate: float)` supplies `advance_updates(count: int) -> None`, `save(path: Path) -> None`, and `restore(path: Path) -> None`.
- CLI `train` adds `--model`, `--budget`, `--initialization` filters. Physical fits require no initialization coordinate.

- [ ] Test a solver spy proves supervised one/four-step losses call the selected model's own `predict`; cross-solver substitutions and full-budget weights fail identity validation. Check no validation label or evaluation metric enters optimizer updates or scales.
- [ ] Test 25-start segment tails at effective batch 256: every start appears exactly once per epoch, batches do not overflow, partial final batches are counted correctly, and four-step target counts exclude the final three starts of each segment. Interruption/resume reproduces selected tensors, optimizer/RNG state and target exposures exactly.
- [ ] Test two-rate nominal-only selection with the existing objective/tie rule, validation every five epochs, earliest plateau after 100 epochs, 1,000-epoch ceilings, a valid slower-rate recovery, all-rate failure and nonqualified-but-finite fits remaining reportable.
- [ ] Test physical fit reused once per session/budget, not once per neural seed; use synthetic endpoints to verify prior/selected solver resolution and failure reporting. Test nominal reference receipts reproduce frozen parameters/scales, work without a trained checkpoint and do not count as completed fits.
- [ ] Run `.venv/bin/python -m pytest -q tests/test_calibration_training.py`; observe RED.
- [ ] Implement the common stateless sampler and own-solver fitting using `optimizer_for`, `batch_objective`, `validate_nominal`, selection and canonical hashing helpers from `poc.train`. **Do not directly use `TrainingSession.advance_updates` for 25-step segments:** its complete eight-start-window accumulation can overflow a batch when one-start tails intervene. Flatten each shuffled epoch's eight-start windows and batch that fixed start list with a resumable cursor; keep epoch permutation and checkpoint state deterministic. Use `optimize_endpoints` for conventional physical calibration on the same observed endpoints.
- [ ] Run training tests plus `tests/test_training.py tests/test_control_retraining.py`; expect pass. Commit: `feat: fit calibration baselines through their own solvers`.

## Task 5: Measure natural-motion learning curves and integration effects

**Files:** Create `poc/calibration_predictions.py`, `tests/test_calibration_predictions.py`; extend CLI `predict`.

**Interfaces:**

- `prepare_evaluation(store: CalibrationStore, *, root: Path) -> Receipt` binds 64 independent 1.6 s trajectories, targets and held-probe commands, without consulting fitted models.
- `prediction_truth(store: CalibrationStore, evaluation_receipt: Receipt, *, root: Path) -> Receipt` records independent simulator endpoints and refinement envelopes.
- `measure_prediction(store: CalibrationStore, fit_receipt: Receipt, evaluation_receipt: Receipt, truth_receipt: Receipt) -> Receipt` emits per-state/horizon/component/partition rows and failure statuses.
- `propagation_diagnostic(store: CalibrationStore, fit_receipt: Receipt, evaluation_receipt: Receipt, truth_receipt: Receipt) -> Receipt` scores identical f/B tensors with both propagators and labels them as a same-weight diagnostic.

- [ ] Test the primary population has 23 anchors per trajectory from 320 through 1,200 ms, 1,472 total, supporting every recorded-command horizon through 400 ms. Distinct held probes at 5/10/20/40 ms and transient stress probes at 80/160 ms retain immutable starting activation. Check recorded-command rollouts end at 80/160/400 ms with query steps 5/10/20/40 ms and split before every actual command change.
- [ ] Test q/v RMSE and p95 from known physical-unit residuals, command/drift decomposition, composition differences, per-predictor failure records and complete coverage. Cross-solver diagnostic rows must never be labeled own-solver trained.
- [ ] Test three-resolution truth envelopes, first/last trajectory audit coordinates and required refinement when error envelopes exceed 1% of the error being interpreted. Do not treat all float64 discrepancies as zero by rounding.
- [ ] Run `.venv/bin/python -m pytest -q tests/test_calibration_predictions.py`; observe RED.
- [ ] Implement model-independent truth generation with `Plant`, using actual logged commands and integer clocks, then compute the raw predictions and diagnostics. Cache truth by evaluation/asset/resolution identity, never by model. Initial-state results remain secondary.
- [ ] Run those tests plus `tests/test_prediction.py tests/test_action_path.py`; expect pass. Commit: `feat: measure calibration learning curves on moving states`.

## Task 6: Measure finite-bank decision regret and sampled feasibility

**Files:** Create `poc/calibration_decisions.py`, `tests/test_calibration_decisions.py`; extend CLI `decisions`.

**Interfaces:**

- `candidate_bank(evaluation_receipt: Receipt, state_id: int, store: CalibrationStore) -> np.ndarray` returns shared `(256,6,2)` torque knots, including zero and collector-PD references.
- `truth_candidates(store: CalibrationStore, evaluation_receipt: Receipt, *, root: Path) -> Receipt` records independent cost/feasibility for every candidate, with cost envelopes.
- `score_candidates(model: nn.Module, state: Tensor, target: np.ndarray, bank: np.ndarray, *, query_step_ns: int) -> dict` returns predicted costs, sampled feasibility, statuses, selected index and work counters.
- `measure_decisions(store: CalibrationStore, fit_receipt: Receipt, evaluation_receipt: Receipt, truth_receipt: Receipt) -> Receipt` records regret, false feasibility and explicit selection/bank failures.

- [ ] Test candidate arrays are byte-identical across model/session-fit/budget identities, contain the two references and respect bounds. Test predicted cost agrees with the original fixed-anchor arithmetic; 5 ms feasibility checks do not add extra running-cost charges. For each main partition 5/10/20/40 ms, assert auxiliary 5 ms anchor queries start at that main step's initial state and only its full-step endpoint advances the rollout. Count every auxiliary scalar prediction and batch call; a coarse path must not silently become sequential 5 ms propagation.
- [ ] With independent truth costs `[3,1,2]`, predicted selection index 0 and all candidates feasible, assert regret equals 2. If selected index 0 is truly infeasible, assert a decision failure with no finite regret. Assert selected tie goes to the lowest index; numerical truth ties remain unresolved within their envelope.
- [ ] Test false-infeasible versus false-feasible counts, prediction exceptions/nonfinite output, all-predicted-infeasible, no-true-feasible-bank, initial-state violations, and a candidate whose state violates the envelope only between 40 ms cost anchors. Preserve all 256 candidates and all 64 decision states in every outcome.
- [ ] Run `.venv/bin/python -m pytest -q tests/test_calibration_decisions.py`; observe RED.
- [ ] Implement a pure study scorer and independent truth replay without changing the old CEM/controller. Evaluate the spec's sampled constraints and fixed cost on their separate clocks, packing auxiliary queries per main step and reusing its endpoint. Never use model scores as the oracle. Numerical cost separations required for ranking must exceed the independently measured envelope, with refinement otherwise.
- [ ] Run those tests plus `tests/test_robot_calibration.py tests/test_stateless_mpc.py`; expect pass. Commit: `feat: compare calibration models by realized decision regret`.

## Task 7: Measure useful end-to-end computation

**Files:** Create `poc/calibration_timing.py`, `tests/test_calibration_timing.py`; extend CLI `time`.

**Interfaces:**

- `measure_timing(store: CalibrationStore, fit_receipts: list[Receipt], evaluation_receipt: Receipt) -> Receipt` records rotated network and scorer measurements.
- `summarize_timing(samples: list[dict]) -> dict` derives median/p95, work counters, batching and critical/noncritical resource categories.

- [ ] Use an injected clock to assert 20 warm-up calls are excluded and 50 repetitions per cell included; rotate method order and keep thread count, dtype, inputs and query partition fixed. Check every batch 1/128/256 and requested model/query cell appears exactly once in the timing coordinate registry. Whole-scorer timing includes auxiliary anchor construction/query batches for 5 ms feasibility at every main partition.
- [ ] Test predicted feasibility/selection overhead stays inside complete scorer timing, while write/hash/audit time has its own recorded category. A fast network plus slow scorer must report the slow scorer. Cache invalidation includes interval, tau, dtype and device; detached precomputation cannot silently replace differentiable h.
- [ ] Run `.venv/bin/python -m pytest -q tests/test_calibration_timing.py`; observe RED.
- [ ] Implement rotated single-threaded inference-mode measurements, detached fixed coefficient caches where applicable and separate whole-bank timings. Disclose processor/platform, observed synchronization/timer granularity and RSS. No elapsed-time stopping or claim that these measurements implement a feedback deadline.
- [ ] Run timing and decision tests; expect pass. Commit: `perf: measure complete calibration decision-scoring costs`.

## Task 8: Aggregate complete evidence and make the gate reviewable

**Files:** Create `poc/calibration_report.py`, `tests/test_calibration_report.py`, `tests/test_calibration_integration.py`; extend CLI `report`, `verify`, full `status`. Future generated outputs: `reports/short_calibration_findings.md`, `reports/short_calibration_result.json`, `reports/short_calibration_metrics.csv`, `reports/short_calibration_learning_curves.svg`, `reports/short_calibration_reproduce.md`. Update `docs/research-trajectory.md` only when actual results exist.

**Interfaces:**

- `expected_coordinates(matrix: str) -> list[dict]` defines the prospective matrix independently of available files.
- `aggregate(store: CalibrationStore, matrix: str) -> dict` validates coverage and recomputes physical metrics, screens and uncertainty from raw records.
- `gate_decision(result: dict) -> dict` returns `proceed_to_deadline_spec`, `representation_only`, `cheap_solver_explains_gain`, `no_practical_headroom`, `unresolved`, or `no_advantage`, with machine-readable reasons; it does not launch Stage C.
- `write_report(result: dict, destination: Path) -> list[Path]` renders Markdown/JSON/CSV/SVG and includes exact portable commands and external-artifact boundaries.
- `verify(store: CalibrationStore, matrix: str) -> dict` recomputes receipts/raw summaries and returns explicit missing/failed/invalid coverage. Failed performance may be a complete valid experiment; missing artifacts are incomplete.

- [ ] Test exact pilot `18 neural + 3 physical` and development `180 neural + 15 physical` fit coverage; reject duplicate/missing/tampered outcomes. Failed and optimization-limited cells count toward coverage and remain visible, with undefined metric rows marked rather than imputed.
- [ ] Test physical thresholds `.01/.10` RMSE and `.03/.30` p95 at 160 ms, zero prediction failures, <=1% decision-failure screen and <=1% regret/zero-command-cost screen. Require the same model/query-step configuration for prediction, decisions and timing. Zero reference cost <=`1e-8` and no feasible true bank yield unresolved screens. Assert a never-passing or nonmonotone curve retains every budget and interval-censored threshold brackets.
- [ ] Test session clustering: average initialization replicas within session, resample all paired model/budget/task rows together using 10,000 fixed-seed bootstrap draws; increasing replica/task count cannot increase the number of independent sessions. Pilot uncertainty is descriptive only and contains no powered claim.
- [ ] Test a fitted-physics win, a matching f/B RK4 win, an actuator-aware NODE win, and slow scoring each produce the appropriate scoped conclusion. No prose or figure may substitute a held-response improvement for improved decisions or call simulator motion seconds hardware commissioning time.
- [ ] Run `.venv/bin/python -m pytest -q tests/test_calibration_report.py tests/test_calibration_integration.py`; observe RED.
- [ ] Implement raw-derived coverage/statistics/screens and SVG generation with existing/standard libraries; use explicit physical-unit axes and show all baselines/failures. Add a tiny synthetic end-to-end fixture running prepare/collect/fit/predict/decisions/time/report/verify with no private files. Keep reduced fixture recipes labeled `unit` and ineligible for empirical claims.
- [ ] Run all new tests and `.venv/bin/python -m pytest -q`; expect all pass. Commit: `feat: report calibration tradeoffs and continuation gates`.

## Task 9: Run the bounded pilot, then the development study

**Files:** Generated mutable artifacts in the explicit output root; tracked summary artifacts from Task 8, a portable exported `config/short_calibration_protocol.json` containing the frozen recipe/namespaces without local paths, and `docs/research-trajectory.md` when complete. Preserve this plan's unchecked boxes until actual execution.

**Interfaces:** The full CLI from Tasks 3–8 and the verified `Receipt` contracts. All commands below are future execution instructions, not commands already run while writing this plan.

- [ ] Run the focused solver/data/training/decision tests and then the full suite before expensive work. Verify pinned environment with `.venv/bin/python -m pip check`; expect no broken requirements. Record the source revision and spec/recipe hashes.
- [ ] Prepare and collect pilot inputs:

```bash
.venv/bin/python -m poc.calibration_study prepare --matrix pilot --output-root study/short-calibration
.venv/bin/python -m poc.calibration_study collect --matrix pilot --output-root study/short-calibration
```

Expected: one complete 600-second session, nested budget receipts and a prospective 21-fit matrix, without historical private files.

- [ ] Run `train`, then `predict`, `decisions`, `time`, `report`, and `verify`, each with `--matrix pilot --output-root study/short-calibration`. Independent producers can be developed before training completes, but measurement awaits verified selected fits. Expect all 21 fit cells, complete scoring/truth coverage and an explicit pilot conclusion; unsuccessful fits remain outcomes.
- [ ] Export and commit the pilot report with an honest resource/throughput projection for completing five sessions and two initializations. Diagnose software/numerical failures before expansion using systematic-debugging; a scientific null is not a software bug. If the gate is already decisively null, complete the declared A/B development matrix or record a prospective scope amendment and call the shortened study a pilot, never a complete development study.
- [ ] Prepare `--matrix development` only after the pilot closes and the new matrix request binds the unchanged recipe. Run collect/train/predict/decisions/time/report/verify with that matrix. Expect 195 fit cells total, reusing the pilot's unchanged fits by hash; do not repeat the pilot as a new independent session or overwrite its report.
- [ ] Fresh-process reproduce one fit for `affine_rk4`, `node_actuator_aware` and `physics_residual`, with pilot session 0/budget 30/initialization 0, then first/last prediction/decision coordinates. Compare canonical selected tensors and numerical outputs; execute designated three-resolution truth checks. Record resources for the replicas separately from primary commissioning estimates.
- [ ] Run `verify --matrix development`, the full pytest suite and `git diff --check`; expect complete hash-bound coverage, all tests passing and no whitespace errors. Verify all historical source/results remain unchanged. Export final summaries and commit: `research: report short-calibration AffineFlow study and gate decision`.
- [ ] Finish with the actual A/B outcome and limits. If the gate supports C, write its new spec before implementing the scheduler/controller. If representation alone survives, name that contribution; if physics or the new cheap NODE removes the proposed advantage, preserve that result and stop this plan.

## Traceability and deferred decisions

| Spec requirement | Owning tasks |
|---|---|
| Stable analytic coefficients, work counters and same-weight propagation controls | 1, 5 |
| All seven baselines plus unfitted nominal reference | 2, 4, 6, 8 |
| Independent sessions, validation included, train-only scales and public-checkout execution | 3, 4, 9 |
| Own-solver fitting, bounded opportunity, failures and deterministic resume | 4, 9 |
| Natural motion, command-path clocks and numerical truth qualification | 5, 6, 9 |
| Independent finite-bank regret and explicit sampled-feasibility failures | 6, 8 |
| Whole-scorer versus isolated-network timing | 7, 8 |
| Complete curves, session uncertainty, provisional thresholds and honest gate | 8, 9 |
| Fresh-process replication, raw integrity and portable reproduction | 3, 8, 9 |

Stage C's deadline scheduler, stale-output/fallback policy, shared compiled backend and 20 ms/400 ms task freeze are deliberately deferred until A/B provides measured headroom. Stage D's n-joint/general-actuator interfaces, hardware telemetry, independently qualified limits and task-derived commissioning acceptance require a selected platform. These are specified research milestones in the design, not unassigned implementation tasks in this plan.

The present documentation request ends when the spec and plan are reviewed, checked and committed. It does not run Task 9 or change the README's empirical headline.

### Pre-empirical consolidated repair contract (2026-10-12)

Before Task 9 freeze, use the spec's evidence-qualification/storage amendment: complete competitor frontier gates, bounded sampled-feasibility qualification, recoverable attempt resource spans, and fixed lossless level-1 gzip for prediction/candidate `rows.jsonl.gz`. Task 9 uses `resources.nonoverlapping_producers` for totals; `resources.categories[].producer_attempts` retains inclusive stage spans, and neither these nor nested rate spans may be added to that total. Producer attempts carry parent-job/attempt provenance, including nested truth refinements. Keep `accounting_status` in pilot projections; `lower_bound_unknown_tail` is incomplete execution accounting. Decode compressed rows through `poc.calibration_records.read_records` or standard `gzip.open`; no compression option, dependency, numerical tensor change or timing-envelope change is introduced. Report actual compressed pilot bytes and costs separately from unit bookkeeping estimates.
