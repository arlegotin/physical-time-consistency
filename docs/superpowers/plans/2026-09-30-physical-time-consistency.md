# Physical-time consistency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a reproducible PTCO-1.0 decision about the discrete power head's action sensitivity and control benefit on the specified two-link robot.

**Architecture:** An explicit milestone CLI coordinates focused physics, data, model, measurement and controller modules. Immutable runs and hash-closed decisions enforce the evidence cascade; standard-library reporting remains usable when scientific dependencies or resources are unavailable. No general experiment framework is introduced.

**Tech Stack:** CPython 3.11; CPU PyTorch 2.8.0, NumPy 2.2.6, SciPy 1.15.3, pytest 8.4.1, MuJoCo 3.14.0; standard-library orchestration; Git.

**Spec:** [2026-09-30-physical-time-consistency-design.md](../specs/2026-09-30-physical-time-consistency-design.md). The supplied S00–S15 protocol remains in the private archive and earlier Git revisions; the derived design has its own identity.

## Global Constraints

- Protocol `PTCO-1.0`; primary method `power_correct`, structured NODE remains a comparator.
- CPU only; float32 training, float64 audit; CPython 3.11; exact dependency pins above.
- `h0=0.04 s`, `tau=0.08 s`, initial physics step `0.0005 s`, sensing/replanning `0.04 s`, history `0.32 s`.
- Primary model step `0.005 s`; sensitivity primary horizons `0.005/0.01 s`; MPC horizon `0.24 s`; episode `1.6 s`; torque bounds `[-2,2] N·m`.
- Initial confirmation seeds `0,1,2,3,4`, twenty paired episodes per seed; development seed `9000` is never confirmation.
- Seven conventional competitors in spec order; statistical rejection family `K=280`, shared alpha ledger including negative shortcuts.
- Neural AdamW lr `0.001`, weight_decay `0.000001`, clip `1.0`, effective batch `256`, initial `100` epochs, rollout weight `0.25`, rollout steps `4`, bootstrap resamples `20000`.
- Every task inherits the spec's scientific, data-access, provenance, numerical, resource and resumption contracts. Test success alone never authorizes the next scientific milestone.
- No implementation/installation/training is performed while preparing this plan. Start execution only after the planning handoff; record actual resource authority before package acquisition or expensive runs.
- Use isolated worktree handling at execution time if needed. Preserve existing author identity; synthetic local `PoC Executor <poc-executor@example.invalid>` is only for absent settings. Stage source/config/test paths explicitly; exclude data, wheels, venv, checkpoints and active locks. Never push.

## Review Focus

- An interrupted write must not leave a valid-looking completed run; pin marker ordering and recovery in Tasks 1 and 26.
- A query at a history-window boundary must use only fully completed eligible tokens; pin exact times and array ownership in Task 8.
- Mixed zero/nonzero intervals within a batch must preserve each zero row and its context; pin masks in Tasks 4 and 9.
- Partial candidate failure and deterministic equal-cost ties must keep candidate IDs and complete episode costs; pin in Tasks 17 and 18.
- A subset negative-confirmation run must share the full rejection family and spending ledger; pin MR/control-negative in Tasks 14 and 19.

## Execution order and file boundaries

The units are dependent parts of one experiment, not independent products. Do not implement later scientific stages merely because their tasks are listed here. Complete M00/G00 before M10; M10/G10 before robot work; M20/G20 before data/training; obey the M30/MR and M40/control-negative branches before full-suite work. Task 26's minimal locks/journals are introduced by Task 1; its later work expands tested recovery and MX dispatch.

| Tasks | Milestone | Files with a single responsibility |
|---|---|---|
| 1–2 | M00 | io: atomic/hash operations; contracts: schemas/registries; gates: decision transitions; bootstrap: actual capabilities/locks; main: lazy dispatch; report: early reporting |
| 3–4 | M10 | analytic: exact maps; clocks: physical schedules; common: prediction types; power: intervention wrapper |
| 5–6 | M20 | envs: plant/reset/snapshot; oracle_rigidbody: independent equations/reference qualification |
| 7–14 | M30/MR | generate/dataset: provenance/access; history: query-safe context; model files: declared predictors; train: shared loss/selection; metrics/statistics: estimands/bounds; sensitivity/prediction: E01/E02 |
| 15–19 | M40 | physics_fit: learned references; action_path: segments; mpc: search; control: physical episodes |
| 20–22 | M50 | attention/ssm: remaining families; contracts: completed matrix/freeze closure |
| 23–25 | M60/M70/M80 | Existing measurement/control handlers with complete frozen cells |
| 26 | MX | continuation: one registered diagnostic request, no agent loop |
| 27 | M90 | replicate: fresh-process replay/retrain and independent summaries |
| 28 | MF | report: terminal precedence, honest files and reproduction script |

JSON-facing types `JobSpec`, `RunRecord`, `DecisionRecord`, `CriterionRecord`, `MXRequest` are stdlib TypedDict definitions in `poc/contracts.py`, with exact fields from spec §§11–14. No task introduces an undeclared external orchestration dependency. Tests may add small deterministic fixtures to their owning module; no full robot-training matrix belongs in pytest.

## Task 1: Standard-library evidence, CLI and early-report foundation

**Files:** Create `README.md`, `.gitignore`, `requirements.in`, `config/base.json`, `config/authorization.json`, `config/milestones.json`, `config/criteria.json`, `schemas/manifest.schema.json`, `schemas/decision.schema.json`, `poc/__init__.py`, `poc/__main__.py`, `poc/bootstrap.py`, `poc/io.py`, `poc/contracts.py`, `poc/gates.py`, `poc/report.py`, `tests/test_io_gates.py`. The supplied protocol is historical material, not a required working-tree file.

**Interfaces:** `io.read_json(path: Path)->dict`, `atomic_json(path: Path,value: dict)->None`, `sha256_file(path: Path)->str`, `job_key(inputs: dict)->str`, `job_lock(job_key: str,root: Path)->ContextManager`, `complete_run(root: Path,record: RunRecord)->None`; `contracts.validate_record(kind: str,value: dict,root: Path)->None`; `gates.decide(gate_id: str,evidence: list[RunRecord],root: Path)->DecisionRecord`; `report.write_report(root: Path,decisions: list[DecisionRecord])->dict`; CLI `main(argv: list[str]|None=None)->int`.

- [ ] **1. Write failing stdlib unittest cases** `test_nonfinite_json_rejected`, `test_missing_evidence_never_go`, `test_completion_marker_written_last`, `test_cli_help_without_scientific_imports`, `test_job_key_ignores_run_directory`.

```python
with self.assertRaises(ValueError):
    read_json(self.write_text('bad.json', '{"metric":NaN}'))
self.assertEqual(job_key({"h_ns":40_000_000}),
                 job_key({"h_ns":40_000_000}))
self.assertNotEqual(decide('G00', [], self.root)['outcome'], 'GO')
```

- [ ] **2. Run red:** `python3.11 -m unittest discover -s tests -p test_io_gates.py -v`; the new missing contracts fail, rather than requiring scientific imports.
- [ ] **3. Implement the contracts and exact configs.** Use strict JSON and same-filesystem atomic renames; semantic keys exclude timestamp/output-directory fields. Implement exclusive job locks, journals and COMPLETED-last behavior now. Dispatch only declared IDs, reject unimplemented handlers, and keep MF/bootstrap paths stdlib. Record only actual authorization, leaving execution prohibited until required fields are valid. requirements.in contains exactly the five pins. Base configuration matches the supplied protocol, including complete horizon/partition lists.
- [ ] **4. Run green:** the unittest command passes; `python3.11 -S -m poc --help` and `python3.11 -S -m poc run MF --help` exit zero without importing NumPy/Torch/MuJoCo. Empty scientific evidence is not reported as a verdict. Exercise all five gate outcomes with synthetic records and reject the supplied null-provenance schema example as real evidence.
- [ ] **5. Commit:** explicitly stage only this task's source, schemas, configs, tests and archived protocol; message `Initialize PTCO protocol and evidence contracts`.

## Task 2: Actual environment, complete wheel lock and G00

**Files:** Modify `poc/bootstrap.py`, `poc/gates.py`, `poc/report.py`, `README.md`, `tests/test_io_gates.py`; generate `locks/requirements.resolved.txt`, `locks/requirements.lock`, `locks/environment.json`, wheel/license/hash manifests and M00 evidence.

**Interfaces:** `bootstrap.run_m00(root: Path,authorization: dict)->RunRecord`; `bootstrap.lock_wheels(wheel_dir: Path,resolved: list[tuple[str,str]])->dict`; `bootstrap.verify_offline_lock(root: Path)->dict`. Consume Task 1's run writer/validators; scientific imports occur only after environment/thread setup.

- [ ] **1. Write failing tests** for mismatched wheel METADATA, conflicting versions, omitted installer distribution, incompatible platform, unauthorized package access and failed offline recreation. A corrupt wheel or unknown capability must make G00 non-GO.
- [ ] **2. Run red:** `python3.11 -m unittest discover -s tests -p test_io_gates.py -v` fails on missing lock/preflight behavior.
- [ ] **3. Implement actual preflight.** Check Python/venv/architecture/disk/authority, install declared binary pins using the platform route, save pip output, freeze/download every distribution, generate METADATA-derived SHA locks and retain license notices. Run empty-world nstep, deterministic Torch/AdamW/serialization/timer checks and a fresh temporary offline venv. Missing Python uses the spec's provisional blocker route; missing compatible wheels stops without substitution.
- [ ] **4. Verify:** `python -m poc run M00 --dry-run`, `python -m poc run M00`, `python -m pytest -q tests/test_io_gates.py`, `python -m poc decide G00`, `python -m poc status --verify`. Proceed only if the actual G00 decision is GO and hash-locked offline recreation passed; otherwise produce MF and preserve exact blocker/diagnosis.
- [ ] **5. Commit:** source/config updates and textual lock/manifests only; `Add reproducible CPU preflight and offline lock`.

## Task 3: Exact analytic mechanics and integer physical schedules

**Files:** Create `poc/analytic.py`, `poc/clocks.py`, `assets/analytic_direct.xml`, `assets/analytic_lagged.xml`, `tests/test_analytic.py`, `tests/test_clocks.py`.

**Interfaces:** `analytic.coefficients(h_s: float,tau_s: float=0.08)->tuple[float,float,float]` returns Kw, Kv, Kq; `analytic.transition(state: np.ndarray,command: np.ndarray,h_s: float,actuator: str)->np.ndarray`; `analytic.action_jacobian(h_s: float,actuator: str,n_joints: int=2)->np.ndarray`; `clocks.to_ns(seconds: float)->int`; `clocks.partition(duration_ns: int,step_ns: int)->tuple[int,...]`; `clocks.jitter_partition(seed_entropy: tuple[int,...])->tuple[int,...]`.

- [ ] **1. Write failing exact/Decimal/composition tests**, covering 1,000 seed-8001 cases, all declared horizons, zero/tiny intervals, both signs/corners and nonzero w. Independent Decimal evaluation must verify stable series, not call the implementation under test.

```python
np.testing.assert_allclose(transition(np.array([1.,2.]),np.array([3.]),.04,'direct'),
                           [1.0824,2.12],atol=1e-12,rtol=1e-12)
assert sum(jitter_partition((1201,0,0,4))) == 40_000_000
assert partition(40_000_000,5_000_000) == (5_000_000,)*8
```

- [ ] **2. Run red:** `python -m pytest -q tests/test_analytic.py tests/test_clocks.py` fails for absent implementations.
- [ ] **3. Implement §3.3 formulas**, expm1/series recurrence with relative next-term `1e-16`, strict interval/shape validation, uniform/jitter clocks and analytic MJCF fixtures. No training data or robot models.
- [ ] **4. Run green:** same pytest command; coefficients/autograd exact checks satisfy `1e-10` relative and Decimal/state checks `1e-12`. Tiny finite differences are explicitly unresolved when subtraction uncertainty prevents the `1e-6` check.
- [ ] **5. Commit:** task files only; `Add exact direct and lagged dynamics fixtures`.

## Task 4: Shared prediction contract and same-weights power intervention

**Files:** Create `poc/models/__init__.py`, `poc/models/common.py`, `poc/models/power.py`, `tests/test_power.py`; modify `poc/bootstrap.py`, `poc/gates.py` to register M10 execution.

**Interfaces:** `Prediction(next_state: Tensor,status: tuple[str,...],network_calls: int,vector_field_calls: int,scalar_candidate_evaluations: int,history_tokens_processed: int)` and `Scales(q: Tensor,v: Tensor,v_input: Tensor)` in common; `PowerHead(network: nn.Module,actuator: str,scales: Scales)` implements the spec's exact `predict` signature. `bootstrap.run_m10(root: Path)->RunRecord` collects complete deterministic evidence.

- [ ] **1. Write failing identity/mutation tests** using a nonzero command-dependent affine network and unit scales: float64 outputs, state/command Jacobians, weight gradients, four-step loss; all three modes at h0; wrong powers/zero gain/w=u/wrong sign; mixed h=0/.04 batch.

```python
for field in ('output','state_jac','command_jac','weight_grad','four_step_loss'):
    np.testing.assert_allclose(identity_fixture['low'][field],
                               identity_fixture['lagged'][field],atol=1e-12,rtol=1e-12)
assert prediction.network_calls == 2
```

- [ ] **2. Run red:** `python -m pytest -q tests/test_power.py`.
- [ ] **3. Implement two-pass §3.2**, mode validation, scales, exact activation update and no-op row masks. Provide analytic Jacobian/unit mutation classifications and slide-fixture refinement evidence without full architecture scaffolding.
- [ ] **4. Verify:** `python -m pytest -q tests/test_analytic.py tests/test_power.py tests/test_clocks.py`; `python -m poc run M10`; `python -m poc decide G10`. G10 must cite arrays, gradient/composition/mutation/timing/fixture hashes, not pytest status alone.
- [ ] **5. Commit:** task files; `Verify nominal power-head identity and timing invariants`.

## Task 5: Exact robot assets, plant adapter and isolated snapshots

**Files:** Create `assets/two_link_direct.xml`, `assets/two_link_lagged.xml`, `poc/envs.py`, `tests/test_env.py`.

**Interfaces:** `Plant(actuator: str,physics_step_ns: int=500_000,asset_root: Path=Path('assets'))` with exactly spec §5.2 methods; `compiled_manifest()->dict`. Consume clocks and strict evidence serializers.

- [ ] **1. Write failing compiled/reset/snapshot tests:** nq/nv/nu2, na0/2, zero gravity/contact/limits, exact armature/damping/transmission and filter tau; branch ±epsilon order invariance; copied state storage; complete integration-state replay and warning detection.

```python
snapshot = plant.snapshot()
first = plant.advance(np.array([.2,-.3]),10_000_000)
plant.restore(snapshot)
second = plant.advance(np.array([.2,-.3]),10_000_000)
np.testing.assert_array_equal(first['state'],second['state'])
```

- [ ] **2. Run red:** `python -m pytest -q tests/test_env.py`.
- [ ] **3. Implement exact XML and adapter** from spec, using fresh data/verified restoration, nstep only for constant segments, explicit qfrc_applied, strict bounds/ticks and integer time. Preserve compiled capsule inertia, do not replace it with rods.
- [ ] **4. Run green:** same tests; verify nonzero initial activation/command reversal, tick rejection and floating/int time ≤1e-10. Produce diagnostic plant records but no training labels.
- [ ] **5. Commit:** task files; `Implement smooth two-link direct and filtered plants`.

## Task 6: Independent robot oracle, numerical calibration and G20

**Files:** Create `poc/oracle_rigidbody.py`, `tests/test_oracles.py`, and `poc/sensitivity.py` with its reference-calibration helper only; modify `poc/gates.py`, `poc/__main__.py`; produce `config/numerical.json`.

**Interfaces:** `oracle_rigidbody.parameters_from_compiled(manifest: dict)->dict`; `rhs(t_s: float,qv: np.ndarray,command: np.ndarray,initial_w: np.ndarray|None,parameters: dict)->np.ndarray`; `integrate(state: np.ndarray,command: np.ndarray,h_s: float,parameters: dict,actuator: str,rtol: float=1e-11,atol: float=1e-13,max_step_s: float=.00025)->dict`; `sensitivity.qualify_reference(root: Path)->RunRecord` is M20's handler.

- [ ] **1. Write failing independent fixtures** for inertia-frame rotation, SPD, Coriolis sign, damping/armature, unit-slide exact response, activation and DOP853 status. Assert no MuJoCo calls within the oracle RHS; numerical qualification cannot accept a stable unresolved zero derivative.
- [ ] **2. Run red:** `python -m pytest -q tests/test_env.py tests/test_oracles.py`.
- [ ] **3. Implement equations and reference-only calibration.** At least64 cold and64 warm probes/actuator; sweep epsilon .01/.02/.04 and physics .5/.25 ms, refine as needed; independent tightened integration; choose largest qualifying slope window before model audit. Persist every ±endpoint/solver status/envelope and qualification manifest.
- [ ] **4. Verify:** focused tests; `python -m poc run M20 --dry-run`, `python -m poc run M20`, `python -m poc decide G20`. Require spec §9 budgets and resolved signal on both plants before generating corpora.
- [ ] **5. Commit:** source/tests and qualified textual settings; `Qualify independent robot dynamics and derivative reference`.

## Task 7: Deterministic corpora, train-only scales and role access

**Files:** Create `poc/generate.py`, `poc/dataset.py`, `schemas/dataset.schema.json`, `tests/test_data.py`.

**Interfaces:** `generate.generate_corpus(root: Path,actuator: str,condition: str,role: str,n_trajectories: int)->RunRecord`; `dataset.load_corpus(manifest_path: Path,phase: str,role: str,purpose: str)->dict[str,np.ndarray]`; `dataset.fit_normalization(train_manifest: Path)->dict`. Consume qualified numerical decision and Plant; no access without explicit phase/role/purpose.

- [ ] **1. Write failing tests** for PCG64 entropy/repeatability, whole-trajectory separation, dtypes/time sums, tampered/truncated arrays, role/dense-label rejection and normalization invariance to validation changes.

```python
with pytest.raises(ValueError):
    load_corpus(confirmation_manifest,phase='confirmatory',role='confirmation',purpose='training')
assert arrays['state'].shape == (16,33,6)  # lagged tiny fixture
np.testing.assert_array_equal(arrays['time_ns'][:,1:],
                              np.cumsum(arrays['duration_ns'],axis=1))
```

- [ ] **2. Run red:** `python -m pytest -q tests/test_data.py`.
- [ ] **3. Implement spec §6** seeds/ranges/32-command arrays, immutable manifest/access events, train-only q/v/input scales and full generation-resume trajectory boundaries. Freeze before confirmation generation. A confirmatory-phase job with training/selection purpose still cannot load confirmation or dense oracle labels; reject undeclared phase/role/purpose combinations.
- [ ] **4. Run green:** same tests and tiny F/A corpus replay. Full train/validation/development generation is part of M30 after resource projection; never generate confirmation here.
- [ ] **5. Commit:** task files; `Add deterministic datasets and audited label access`.

## Task 8: Immutable physical-time history and candidate isolation

**Files:** Create `poc/history.py`, `tests/test_history.py`; modify `poc/models/common.py` to import HistoryBatch.

**Interfaces:** `HistoryToken(start_ns: int,state: np.ndarray,mean_command: np.ndarray,duration_ns: int,provenance: str,packet_id: str)`; `HistoryBatch(tokens: tuple[tuple[HistoryToken,...],...],query_time_ns: np.ndarray)`; `HistoryBatch.eligible()->HistoryBatch`; `append_completed(token: HistoryToken)->HistoryBatch`; `fork()->HistoryBatch`. No mutating query method.

- [ ] **1. Write failing exact-boundary tests**: token start exactly query−320ms included if completed; old/incomplete token excluded; zero query no commit; F/A varying token counts; high-rate mean command; independent candidates and masked padding.

```python
assert history_fingerprint(before) == history_fingerprint(after_repeated_queries)
assert outputs_in_original_order == outputs_after_candidate_permutation
assert not np.shares_memory(branch_a.tokens[0][0].state,branch_b.tokens[0][0].state)
```

Test helpers `history_fingerprint` and permutation restoration live in this test module and use canonical immutable token values.
- [ ] **2. Run red:** `python -m pytest -q tests/test_history.py`.
- [ ] **3. Implement actual-time filtering and branch copies.** F/A training appends actual dataset anchors; E02/MPC appends at common40ms only; observed provenance remains separate from imagined. Copy/freeze arrays on ingestion, recompute time ages and never crop partial tokens.
- [ ] **4. Run green:** same tests, including a320ms window containing16 augmented tokens and candidate query counts that differ across partitions while real packet history stays identical.
- [ ] **5. Commit:** task files; `Make physical-time histories read-only across candidate queries`.

## Task 9: Core conventional, power and NODE predictors

**Files:** Create `poc/models/mlp.py`, `poc/models/sequence.py`, `poc/models/ode.py`, `tests/test_models.py`; modify `common.py`, `power.py`.

**Interfaces:** `build_model(model_id: str,actuator: str,scales: Scales,seed: int)->nn.Module` in models/__init__; every result implements `predict` from spec §7. Include mlp_dt, mlp_h, GRU, LSTM, generic/structured NODE and real power MLP. LSTM implementation may wait until M50; its registry entry must already declare the exact recipe. `common.features(state,command,h_s,scales)->Tensor` uses specified ordering.

- [ ] **1. Write failing shapes/counts/masks/gradient tests**, physical u/2 derivatives, mixed zero/nonzero intervals, batched/scalar parity and stage activation. Assert direct MLP18308, GRU18820, genericNODE18180, structured17922, laggedMLP18564; compute every other count rather than padding parameters.

```python
assert sum(p.numel() for p in mlp.parameters()) == 18_308
torch.testing.assert_close(prediction.next_state[zero_row],state[zero_row],rtol=0,atol=0)
assert structured_lagged_field_command_derivative_at_fixed_w == 0
assert rk4_prediction.vector_field_calls == 4
```

- [ ] **2. Run red:** `python -m pytest -q tests/test_models.py tests/test_power.py tests/test_history.py`.
- [ ] **3. Implement exact tanh architectures/decoders and explicit classical RK4**, no h in fields, no command shortcut in lagged structured acceleration, exact stage/endpoint w, bounded finite histories. Network outputs update unwrapped q. Counters count every power pass, field call and processed token.
- [ ] **4. Run green:** same tests plus independent exact-field RK4 and two-half-step comparisons; masked-token changes must not affect outputs.
- [ ] **5. Commit:** model/test files; `Implement faithful core time-aware dynamics predictors`.

## Task 10: Shared endpoint/rollout training and nominal selection

**Files:** Create `poc/train.py`, `tests/test_training.py`; modify dataset/history/common interfaces only where required.

**Interfaces:** `train.endpoint_loss(predicted: Tensor,target: Tensor,scales: Scales)->Tensor`; `rollout_loss(model: nn.Module,window: dict,history: HistoryBatch)->tuple[Tensor,int]`; `fit(job: JobSpec,root: Path,resume: bool=False)->RunRecord`; `qualify_nominal(training_run: RunRecord)->dict`; `run_learning_controls(model_id: str,actuator: str,root: Path)->RunRecord`.

- [ ] **1. Write failing saved-batch loss/gradient/update tests**, one/four weighting, actual-duration branch histories, last-three-start exclusion only from four-step, masks/effective batches, checkpoint/RNG resume and checkpoint tie selection.

```python
assert supervised_one_step_targets == 32
assert valid_four_step_starts == 29
torch.testing.assert_close(total_loss,one_loss+.25*four_loss)
assert selected_epoch_for_relative_tie == earlier_epoch
assert continuous_valid_target_count == resumed_valid_target_count
```

- [ ] **2. Run red:** `python -m pytest -q tests/test_training.py`.
- [ ] **3. Implement the exact AdamW recipe and accounting**, 32×8 effective windows shared across methods, grad-clip/nonfinite handling, validation every5 epochs, best/last checkpoints, canonical tensor and archive hashes, weights_only loading and JSON RNG state. Implement analytic controls and nominal plateau/overfit checks with the fixed lr grid and moving100-epoch windows; selection reads only declared nominal data.
- [ ] **4. Run green:** focused tests and all core learning controls. Independently verify power-mode gradients before/after the same optimizer update. Distinguish a known RK4 floor from a failed model fit; no robot superiority claim from the tiny controls.
- [ ] **5. Commit:** task files; `Add matched supervised training and resumable nominal selection`.

## Task 11: Jacobian metrics, finite-time slopes and mutation checks

**Files:** Create `poc/metrics.py`, `schemas/metrics.schema.json`, `tests/test_sensitivity.py`; extend `poc/sensitivity.py`.

**Interfaces:** `sensitivity.action_jacobians(model,state,command,h_s,history)->dict`; `reference_jacobians(plant: Plant,state: np.ndarray,command: np.ndarray,h_ns: int,epsilon_nm: float)->dict`; `metrics.jacobian_metrics(model_jac: np.ndarray,reference_jac: np.ndarray,envelope: dict)->dict`; `fit_slope(h_s: np.ndarray,norms: np.ndarray,norm_bounds: np.ndarray)->dict`.

- [ ] **1. Write failing exact-coefficient tests** for direct/lagged maps; zero/wrong-sign/swapped matrices; physical command units; frozen activation/history; unresolved denominator; zero-gain slope status; numerical intervals crossing thresholds. Assert a norm-matched −J fails full matrix error and zero action gain does not become a dropped probe.
- [ ] **2. Run red:** `python -m pytest -q tests/test_sensitivity.py`.
- [ ] **3. Implement float64 autograd and independent finite differences**, raw ±endpoints, signal checks, normalized error/gain/matrix alignment and signed slope interval propagation. Every unresolved/failed row keeps its identity/status. Add the specified mutations without training on their labels.
- [ ] **4. Run green:** same tests; analytic/autograd/reference comparisons verify derivative w.r.t N·m rather than normalized u, exact activation Jacobian and reference-window locking.
- [ ] **5. Commit:** task files; `Measure new-command Jacobians with numerical envelopes`.

## Task 12: Atomic registry, paired uncertainty and sequential decisions

**Files:** Create `poc/statistics.py`, `tests/test_statistics.py`; modify `poc/contracts.py`, `poc/gates.py`, `config/criteria.json`.

**Interfaces:** `contracts.expand_criteria()->list[CriterionRecord]`; `statistics.contrast_bounds(values: np.ndarray,alpha: float,bootstrap_indices: np.ndarray,numerical_envelope: dict,bounded_fraction: bool=False)->dict`; `fieller_improvement(baseline: np.ndarray,method: np.ndarray,alpha: float)->dict`; `plan_precision(summary: dict,pilot: dict,look: int)->dict`; `gates.evaluate_criterion(criterion: CriterionRecord,summary: dict,look: int)->dict`.

- [ ] **1. Write failing independent statistical fixtures**:exact threshold equality/strict positivity, matrix registry dimensions, normal t calculation, Fieller finite/unbounded intervals, rare-bootstrap tails, constant/boundary fractions, influence triggers, zero denominators and numerical expansion.

```python
assert len([c for c in expand_criteria() if c['statistical']]) == 280
assert first_look_alpha == .025
assert reject_alpha == .025/280
assert all_one_fraction_lower == pytest.approx(.025**(1/5))
assert not passes_gain_coverage_with_five_perfect_seed_fractions
```

- [ ] **2. Run red:** `python -m pytest -q tests/test_statistics.py tests/test_io_gates.py`.
- [ ] **3. Implement spec §§11–12 exactly**,112 relative/strict atoms and all remaining families,20,000 shared cluster indices, quantile linear/ddof1, rare-tail rule, Fieller precision, influence checks and ascending sample planning. Keep atomic bounds and overall intersection-union decision distinct. Ledger commits before aggregation and identical input is idempotent.
- [ ] **4. Run green:** same tests plus independently calculated small synthetic samples around every criterion boundary. A precise negative may support FAIL; an imprecise negative remains unresolved; missing/stale data stays INVALID/BLOCKED before arithmetic.
- [ ] **5. Commit:** task files; `Preregister 280 atomic contrasts and conservative paired gates`.

## Task 13: Complete E01 and common-anchor E02 runners

**Files:** Extend `poc/sensitivity.py`; create `poc/prediction.py`; extend sensitivity/history tests.

**Interfaces:** `sensitivity.run_e01(job: JobSpec,root: Path,resume: bool=False)->RunRecord`; `prediction.run_e02(job: JobSpec,root: Path,resume: bool=False)->RunRecord`; `metrics.summarize_probes(rows: dict,expected_cells: list[str])->dict`. Consume model factory, frozen histories, qualified reference and registry.

- [ ] **1. Write failing tiny end-to-end fixtures** for cold/warm selection,512 full-workload probe IDs, all h values, per-stratum medians, truth-versus-consistency and uniform/jitter schedules; insert a fresh substep observation and assert invalidity.
- [ ] **2. Run red:** `python -m pytest -q tests/test_sensitivity.py tests/test_history.py`.
- [ ] **3. Implement typed NPZ coordinates and complete expected-cell iteration.** Warm state is first anchor≥320ms with genuine preceding history. Predictions receive no dense truth, append history only at common anchors, and preserve failures. E01 float64 weights remain the same learned tensors; F modes share checkpoint/hash. Cache reference measurements only on full semantic identity.
- [ ] **4. Run green:** focused tests; tiny exact model rollouts remain accurate under every partition and mutated no-motion model exposes consistent-but-wrong behavior. Resume complete Jacobian cells without accepting a partial ±pair.
- [ ] **5. Commit:** task files; `Add complete sensitivity and common-anchor transfer experiments`.

## Task 14: M30 development cascade and MR rejection-only confirmation

**Files:** Modify `poc/__main__.py`, `poc/train.py`, `poc/sensitivity.py`, `poc/gates.py`, `poc/contracts.py`; create `config/matrix.json`; write `config/rejection_freeze.json` only upon MR entry. Tests extend data/training/statistics/io gates.

**Interfaces:** `train.run_m30(root: Path,resume: bool=False)->RunRecord`; `sensitivity.run_mr(root: Path,resume: bool=False)->RunRecord`. Both enumerate explicit jobs from the same registry; no hidden Cartesian cells or confirmatory seed substitution.

- [ ] **1. Write failing transition tests**:one development seed cannot reject a claim; weak G30 routes MR; MR uses K280 and retained look ledger; subset support cannot emit PASS; qualified core rejection may stop without attention/SSM/MPC.
- [ ] **2. Run red:** `python -m pytest -q tests/test_io_gates.py tests/test_data.py tests/test_statistics.py`.
- [ ] **3. Implement fixed core matrices, pre-audit nominal freeze and resource projection.** Generate four train/validation/development corpora, qualify fits, then open exploratory E01/E02. GO screen is ≥25% error reduction, both plants/5 and10ms, gain median[.5,2], correct F direction. Otherwise MR freezes nominal recipes and independent seeds0–4, then generates/seals confirmation probes; no development evidence relabeling.
- [ ] **4. Verify:** `python -m poc run M30 --dry-run`, focused tests, then M30/G30 only within allocation. If required, `python -m poc run MR --dry-run`, MR/GR. Scientific rejection requires actual conservative/precision-qualified evidence; unresolved results create a named MX request. Do not proceed to Task15 if valid rejection already establishes FAIL.
- [ ] **5. Commit:** source/config/tests and freeze manifests only; `Implement exploratory and rejection-only mechanism gates`.

## Task 15: Training-only ridge and calibrated mechanics references

**Entry:** G30 exploratory GO or GR core GO; no valid mandatory rejection.

**Files:** Create `poc/physics_fit.py`; extend `tests/test_models.py`, `tests/test_training.py`, `tests/test_oracles.py`.

**Interfaces:** `physics_fit.fit_ridge(train_manifest: Path)->dict`; `fit_physics(job: JobSpec,root: Path)->RunRecord`; `PhysicsFit(parameters: Tensor,actuator: str,scales: Scales)` and `AccelRidge(coefficients: np.ndarray,actuator: str,scales: Scales)` implement the shared predict API. Separate Torch dynamics take learned physical parameters, not the privileged oracle object.

- [ ] **1. Write failing fixtures**:SPD mass matrices across angles, correct inverse-softplus initialization, unregularized constant, exact lagged training mean effort and forbidden compiled-parameter/acceleration access.

```python
np.testing.assert_allclose(initial_physical_parameters,[.10,.02,.02,.05,.05])
assert np.linalg.eigvalsh(mass_matrices).min() > 0
assert physics_reference_partition_discrepancy < numerical_budget
```

- [ ] **2. Run red:** focused model/training/oracle tests.
- [ ] **3. Implement spec §7.3**, augmented least-squares ridge and five-parameter Adam fitting; batch256/lr.01/100epochs/validation5, training RK4≤5ms with .5% normalized fit convergence, evaluation≤1.25ms independent of query partition. Preserve nominal failures and diagnose rather than discard.
- [ ] **4. Run green:** focused tests, then A reference fits with logged actual compute/data access. Fit F references before M50 freeze. A deterministic fitted reference is not copied five times to fabricate independent seeds.
- [ ] **5. Commit:** task files; `Add calibrated physics and acceleration references`.

## Task 16: Common action compiler and physical cost

**Files:** Create `poc/action_path.py`, `tests/test_action_path.py`; begin `poc/control.py`, `tests/test_control.py` with scoring only.

**Interfaces:** `ActionSegment(start_ns: int,duration_ns: int,command: np.ndarray)`; `action_path.compile_knots(knots: np.ndarray,actuator_interval_ns: int,horizon_ns: int=240_000_000)->tuple[ActionSegment,...]`; `path_checksum(segments)->str`; `effort_integral(segments,start_ns: int,end_ns: int)->float`; `control.anchor_cost(states: np.ndarray,target: np.ndarray,segments,terminal_weight: float=5.0)->float`.

- [ ] **1. Write failing independent path/cost fixtures**:six knots/12 variables,6/12/24 holds, boundary endpoint interpolation, last-knot hold, exact command-square integration and common anchors. Constant-state cost stays identical under prediction partition changes; episode terminal term occurs once.

```python
assert [len(compile_knots(knots,h)) for h in (40_000_000,20_000_000,10_000_000)] == [6,12,24]
assert effort_integral(constant_path,0,240_000_000) == pytest.approx(.24*np.dot(u,u))
assert len(cost_anchor_times) == 6
```

- [ ] **2. Run red:** `python -m pytest -q tests/test_action_path.py tests/test_control.py`.
- [ ] **3. Implement integer paths**, strict knot bounds/shape, time-weighted completed means, segment subdivision without crossing boundaries, and six planning/40 realized endpoint costs from spec §10. Use the same compiler/checksum in all consumers.
- [ ] **4. Run green:** tests include intervals that must split at actuator/disturbance boundaries; reject invalid duration/knot shapes rather than silently round or clip state.
- [ ] **5. Commit:** task files; `Unify actuator paths and physical-time cost anchors`.

## Task 17: Exact CEM and independent analytic optimization oracle

**Files:** Create `poc/mpc.py`, `tests/test_mpc.py`; extend action_path/control tests.

**Interfaces:** `mpc.search(model: nn.Module,state: Tensor,target: Tensor,history: HistoryBatch,previous_knots: np.ndarray|None,standard_normals: np.ndarray,settings: dict,model_partition: tuple[int,...],actuator_interval_ns: int)->dict`; `analytic_box_optimum(state: np.ndarray,target: np.ndarray,actuator_interval_ns: int=40_000_000)->dict`; `qualify_cem(root: Path,settings: dict)->RunRecord`.

- [ ] **1. Write failing CEM/QP tests**:candidate0mean/1zero, fixed Z consumption, population std/floor, finite subset, one/none finite, global IDs, ties, retained best evaluated plan, warm-start shift, scalar/batch/context permutation and QP ×2 cost/optimality.

```python
assert equal_cost_winner_global_id == 0
assert singleton_elite_std.min() == .10
assert all_failed_result['status'] == 'all_candidates_failed'
assert all_failed_result['fallback'] == 'zero_command'
assert qp_projected_optimality <= 1e-6
```

- [ ] **2. Run red:** `python -m pytest -q tests/test_mpc.py tests/test_action_path.py`.
- [ ] **3. Implement spec §10 search with64×2/8 elites**, saved normal banks and exact tie rules, no unscored elite-mean execution. Independently assemble affine exact-integrator residuals for bounded lsq_linear, tol1e-12/max_iter1000, and validate by separate rollout. Qualify16tasks×5streams and implement fixed common-budget escalation, selecting first qualifying oracle-only setting.
- [ ] **4. Run green:** tests plus analytic qualification evidence (≥75% median, ≥50% each nonzero-headroom, spread<5pp). If budget fails, repair/evaluate next declared budget only after projection; do not calibrate with learned-model ranking.
- [ ] **5. Commit:** task files; `Implement deterministic CEM and independent planner calibration`.

## Task 18: Closed-loop episodes, failure accounting and event clocks

**Files:** Complete `poc/control.py`, `tests/test_control.py`; extend `poc/metrics.py`.

**Interfaces:** `control.run_episode(job: JobSpec,plant: Plant,model: nn.Module,episode_id: int,root: Path,resume: bool=False)->dict`; `run_e03(job: JobSpec,root: Path,resume: bool=False)->RunRecord`; `metrics.summarize_episodes(rows: list[dict],expected_cells: list[str])->dict`. Episode job contains seed, actuator, training/loss/model/checkpoint, condition, partition, actuator rate and all frozen hashes.

- [ ] **1. Write failing episode fixtures**:forty fresh decisions, t0/q/v/w, target seeds, one terminal cost, no extra observations, final200ms stepwise success, first40ms plan outcome, and complete logging of candidate/model failures.

```python
assert observation_times_ns == list(range(0,1_600_000_000,40_000_000))
assert simulated_elapsed_ns == 1_600_000_000
assert fallback_episode['complete'] and fallback_episode['fallback_count'] == 1
assert math.isfinite(fallback_episode['realized_cost'])
```

- [ ] **2. Run red:** `python -m pytest -q tests/test_control.py tests/test_mpc.py tests/test_history.py`.
- [ ] **3. Implement exact episode schedule**, only selected first40ms path execution, real observation histories and scoring-only dense truth. Save every sampled knot/candidate cost/status/ID, decision, observation/action/true-state trace. Guard runaway imagined states as model failure; true-plant warning invalidates; partial episodes cannot aggregate. Save independent full240ms counterfactual oracle replay only on audit subsets, distinct from future replanning.
- [ ] **4. Run green:** tiny exact-policy episodes; nominal shared-power identical candidates/actions/trajectories; timer delay produces no physical latency; failed imagined candidates never disappear from summaries. Exercise a partial-failure search that still has valid chosen actions.
- [ ] **5. Commit:** task files; `Execute auditable fixed-clock MPC episodes and preserve failures`.

## Task 19: M40 qualification and control-negative shortcut

**Files:** Extend mpc/control/gates/contracts/main and their tests; write `config/planner.json` after qualification, `config/control_rejection_freeze.json` only upon branch entry.

**Interfaces:** `control.run_m40(root: Path,resume: bool=False)->RunRecord`; `run_control_negative(request: MXRequest,root: Path)->RunRecord`; consume explicit core job identities and the shared look ledger.

- [ ] **1. Write failing gate/branch fixtures**:unqualified search cannot authorize model conclusions; weak valid pilot creates control-negative; one qualified conventional comparator can reject full superiority; positive subset authorizes M50 only; looks do not reset after MR.
- [ ] **2. Run red:** `python -m pytest -q tests/test_control.py tests/test_io_gates.py tests/test_statistics.py`.
- [ ] **3. Implement true-plant/physics calibration and ten paired exploratory episodes/actuator**, core A/shared40/5ms and F attribution direction, with G40 screens≥10% shifted improvement, nominal within10%. On weak valid results freeze core confirmation, seeds0–4×20, both actuators, same complete family, and run only missing compatible cells. Price one full episode and all branch jobs first.
- [ ] **4. Verify:** `python -m poc run M40 --dry-run`, M40/G40, and focused tests. If control-negative is requested, invoke it through registered MX only; support/reject using C04's final thresholds and conservative precision, never the pilot floors. Stop remaining unnecessary construction on an admissible rejection.
- [ ] **5. Commit:** source/tests/qualified manifests; `Qualify robot planning and support early control rejection`.

## Task 20: Remaining conventional MLP/LSTM and exact attention

**Entry:** G40 GO or positively supported control-negative branch. No mandatory rejection.

**Files:** Complete `poc/models/sequence.py` and mlp variants; create `poc/models/attention.py`, `tests/test_attention_ssm.py`; extend model tests.

**Interfaces:** `build_model` adds all specified remaining conventional IDs without changing existing ones; `AttentionDynamics(actuator: str,scales: Scales)` shares predict and HistoryBatch contracts. No trainable positional table or extra final normalization.

- [ ] **1. Write failing counts/masks/causality/age tests:** LSTM18428, attention17764, direct MLP18308; masked padding permutation invariant; relative ages change on query-time advance; no future-state/token attention; batched/scalar equality.
- [ ] **2. Run red:** `python -m pytest -q tests/test_attention_ssm.py tests/test_models.py` (SSM tests arrive Task21).
- [ ] **3. Implement §7 architectures**, four-head32-channel pre-norm block,16specified sin/cos pairs, FF32→128→32/GELU, no dropout, and LSTM56/mlp_h exact decoders. Keep full state and bounded history available.
- [ ] **4. Run green:** focused tests, then each new family's tiny analytic/normal-corpus learning controls with nominal recipe; any optimization repair uses the shared declared opportunity.
- [ ] **5. Commit:** task files; `Complete conventional time-aware models and causal attention`.

## Task 21: Interval-aware real-block SSM and diagnostic controls

**Files:** Create `poc/models/ssm.py`; extend attention_ssm/model tests and factory. Implement development-only `ssm_fixed` and `mlp_blind` dispatch.

**Interfaces:** `ssm.discretize(alpha: Tensor,omega: Tensor,h_s: Tensor)->tuple[Tensor,Tensor]`; `SSMDynamics(actuator: str,scales: Scales,fixed_interval: bool=False)` implements predict; interval units remain seconds. Two distinct width48 blocks each consume the actual token duration.

- [ ] **1. Write failing scipy-expm reference/gradient tests**, zero and tiny h, rotation sign, stable decay, non-elementwise integral, input-duration response, state reset, count18180 and diagnostic exclusion from competitive registry.

```python
np.testing.assert_allclose(F_reference,F_from_augmented_expm,rtol=1e-12,atol=1e-14)
assert sum(p.numel() for p in direct_ssm.parameters()) == 18_180
assert 'ssm_fixed' not in conventional_competitors
```

- [ ] **2. Run red:** `python -m pytest -q tests/test_attention_ssm.py tests/test_models.py`.
- [ ] **3. Implement §7.2 E/F**, near-zero matrix series with validated fixed float32 terms and autograd, lambda/omega/D initialization, independent block states and read-only physical-time replay. Diagnostic fixed timing/time-blind variants stay exploratory.
- [ ] **4. Run green:** reference/mask/candidate tests and family learning controls; record actual lagged count/timescales/work. No claim of published S5 replication.
- [ ] **5. Commit:** task files; `Add exact interval-dependent real-block state-space predictor`.

## Task 22: M50 complete matrix, nominal qualification and freeze closure

**Files:** Modify contracts/train/gates/main, config/matrix.json, config/criteria.json; create `config/frozen.json` only after qualification. Extend test_data/models/training/io_gates.

**Interfaces:** `contracts.freeze_inputs(root: Path,matrix: dict,criteria: list[CriterionRecord])->dict`; `validate_freeze(root: Path,freeze: dict)->None`; `train.run_m50(root: Path,resume: bool=False)->RunRecord`. Freeze names exact selected/canonical checkpoint hashes, not mtime or favorable evaluation scores.

- [ ] **1. Write failing complete-matrix fixtures** for absent competitor/unqualified fit, incorrect count/mask, unsealed confirmation, changed scaler/decoder and incompatible reused MR checkpoint; statistical count stays280.
- [ ] **2. Run red:** `python -m pytest -q tests/test_io_gates.py tests/test_models.py tests/test_training.py tests/test_data.py`.
- [ ] **3. Implement all required S07 matrix jobs** across both actuators, seeds 0–4, references and F/A losses; reuse compatible complete prior jobs only by semantic key. Select and qualify using nominal validation, enumerate every future required evidence cell, freeze code/data/environment/numerical/planner/criteria/workloads. Existing MR inputs are sealed; future confirmation arrays are generated from frozen algorithms/namespaces and add their hashes through parent-linked manifests. No fine-time baseline tuning. Commit freeze before new confirmation access.
- [ ] **4. Verify:** `python -m poc run M50 --dry-run`; complete authorized jobs; focused tests;`python -m poc validate --scope protocol`;`python -m poc decide G50`. Missing qualifications cannot be hidden by deleting a competitor; an existing qualified mandatory rejection can stop the cascade instead.
- [ ] **5. Commit:** source/config/tests/frozen textual manifests;`Freeze full PTCO confirmation matrix and selected nominal fits`.

## Task 23: M60 complete action-effect confirmation

**Files:** Extend sensitivity/prediction/metrics/statistics/gates/main and tests; generate M60 raw artifacts and decisions.

**Interfaces:** `sensitivity.run_m60(root: Path,resume: bool=False)->RunRecord`; consume validated frozen matrix, selected tensor hashes, sealed probes, accepted reference and shared look ledger. All reused MR cells retain their producing-run identity.

- [ ] **1. Write failing coverage/gate fixtures** for missing warm stratum, missing seed/component/horizon, unresolved reference, NODE refinement changing ranking, opposite actuator powers and stale checkpoint. A favorable slope alone cannot satisfy absolute/gain criteria.
- [ ] **2. Run red:** `python -m pytest -q tests/test_sensitivity.py tests/test_statistics.py tests/test_io_gates.py`.
- [ ] **3. Implement frozen E01/E02 batches**, F same-weight modes, A one-step/shared ablations, all conventional comparisons, cold/warm disaggregation, and one-step/two-half-step NODE audit. Propagate all numerical envelopes before atomic criteria. G60 GO requires C02 and K03.identity/jacobian; admissible mandatory rejection stops before full MPC.
- [ ] **4. Verify:** `python -m poc run M60 --dry-run`, M60, evidence validation, G60. Insufficient gain-fraction precision creates the predeclared additional-seed request; never replace the small-sample bound with observed100% coverage.
- [ ] **5. Commit:** implementation/tests and textual evidence indices;`Confirm full-suite action sensitivity under frozen rules`.

## Task 24: M70 fixed-clock control and causal attribution

**Files:** Extend control/action_path/metrics/statistics/gates/main and tests; no architecture/training-objective changes.

**Interfaces:** `control.run_m70(root: Path,resume: bool=False)->RunRecord`; `metrics.attribution(fixed_episode_rows: list[dict])->dict`. Consume validated G60, frozen planner/weights/workloads and episode runner.

- [ ] **1. Write failing fixed-clock/attribution fixtures**:shared F nominal equality, correct D/A/B and positive-B requirement, predicted-versus-realized first hold, all conventional ratios, model failure versus plant invalidity and unresolved attribution denominator.

```python
assert attribution['A'] == pytest.approx(D_low-D_correct)
assert attribution['B'] == pytest.approx(D_low-D_structured)
assert unresolved_B_attribution_ratio is None
```

- [ ] **2. Run red:** `python -m pytest -q tests/test_control.py tests/test_action_path.py tests/test_statistics.py`.
- [ ] **3. Implement M70 decisive40/5ms cells**, A/shared full suite plus F power/structured attribution, seed-paired20episodes and first-five curves/oracle audits. Quantify A, B, A−.5B and nominal same-power candidate/action/episode equality. Preserve complete fallback episodes and compare realized costs rather than overlapping planning scores.
- [ ] **4. Verify:** focused tests; M70 dry-run/measured episode/projection/run/G70. Apply20% shifted,5% nominal and all task/attribution/reliability conditions with actual bounds and rejection precision. Stop unused batches on valid mandatory failure; named precision/numerical request otherwise. G70 GO is required for Task25.
- [ ] **5. Commit:** source/tests/manifests; `Confirm fixed-clock control and power-only attribution`.

## Task 25: M80 actuator rates, perturbations and numerical convergence

**Files:** Extend control/action_path/metrics/gates/main and tests; no model, objective or task changes.

**Interfaces:** `control.run_m80(root: Path,resume: bool=False)->RunRecord`; `validate_observation_events(events: list[dict],condition: str)->None`. Consume G70 and frozen episode/planner APIs; reuse E03 only under identical action compiler/code/cost/manifests.

- [ ] **1. Write failing isolated condition fixtures** for physical hold rates, noisy packet reuse, scoring-only true state, balanced signs, external-force boundaries and finer-physics expansion decisions.

```python
assert noise_packet_times == fresh_observation_times
assert disturbance_intervals == [(640_000_000,720_000_000)]
assert sorted(signs_per_twenty_episode_block) == [-1]*10+[1]*10
assert high_rate_observation_count == 40
assert high_rate_optimized_variables == 12
assert near_threshold_decision_requires_full_finer_physics_cell
```

- [ ] **2. Run red:** `python -m pytest -q tests/test_control.py tests/test_action_path.py tests/test_statistics.py`.
- [ ] **3. Implement exactly four additional primary families:** actuator20/10ms, isolated noise and isolated disturbance; no Cartesian product. Keep sensing/replanning40ms/model5ms and same-rate competitors/references. Run next-finer physics first-five subsets and affected full cells at specified triggers/near thresholds; NODE solver resolution is separate. Reliability rechecks every corrected mandatory episode.
- [ ] **4. Verify:** focused tests; M80 dry-run/projection/run/G80. Require all10% practical/noise/disturbance floors and convergence with conservative bounds. Intrinsic corrected fallback is a reliability rejection; coding defects invalidate the affected runs. Preserve clean supported claims even if robustness fails.
- [ ] **5. Commit:** source/tests/manifests; `Confirm practical actuator rates and isolated perturbations`.

## Task 26: Named MX contracts, dependency invalidation and crash recovery

**Placement:** Basic job locks/journals exist from Task1. Implement branch dispatch when its first real request occurs; finish recovery coverage before M90. This task does not require waiting for M80 to repair an earlier defect.

**Files:** Create `poc/continuation.py`, `tests/test_recovery.py`; extend io/contracts/gates/main; register requests in config/milestones.json and immutable config/revisions records.

**Interfaces:** `continuation.validate_request(request: MXRequest,root: Path)->None`; `run_request(request: MXRequest,root: Path)->RunRecord`; `contracts.affected_descendants(changed_inputs: dict,runs: list[RunRecord])->list[str]`; `io.resume_boundary(job: JobSpec,root: Path)->dict`. Existing model/physics modules implement the named branch measurement, not an autonomous search loop.

- [ ] **1. Write failing fault-injection fixtures**:crash during rename/hash/marker, live/foreign lock recovery, duplicate expensive launch, training RNG/permutation resume, partial ±pair, mid-CEM restart, stale freeze/undeclared import and semantic invalidation. Test every registered branch rejects empty parameters/evidence/projection and unknown branches.
- [ ] **2. Run red:** `python -m pytest -q tests/test_recovery.py tests/test_io_gates.py`.
- [ ] **3. Implement spec §§13–14 resumption/branches**, preserving all attempts; repair minimized fixture first, numerical reference-only refinements, solver equal opportunity and nominal fit interpretation, optimizer fixed nominal grid, precision request committed before outcomes, planner fixed sequence, control-negative subset, and first-divergence reproduction. Unexpected branches require all twelve contract elements and version/fresh data rules. A valid negative result is not reparable bookkeeping.
- [ ] **4. Run green:** recovery tests plus each branch's small dispatch fixture;`python -m poc run MX --dry-run` on a genuine pending request enumerates exactly that contract's jobs;`next --dry-run` returns its exact stored return milestone. Unknown/unregistered request cannot launch anything.
- [ ] **5. Commit:** task files;`Enforce named continuation and deterministic interrupted-job recovery`.

## Task 27: M90 fresh-process replay, training replication and audit closure

**Entry:** G80 GO; no unresolved mandatory criterion. A scientific FAIL already established follows MF instead.

**Files:** Create `poc/replicate.py`, `tests/test_end_to_end.py`; extend test_recovery and contracts/gates/main.

**Interfaces:** `replicate.reference_aggregate(raw_paths: list[Path],criteria: list[CriterionRecord])->dict` is a separately coded simple aggregator; `run_m90(root: Path,resume: bool=False)->RunRecord`; `compare_replay(original: dict,replay: dict)->dict`; `compare_retraining(original: RunRecord,retrained: RunRecord)->dict`.

- [ ] **1. Write failing tiny end-to-end fixtures** comparing production/reference summaries, edited archived input, archive metadata versus canonical tensors, wrong thread/env fingerprint, replay tolerance and checkpoint resume. Each tolerance boundary is asserted independently.

```python
assert replay_max_q_diff <= 1e-6
assert replay_max_v_diff <= 1e-5
assert replay_max_command_diff <= 1e-6
assert replay_relative_cost_diff <= 1e-4
assert retrain_relative_validation_diff <= 1e-4
assert retrain_relative_control_cost_diff <= 1e-3
```

- [ ] **2. Run red:** `python -m pytest -q tests/test_end_to_end.py tests/test_recovery.py`.
- [ ] **3. Implement fresh offline process/venv recreation**, every primary independent reaggregation, first64probe/stratum replay, firstepisode/seed/actuator/mandatory condition replay and seeds0/4 actual retraining for corrected/comparator union and F attribution cells. Check max parameter≤1e-6*(1+max|parameter|), unchanged decisions, all coverage/hash/resource closure; no cached training masquerades as replication.
- [ ] **4. Verify:** tiny tests; M90 dry-run/projection/run; evidence validation; G90. Larger discrepancies create reproduction MX with first divergent unit, never post-hoc tolerance changes. G90 GO alone makes PASS eligible, not yet assigned.
- [ ] **5. Commit:** source/tests/manifests;`Reproduce decisive behavior and close the evidence audit`.

## Task 28: Honest terminal report and safe reproduction commands

**Placement:** Minimal MF is usable from Task1. Complete/report fields incrementally; this task's terminal/report tests are required for any early termination, not only after M90.

**Files:** Complete `poc/report.py`; extend `tests/test_io_gates.py`, `tests/test_end_to_end.py`; produce `reports/final_report.md`, `claims.csv`, `run_index.csv`, `reproduce.sh`, MF evidence and GF decision.

**Interfaces:** `report.terminal_verdict(decisions: list[DecisionRecord],criteria: list[CriterionRecord],blockers: list[dict])->str|None`; `write_report(root: Path,decisions: list[DecisionRecord])->dict` completes Task1 API; `validate_release(root: Path)->dict` in contracts independently checks report/CSV/decision mapping.

- [ ] **1. Write failing verdict/report fixtures**:qualified FAIL with unrelated blocker; all-criteria PASS; missing mandatory robustness not PASS; setup-only BLOCKED; valid unresolved with executable action NONTERMINAL; irreducible valid uncertainty INCONCLUSIVE; invalid coding evidence never scientific FAIL. Verify reproduction defaults to validation/reaggregation and requires explicit current authorization for expensive reruns.
- [ ] **2. Run red:** `python -m pytest -q tests/test_io_gates.py tests/test_end_to_end.py`.
- [ ] **3. Implement thirteen report sections and exact CSV fields**, all attempts/resources/missingness/evidence pointers, source versus observed claims, and actual-manifest-derived reproduction commands. Script starts with `set -euo pipefail`; hash verification/offline restore; default no experiment rerun;`--rerun-experiments` checks allocation and job locks. Stdlib blocked report must not reference unavailable CLI/environment as if it exists.
- [ ] **4. Verify:** focused tests;`python -m poc run MF`;`python -m poc validate --scope release`;`python -m poc decide GF`. Read generated text/CSV/terminal JSON and independent verdict mapping; all agree. Report GO certifies honest reporting, not the scientific outcome.
- [ ] **5. Commit:** report source/tests and generated textual artifacts that actually exist;`Report the scoped PTCO decision and reproduction record`. Do not commit fictitious results or unexecuted reproduction steps.

## Specification coverage and handoff

| Supplied protocol sections | Spec sections | Plan owners |
|---|---|---|
| S00–S02 intent/claims/acceptance | 1–3,11–12,15 | Tasks1,4,11–14,23–25,28 |
| S03 provisioning/authority | 4 | Tasks1–2 |
| S04 mechanics/robot/oracles | 3,5,9 | Tasks3–6,11 |
| S05 data/history | 6 | Tasks7–8,13 |
| S06 model fairness/references | 7 | Tasks4,9,15,20–21 |
| S07 training/freeze | 8 | Tasks10,14,22 |
| S08 estimands/numerics/statistics | 9,11–12 | Tasks6,11–13,23–25 |
| S09 control/robustness/resources | 10 | Tasks15–19,24–25 |
| S10 interfaces/schemas | 13 | Tasks1–2,7,11–14,22,26,28 |
| S11–S12 transitions/contracts | 14 | Tasks1–2,4,6,14,19,22–28 |
| S13 resumption/conditional work | 12–14 | Tasks1,10,12,14,19,26–27 |
| S14 reports/verdicts | 15 | Tasks1,28 |
| S15 provenance/limits | 1,16 | Tasks1–2,6,27–28 |

Every task owns an independently checkable deliverable and a commit. Commands are future verification instructions; unchecked boxes do not denote executed work. This plan has no scientific timing estimates beyond the supplied protocol's illustrative work counts: substitute measured preflight/training/reference/episode throughput before scheduling the corresponding batch.

The plan was self-reviewed for coverage, file/interface consistency, threshold values, all 280 statistical atoms, gate order, missing-data behavior and the five review-focus classes. Document verification does not implement or test these interfaces.

For later execution, native `superpowers:executing-plans` is a reasonable fit because the numerical/data/controller units share strict interfaces and scientific runs are serial. A subagent-driven execution method is also available after the user selects it; this planning task does not dispatch agents or begin implementation. Review the spec and plan at handoff before starting Task1.
