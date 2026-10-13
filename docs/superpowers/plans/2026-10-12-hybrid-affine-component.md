# Hybrid AffineFlow Component Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Implement tasks serially with task-scoped reviews and meaningful commits.

**Goal:** Execute one bounded, prospectively gated comparison of AffineFlow propagation with shared nominal physics and small residuals, then close or justify independent confirmation.

**Architecture:** Add five hybrid predictors to the existing registry without changing its legacy model list. Add an explicit hybrid matrix to the existing hash-bound calibration pipeline, recipe-driven coverage and a separate component gate. Reuse all truth, scoring, timing and resource producers; freeze reviewed source before fresh empirical preparation.

**Tech Stack:** Existing native Python3.11/PyTorch/NumPy/MuJoCo CPU environment; no dependencies.

**Spec:** `docs/superpowers/specs/2026-10-12-hybrid-affine-component-design.md`.

## Global Constraints

- Work on existing `continue`; no new worktree/branch, push, merge, dependency/asset/historical-report changes.
- Matrix hybrid: sessions1/2/3, initialization0, budgets30/120s, five declared models:30fits plus6nominal references. Fresh evaluation seed_session_id=1. Default pilot/development recipes and prediction behavior remain compatible.
- Primary three6→32→32→6 coefficient networks have1,478 trainable parameters and identical seeded tensors; practical pair8→32→32→2 has1,410. All residual output layers initialize zero; priorparameters (.10,.02,.02,.05,.05) frozen, known sharedtau=.08.
- Existing own-solver float32/float64, one+.25four objective, LR(.001,.0003),256batch,5epochvalidation, plateau and1,000/rate cap remain; retain failures/limits, no replacementdraws.
- Mainclaim budget30/40ms. All3sessions must satisfy provisional screens and5% conservative-error bands against hybrid_accel_rk4; paired decision quality follows spec. Median saving≥25% versus fastest acceptable othermodel at that same configuration; p95 no worse. Otherbudgets/partitions diagnostic, no headline switch.
- Repair known post-commit resource-observation accounting edge before freeze; no source changes after preparation. Rawroot `/private/tmp/physical-time-consistency-hybrid-component`; replicas separate; trackedreports `reports/hybrid_component/`.

## Review Focus

- Zero/mixed/tiny intervals and gradients: zero states remain exact, solver work truthful, commonprior field agrees with existing acceleration (Task1).
- Recipe/matrix changes and defaults: source/recipe mismatch refuses reuse; exact30fit cohort, default21/195 preserved; filters cannot invent fits (Task2).
- Post-commit/pre-observation interruption: no doublecount of checkpoint persistence or observational writes, including nested trackers (Task2).
- Near-tied truth/invalid selected action/missing anchor: paired-cost cancellation valid only for identical feasible selections; undefined inputs cannot produce positivegate (Task3).
- Mixed partitions, faster unacceptable baseline and pooled timing groups: use fixedprimary, all50calls/session, no fastest-repeat selection, every comparison retained (Task3).

## Task 1: Implement matched hybrid predictors

**Files:** Create `poc/hybrid_residual.py`, `tests/test_hybrid_residual.py`; modify `poc/physics_fit.py`, `poc/calibration_models.py`, `tests/test_calibration_models.py`.

**Interfaces:** `PhysicsFit.acceleration_coefficients(qv: Tensor) -> tuple[Tensor, Tensor]` returns nominaldrift and densegain using onecombinedsolve. Classes `HybridAffineFlow`, `HybridAffineRK4`, `HybridAffineAware`, `HybridAccelRK4`, `HybridAccelAware` accept `Scales`; predictors retain existing `predict`/Prediction contract. Add `HYBRID_MODEL_IDS` and `ALL_MODEL_IDS` while preserving `MODEL_IDS`; `build_model`/`model_metadata` support newids. `propagation_family(model_id: str) -> tuple[str,...]` returns legacypair, hybridthree or empty.

- [ ] Write failing tests: field f+B*w equals independent legacy acceleration at randomstates/denseeffort; physicalforward arithmetic remains identical; newprimaryweights/shapes/counts pair exactly; genericreferenceweights pair; zeroresidual=nominalfield, no privileged plantparameters.
- [ ] Add independent constantdenseB/constantdrift analytic actuator-response checks, RK4 denseintegration check, zero/mixed/tinyh, invalidinputs, float64gradients and actual2/4/2networkcalls. Run `.venv/bin/python -m pytest tests/test_hybrid_residual.py tests/test_calibration_models.py` and confirm RED before implementation.
- [ ] Implement common nominal force factoring/coefficient solve and smallmodels. Reuse existing stage maps and stable actuatorhelpers; no extraNNcalls or per-state caching. Constructor/factory preserves callerRNG, independent storage, frozenprior and metadata.
- [ ] Run focusedhybrid/physics/solver/model tests; verify defaultmodel behavior and `git diff --check`; commit `feat: add physics-prior residual solver controls`.

## Task 2: Add the bounded cohort and raw evidence support

**Files:** Modify `poc/calibration_study.py`, `poc/calibration_training.py`, `poc/calibration_predictions.py`, `poc/calibration_report.py` (coverage only), `poc/calibration_resources.py`; tests `test_hybrid_matrix.py`, existing calibration/resource tests.

**Interfaces:** `prospective_recipe(*, unit_fixture: dict | None = None, matrix: str = 'pilot') -> dict` preserves legacydefault and emits spec's hybridvariant; `prepare/open_store` and allCLI stages accept matrixhybrid. A shared `_coordinates(recipe,matrix)` defines membership. Training accepts ALL_MODEL_IDS but requires preparedmembership. Propagation producers/readers use `propagation_family`; everyselectedprimaryfit owns three-map diagnostics. Task3 owns aggregate gate/export interpretation.

The hybrid recipe's `hybrid_component` block records `primary_model_id='hybrid_affine_flow'`, `anchor_model_id='hybrid_accel_rk4'`, `session_ids=[1,2,3]`, `initialization_ids=[0]`, `primary_budget_s=30`, `primary_query_step_ns=40000000`, `error_ratio_limit=1.05`, `median_ratio_limit=.75`, and `p95_ratio_limit=1.0`. No such block is added to unchanged legacy recipes. Its full model list is `HYBRID_MODEL_IDS` and budgets `[30,120]` before explicit unit overrides.

- [ ] RED exact30hybridcoordinates/6nominal, freshsessions/evaluation/banks, immutablematrix refusal, legacyrecipe equality/21/195membership and outside-cohort filters. RED three-map same-weight coefficientdiagnostic metadata/raw coverage and invalidswap/omission detection.
- [ ] RED known10active+2persistence post-commit observational interruption reports12, not14; add pre-observation/nested/failedcommit cases proving observationtime excluded and restart occurs onexception.
- [ ] Add explicitmatrixrecipe/CLI support without changing olddefaultvalues. Generalize coverage using recipe.models and propagationfamily, retaining invalid/missing outcomes. Correct trackerreset in finally; preserve retainedinterruptedattempts/unknown-tail semantics.
- [ ] Run affectedmatrix/training/store/prediction/resource tests and genuine reduced-unit hybrid acquisition/fitting/prediction/decision/timing workflow (report handled Task3); commit `feat: add bounded hybrid comparison cohort and evidence`.

## Task 3: Implement prospective component decisions and reports

**Files:** Create `poc/hybrid_report.py`, `tests/test_hybrid_report.py`; modify `poc/calibration_report.py`; test existing report/publiccheckout workflows.

**Interfaces:** `paired_cost_comparison(candidate_outcomes, anchor_outcomes) -> dict` aligns session/state/query identities, checks validity, reports paired mean difference/envelope and retainedranking statuses. `hybrid_gate(result: dict) -> dict` uses complete raw-derivedconfigs/outcomes/timing, returns declared four outcomes plus eachmodel/session eligibility, primarypoolstats, fastestacceptable comparator and saving/p95ratio. Existing aggregate/write_report call componentgate only forhybridmatrix; defaultgates remain unchanged.

For hybrid reports only, enrich independently validated decision outcomes with `selected_truth_cost`, `selected_truth_envelope`, `selected_command_sha256`, and `truth_receipt` from their verified shared candidate truth. Invalid selections have null selected fields. Preserve stored-summary verification before making enriched copies. Paired identity also binds budget/initialization and the common truth receipt; equal command digests may cancel cost even when candidate indices differ. Retain verified complete-scorer samples in each hybrid timing group as `samples` containing coordinate-bound repetition, wall_s, cpu_s and status, sorted by repetition; pool actual durations, never group medians. Default legacy outcome/timing export schemas remain unchanged.

- [ ] RED same-feasible-selection costcancellation despite oracle envelope; distinctselections use independentcost intervals; unsafe/no-selection/missing/duplicate/unresolvedfeasibility/emptytruth/failedanchor cannotpass; rankingties alone usable with disclosure.
- [ ] RED fixed30s/40ms primary, all3session checks,5%error boundaries,0.75median boundary, p95 guard, cheapacceptablebaseline preventinggain, fasterpoorquality baseline excluded, missingtiming/samples unresolved, cappedfits retained, positive120s-only result cannot switchheadline, unitfixtures ineligible.
- [ ] Implement gate and recipe-drivencurves/ratios/SVG/model legends/portable hybridreproduction commands. Retain raw reports and all eligible/ineligible alternatives; export component_findings.md/component_gate.json in addition to standardfivefiles. Make newhybridCSV LF, legacyCSV unchanged.
- [ ] Run genuine copied-public-checkout reduced hybrid workflow through report/verify; affectedchecks and fullpytest once, compile/diffchecks. Commit `research: freeze hybrid component gate and reproducible reporting`.
- [ ] Complete whole-source review before preparation; one consolidatedfixwave/scopedreview if needed. Record exact source/spec/recipe hashes. Noempirical jobs until reviewedfreeze.

## Task 4: Execute, reproduce and close the component study

**Files:** New `reports/hybrid_component/` exports/recipe/commands/resources/reproduction/findings and `docs/research-trajectory.md`; preserve allprevious results. Native `.venv` and fixedone-thread CPU.

- [ ] Prepare and collect `python -m poc.calibration_study prepare/collect --matrix hybrid --output-root /private/tmp/physical-time-consistency-hybrid-component`. Freeze source before firstprepare. Run full30fit train, predict, decisions, one completepooled time, report and verify. Commands/logs/receipts are durable and explicit. Don't change source, lowercaps or replacefailedcells after seeing outcomes.
- [ ] Report actual allmodel/budget/session/partition coverage, primaryquality eligibility and25%gate, fullruntime/qualitycurves, failures/limits, numericaltruth audits and nonoverlappingresources. Nohardware/deadline/confirmatory claim. Mainquestion resolves once; secondarywins do not redefineit.
- [ ] Fresh separate-root prepare/collect and threeprimary session1/budget30/init0 fits; compare selectedtensors/validation plus first/last predictions/decisions/status/envelopes/selection/regret/work/querylogs across allmainpartitions and designatedthreegridtruth. Exactexecuted supportdrivers and checks archived; resources separate.
- [ ] Final rawverify, artifact/hash/link/CRLF-aware rangecheck; no unchangedfullsuite repetition. Commit actualfindings and honest researchtrajectory. Positivegate permits only a futureconfirmation spec; validnonpositive ends activeinvestment without deleting scientificartifacts. Unresolvedoutcomes remain unresolved and retained.
- [ ] Independentfinalartifactreview. Archive exactreports/rulings and cleanup onlythisplan's agent-ownedscratchworkspace; leavecurrentbranch/rawstores/siblingplans intact. Return actualresult, localcommits and evidence.
