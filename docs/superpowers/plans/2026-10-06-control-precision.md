# Control precision implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan inline, task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the prospectively sized five-seed continuation for the two fine-time MLP comparisons without changing the original experiment or reproduction inputs.

**Architecture:** A fixed request/nominal/control coordinator launches existing scientific CLIs in subprocesses. A separate consumer verifies complete raw evidence, qualified plant refinement and ten paired seed means, then appends a two-criterion update to the original eight-contrast spending ledger.

**Tech stack:** Existing Python 3.11, CPU PyTorch, NumPy, SciPy, MuJoCo and pytest; no new dependency.

**Spec:** `docs/superpowers/specs/2026-10-06-control-precision.md`.

## Global constraints

- Initial seeds 0–4 retained; new control seeds 5–9; twenty paired tasks per seed.
- Only `mlp_dt,power_correct`, both plants, A/shared and `h_model=5 ms`; existing hardware clocks and planner unchanged.
- No altered model, optimizer selection, data, threshold, solver, original registry or reproduction source.
- New scientific jobs are serial and wait for the existing reproduction queue to complete.
- The family stays eight, look 2 alpha is 0.0125 and its two-direction tail is 0.00078125.
- Six unchanged comparisons keep their original look 1 evidence.
- Preserve failed/incomplete attempts; no sign from missing data or unqualified numerical evidence.

## Review focus

- Reused head fits must be the complete eligible ascending cohort; a favorable-score substitution fails.
- A copied confirmation label or fitted-model cache must fail before training.
- New files or dirty Git metadata must not invalidate an unrelated old run; actual semantic changes must fail.
- An active reproduction worker blocks scientific dispatch, without blocking read-only preparation or independent tests.
- Duplicate/partial cells, changed real clocks, zero variance and influence instability cannot manufacture a resolved sign.
- Report regeneration must reuse the same statistical look and preserve cached unchanged comparisons.

## Task 1: Prospective request and nominal checkpoint freeze

**Files:** Create `poc/control_precision.py`, `tests/test_control_precision.py`, `config/revisions/MX-control-precision-0001.json`; register that exact sub-contract in `config/milestones.json`.

**Interfaces:**

- `build_request(root: Path) -> dict`: binds the saved planning analysis, original result/design/numerics, source/data hashes, exact new IDs and twelve-part contract.
- `verify_request(root: Path, request: dict) -> dict`: rejects unknown dimensions, changed evidence, reordered/replaced seeds, missing allocation and an incompatible parent look.
- `prepare_nominal(root: Path, request: dict) -> dict`: creates one cache-free, independently copied nominal context and exact twenty-coordinate registry; records eligible external direct-head references.
- `run_nominal(root: Path, registry: dict, *, resume: bool) -> dict`: uses the unchanged nominal engine and returns the fully qualified checkpoint freeze.

- [ ] Write `test_request_keeps_old_clusters_and_adds_only_seeds_5_to_9`: assert both target criteria, 400 new episodes, unchanged family/tails and rejection of a replaced or exploratory seed.
- [ ] Write `test_nominal_inventory_excludes_labels_and_weights`: reject confirmation/development arrays and neural checkpoints; require train/validation and complete early validity closure for both plants.
- [ ] Write `test_reuse_is_complete_and_semantically_qualified`: accept only eligible direct-head seeds 5–9, match nominal score/scaler/tensors and reject missing, changed or score-selected reuse.
- [ ] Write `test_live_reproduction_blocks_dispatch_but_allows_dry_run` and a source-inventory fixture proving additive root files leave the two existing source contexts unchanged.
- [ ] Run `.venv/bin/python -m pytest -q tests/test_control_precision.py`; record RED from missing implementation or the intended assertion.
- [ ] Implement the interfaces above using existing atomic I/O, locks, inventory copying, `train_cell` and nominal qualification. Scientific producers run in their own subprocesses; configuration mappings contain filepath-to-object entries only.
- [ ] Run the focused tests and affected context/nominal tests. Commit the implementation and the prospective request before new outcomes.

## Task 2: Exact episode registry and serial recovery

**Files:** Extend `poc/control_precision.py` and its tests. Generated inputs are `config/control_precision_control_jobs.json`, `config/control_precision_finer_jobs.json` and explicit requests under `config/control_precision_jobs/`.

**Interfaces:**

- `prepare_control(root: Path, request: dict, selected: dict) -> dict`: freezes exactly 400 original-S09 5 ms requests for seeds 5–9.
- `prepare_finer(root: Path, request: dict, primary: dict, *, full: bool=False, actuators: tuple[str, ...]=('direct','lagged')) -> dict`: derives the first-five-episode requests at 125 µs (100 for both plants), or full coverage of the explicitly affected plants.
- `run_episode_registry(root: Path, registry: dict, *, resume: bool) -> dict`: executes and verifies exact requests through `poc.control_jobs`, retaining attempts and compatible completed jobs.

- [ ] Write `test_exact_400_and_100_cells_match_checkpoint_freeze`: pin target coordinates, CEM 128/4/16, hardware 40 ms, physical 250/125 µs, twenty task IDs and root 6101; changed checkpoint/path/clocks fail.
- [ ] Write `test_resume_reuses_completed_jobs_and_preserves_interruption`: reject a stale live lock, modified request, conflicting completion or partial episode coverage; never choose a better duplicate attempt.
- [ ] Run the focused registry tests and observe RED before adding the implementation.
- [ ] Implement the exact subprocess runner with immutable complete receipts, observed return codes and full manifest/raw-cost verification; do not alter the scientific producer.
- [ ] Exercise a genuine tiny complete MuJoCo fixture across the coordinator/producer manifest boundary, then the changed-request and interrupted-resume cases.
- [ ] Run focused control/job/recovery tests and the complete suite once; commit the verified runner before preparing real control jobs.

## Task 3: Numerical qualification and ten-seed inference

**Files:** Create `poc/control_precision_result.py`, `tests/test_control_precision_result.py`; generate distinct precision reports and immutable evidence snapshots. Original reports and the reproduction-ready pointer remain unchanged.

**Interfaces:**

- `aggregate_pairs(original: list[dict], appended: list[dict], request: dict) -> dict`: exactly two actuator contrasts, ten ordered seed means, twenty complete paired tasks each.
- `qualify_numerics(root: Path, request: dict, primary: dict, finer: dict) -> dict`: measures outward old/new plant-step envelopes and requests explicit full/further refinement for affected or near-boundary cells.
- `evaluate_precision(costs: dict, *, look_index: int, envelopes: dict) -> dict`: calls existing `contrast_bounds` at the unchanged eight-family tail, with the common seed-cluster indices and original influence/rare-tail rules.
- `write_report(root: Path, request: dict, evidence: dict) -> dict`: verifies all hashes/coverage first, reuses `reserve_control_look` with the original design, and records updated two-criterion results plus six cached look-1 references.

- [ ] Write `test_no_partial_or_stale_ten_seed_sign`: reject missing/duplicate pairs, an omitted fallback, wrong task namespace, incomplete numerical scope or unqualified checkpoint.
- [ ] Write `test_outward_envelope_and_near_boundary_expansion`: a measured discrepancy widens bounds; absent numerical evidence never becomes zero; affected new cells require full refinement.
- [ ] Write `test_fixed_tail_cluster_bootstrap_and_influence`: pin 0.00078125, entropy `[9102,2]`, 20,000 common indices, zero-variance/rare-tail rules, and unstable leave-one-seed-out classification.
- [ ] Write `test_report_is_idempotent_and_six_old_cells_stay_at_look1`: preserve original records, family eight, prior spending and exactly one appended look for identical evidence.
- [ ] Run the new module tests for RED, implement the consumer interfaces, then run GREEN and affected statistics/ledger tests.
- [ ] Run the complete suite once after the final source change and commit. Freeze all request/producer/consumer hashes before new scientific execution.

## Task 4: Execute the precision continuation and return

- [ ] Complete and verify the existing 60 replay episodes, twelve fresh fits and 480 repeated-control episodes. Diagnose any actual reproduction defect first; do not edit its frozen inputs.
- [ ] Prepare the new nominal context, verify eligible direct-head references, project actual context/training resources and run only missing nominal fits serially.
- [ ] Independently reproduce the selected validation objectives and tensors; freeze all twenty selected checkpoint coordinates before opening new control outputs.
- [ ] Measure a representative new control job, check actual free resources, then execute the remaining compatible 400-job registry serially with all attempts retained.
- [ ] Execute and verify the first-five finer cohort; run each explicitly required affected/full/further refinement before accepting the numerical envelope.
- [ ] Reaggregate all old and new raw paired costs, perform the ten-seed look using the committed rule, and preserve unresolved or unfavorable findings alongside supported ones.
- [ ] Reproduce newly decisive behavior from the existing locked environment and frozen manifests. A result outside the original tolerances requires diagnosis, not post-hoc tolerance widening.
- [ ] If a live sign or influence question remains, write the next independently justified sample request before any new outcomes. Otherwise record the limited decision, finish whole-branch review and release the honest report.
- [ ] Record measured resources and meaningful commits throughout. No arbitrary seed cap, full-suite PASS or equivalence claim follows from this subset.
