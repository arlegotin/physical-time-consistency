# Final bounded hybrid component execution

The fixed primary gate resolves to **`no_component_advantage`**: `primary_flow_not_acceptable_in_every_session`. The primary configuration remains budget 30 seconds / main partition 40 milliseconds. This valid nonpositive bounded result ends active investment in this line; all scientific artifacts, secondary observations and unresolved comparisons remain retained. No confirmation, hardware, deadline or controller stage was launched.

Flow passes every primary conservative prediction error band, but session 3 fails the paired decision-quality bar: 64 valid pairs, 62 identical commands, mean selected truth-cost difference `4.764970410914889e-5` versus mean joint numerical envelope `3.258027528318941e-15`. Identical commands contribute exactly zero cost difference and envelope. Sessions 1 and 2 retain unresolved paired comparisons because the anchor has one invalid selected decision in each 64-state group (1.5625%). These unresolved comparisons are preserved; the demonstrated session-3 loss independently prevents all-session acceptance.

No primary method is acceptable in every session. The gate therefore has no fastest acceptable alternative and its median ratio, saving fraction and p95 ratio remain undefined. The runtime values below are complete measured diagnostics; they do not rescue failed quality eligibility.

| Primary model | Eligibility | Session statuses | Pooled scorer median (ms) | p95 (ms) | Calls |
|---|---|---|---:|---:|---:|
| hybrid_affine_flow | fail | 1: unresolved, 2: unresolved, 3: fail | 16.401875 | 17.036167 | 150 |
| hybrid_affine_rk4 | fail | 1: fail, 2: fail, 3: fail | 28.842208 | 29.873254 | 150 |
| hybrid_affine_aware | fail | 1: unresolved, 2: fail, 3: fail | 16.114854 | 16.697204 | 150 |
| hybrid_accel_rk4 | fail | 1: fail, 2: fail, 3: pass | 25.397833 | 26.460521 | 150 |
| hybrid_accel_aware | fail | 1: fail, 2: fail, 3: fail | 14.387041 | 14.922275 | 150 |

Each pool contains the same 50 actual complete-scorer calls in each of three calibration sessions, with rotated-order provenance. Group medians were not pooled. The one full timing request contains all 36 selected predictors/references.

| Flow session | q RMSE / anchor | q p95 / anchor | v RMSE / anchor | v p95 / anchor | Quality status |
|---|---:|---:|---:|---:|---|
| 1 | 0.796758 | 0.849791 | 0.846946 | 0.879754 | unresolved |
| 2 | 0.615608 | 0.681839 | 0.759517 | 0.786858 | unresolved |
| 3 | 0.641894 | 0.661117 | 0.765097 | 0.759487 | fail |

All ratios satisfy the provisional 1.05 band. The complete [component findings](component_findings.md) and [gate JSON](component_gate.json) retain every method/session/configuration comparison and reason. The [raw diagnostic report](short_calibration_findings.md), [48-row CSV](short_calibration_metrics.csv), [full metrics/result JSON](short_calibration_result.json) and [learning curves](short_calibration_learning_curves.svg) preserve both budgets, all 5/10/20/40-ms partitions, nominal references and same-weight propagation. Secondary runtime or prediction improvements cannot redefine the main claim.

Coverage is complete: three full 600-segment independent sessions; 30 neural fits plus six unfitted nominal references; 36 predictions, 18 same-weight coefficient-family diagnostics (three maps each), 36 decisions; 1,008 timing groups in one pooled request. The result contains 12,960 prediction-metric groups, 9,216 decision outcomes and 48 own-solver model/budget/partition configurations. Evaluation uses all 64 trajectories, all declared anchors/cases and 256-candidate banks, with fresh seed_session_id=1. There are no missing or invalid cells, failed fits or failed optimizer attempts. Six finite fits reached optimization limits and are retained without replacement or lower caps:

- Session 1, budget 30, init 0: `hybrid_accel_aware`.
- Session 1, budget 120, init 0: `hybrid_affine_aware`.
- Session 1, budget 120, init 0: `hybrid_affine_flow`.
- Session 1, budget 120, init 0: `hybrid_affine_rk4`.
- Session 3, budget 30, init 0: `hybrid_accel_aware`.
- Session 3, budget 30, init 0: `hybrid_accel_rk4`.

The [recipe](hybrid_recipe.json) fixes the common nominal parameters (.10,.02,.02,.05,.05), tau=.08, matched 1,478-parameter coefficient networks, 1,410-parameter practical acceleration residuals, zero residual outputs, seed `2026101200+2*session_id+initialization_id`, float32 fitting/float64 scoring, native one-thread CPU, batches ≤256, own-solver one+.25four objective, learning rates (.001,.0003), five-epoch validation, plateau only after ≥100 epochs and 1,000 epochs per rate. No source, dependency, branch, worktree, cap, draw, historical store or historical report was changed.

The [approved freeze](source-freeze.json) binds scientific revision `97be931dcaa5d61f364397e606de17da5ce6e1e2` and exact source/runtime/asset/recipe/spec hashes. Prior source evidence is explicitly 1,490 full-suite passes at `f11252a`, followed by 87 focused p95-fix passes and 124 pooled-request-fix passes at `97be931`; the unchanged suite was not rerun after empirical work. The reviewed post-commit resource-observation fix applies to this freeze. Historical pilot limitations remain disclosed in their original reports.

Fresh reproduction uses a separate root, independently collects all 600 session-1 segments, and refits the three session-1/budget-30/init-0 coefficient models in fresh CLI processes. The [numerical replication](numerical_replication.json) reports `all_equal=true`, identical collected/evaluation arrays, independently regenerated candidate banks, all selected tensors/canonical hashes and selected validation. First/last prediction rows include endpoints, drift, command response and composition, truth/errors/envelopes/statuses; complete first/last 256-candidate decisions include selections, realized regret, status, feasibility, work and query logs under all four main partitions. Designated first/last trajectories receive independent 250000/125000/62500-ns truth replay and retained adaptive grids.

| Replica model | Selected epoch | Canonical tensor SHA-256 | Endpoint component checks | Additional term checks | Candidate checks |
|---|---:|---|---:|---:|---:|
| hybrid_affine_flow | 500 | `681e1743266802646cd425bee9d5cb611395f095e51cee004d09b64c76d589f0` | 108 | 108 | 2048 |
| hybrid_affine_rk4 | 300 | `6215822fa27ea62d409e5a4375a742456d1d2480ccf1659ada865505c3f8c319` | 108 | 108 | 2048 |
| hybrid_affine_aware | 500 | `223ff0b4c150f22bb7b68608f838b9ce195b884bacb225c39d4ed151a2e4185e` | 108 | 108 | 2048 |

Replication is numerical repeatability in the same frozen native environment, not an additional independent scientific session. Its recipe/matrix membership remains the full hybrid declaration, while only the three declared replica fits are executed: replica raw verification therefore correctly reports incomplete main-matrix coverage, with no invalid retained products. Its numerical support artifacts verify the prescribed subset.

The [resource audit](resource_audit.json) keeps primary and replica attempts separate and excludes nested parent/child overlap. Producer spans include active work and measured checkpoint persistence, exclude ledger observation writes and post-producer finalization, and use monotonic wall/process CPU. Inclusive CLI child costs are a separate end-to-end software-process view and cannot be added to overlapping producers. Original finalization observations remain separate in the result. Calendar timestamps establish chronology only; the decision-stage calendar interval differs from its monotonic duration and is not used to infer compute costs. No total commissioning gain is claimed.

| Primary top-level producer stage | Wall (s) | CPU (s) | Attempts |
|---|---:|---:|---:|
| collect | 29.143508 | 29.015912 | 3 |
| decision_truth | 1997.689651 | 1996.097291 | 1 |
| decisions | 484.513843 | 484.132801 | 36 |
| evaluation | 1.613431 | 1.610073 | 1 |
| fit | 2545.107376 | 2536.675989 | 30 |
| matrix | 0.000299 | 0.000284 | 1 |
| nominal | 0.005457 | 0.005275 | 6 |
| prediction_truth | 36.307524 | 36.276611 | 1 |
| predictions | 267.679312 | 267.251573 | 36 |
| propagation | 381.783195 | 381.142562 | 18 |
| recipe | 0.010760 | 0.010686 | 1 |
| timing | 501.316710 | 501.215202 | 1 |

Independent numerical replica active/persistence cost: wall 88.680155 s, CPU 88.664197 s. This inclusive numerical resource span is separate from primary commissioning/evaluation; its nested fresh-evaluation work must not be added again. Primary raw storage: 1385257675 bytes; replica raw storage: 6972751 bytes. Current volume headroom and every resource attempt are retained in the audit.

All exact [executed commands](execution_commands.json), stdout/stderr, prepared/source/asset manifests and [support provenance](support_provenance.md) are archived. The support orchestrator initially completed all six scientific stages, then failed before report CLI launch because the destination path became a log pathname. Its [exact original driver](run_study_initial.py), [traceback](orchestration_failure_traceback.txt) and [failure account](orchestration_failure.json) are preserved. Only support filename handling was corrected; report launched afterward with no empirical stage repeat. The failed wrapper tail is unmeasured, and no report-child/producer attempt existed at that failure. All actual CLI records and numerical producer attempts completed; no failed numerical product was replaced.

| Scope | Exact command | Monotonic wall (s) | Child CPU (s) | Exit |
|---|---|---:|---:|---:|
| primary | `/workspace/physical-time-consistency/.venv/bin/python -m poc.calibration_study collect --matrix hybrid --output-root /private/tmp/physical-time-consistency-hybrid-component` | 32.254756 | 31.752349 | 0 |
| primary | `/workspace/physical-time-consistency/.venv/bin/python -m poc.calibration_study decisions --matrix hybrid --output-root /private/tmp/physical-time-consistency-hybrid-component` | 2521.418444 | 2518.483784 | 0 |
| primary | `/workspace/physical-time-consistency/.venv/bin/python -m poc.calibration_study predict --matrix hybrid --output-root /private/tmp/physical-time-consistency-hybrid-component` | 721.075508 | 718.762746 | 0 |
| primary | `/workspace/physical-time-consistency/.venv/bin/python -m poc.calibration_study prepare --matrix hybrid --output-root /private/tmp/physical-time-consistency-hybrid-component` | 0.822758 | 0.812755 | 0 |
| primary | `/workspace/physical-time-consistency/.venv/bin/python -m poc.calibration_study report --matrix hybrid --output-root /private/tmp/physical-time-consistency-hybrid-component --destination /workspace/physical-time-consistency/reports/hybrid_component` | 528.576981 | 528.363982 | 0 |
| primary | `/workspace/physical-time-consistency/.venv/bin/python -m poc.calibration_study time --matrix hybrid --output-root /private/tmp/physical-time-consistency-hybrid-component` | 519.487687 | 519.056590 | 0 |
| primary | `/workspace/physical-time-consistency/.venv/bin/python -m poc.calibration_study train --matrix hybrid --output-root /private/tmp/physical-time-consistency-hybrid-component` | 2605.729381 | 2590.003055 | 0 |
| primary | `/workspace/physical-time-consistency/.venv/bin/python -m poc.calibration_study verify --matrix hybrid --output-root /private/tmp/physical-time-consistency-hybrid-component` | 526.155614 | 526.068023 | 0 |
| replica | `/workspace/physical-time-consistency/.venv/bin/python -m poc.calibration_study collect --matrix hybrid --output-root /private/tmp/physical-time-consistency-hybrid-component-replication --session 1` | 9.866318 | 9.772617 | 0 |
| replica | `/workspace/physical-time-consistency/.venv/bin/python /workspace/physical-time-consistency/reports/hybrid_component/replication_driver.py /private/tmp/physical-time-consistency-hybrid-component /private/tmp/physical-time-consistency-hybrid-component-replication` | 89.556936 | 89.529759 | 0 |
| replica | `/workspace/physical-time-consistency/.venv/bin/python -m poc.calibration_study prepare --matrix hybrid --output-root /private/tmp/physical-time-consistency-hybrid-component-replication` | 0.683380 | 0.650189 | 0 |
| replica | `/workspace/physical-time-consistency/.venv/bin/python -m poc.calibration_study train --matrix hybrid --output-root /private/tmp/physical-time-consistency-hybrid-component-replication --session 1 --model hybrid_affine_aware --budget 30 --initialization 0` | 34.638185 | 34.320666 | 0 |
| replica | `/workspace/physical-time-consistency/.venv/bin/python -m poc.calibration_study train --matrix hybrid --output-root /private/tmp/physical-time-consistency-hybrid-component-replication --session 1 --model hybrid_affine_flow --budget 30 --initialization 0` | 33.238585 | 32.623803 | 0 |
| replica | `/workspace/physical-time-consistency/.venv/bin/python -m poc.calibration_study train --matrix hybrid --output-root /private/tmp/physical-time-consistency-hybrid-component-replication --session 1 --model hybrid_affine_rk4 --budget 30 --initialization 0` | 42.904722 | 42.617039 | 0 |
| replica | `/workspace/physical-time-consistency/.venv/bin/python -m poc.calibration_study verify --matrix hybrid --output-root /private/tmp/physical-time-consistency-hybrid-component-replication` | 2.773983 | 2.715152 | 0 |

Final primary raw verify has complete coverage and no invalid cells. Separate replica verify has no invalid products and preserves its intentional partial scope. The archived export checker verifies exact source/runtime/recipe/spec bindings, exported hashes, gate equality, all 48 CSV coordinates/metric values and local links; line ranges are counted with splitlines, treating CRLF as one newline. The artifact manifest excludes itself. Independent final artifact review and static/scratch archival belong to the controller; its review directory is preserved separately and is not part of this executor commit.

Portable reproduction commands are in [the frozen CLI reproduction export](short_calibration_reproduce.md). Exact support drivers are [serial CLI orchestration](run_study.py), [fresh numerical replay](replication_driver.py), [numerical process orchestration](run_numerical.py), [resource/archive audit](audit_driver.py), [export verification](verify_exports.py) and [this report renderer](write_execution_report.py). Raw roots remain `/private/tmp/physical-time-consistency-hybrid-component` and `/private/tmp/physical-time-consistency-hybrid-component-replication`; arrays, weights and raw rows stay external and preserved.

Three independent noiseless known-lag simulator sessions are exploratory, not powered confirmation or customer tolerances. Optimization limits, invalid sampled selections, unresolved paired comparisons, model-capacity differences, independent numerical envelopes, timing on one frozen representative bank and the bounded optimizer recipe all constrain interpretation. The validated negative primary outcome ends active investment without deleting scientific artifacts or denying secondary prediction/runtime improvements. No hardware/deadline/feedback-controller/confirmation claim follows.
