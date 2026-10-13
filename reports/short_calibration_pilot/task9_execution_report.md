# Task 9 empirical execution report

Status: the complete frozen 21-fit bounded pilot and prescribed fresh-root numerical replication completed. Development has not been prepared or executed. Root prospectively closed execution at the pilot after verified null and fresh replication, before development preparation. The durable scope amendment is `docs/superpowers/plans/2026-10-12-short-calibration-pilot-scope-amendment.md` (commit `2a4c8ba`). Remaining 174 fits and full development/C/D remain unexecuted. This is a scientific investment decision with adequate resources, not a resource block. Artifact review is root-owned and follows this report.

## Frozen preparation and commands

User execution authorization superseded the original documentation-only note. Work stayed on `continue`, with no worktree/branch/push, source/asset/dependency change, lowered cap, replacement draw or helper agent. Native `.venv/bin/python` used the frozen deterministic single-thread CPU CLI, float32 fitting and float64 scoring. Scientific revision: `c75fc323832bc7543ed1812ec47ff4d9969e0a30`. Portable recipe: `config/short_calibration_protocol.json`; all frozen preflight bindings/spec/plan/authority hashes are in `reports/short_calibration_execution.md` and `reports/short_calibration_pilot/preflight_record.json`.

Primary raw root: `/private/tmp/physical-time-consistency-short-calibration`. Exact command sequence: `.venv/bin/python -m pip check`; then `.venv/bin/python -m poc.calibration_study {prepare,collect,train,predict,decisions,time,report,verify} --matrix pilot --output-root /private/tmp/physical-time-consistency-short-calibration`, each phase serial. Report added `--destination reports/short_calibration_pilot`. Original command timestamps/monotonic durations/CPU/exit codes are archived in `primary_command_records.json`. Main session receipt is `fa89db97974ee49a2e111c18cac26ada60b74fd535dd8972af4445beb37195ac`, manifest `8d49f65a38687f586161d290d88c9b87fbc855ac80de8df0651d414d84dfc172`.

## Actual coverage and selection

Report and independent raw verification exited zero. Coverage: 21 finite fits, 3 nominal references, 24 prediction products, 6 paired same-weight propagation products, 24 decision products and one exact complete-matrix pooled timing request (672 groups, 33,600 measured samples). There are no missing, invalid or failed attempts. One capped 30-second actuator-aware NODE remains optimization-limited at selected epoch 1000/rate 1. Full selected epochs, validation dictionaries, both-rate attempts and receipt hashes are archived in `pilot_fit_selection.json`; neural and physical selected objectives are different fitting objectives, so comparative claims use common held-out physical metrics.

Prediction uses all 1,472 declared warm primary anchors and all secondary/cold/command-path cases. Candidate decisions cover 64 warm anchors ×256 paths, all four partitions, with every truth path audited at 250/125/62.5us. All 16,384 candidate feasibility statuses resolve, zero empty banks, zero historical feasibility disagreements. Eleven additional hash-bound prediction refinement products remain retained. Truth-bank hash: `e3123a2801fae5b561a01056d86e2997b1e5b0745e981690a32ba4f0a2375587` (see authoritative result for the full receipt and payload inventory).

## Outcome and limits

Frozen gate: `no_advantage`, reason `no_earlier_budget_or_faster_equal_screen_without_decision_loss`. All 96 configs resolved: 61 pass, 35 fail. At 30 seconds, f/B Flow and RK4, fitted physics and physical residual all pass combined screens; nominal fails. Passing 30 seconds is interval censored `(0,30]`, not an exact minimum. Structured NODE first passes at 600 seconds, giving qualified representation evidence against that comparator. Residual has no worse q/v RMSE/p95, conservative regret, failure fraction and scorer median/p95 than Flow at 20/40 ms at all three budgets. At 30 seconds/40ms, Flow versus residual: q RMSE 0.00264818584/0.00121162837 rad; v RMSE 0.0296816181/0.013435129 rad/s; median 41.385/29.375 ms, zero decision failures each. Physical fit is much more accurate but slower; it alone did not trigger `no_practical_headroom`. Complete screens/frontiers and same-budget comparisons remain in `pilot_execution_findings.md`, `pilot_screen_frontier.json` and the unchanged full raw-recomputed exports.

One session/one neural initialization supports a provisional simulator pilot, with no powered confirmation, hardware commissioning, continuous/global-optimal feasibility, automatic adaptation, deadline scheduler or larger-arm claim. Stage C did not launch.

## Resources and accounting

Audit: 132 primary producer/rate ledgers, all completed with no unknown tails, interruptions, failed events or stderr. N1 remains parked: a caught interruption during post-commit resource-observation persistence can double-count persistence. No such path occurred in these records; no commissioning-compute advantage is claimed. Top-level parent sums exclude 11 nested producers:85 nonoverlapping attempts total 6130.002149 wall/6112.696257 CPU seconds. Fit producer totals 2763.467470 wall/2754.627613 CPU. Timing finalization separately 0.047070 wall seconds; observation and post-producer finalization exclusions remain explicit.

Additional 174 fits project 6.819209 serial hours. Additional producer work projects 9.707946 hours, including a new complete 210-coordinate development timing request. Requested-development workload projects 11.251894 hours; cumulative campaign including retained pilot timing 11.410724 hours, plus future setup/report/verify/conditional refinement. This is approximate observed-work scaling, not guaranteed elapsed completion. Resource headroom is adequate: 753,982,391 raw bytes; actual lossless streams 518,604,389 compressed / 4,274,479,414 decoded bytes (ratio 8.242274); cumulative retained-artifact projection 6,482,875,291 bytes; raw free 60,740,956,160 bytes. Overall native high-water RSS 790,642,688 bytes is not a per-model memory result.

Training UTC subtraction differs from monotonic duration by 2214.692982 seconds; cause unverified. Costs use recorded monotonic/process CPU, never calendar subtraction. Resource audit/projection mathematics, exact executed audit bytes and transparent supplemental arithmetic are preserved in `pilot_resource_projection.json`, `audit_driver.py` and `support_provenance.md`.

## Fresh independent numerical replication

Fresh root: `/private/tmp/physical-time-consistency-short-calibration-replication`. Separate CLI processes prepared/collected the unchanged full pilot recipe, then fit session 0/budget 30/init 0 for `affine_rk4`, `node_actuator_aware`, `physics_residual`, each with unchanged own frozen solver/caps. Command wall seconds 51.735433, 43.603443, 52.738413; selected epochs 680, 1000, 545, with the same actuator-aware limit flag. This partial replica matrix claims three fits, not complete pilot coverage.

Exact executed `replication_driver.py` ran after those fits in a fresh process. All selected canonical hashes, selected tensor values and validation dictionaries matched; regenerated session/evaluation arrays and candidate banks matched. Each model reproduced 108 endpoint component coordinates at states 0/1534/1535 and 2048 candidate coordinates at states 0/1512 (all 4 partitions), with maximum output/cost differences 0. Prediction/status/errors/truth/envelopes and candidate feasibility/selection/regret/work/query logs were compared. Designated independent truth: 48 first/last-trajectory states ×15 cases plus 512 candidate paths, each replayed at 250/125/62.5us; all values exact. Maximum cross-resolution discrepancies: prediction2.04281e-12 then1.48770e-13; decision cost1.15641e-12 then6.21725e-14. Replica replay 95.807505 wall seconds / 95.160868 CPU, completed clean ledger; 14 replica producer/rate ledgers clean, all stderr empty. Replica resources are excluded from primary commissioning estimates. Evidence: `numerical_replication.json`, `replication_execution.json`, `numerical_replication_resources.json`, exact orchestration/stage drivers.

## Verification and commits

The unchanged-source software readiness evidence is 1,390 tests passed in 334.55 seconds; `pip check` clean. Per the explicit handoff/root ruling it was not rerun after unchanged empirical execution. Final primary raw verify closed complete; replica process exited 0 with `all_equal=true`; focused new-export JSON/coverage validation passed; final source/asset/runtime/recipe bindings match both prepared stores (declared output roots differ); historical source/results diff empty; `git diff --check` passed.

Commits: `77623d5` preparation/protocol/provenance; root `60845d6` prior implementation audit; `aa3f561` verified pilot/export/frontier/resource projection. Root `2a4c8ba` records the prospective scope amendment. The final replica/archive/checklist/trajectory commit has subject `research: report short-calibration pilot replication and bounded null`; its hash is returned with the final task status. Root dispatches artifact review after this report; no duplicate software review was started.

## Execution checklist

- [x] Pre-expensive readiness, environment and frozen bindings recorded.
- [x] Full default pilot prepare/collect, all 21 fits and scoring/truth/timing/report/raw verify completed.
- [x] Pilot reports and honest resource projection committed before expansion.
- [x] Three prescribed fresh-process numerical replicas and designated truth checks completed.
- [ ] Full development prepare/collect/additional 174 fits/complete 210-coordinate pooled timing/development report and verify: explicitly unexecuted/deferred under `2a4c8ba`; no full development claim.
- [x] Final bounded-pilot scope ruling, durable task report/trajectory/checklist export; final commit subject recorded above.
- [ ] Root artifact review (root-owned subsequent step).

The stop risks missing an advantage in other calibration draws or below the lowest 30-second budget. Numerical replication uses the same declared session seed and is not an additional independent cluster. Renewed work requires a prospective acquisition/scoring declaration. Original frozen spec/plan hashes remain chronology; amended documentation is separately inventoried.
