# Verified bounded pilot: execution and expansion assessment

The complete 21-fit pilot resolves the frozen provisional simulator gate to `no_advantage`: `no_earlier_budget_or_faster_equal_screen_without_decision_loss`. One calibration session and one neural initialization per budget/method support this bounded pilot; the five-session, two-initialization development matrix has not been prepared or run. The three prescribed fresh-root numerical replicas closed with exact selected-tensor and output equality; scope adjudication is recorded separately below.

Raw verification closed complete coverage with 21 finite fits, three unfitted nominal references, 24 prediction products, six same-weight propagation products, 24 decision products and 672 timing groups in one pooled timing request (33,600 measured samples). There are no missing/invalid products or failed attempts. The 30-second actuator-aware NODE remains optimization-limited at the unchanged cap.

## All combined screens

Each cell lists the 30/120/600-second combined prediction, feasibility and regret screens in that order (`P` pass, `F` fail, `U` unresolved). The primary evaluation has 1,472 naturally moving anchors; decisions use the declared 64 initial warm anchors and 256 candidate paths each.

| Method | 5 ms | 10 ms | 20 ms | 40 ms |
|---|---|---|---|---|
| affine_flow | P/P/P | P/P/P | P/P/P | P/P/P |
| affine_rk4 | P/P/P | P/P/P | P/P/P | P/P/P |
| node_structured | F/F/P | F/F/P | F/F/P | F/F/P |
| node_quadrature | F/F/P | F/F/P | F/F/P | F/P/P |
| node_actuator_aware | F/F/P | F/F/P | F/F/P | F/F/P |
| physics_fit | P/P/P | P/P/P | P/P/P | P/P/P |
| physics_residual | P/P/P | P/P/P | P/P/P | P/P/P |
| physics_nominal | F/F/F | F/F/F | F/F/F | F/F/F |

There are 61 passing, 35 failing and zero unresolved configurations. Passing at 30 seconds gives an interval-censored threshold `(0, 30]`, rather than an exact minimum calibration requirement. Both f/B methods pass at 30 seconds; the plain structured NODE first passes at 600 seconds. That is pilot evidence for representation against that comparator. Fitted physics and the physics residual also pass at 30 seconds, so it establishes no data-budget advantage over the strongest prior-using alternatives. The actuator-aware NODE first passes at 600 seconds.

## Strongest comparisons and complete frontiers

The physics residual improves q/v RMSE and p95, conservative regret, median and p95 scorer time versus AffineFlow at the same 30/120/600-second budgets and 20/40-ms main partitions, with equal zero decision failures. All six comparisons are recorded in `pilot_screen_frontier.json`; all 96 configurations, including failures, are preserved in the raw-recomputed result and CSV.

| Budget s | Main step ms | Method | q RMSE rad | v RMSE rad/s | Regret upper | Scorer median ms | Scorer p95 ms |
|---:|---:|---|---:|---:|---:|---:|---:|
| 30 | 40 | affine_flow | 0.00264818584 | 0.0296816181 | 0.00184630803 | 41.385083 | 44.912303 |
| 30 | 40 | physics_residual | 0.00121162837 | 0.013435129 | 0.00148329295 | 29.375020 | 31.073877 |
| 120 | 40 | affine_flow | 0.000644961238 | 0.00890924031 | 6.54270034e-07 | 41.434958 | 44.204108 |
| 120 | 40 | physics_residual | 0.000402121566 | 0.00458638339 | 1.69204169e-13 | 29.460542 | 31.652598 |
| 600 | 40 | affine_flow | 0.000357166144 | 0.00734852329 | 0.00148394722 | 42.288000 | 45.197021 |
| 600 | 40 | physics_residual | 0.000173117873 | 0.00200334202 | 6.54270034e-07 | 29.890729 | 32.986883 |

Fitted physics is extraordinarily accurate in this fixed noiseless plant: at 30 seconds its q/v RMSE is approximately `1.094234e-7` rad / `1.261094e-6` rad/s. It selects the oracle actions to numerical resolution, but its full scorer is slower: median 90.590 ms at a 5-ms partition and 248.827 ms at 40 ms. Physics alone therefore did not trigger the frozen `no_practical_headroom` gate. The residual provides the stronger quality/runtime comparison. Nominal unfitted physics fails the combined screens.

The full screen-passing Pareto frontier within every declared main partition is exported to `pilot_screen_frontier.json`, using budget, complete-scorer median, q/v RMSE, failures and conservative regret. It does not alter the gate or select experimental coordinates. AffineFlow is faster than its own-representation RK4 comparator, while the residual removes a unique practical advantage in this pilot. No representation-only contribution survives the strongest complete frontier under the frozen gate.

## Truth and numerical qualification

Prediction truth used 250/125/62.5 us and the designated first/last trajectories, with 11 additional hash-bound refinement products where the declared 1% rule requested them. Independent candidate truth audited every one of 16,384 paths at those three resolutions. All candidate feasibility statuses resolved; there are zero empty feasible banks or historical feasibility disagreements. All candidates ended at the 62.5-us audited resolution without additional refinement. This certifies the declared sampled-feasibility evaluation, with dense diagnostics retained, rather than globally optimal or continuously feasible control.

## Actual resources and prospective expansion

The audit examined all 132 primary producer/rate ledgers and every closed execution/error log: all attempts completed, no unknown tails, interruptions, failed events or stderr. The parked N1 interruption/persistence-accounting defect remains in frozen software; this run did not exercise its known failure path. Inclusive rate/child spans were not added to parents. The raw-recomputed nonoverlapping totals are 6,130.002149 wall seconds and 6,112.696257 CPU seconds across 85 top-level attempts; 11 nested producers are excluded from that sum. Resources exclude ledger observations and declared post-producer finalization. Timing finalization is separately measured at 0.047070 wall seconds.

| Producer category | Pilot monotonic wall s | Projected requested-development wall s |
|---|---:|---:|
| collect | 8.482543 | 42.412715 |
| fit | 2763.467470 | 27312.619862 |
| predictions | 198.473036 | 1620.509698 |
| propagation | 85.354102 | 853.541021 |
| decision_truth | 1996.506904 | 1996.506904 |
| decisions | 473.156218 | 3645.294574 |
| timing | 571.788633 | 5003.150542 |

The remaining 174 fits project to 6.819209 additional serial fitting hours. All additional producer work projects to 9.707946 hours, including a new complete development timing request; the original pilot timing remains retained. Requested-development producer totals project to 11.251894 hours; cumulative pilot-plus-development producer work projects to 11.410724 hours. Reporting/raw verification and CLI setup/finalization are separate additional costs. These estimates reuse the observed pilot stopping/work per coordinate; new sessions/initializations and adaptive truth requests can differ substantially. They are workload projections, not guaranteed completion times.

Actual raw files occupy 753,982,391 bytes. Prediction/candidate gzip streams occupy 518,604,389 bytes and decode losslessly to 4,274,479,414 bytes: an empirical ratio of 8.242274. Cumulative retained artifact projection is approximately 6482875291 bytes, before new conditional refinement/report/driver overhead. The raw volume has 60,740,956,160 free bytes and the repository 2,830,254,080; storage headroom is adequate. Native process high-water RSS reached 790,642,688 bytes, not a per-model memory measurement.

UTC subtraction for training differs from its monotonic command timer by 2,214.692982 seconds. The cause is unverified. Resource projections use recorded monotonic and process-CPU durations, never UTC subtraction. Full training CLI wall was 2,790.735497 seconds; its full child CPU was not separately captured, while fit-producer CPU was measured. Subsequent closed CLI wall/CPU measurements and the full audit are retained in `pilot_resource_projection.json`.

## Scope and remaining work

This is a provisional simulator pilot, with one independent calibration cluster and no powered confirmation. No deadline scheduler, controller, shared compiled backend, hardware commissioning, automatic payload adaptation or larger arm experiment has run. Independent fresh-root replicas for affine RK4, actuator-aware NODE and physics residual completed. Their selected canonical hashes, every selected tensor and validation dictionary matched exactly. Each reproduced 108 endpoint component coordinates and 2,048 decision-candidate coordinates, including statuses, errors, truth envelopes, feasibility, selection, regret and work/query logs, with maximum numerical difference zero. The first/last designated three-resolution prediction and decision truth audits also matched exactly. Replica resources are separate from primary commissioning estimates. Root closed this task at the bounded pilot after exact replication under the [prospective scope amendment](../../docs/superpowers/plans/2026-10-12-short-calibration-pilot-scope-amendment.md), before any development preparation. The remaining 174 fits, full 195-fit matrix, deadline and larger-arm stages are explicitly unexecuted. The decision uses the decisive frozen pilot null with adequate resources; it does not establish a cross-session null and can miss benefits in other draws or below 30 seconds.

The frozen source revision is `c75fc323832bc7543ed1812ec47ff4d9969e0a30`; protocol/spec/recipe/source/runtime hashes and exact commands are in `../short_calibration_execution.md` and `../../config/short_calibration_protocol.json`. Earlier implementation audits/rulings are preserved in `../../docs/audits/short-calibration-implementation.md`.
