# Affine-flow refinement: prediction gains, no established control gain

The revised method has supported short-horizon prediction and batched-inference gains over the strongest neural comparator tested here, the mechanics-aware NODE. The complete 440-episode controller comparison does **not** establish better control: cost advantages are below 1%, their uncertainty includes no improvement, and NODE succeeds on more tasks. Independent reproduction passes. The full 1,010-test suite and fresh final review pass. This targeted study is complete; it is a separate refinement of the original power-head method.

On the same known-lag two-link robot, eleven independent training pairs were evaluated on 512 fresh physical states/commands, with 256 cold and 256 warm probes. Each row below is a mean of equally weighted per-training-seed statistics. The training data and probe library are fixed; the uncertainty concerns independent model fitting on that corpus.

| At 5 ms | Affine-flow | Structured NODE | Improvement | Central 95% paired Fieller interval |
|---|---:|---:|---:|---:|
| Position error | 2.928249e-06 rad | 4.996281e-06 rad | 41.4% lower | 29.6% to 51.3% |
| Velocity error | 0.001248619 rad/s | 0.001929835 rad/s | 35.3% lower | 22.1% to 46.3% |
| Inference, batch 1 | 351.4 µs | 318.2 µs | -10.5% reduction | -12.7% to -8.2% |
| Inference, batch 128 | 688.1 µs | 976.3 µs | 29.5% reduction | 27.4% to 31.6% |

Positive reduction means affine-flow is better; single-query inference is slower. The three-component primary claim requires positive position-error, velocity-error and batch 128 speed contrasts together. All three pass the preregistered look 3 one-sided alpha 0.00625 checks, including numerical envelopes and leave-one-seed-out stability. No seed was removed and no percentage superiority floor was used. The table's central95% ratio intervals are descriptive uncertainty intervals, rather than simultaneous bounds for every reported row.

The state breakdown limits the averaged gain. Cold probes use random initial positions, velocities and activations; warm probes come after at least 320 ms of random held commands. Both predictors are stateless and receive the same complete current physical state. At 5 ms:

| Probe distribution | Component | Affine-flow mean error | NODE mean error | Affine-flow reduction |
|---|---|---:|---:|---:|
| cold | q | 3.138456e-06 | 7.391394e-06 | 57.5% |
| warm | q | 2.718042e-06 | 2.601169e-06 | -4.5% |
| cold | v | 0.00133568 | 0.002824967 | 52.7% |
| warm | v | 0.001161557 | 0.001034703 | -12.3% |

Negative reduction means affine-flow has higher error. Its cold-state advantage drives the positive equally weighted primary average; NODE is more accurate on warm states. These are descriptive secondary comparisons. They do not support a claim that affine-flow wins in every state regime. Drift and new-command response are retained separately in [the full state breakdown](affine_precision_001_state_breakdown.json); errors are vector norms and do not add arithmetically.

The new-command response comparison measures how much a newly chosen torque changes the predicted endpoint, holding the initial activation and state fixed. These descriptive5 ms averages favor affine-flow in both state groups:

| State group | Position-response error reduction | Velocity-response error reduction |
|---|---:|---:|
| cold | 85.4% | 82.9% |
| warm | 29.5% | 16.5% |

This is a finite held-command effect comparison, rather than a local action Jacobian or an isolated change of exponents. Improved command response can coexist with worse background-motion prediction, as the warm-state results show. No new selected significance claim is made for these secondary rows.

The absolute position errors are only a few millionths of a radian. A large relative improvement here may make little difference to a feedback controller. The completed development comparison showed percent-level gains and losses; the new independent control experiment measures that practical question directly.

Both models receive the same positions, velocities, observed activation, known 80 ms motor time constant, training labels, objective, controller and candidate streams. Affine-flow has 18,182 parameters and two field-network calls; NODE has 17,922 and four RK4 field calls. Affine-flow assumes instantaneous acceleration is linear in torque and integrates the known motor response analytically. This is a revised learned acceleration model, not the original same-weights power intervention.

Actual training costs under the common nominal convergence policy:

| Model | Mean CPU time per qualified fit | Median | Epoch range | Mean nominal validation objective |
|---|---:|---:|---:|---:|
| Affine-flow | 6.98 min | 6.18 min | 100–700 | 0.003267 |
| Structured NODE | 10.68 min | 8.45 min | 100–600 | 0.002545 |

These are measured fitting costs with nominal-only checkpoint selection. NODE achieves the better mean nominal validation objective. This is not an equal-accuracy training-speed comparison, and neither model was forced to stop at the other's epoch count.

The practical experiment covers 440 paired episodes: both models at 5 and40 ms internal queries, eleven training pairs and ten independently drawn tasks per pair. Sensing, replanning and actual command holds stay 40 ms; the same six torque knots and CEM 128×4 search are used throughout. Coarse NODE and coarse affine-flow are both included, so either method can benefit from cheaper queries.

| Controller | Mean realized cost | Tracking q RMSE, rad | Mean terminal q error, rad | Successful tasks | Mean episode-median planning time |
|---|---:|---:|---:|---:|---:|
| Affine-flow,5 ms | 0.234749 | 0.196600 | 0.010741 | 52/110 | 2.119 s |
| NODE,5 ms | 0.236093 | 0.196557 | 0.010456 | 61/110 | 2.175 s |
| Affine-flow,40 ms | 0.234597 | 0.196701 | 0.011109 | 54/110 | 0.673 s |
| NODE,40 ms | 0.236304 | 0.196586 | 0.010761 | 66/110 | 0.684 s |

All 440 primary episodes completed with zero all-candidate-failure fallbacks. Success requires joint-position error below 0.1 rad and velocity below 0.2 rad/s at every accepted physics step of the final 200 ms. The success counts are descriptive, not an added selected hypothesis test. The weighted cost includes position, velocity, command effort and one terminal penalty; lower cost does not mean every task metric improves.

| Prespecified realized-cost contrast | Affine-flow reduction | Central 95% paired Fieller interval | Holm-supported gain at look 3? |
|---|---:|---:|---|
| Affine5 vs NODE5 | 0.57% | -0.16% to 1.36% | No |
| Affine5 vs NODE40 | 0.66% | -0.07% to 1.43% | No |
| Affine40 vs NODE40 | 0.72% | -0.05% to 1.49% | No |

All three cost contrasts are stable under the influence checks, and all central interval widths are below the preregistered 4-percentage-point reporting-precision limit. This is a bounded near-zero practical finding, rather than a demonstrated gain or an equivalence certificate. No additional seed campaign is needed to describe this outcome reliably at the chosen precision.

The batch 128 network gain translates into only 2.6% less whole-planner time at 5 ms. At 40 ms the whole-planner saving is 1.5%. Both coarse-query controllers plan about 3.2 times faster than their corresponding fine-query versions, with similar measured costs; neither baseline is forced to use an expensive fine clock. Even the coarse controller takes about 0.67 s per 40 ms feedback decision. The simulator waits for planning, so these are offline modeling results, not real-time robot performance.

All 44 finer-physics episodes are complete. Their empirical absolute-cost envelope is 2.29e-13; the result passes the frozen convergence checks. A separately coded arithmetic audit recomputes all 440 realized costs from the saved physical anchors, command effort and single terminal term, with maximum discrepancy 2.22e-16. All 44 saved-weight controller replays match exactly. All four independent training replications reproduce identical selected tensors and validation scores. All 132 prediction replay checks match exactly; all 44 finer-physics comparisons pass. [The complete reproduction record](affine_precision_001_reproduction.json) retains every comparison.

Current evidence: [complete look 3 result](affine_precision_001_result.json), [preserved look2 result](provenance/affine-look-002-result.json), [primary episode pointer](affine_precision_001_primary_batch.json), [finer-physics pointer](affine_precision_001_finer_physics_batch.json), [added raw measurement pointer](affine_precision_001_measurement.json), [independent scalar audit](affine_precision_001_independent_arithmetic.json), [training resources](affine_precision_001_training_resources.json), [prospective precision request](../config/affine_precision_request_001.json). The first-look results remain archived separately.

This evidence concerns one fully observed smooth robot with known motor lag. It does not establish a gain on direct torque, contact, unknown delays or physical hardware. The earlier direct-torque development result was worse and is retained in [the development report](affine_development_findings.md).

Reproduction and exact-resumption commands: [instructions](affine_reproduction_instructions.md). The fitted mechanics comparator remains an additional development reference; this fresh neural comparison does not establish superiority over known-topology calibrated physics.

[The standalone 440-clip simulation replay](affine_fresh_demo.html) shows actual saved motions, costs and measured planner time for every seed/task/configuration. Playback uses physical simulation time; it does not hide the slower offline computation. [The CSV](affine_practical_metrics.csv) and [machine-readable measurements](affine_practical_metrics.json) include terminal error, velocity RMS, command effort, success, median/p95 planning time and producer resource records.

The listed fresh experiment producers consumed 11.85 CPU hours in total, with maximum recorded process RSS of 1.28 GiB. This includes the preserved disk-failed replica export and its resumed completion. It excludes earlier development studies and unlisted reporting/audit overhead; it is not a whole-project resource total. No scientific evidence was deleted to recover space. Work contexts and the installed environment now use the owner-authorized main volume; sealed evidence and archived dependency wheels remain in the repository.

The reproducible conclusion is a prediction/compute tradeoff on this particular known-lag robot. It is useful evidence for batched learned simulation, while the complete feedback experiment provides no established practical controller advantage over NODE. Improving the new-command effect alone does not ensure the best overall motion prediction or control. A broader robot comparison would be a separate experiment; it is not needed to state these completed results honestly.

[Final review, limitations of review and execution rulings](affine_refinement_release_review_2026-10-09.md). The archived execution notes preserve every failed/partial operational attempt and decision; no further work is required by this completed refinement plan.
