# Compute-matched NODE follow-up

AffineFlow has a supported controller-cost advantage against every tested NODE option at a matched allocation.

The controller tries motor commands in a learned model, chooses a plan, and applies its first 40 milliseconds to the simulated arm. AffineFlow is our model; NODE learns continuous acceleration and advances it numerically. The cost score combines target error, speed and commanded torque. Lower is better. This comparison gives the models roughly the same CPU planning time.

Complete paired matrix: 4560 episodes, 38 training clusters, three neural models at 5/40 ms. Hardware sensing, planning and command holds remain 40 ms. Budgets were selected from timing before independent task outcomes.

| Model | Query | Candidates × iterations | Mean cost | Tracking RMSE | Successes | Mean episode planner median |
|---|---:|---:|---:|---:|---:|---:|
| affine_flow | 5 ms | 224 × 4 | 0.286100 | 0.218467 rad | 537/760 | 0.2325 s |
| affine_flow | 40 ms | 832 × 4 | 0.282965 | 0.217797 rad | 742/760 | 0.2182 s |
| node_quadrature | 5 ms | 272 × 4 | 0.286717 | 0.218329 rad | 587/760 | 0.2342 s |
| node_quadrature | 40 ms | 832 × 4 | 0.286012 | 0.217627 rad | 746/760 | 0.2154 s |
| node_structured | 5 ms | 128 × 4 | 0.288871 | 0.218736 rad | 386/760 | 0.2362 s |
| node_structured | 40 ms | 672 × 4 | 0.284978 | 0.217662 rad | 738/760 | 0.2169 s |

Actual episode planning means remain within the 10% matching tolerance: True. All timing ratios and p95 measurements are retained in the machine result. No physical wall-clock delay was injected.

| Affine query | NODE comparator | Cost reduction | Central 95% interval | Holm-supported gain? |
|---|---|---:|---:|---|
| 5 ms | node_quadrature, 5 ms | 0.22% | -0.01% to 0.44% | False |
| 5 ms | node_quadrature, 40 ms | -0.03% | -0.35% to 0.28% | False |
| 5 ms | node_structured, 5 ms | 0.96% | 0.65% to 1.28% | True |
| 5 ms | node_structured, 40 ms | -0.39% | -0.69% to -0.10% | False |
| 40 ms | node_quadrature, 5 ms | 1.31% | 1.10% to 1.52% | True |
| 40 ms | node_quadrature, 40 ms | 1.07% | 0.86% to 1.27% | True |
| 40 ms | node_structured, 5 ms | 2.04% | 1.78% to 2.32% | True |
| 40 ms | node_structured, 40 ms | 0.71% | 0.45% to 0.96% | True |

No percentage improvement floor is required. The eight prospectively declared cost comparisons share Holm correction at the recorded look alpha. A positive point estimate with an interval crossing zero remains unresolved. Success is the strict final-200-ms criterion and is retained separately from the weighted cost. A narrow non-detection is not equivalence.

## Prediction and isolated inference

| Comparator | Probe group | Component | Affine error | NODE error | Reduction at 5 ms |
|---|---|---|---:|---:|---:|
| node_quadrature | cold | q | 3.49959e-06 | 7.9831e-06 | 56.16% |
| node_structured | cold | q | 3.49959e-06 | 6.27224e-06 | 44.21% |
| node_quadrature | cold | v | 0.00149901 | 0.00307125 | 51.19% |
| node_structured | cold | v | 0.00149901 | 0.00240458 | 37.66% |
| node_quadrature | warm | q | 2.78193e-06 | 3.88979e-06 | 28.48% |
| node_structured | warm | q | 2.78193e-06 | 2.64e-06 | -5.38% |
| node_quadrature | warm | v | 0.00116938 | 0.00155252 | 24.68% |
| node_structured | warm | v | 0.00116938 | 0.00105129 | -11.23% |
| node_quadrature | all | q | 3.14076e-06 | 5.93644e-06 | 47.09% |
| node_structured | all | q | 3.14076e-06 | 4.45612e-06 | 29.52% |
| node_quadrature | all | v | 0.00133419 | 0.00231188 | 42.29% |
| node_structured | all | v | 0.00133419 | 0.00172794 | 22.79% |

These secondary rows retain all cold and warm states; negative reductions mean AffineFlow is worse. Position uses radians and velocity rad/s. Absolute errors, finite command response and solver discrepancy are available for 5,10,40 ms. They are not proof of universal model superiority.

| Comparator | Batch | Affine µs | NODE µs | Inference-time reduction at 5 ms |
|---|---:|---:|---:|---:|
| node_quadrature | 1 | 331.47 | 187.55 | -76.74% |
| node_structured | 1 | 331.47 | 307.98 | -7.63% |
| node_quadrature | 128 | 666.08 | 516.02 | -29.08% |
| node_structured | 128 | 666.08 | 956.00 | 30.33% |

Quadrature NODE and AffineFlow both use two network evaluations; RK4 NODE uses four. Quadrature uses the same 17,922-parameter acceleration network and seed initialization as RK4; AffineFlow has 18,182 parameters and a torque-affine prior. The comparator was trained through its own solver using the same fixed data, objective, nominal-only optimizer opportunity and selection rule. Training time must be read beside nominal fitting error, rather than called equal-accuracy speed.

| Model | Mean CPU minutes per fit | Mean nominal objective | Epoch range |
|---|---:|---:|---:|
| affine_flow | 5.31 | 0.00350669 | 100–700 |
| node_quadrature | 5.01 | 0.00995902 | 100–700 |
| node_structured | 8.68 | 0.00295939 | 100–800 |

Measured workload per planning decision (40 decisions per episode):

| Model | Query | Batched network calls | Neural rows (calls × batch) | Candidate transitions |
|---|---:|---:|---:|---:|
| affine_flow | 5 ms | 384 | 86016 | 43008 |
| affine_flow | 40 ms | 48 | 39936 | 19968 |
| node_quadrature | 5 ms | 384 | 104448 | 52224 |
| node_quadrature | 40 ms | 48 | 39936 | 19968 |
| node_structured | 5 ms | 768 | 98304 | 24576 |
| node_structured | 40 ms | 96 | 64512 | 16128 |

New sealed jobs consumed 23.534 CPU hours and 23.635 recorded job-wall hours. Saved artifacts occupy 42.233 GiB. Maximum recorded process peak RSS is 979.1 MiB; it is a process high-water mark, not a reset per-job allocation. Previously trained AffineFlow/RK4 fits are listed separately above. Raw effort (squared torque integral, not energy), velocity RMS, terminal error and p95 planning times remain in the machine table.

## Reproduction and evidence

Fresh-process control replay and finer-physics subset: 912 checks pass. Two new-comparator training replications reproduce exact selected tensors. Independent arithmetic recomputes every episode cost; maximum difference 2.22e-16. Empirical cost envelope: 2.11e-13. Zero intrinsic fallbacks: True.

New raw artifacts are on the owner-authorized Data volume, with explicit root, content-addressed jobs, source/model/data bindings and complete-file hashes. Earlier results and frozen producers remain intact. A source-only Git checkout needs the indexed artifact store and original checkpoints to reproduce the measurements.

Machine results: [compute_study_result.json](compute_study_result.json); practical table: [compute_study_metrics.csv](compute_study_metrics.csv); saved simulation replay: [compute_study_demo.html](compute_study_demo.html). Exact commands and scope are in [compute_study_reproduce.md](compute_study_reproduce.md).

Scope: one fully observed smooth known-lag arm, offline CPU MPC and this training corpus. No contact, unknown motor dynamics, physical robot, or stronger-than-all-calibrated-physics claim. A larger robot is not justified merely to rescue a null result.
