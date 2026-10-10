# Physical-time consistency: measured utility results

The corrected discrete head has a supported practical benefit on this simulated two-link robot: when the planner queries dynamics in 5 ms steps, it lowers average tracking cost versus the qualified, time-aware MLP on both direct torque and known motor lag. This is a positive result for the tested implementation and conditions. It does not establish a full PTCO-1.0 PASS or superiority over every conventional predictor.

Status: the ten-seed numerical/statistical audit, all twenty new fresh-process replays and the whole-branch review are complete. The review found no major implementation or evidence defects; public execution-status and checkpoint-reuse disclosures were corrected, and a frozen-renderer presentation issue is deferred in [the review record](control_precision_release_review_2026-10-07.md). This is a limited utility result; the historical PTCO-1.0 verdict remains separate.

## The confirmed comparison

Both models train on the same augmented 20–60 ms transition corpus with the same one-plus-four-step objective and nominal-only checkpoint selection. The robot receives a new observation, replans, and updates its command every 40 ms. The internal model step is 5 ms. Each episode lasts 1.6 s. The same fixed CEM search budget, task draws and random streams are paired across models.

There are ten independent training-seed clusters and twenty paired tasks per seed: 200 episodes per model and actuator, 800 compared episodes in total. All original and added seeds are retained.

The extension selected twenty checkpoint coordinates: fifteen newly trained fits and five reused direct-head fits for seeds 5–9. Those reused fits had previously observed prediction-audit results. Reuse includes the entire eligible ascending cohort; no seed was chosen by a favorable score. The unchanged nominal-only selection and tensor/scaler/source identities were independently verified, and all twenty selections were frozen before opening the new control outcomes. The [nominal registry](../config/control_precision_nominal_jobs.json) and [completion audit](control_precision_nominal_completion_2026-10-06.json) preserve that qualification and disclosure.

| Quantity | Direct torque | Motor lag |
| --- | ---: | ---: |
| MLP mean tracking cost | 0.317404 | 0.379069 |
| Corrected-head mean tracking cost | 0.245728 | 0.280378 |
| Head cost reduction, point estimate | **22.58%** | **26.04%** |
| Seed means favoring the head | 10/10 | 10/10 |
| Qualified signed cost-difference bounds | [0.02915, 0.11420] | [0.04087, 0.15651] |
| Strict task success, head | 40/200 (20%) | 108/200 (54%) |
| Strict task success, MLP | 17/200 (8.5%) | 23/200 (11.5%) |
| Head zero-command fallback events | 0 | 0 |

Positive cost differences favor the head. The percentages are estimates, not lower confidence limits. Both signed bounds support a positive improvement under the prespecified procedure; removing any one seed preserves that interpretation. The cost combines tracking, velocity, command effort and terminal position error; it is not an energy measurement. Strict success requires position and velocity tolerances at every accepted physics step in the final 200 ms. The success counts are descriptive and have no separate population superiority claim.

The [recorded result](control_precision_result.md) and [machine-readable evidence](control_precision_result.json) retain all ten seed means, bounds, numerical envelopes and the six unchanged original comparisons. The [independent reconciliation](control_precision_summary_audit_2026-10-07.json) reopens all 800 actual episode summaries and independently recomputes the signed uncertainty calculation.

## Stronger models and speed

The separate original five-seed control comparison includes a generic neural ODE, or NODE: a network predicts rates of motion, and an integrator turns those rates into the next state. The corrected head predicts the next state directly with explicit powers of elapsed time. These are different implementations of a learned dynamics predictor.

In the original matched 100-episode cells, the head's mean cost is 1.00% higher than the generic NODE with direct torque and 0.13% lower with motor lag. Those comparisons remain statistically unresolved. The data support close point estimates; they do not establish equivalence or superiority. The added MLP/head seeds do not enlarge the NODE comparison.

| End-to-end CPU planning time, original five-seed workload | Direct torque | Motor lag |
| --- | ---: | ---: |
| MLP | 1.918 s | 1.960 s |
| Corrected head | 2.030 s | 1.983 s |
| Generic NODE | 2.058 s | 2.126 s |

These values are medians of per-episode median planning times under the same search budget. The head is descriptively 1.37% faster than NODE on direct torque and 6.71% faster with lag. MLP is fastest: the head takes 5.84% and 1.22% longer, respectively. Two head network passes versus four NODE vector-field calls do not imply twice the overall speed. These offline CPU runs take roughly two seconds per planning decision; real-time execution was not demonstrated. The [original performance report](control_performance_summary_2026-10-06.md) preserves exact timing definitions, target errors, strict-success weaknesses and nominal comparisons.

## What the mechanism evidence adds

A separate forty-seed direct-torque audit finds supported command-sensitivity prediction-error reductions versus the qualified generic NODE of 49.38% on cold probes and 42.90% on warm probes at 5 ms. Cold 10 ms also improves by 17.32%; the 7.87% warm 10 ms estimate remains unresolved. These are Jacobian errors, not control costs. See [the prediction interpretation](mechanism_interpretation.md).

The fixed-weight development intervention preserves nominal 40 ms predictions and improves fine-step control cost by 3.69% direct and 0.49% lagged versus its deliberate low-order mode. That control experiment uses one development seed. It does not establish that the scaling change alone explains the entire confirmed MLP advantage. The structured NODE is substantially better than the fixed-data head in that exploratory test; this finding is retained in [the attribution report](control_attribution_development_result.md).

## Numerical checks and reproduction

The original control suite has complete finer-physics coverage. For the new seeds, all 100 required first-five-task checks at 125 us are complete: the largest relative cost discrepancy is about 3.84e-12, with no success or fallback changes. Both new contrasts qualify under the declared rule and require no full-cell extension. The empirical numerical envelope is included in the signed bounds; it is not a mathematically certified global error bound.

Original reproduction completed sixty selected-weight replays, twelve fresh fits and 480 repeated-policy episodes, with identical selected tensors, nominal scores and control costs. [All twenty new selected-weight replays](control_precision_replay_completion_2026-10-07.json) also passed independent raw/manifest verification, with exactly zero differences in position, velocity, activation, commands and cost. These are internal same-machine reproduction checks, not external laboratory replication. Numerical NODE solver sensitivity remains attached to the original NODE comparisons; no solver-independent field claim is made.

## Statistical integrity and scope

The owner withdrew fixed percentage superiority floors after viewing earlier evidence. The dated amendment is preserved, and the original PTCO gates are unchanged. The utility design tests signed improvement. Its second look spends alpha 0.0125 within the existing eight-comparison family, with two directional tails, 0.00078125 per tail. The rare-tail t rule, central 95% paired seed-bootstrap check, influential-seed check and numerical expansion are all retained. The six untouched comparisons retain five seeds and their original look-1 classifications. Rechecking the same evidence spends no additional alpha.

The inference is conditional on this robot, fixed training corpus and declared workload. Its seed-level t and bootstrap assumptions are approximate. The result does not establish nominal noninferiority, a general NODE advantage, the full seven-competitor suite, changed actuator rates, observation-noise or disturbance robustness, physical robot performance, novelty, or real-time feasibility. Those conditions have not been confirmed by this continuation. The original demanding [PTCO report](final_report.md) and all unfavorable, invalidated and interrupted attempts remain preserved.

## Measured work and verification

The 400 new primary episode producers used 8.83 CPU hours and 1.003 GB of logical artifacts; the 100 initial finer producers used 2.21 CPU hours and 0.276 GB. The replay producer totals are recorded separately in its completion proof. Preparation, training, interpreter startup, independent audits and interrupted attempts are excluded from these producer totals; they are not a complete project resource bill. All arrays and prior attempts are preserved.

From the repository root and the pinned local environment, the existing result can be independently reaggregated without launching training or simulation:

```bash
.venv/bin/python -m poc.control_precision_result --report
```

This checks completed raw evidence and hashes and reuses the same statistical look. Environment/wheel restoration and the historical protocol reproduction instructions remain in the archived project report. Exact requests, selected models, arrays, manifests, logs and look identities are linked from the machine-readable result and the completion proofs.
