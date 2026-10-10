The first paired learned-control task is promising. On direct torque, the corrected power head had lower realized tracking cost than both tested conventional predictors at both internal prediction steps. This is exploratory evidence from one fixed task and training seed 9000; it does not establish a population control advantage or isolate the scaling intervention.

| Model | 40 ms internal step | 5 ms internal step |
|---|---:|---:|
| MLP | 0.05360962 | 0.12317710 |
| Generic NODE | 0.04445603 | 0.04445603 |
| Corrected power head | 0.04054579 | 0.04264654 |

Lower cost is better. At 5 ms the head reduced cost by 4.0703% versus the generic NODE and 65.3779% versus the MLP. At 40 ms the reductions were 8.7957% and 24.3684%. The head's own cost rose by 5.1812% under refinement. The generic NODE retained identical cost, while the MLP's cost rose by approximately 130%. All six episodes completed without fallback.

All runs used the original fixed task, six torque knots, the independently qualified common CEM budget, 40 ms observation/replanning/actuation, and the qualified 0.25 ms plant step. Only the internal model partition changed. Each model had a qualified nominal fit on augmented transitions and the shared one-plus-four-step objective. Checkpoints were pinned before control outcomes.

The raw manifest pointers and measured resources are in [control_representative_measurements.json](control_representative_measurements.json). [control_representative_comparisons.json](control_representative_comparisons.json) recomputes paired costs from verified episodes and explicitly carries no confidence bounds. The remaining nine fixed tasks will test whether this encouraging first comparison persists. Fixed-weight causal attribution, lagged learned-control results, independent seed confirmation and robustness remain unmeasured.
