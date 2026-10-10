The forty-seed study supports the prediction improvement at 5 ms in both probe groups and at 10 ms for cold probes. Warm-probe improvement at 10 ms remains unresolved. The original 50% gate remains INCONCLUSIVE; under the owner's revised objective, no further seed batch is planned solely to settle that withdrawn cutoff.

This is the unchanged direct-torque comparison: augmented 20–60 ms labels, common one-plus-four-step loss, qualified nominal fits, corrected head versus the prespecified time-aware generic NODE. Seeds 0–39 are paired on the same fixed library of 256 cold and 256 warm probes. These measurements concern command-to-position and command-to-velocity sensitivities, rather than realized robot-control cost.

| Probe group | Horizon | Generic NODE error | Corrected head error | Reduction | Original strict-improvement status |
|---|---:|---:|---:|---:|---|
| Cold | 5 ms | 0.04335073 | 0.02194523 | 49.3775% | Pass |
| Warm | 5 ms | 0.04341334 | 0.02479038 | 42.8969% | Pass |
| Cold | 10 ms | 0.02389433 | 0.01975502 | 17.3234% | Pass |
| Warm | 10 ms | 0.02408812 | 0.02219168 | 7.8729% | Unresolved |

Errors are dimensionless normalized Frobenius Jacobian errors: median across probes within each seed, then mean across independent training seeds. Reductions are ratios of those means. The positive one-sided lower bounds on the strict error difference are 0.01773377, 0.01452451 and 0.00201702 for the three passing rows above. No resolved bound supports the warm 10 ms gain. [The machine-readable result](confirmation_look4.json) preserves every paired seed statistic, formal outcome and exact gate provenance.

The original fourth statistical look spends alpha 0.003125, with cumulative alpha 0.046875 and the unchanged rejection family K=280. Original percentage-floor decisions, numerical envelopes, solver checks and seed influence diagnostics remain preserved. The revised acceptance objective was chosen after the earlier prediction results; this report does not relabel the original floor criteria or present that amendment as untouched original-protocol confirmation.

[The independent raw audit](confirmation_look4_raw_audit.json) reconciles all 1,440 model/stratum/horizon summaries from signed q/v Jacobians and saved reference matrices. [The selection-integrity audit](selection_integrity_forty.json) separately reloads all eighty frozen checkpoints in the exact scientific source context and reproduces every recorded nominal validation objective, component score and target count. It verifies archive/canonical tensor hashes, uses validation labels only and leaves tensors unchanged. Its access journal lives outside the frozen source; it neither retrains nor selects different weights or adds a statistical look.

The [twenty-seed result](confirmation_look3.md) and all earlier results remain historical records. Control evidence is separate: [the direct augmented comparison](control_development_direct_result.md) and [the fixed-weight two-plant intervention](control_attribution_development_result.md) are development measurements at one training seed. They do not establish a multi-seed control advantage, lagged-plant prediction confirmation, broad baseline superiority, noise/rate robustness or full fresh-training replication.
