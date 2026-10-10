# Twenty-seed action-sensitivity confirmation

The correction helps, but the required 50% advantage remains unconfirmed. The valid third-look decision is **INCONCLUSIVE**; the project is **NONTERMINAL**. Robot-control, lagged-plant confirmation, practical actuator-rate and robustness evaluations have not run.

This comparison uses direct torque, augmented 20–60 ms training data, the common one-plus-four-step loss, and independent seeds 0–19. Both models have qualified nominal fits. Each seed uses the same 256 cold and 256 warm probes. The baseline is the prespecified time-aware generic NODE.

| Probe stratum | Horizon | Generic NODE error | Corrected head error | Relative reduction |
|---|---:|---:|---:|---:|
| cold | 5 ms | 4.41% | 2.25% | 48.97% |
| cold | 10 ms | 2.44% | 2.04% | 16.62% |
| warm | 5 ms | 4.51% | 2.54% | 43.54% |
| warm | 10 ms | 2.50% | 2.28% | 8.64% |

Errors are normalized Frobenius errors of the command-to-position and command-to-velocity Jacobian matrices. The primary statistic takes the median across probes within each seed, then the mean across seeds. Reductions compare those cluster means. These values do not measure tracking or control performance.

The ordinary improvement criterion passes for cold and warm probes at 5 ms, and cold probes at 10 ms. Warm-probe improvement at 10 ms remains unresolved. None of the four mandatory 50% floor comparisons passes.

At 10 ms, the family-corrected rejection contrasts are negative and the leave-one-seed-out interpretation is stable. The paired Fieller intervals nevertheless have widths about 0.454 (cold) and 0.610 (warm), exceeding the required 0.20 precision limit. The numerical envelopes are included. This prevents a formal rejection under the frozen protocol even though the estimated advantage is substantially below the floor.

The decision uses look 3, alpha 0.00625, the unchanged full family of 280 statistical criteria, paired training-seed cluster inference, and preserved looks 1–2. No threshold, model recipe, training corpus, checkpoint selection rule or evaluation horizon changed.

The next prospective request retains every existing seed and adds seeds 20–39. It reserves look 4 with alpha 0.003125 before any new outcomes. Measured training across the latest ten seeds took about 2.02 hours including all 48 training attempts; the next twenty seeds project about 4.05 training hours plus audits. This is an estimate, with no owner-imposed time cap. More seeds may still be required.

The primary errors were independently recomputed from 720 saved model/stratum/horizon summaries and checked against their cached values. [Machine-readable results](confirmation_look3.json) preserve every paired seed value, formal criterion outcome, decision hash and precision diagnostic.

The completed [ten-seed result](confirmation_look2.md), [first five-seed result](confirmation_action_audit.md) and [discovery audit](exploratory_action_audit.md) remain archived.
