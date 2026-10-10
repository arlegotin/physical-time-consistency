# Ten-seed action-sensitivity confirmation

**Mixed evidence; no terminal PoC verdict.** The correction helps at 5 ms, but its estimated advantage at 10 ms is far below the required 50%. Robot control has not been tested.

The audit uses the direct-torque two-link plant, augmented 20–60 ms training and the common one-plus-four-step loss. Ten independent training seeds (0–9) compare the discrete power head with the prespecified time-aware generic NODE. Each seed is evaluated on the same 256 cold and 256 warm probes. The table shows the mean of per-seed median normalized command-Jacobian errors; the reduction is the ratio of these means.

| Probes | Interval | Generic NODE error | Corrected-head error | Relative reduction |
|---|---:|---:|---:|---:|
| cold | 5 ms | 4.287% | 2.138% | 50.12% |
| cold | 10 ms | 2.371% | 1.918% | 19.09% |
| warm | 5 ms | 4.469% | 2.387% | 46.60% |
| warm | 10 ms | 2.458% | 2.121% | 13.73% |

These are action-response prediction errors, not tracking costs or control improvements. All 360 raw model/stratum/horizon summaries were independently recomputed from signed 2×2 position and velocity Jacobians before producing the table.

The formal second look is **valid INCONCLUSIVE**, with alpha 0.0125 and the unchanged 280-criterion family. A strictly positive reduction at 5 ms passes in both probe strata. The mandatory 50% criteria remain unresolved. At 10 ms, family-corrected upper linear-contrast bounds are negative, but leaving out one seed can change the interpretation; paired Fieller interval widths (approximately 0.963 cold and 1.288 warm) also exceed the required 0.20. These safeguards prevent a premature failure verdict.

The prospective [twenty-seed continuation](../config/revisions/MX-precision-0006.json) retains every compatible earlier observation and the alpha ledger, adding seeds 10–19 under the same scientific design. It does not change the threshold, model, data or checkpoint-selection rule.

Source decision: `study/precision-look-2/decisions/GR/49915e9ab668e618661e58fdf0f73ae72e0e707c260655b7fed71d7e993d3911.json`; SHA-256 `3f92eae4e735caa65449f5e34769943e8b48e68cf6fb9e395ff0a62bc7a24155`. The [machine-readable result](confirmation_look2.json) retains paired seed values and the exact criterion statuses. The [first five-seed report](confirmation_action_audit.md) remains preserved. Lagged confirmation, the complete competitor suite and all control/robustness/replication conditions are not yet evaluated.
