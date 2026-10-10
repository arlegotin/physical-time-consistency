The corrected head avoids the large next-interval prediction error that the MLP develops under the fine model clock. The generic NODE also remains accurate. This is a reaggregation of completed development episodes, not new control confirmation.

Each number below is the mean of ten episode-level prediction RMSEs at training seed 9000. At every real feedback decision, the audit compares the selected plan's predicted first 40 ms anchor with the actual outcome of those same executed commands. It excludes future motion changed by replanning.

| Direct / augmented shared-loss model | Internal step | Position prediction RMSE (rad) | Velocity prediction RMSE (rad/s) |
|---|---:|---:|---:|
| MLP | 40 ms | 0.000170 | 0.00577 |
| MLP | 5 ms | 0.008838 | 0.37958 |
| Generic NODE | 40 ms | 0.000360 | 0.02139 |
| Generic NODE | 5 ms | 0.000360 | 0.02139 |
| Corrected head | 40 ms | 0.000211 | 0.00935 |
| Corrected head | 5 ms | 0.000401 | 0.01894 |

This is consistent with the [measured control comparison](control_development_direct_result.md): the head lowers fine-step cost relative to the MLP, while its mean cost is slightly higher than the NODE's. Its next-interval velocity error is slightly smaller than the NODE's, and its position error is slightly larger. Better performance on one prediction component therefore does not establish better overall control.

The audit reconciles all 8,800 decisions across the sixty augmented/shared direct episodes and the 160 fixed-data intervention episodes on both plants. It checks saved predictions, actual next states, executed holds, physical clocks and both logged first-interval costs. Fallbacks retain actual costs and explicit missing forecasts; none occurred in these completed development cells. The sum of first-interval costs excludes the episode terminal cost and is not presented as the primary realized episode cost.

The policies choose different commands and visit different states. These prediction errors describe each policy's own workload; they cannot alone isolate the cause of a between-policy control difference. All results here are descriptive at one training seed, with no new statistical look or checkpoint selection.

[A separate hardware-path audit](control_partition_policy_identity.json) finds that the NODE's 40/5 ms commands differ by at most 2.23e-16 N·m across all 400 decisions, with plant differences at floating-point roundoff. The head and MLP change their selected commands materially. This is consistent with the NODE's unchanged pilot cost, and describes this fixed planner/workload rather than proving exact composition of its model. Run `.venv/bin/python reports/audit_control_partition_policy.py` to reproduce the path comparison.

[The compact summary](control_prediction_gap_summary.json) pins the complete derived audit, its source receipts and evaluator hashes. The larger local `control_prediction_gap_development.json` retains every decision and manifest dependency. Reproduce from existing immutable episode artifacts:

```bash
.venv/bin/python reports/audit_control_prediction_gap.py
```
