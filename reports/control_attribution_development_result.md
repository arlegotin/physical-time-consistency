Changing only the temporal powers modestly improves mean fine-step control cost across the ten original tasks on each plant. The correct power mode lowers mean cost versus the same-weight low-order mode by 3.6888% on direct torque and 0.4942% with actuator lag. It wins six tasks and loses four on each plant. All forty raw nominal comparisons pass: the three power modes use the same checkpoint and produce matching nominal candidate scores, actions, observations and plant trajectories to the required tolerance.

| Plant / mode | Mean cost, 40 ms internal | Mean cost, 5 ms internal | Successes, 40 / 5 ms |
|---|---:|---:|---:|
| Direct / low | 0.21215080 | 0.26996350 | 2 / 1 |
| Direct / correct (2,1) | 0.21215080 | 0.26000511 | 2 / 0 |
| Direct / mismatched (3,2) | 0.21215080 | 0.47270521 | 2 / 0 |
| Direct / structured NODE | 0.20934235 | 0.20942910 | 5 / 5 |
| Lagged / low | 0.24135991 | 0.30392936 | 4 / 1 |
| Lagged / mismatched (2,1) | 0.24135991 | 0.31698199 | 4 / 0 |
| Lagged / correct (3,2) | 0.24135991 | 0.30242748 | 4 / 2 |
| Lagged / structured NODE | 0.24318574 | 0.24274833 | 4 / 8 |

Lower cost is better. Every row contains ten tasks at training seed 9000; these tasks are not ten independent training replicas. The correct powers beat the deliberately mismatched powers by 44.9964% on direct torque and 4.5916% on the lagged plant. However, corrected fine-step costs remain 24.1495% and 24.5848% above the structured NODE, respectively, and rise 22.5567% and 25.3015% from their own nominal costs. No episode used the all-candidate-failure fallback. The unfavorable strict-success counts are retained alongside weighted costs.

The prespecified degradation contrasts are descriptive:

| Plant | Power-only improvement A | Structured improvement B | A − 0.5 B |
|---|---:|---:|---:|
| Direct | 0.00995839 | 0.05772595 | −0.01890458 |
| Lagged | 0.00150188 | 0.06300686 | −0.03000155 |

A and B are positive in this measured workload, but the isolated power change accounts for much less improvement than the structured comparator. No population confidence bound on B has been established; the fraction explained remains null. The first-task direct loss and lagged gain in [the earlier report](control_attribution_first_task.md) remain valid individual observations. Completing all original tasks changed their aggregate interpretation without replacing any task or checkpoint.

This experiment isolates a temporal-law change in the specified fixed-data, one-step-trained heads. The low-order mode is an intervention control, not a competitive baseline. It does not establish that conventional predictors naturally have that law or that the correction improves control generally. These checkpoints differ from the augmented/shared-loss models in [the direct conventional-baseline comparison](control_development_direct_result.md). Together, the results support a useful prediction mechanism and a modest directionally favorable isolated control effect, while leaving superiority over strong learned control baselines unestablished.

All 160 registered episodes completed under unchanged 40 ms sensing, replanning and command clocks, 0.25 ms qualified plant steps and the common 128-candidate, four-iteration planner. Producer receipts record 8,552.04 seconds of wall time and 400,855,286 artifact bytes in total, including the sixteen reused first-task episodes. These totals exclude orchestration/startup and independent-audit overhead. The measured sum is about 2.38 hours; it is resource accounting rather than a performance acceptance threshold.

[The complete receipt](control_attribution_development.json) retains every episode, resource record, task contrast and all forty nominal identity checks. [The independent audit](control_attribution_development_audit.json) recomputes costs and success from raw plant trajectories and executed commands, checks every cell and task pairing, and verifies artifact, semantic dependency and wheel hashes. The original [registry](../config/control_attribution_development_jobs.json) preserves all 160 requests and explicit reuse of sixteen prior completed episodes.

Repeat the raw audit from the repository root:

```bash
.venv/bin/python reports/audit_control_attribution_development.py
```
