Changing only the powers improves fine-step control on the first lagged-actuator task, but worsens it on the first direct-torque task. All three modes share each plant's fixed-data, one-step-trained checkpoint. Their nominal candidates, selected actions, observations and realized trajectories match to the required tolerance. This isolates an effect of the power change on these checkpoints; it does not establish that the effect helps across tasks or training seeds.

| Plant / mode | Cost, 40 ms internal | Cost, 5 ms internal |
|---|---:|---:|
| Direct / low | 0.04783830 | 0.07755193 |
| Direct / correct (2,1) | 0.04783830 | 0.09882203 |
| Direct / mismatched (3,2) | 0.04783830 | 0.49583856 |
| Direct / structured NODE | 0.03765055 | 0.03765055 |
| Lagged / low | 0.04499719 | 0.06300056 |
| Lagged / mismatched (2,1) | 0.04499719 | 0.06234662 |
| Lagged / correct (3,2) | 0.04499719 | 0.05149469 |
| Lagged / structured NODE | 0.04412067 | 0.04412067 |

Lower cost is better. The correct powers reduce fine-step cost by 18.2631% versus the low-order mode on the lagged task; they increase it by 27.4269% on the direct task. Both correct heads have higher fine-step cost than their structured NODE comparator here. No episode used the all-candidate-failure fallback.

For the prespecified degradation contrasts, direct A is −0.02127010 and B is 0.02971363; lagged A is 0.01150587 and B is 0.01800337. A−0.5B is −0.03612691 and 0.00250419, respectively. These are descriptive values from one training seed and one fixed task per plant. No positive statistical bound on B has been established, so the reported fraction explained remains null.

[control_attribution_measurements.json](control_attribution_measurements.json) retains all sixteen episode pointers, resources and contrasts. [control_attribution_measurements_audit.json](control_attribution_measurements_audit.json) independently checks raw realized costs, artifact and semantic hashes, and all four nominal low/direct/lagged comparisons, including the deliberately mismatched mode. Run `.venv/bin/python reports/audit_control_attribution_measurements.py` from the repository root to repeat the audit.

The original episode-zero tasks, seed 9000, known actuation chains, 40 ms hardware clocks, qualified physics step and frozen common planner were preserved. These fixed-data checkpoints differ from the augmented/shared-loss checkpoints in the [ten-task control comparison](control_development_direct_result.md). Correct asymptotic powers alone do not guarantee correct finite-time gains or useful control. The [full ten-task intervention](control_attribution_development_result.md) has now completed: the mean correct-versus-low cost reduction is 3.7% on direct torque and 0.5% with lag. No model, task or threshold was changed in response to this first-task observation.
