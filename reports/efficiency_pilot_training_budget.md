# Frozen pilot training budgets

Selected-fit training ancestry only; synthetic controls and unselected optimizer candidates are excluded. Historical resource measurements across runs, not a prospective training-speed benchmark.

All neural checkpoints were selected using nominal validation only under the same allowed optimizer recipe. Converged models used different numbers of epochs. Lower validation error alone does not establish better counterfactual control. The physics reference has a much stronger prior and a different fitted objective.

| Plant:model | Completed epochs | Selected epoch | Updates | Recorded CPU seconds | CPU seconds / epoch |
|---|---:|---:|---:|---:|---:|
| direct:mlp_h | 700 | 590 | 89600 | 479.6 | 0.69 |
| direct:node_generic | 100 | 75 | 12800 | 237.8 | 2.38 |
| direct:node_structured | 100 | 90 | 12800 | 149.9 | 1.50 |
| direct:physics_fit | 100 | 20 | 12800 | 247.2 | 2.47 |
| direct:power_correct | 200 | 165 | 25600 | 268.5 | 1.34 |
| lagged:mlp_h | 400 | 400 | 51200 | 242.0 | 0.60 |
| lagged:node_generic | 500 | 400 | 64000 | 901.0 | 1.80 |
| lagged:node_structured | 200 | 200 | 25600 | 294.4 | 1.47 |
| lagged:physics_fit | 100 | 5 | 12800 | 254.9 | 2.55 |
| lagged:power_correct | 400 | 340 | 51200 | 351.8 | 0.88 |

Full ancestry paths and hashes: [training receipts](efficiency_pilot_training_budget.json). These measurements distinguish optimization effort from the separate inference benchmark. They do not justify a universal training-speed multiplier.
