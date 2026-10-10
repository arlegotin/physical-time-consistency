# Direct-torque pilot results

Direct torque only. One development training seed, ten paired tasks per condition, 200 policy runs. No population significance or equivalence claim. Lagged-actuator grid and eight true-plant audits are not covered by this subset report.

All 200 complete episode manifests and raw artifacts validate. Costs below were independently recomputed from true simulated states and actual executed commands. Zero fallbacks occurred. This preserves both gains and losses; it is not the final full-pilot result.

| Commands, ms | Model | Actual maximum query, ms | Mean cost ↓ | Mean episode median planning, s ↓ | Success / 10 | Mean q RMSE, rad ↓ |
|---:|---|---:|---:|---:|---:|---:|
| 10 | mlp_h | 5 | 0.211594 | 2.511 | 3 | 0.1797 |
| 10 | node_generic | 5 | 0.206823 | 2.284 | 2 | 0.1810 |
| 10 | node_structured | 5 | 0.211315 | 2.438 | 1 | 0.1801 |
| 10 | physics_fit | 5 | 0.208220 | 2.377 | 3 | 0.1808 |
| 10 | power_correct | 5 | 0.212159 | 2.480 | 2 | 0.1784 |
| 10 | mlp_h | 10 | 0.211385 | 1.612 | 2 | 0.1793 |
| 10 | node_generic | 10 | 0.206823 | 1.522 | 2 | 0.1810 |
| 10 | node_structured | 10 | 0.211315 | 1.452 | 1 | 0.1801 |
| 10 | physics_fit | 10 | 0.208220 | 1.652 | 3 | 0.1808 |
| 10 | power_correct | 10 | 0.208965 | 1.522 | 4 | 0.1802 |
| 40 | mlp_h | 5 | 0.213362 | 2.212 | 4 | 0.1798 |
| 40 | node_generic | 5 | 0.212047 | 2.215 | 2 | 0.1800 |
| 40 | node_structured | 5 | 0.209514 | 2.083 | 2 | 0.1798 |
| 40 | physics_fit | 5 | 0.210443 | 2.292 | 6 | 0.1800 |
| 40 | power_correct | 5 | 0.213950 | 2.107 | 0 | 0.1784 |
| 40 | mlp_h | 40 | 0.211222 | 0.740 | 2 | 0.1795 |
| 40 | node_generic | 40 | 0.212047 | 0.671 | 2 | 0.1800 |
| 40 | node_structured | 40 | 0.209514 | 0.669 | 2 | 0.1798 |
| 40 | physics_fit | 40 | 0.210443 | 0.937 | 6 | 0.1800 |
| 40 | power_correct | 40 | 0.210674 | 0.645 | 3 | 0.1805 |

| Commands, ms | Actual maximum query, ms | Head vs competitor | Cost improvement, % | Head lower-cost tasks / 10 | Planning-time reduction, % |
|---:|---:|---|---:|---:|---:|
| 40 | 40 | mlp_h | +0.259 | 6 | +12.75 |
| 40 | 40 | node_generic | +0.647 | 6 | +3.78 |
| 40 | 40 | node_structured | -0.554 | 5 | +3.49 |
| 40 | 40 | physics_fit | -0.110 | 6 | +31.12 |
| 40 | 5 | mlp_h | -0.275 | 6 | +4.75 |
| 40 | 5 | node_generic | -0.897 | 5 | +4.90 |
| 40 | 5 | node_structured | -2.117 | 3 | -1.16 |
| 40 | 5 | physics_fit | -1.666 | 3 | +8.09 |
| 10 | 10 | mlp_h | +1.145 | 5 | +5.63 |
| 10 | 10 | node_generic | -1.036 | 2 | +0.06 |
| 10 | 10 | node_structured | +1.112 | 5 | -4.77 |
| 10 | 10 | physics_fit | -0.358 | 4 | +7.90 |
| 10 | 5 | mlp_h | -0.267 | 4 | +1.26 |
| 10 | 5 | node_generic | -2.580 | 2 | -8.56 |
| 10 | 5 | node_structured | -0.399 | 4 | -1.72 |
| 10 | 5 | physics_fit | -1.892 | 2 | -4.31 |

Positive improvement favors the head; negative improvement is a measured loss. Comparing equal query sizes alone is incomplete: each NODE also receives its coarsest legal queries. With commands every 10 ms, a nominal 40 ms partition is split into actual 10 ms prediction queries. Observations and replanning remain every 40 ms.

The head has small gains and losses against the learned alternatives. It does not have the lowest cost against the tested learned competitors in any of these four direct-torque conditions. Its separate inference speed advantage does not imply a similarly large end-to-end planning speedup. The fitted mechanics reference has a stronger prior than the neural methods.

Task success is the prespecified strict final-200-ms test on both position and velocity at every physics step. Ten tasks do not give a reliable population success-rate claim. Cost and success measure different aspects of the motion.

Raw values, paired task differences, complete manifest paths and hashes: [direct subset JSON](efficiency_pilot_direct_subset_2026-10-07.json). [Inference measurements](efficiency_pilot_inference.md) and [selected training budgets](efficiency_pilot_training_budget.md) remain separate. The actuator-lag runs and true-plant audits are still required before the complete pilot report.
