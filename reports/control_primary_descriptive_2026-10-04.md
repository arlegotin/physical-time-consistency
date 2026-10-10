# Completed primary control results: qualification pending

All 1,200 primary episodes are complete and raw-audited. These are descriptive results for five independent training seeds and twenty paired tasks per seed, with unchanged 40 ms hardware clocks. Physical-reference comparisons, NODE solver checks, statistical inference and reproduction remain pending. No scientific verdict or statistical look is assigned here.

At 5 ms, the corrected head has 18.38% lower mean physical cost than the MLP on direct torque and 20.46% lower cost with actuator lag. Against the generic NODE, its mean cost is 1.00% higher on direct torque and 0.13% lower with lag. This is encouraging against the MLP and mixed against the stronger comparator; the point estimates alone do not establish a population gain.

| Plant | Model | Cost at 40 ms | Cost at 5 ms | Successes at 40/5 ms (each /100) | Fallbacks at 40/5 ms |
|---|---|---:|---:|---:|---:|
| direct | mlp_dt | 0.242927221 | 0.300609867 | 27/12 | 0/0 |
| direct | node_generic | 0.242924800 | 0.242906312 | 26/26 | 0/0 |
| direct | power_correct | 0.244611631 | 0.245343434 | 12/25 | 0/0 |
| lagged | mlp_dt | 0.279521287 | 0.350222815 | 52/12 | 0/0 |
| lagged | node_generic | 0.279045975 | 0.278932651 | 55/62 | 0/0 |
| lagged | power_correct | 0.278367588 | 0.278567933 | 60/51 | 0/0 |

Lower cost is better. Success uses the stricter final-200-ms position and velocity rule; cost and success are separate outcomes. The lagged head has fewer fine-time successes than the NODE despite their similar mean costs. All episodes are retained; no corrected-model fallback occurred.

The [machine-readable summary](control_primary_descriptive_2026-10-04.json) includes every seed mean and all eight descriptive head-versus-baseline comparisons. Its evidence hashes link the [complete raw audit](control_confirmation_raw_audit.json), [primary coverage verification](control_primary_completion_2026-10-04.json), and [initial finer-physics check](control_finer_subset_review_2026-10-04.json). The finer-physics subset stayed within its numerical budgets; that check is separate from NODE integration and control inference.

This registered design compares the MLP, generic NODE and corrected head on both plants. It does not cover all seven original competitors, changed actuator rates, observation noise or disturbances. Historical PTCO decisions remain preserved.
