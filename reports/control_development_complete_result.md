The complete development workload supports a useful control benefit over the MLP and recurrent GRU when the model is queried at 5 ms. It does not establish superiority over the neural ODE models. These are measured results from one development training seed, 9000, with ten paired tasks per plant, augmented 20–60 ms training data, and the shared one-plus-four-step loss.

| Plant | Model | Mean cost, 40 ms | Mean cost, 5 ms | Task successes, 40 / 5 ms |
|---|---|---:|---:|---:|
| Direct torque | MLP | 0.21272743 | 0.30063297 | 2 / 0 |
| Direct torque | GRU | 0.21050318 | 0.22681326 | 3 / 2 |
| Direct torque | Generic NODE | 0.21204692 | 0.21204692 | 2 / 2 |
| Direct torque | Corrected head | 0.21067437 | 0.21394988 | 3 / 0 |
| Direct torque | Structured NODE | 0.20951355 | 0.20951355 | 2 / 2 |
| Lagged actuation | MLP | 0.24044464 | 0.32566581 | 6 / 0 |
| Lagged actuation | GRU | 0.24061344 | 0.28349435 | 3 / 3 |
| Lagged actuation | Generic NODE | 0.24317750 | 0.24301811 | 5 / 6 |
| Lagged actuation | Corrected head | 0.24182466 | 0.24120193 | 6 / 4 |
| Lagged actuation | Structured NODE | 0.24206924 | 0.24133738 | 5 / 6 |

Lower cost is better. At 5 ms, the head lowers mean cost by **28.83% versus the MLP and 5.67% versus the GRU on direct torque**, and by **25.94% versus the MLP and 14.92% versus the GRU on lagged actuation**. It wins all ten MLP comparisons on each plant, nine of ten direct GRU comparisons, and all ten lagged GRU comparisons. This is encouraging evidence that the correction helps these transition predictors transfer to finer internal timesteps.

The stronger comparisons remain mixed. Against the generic NODE, the head costs 0.90% more on direct torque and 0.75% less on lagged actuation. Against the structured NODE, it costs 2.12% more on direct torque and 0.06% less on lagged actuation. These small differences have no independent-seed uncertainty estimate in this development workload. They do not establish either equality or a reproducible head advantage.

At the original 40 ms model interval, all five model mean costs lie within 1.6% of each other on each plant. This is a descriptive workload observation, not a statistical nominal-noninferiority finding. The head's own cost changes by +1.55% on direct torque and −0.26% on lagged actuation when its internal interval is refined to 5 ms. The corresponding MLP increases are 41.32% and 35.44%; the GRU increases are 7.75% and 17.82%.

Strict task success requires the position and velocity tolerances at every accepted physics step in the final 200 ms. The head achieves zero of ten direct and four of ten lagged successes at 5 ms, compared with two and six for each NODE. The cost improvements over the MLP and GRU therefore do not establish better task success than the NODEs. All 200 episodes are retained; none used the all-candidate-failure fallback.

Every episode used 40 ms sensing, replanning and actuator holds, six torque knots, the independently qualified 128-candidate/four-iteration CEM budget, and the 0.25 ms plant step. Only the internal model interval changed. Checkpoints were selected using nominal validation. No architecture, target distribution, cost, optimizer, or threshold was selected from these control results.

[control_development_complete_audit_v2.json](control_development_complete_audit_v2.json) verifies all 200 episode manifests and artifact hashes, frozen requests for the newer 140 episodes, relevant configurations, historical dependency compatibility, and wheel hashes. It independently reconstructs each realized cost from the saved true anchors and executed commands, checks clocks and observation packet identities, verifies task success and fallback counts, and requires the complete paired workload before aggregation. Reproduce the audit with `.venv/bin/python reports/audit_control_development_complete.py`; an unchanged audit preserves the existing output. This version strengthens input validation and retains [the original audit](control_development_complete_audit.json); all 200 verification records and measured summaries agree exactly. [The reporting tests](../tests/test_development_report.py) reject missing, duplicate, incompatible, unpaired and nonfinite evidence, truncated partitions, malformed success/fallback statuses and unintended power overrides, and retain losses, fallbacks and unresolved percentages.

This completes the original core-model development control comparison. Independent control confirmation uses separate seeds and tasks and remains in progress. Its registered subset covers the head, MLP and generic NODE on both plants; it will not confirm the GRU or structured NODE comparisons in this table. Numerical, physical-reference and reproduction checks still precede any confirmed utility claim. No statistical look or overall PoC verdict is assigned by this development audit. The [fixed-weight intervention results](control_attribution_development_result.md) remain separate evidence: changing only the powers produced modest gains, and the structured NODE performed substantially better there.
