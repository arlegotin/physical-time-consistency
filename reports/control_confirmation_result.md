# Limited control utility confirmation

Two-link direct and known-lag plants; A/shared corrected head versus specified qualified MLP and generic NODE; no full competitor-suite claim.

This result uses 5 independent training seeds with 20 paired episodes per seed and condition: 1,200 complete episodes. Twenty episodes improve each seed mean; they do not create twenty independent training replications.

| Plant | Competitor | Model step (ms) | Evidence | Baseline mean cost | Head mean cost | Baseline minus head | Lower bound | Upper bound |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct | mlp_dt | 40 | unresolved | 0.242927221 | 0.244611631 | -0.0016844108 | -0.0101652745 | 0.00679645292 |
| direct | mlp_dt | 5 | unresolved | 0.300609867 | 0.245343434 | 0.0552664331 | -0.0180420451 | 0.128574911 |
| direct | node_generic | 40 | unresolved | 0.2429248 | 0.244611631 | -0.00168683088 | -0.00931977811 | 0.00594611635 |
| direct | node_generic | 5 | unresolved | 0.242906312 | 0.245343434 | -0.00243712125 | -0.0100125413 | 0.00513829885 |
| lagged | mlp_dt | 40 | unresolved | 0.279521287 | 0.278367588 | 0.00115369954 | -0.00602601232 | 0.0083334114 |
| lagged | mlp_dt | 5 | unresolved_influence | 0.350222815 | 0.278567933 | 0.0716548827 | -0.0114632247 | 0.15477299 |
| lagged | node_generic | 40 | unresolved | 0.279045975 | 0.278367588 | 0.000678387329 | -0.0073248041 | 0.00868157876 |
| lagged | node_generic | 5 | unresolved | 0.278932651 | 0.278567933 | 0.000364718529 | -0.00589783128 | 0.00662726834 |

Positive cost differences favor the head; negative differences favor the competitor. Ratios are descriptive. Signed bounds include empirical numerical discrepancies and paired training-seed uncertainty. Unresolved variance or influential clusters prevent a supported sign.

Control look 1: alpha 0.025, per-direction tail 0.0015625, eight comparisons and two directional tails. The original prediction ledger remains separate.

Fallback intervals: 0. Successful episodes: 420. All episodes, including model-failure fallbacks, remain in the cost means.

Reference episodes: physics_fit: 200, zero_command: 200, sim_oracle: 50. One fitted physical model per plant; task streams are not independent physics-model fits.

Original PTCO all-claims PASS is not established. This three-model clean fixed-clock comparison does not establish the untested full competitor suite, changed actuator rates, observation noise, external disturbances, or physical-robot behavior.


The closed-loop NODE solver audit retains the original one-substep-trained discrete model and costs. The largest observed subset cost discrepancy expands nominal NODE uncertainty; it is an empirical envelope, not a certified bound.

Solver refinement materially changes the nominal validation objective: the learned field is discretization-specific. Refined evaluation is diagnostic and does not replace the comparator. A continuous-field claim requires the separate registered solver-training interpretation.
