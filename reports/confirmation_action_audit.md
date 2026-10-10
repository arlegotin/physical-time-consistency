# First confirmatory action-sensitivity result

**Encouraging, but mixed.** The discrete correction reduces action-sensitivity error substantially at 5 ms, while a strong time-aware generic NODE catches up at 10 ms. This is a model-sensitivity result; robot-control benefit has not been tested.

Status: **NONTERMINAL**. The valid first confirmatory gate is **INCONCLUSIVE**, with five paired independent training seeds. No POC PASS or POC FAIL is supported yet.

| Query interval | Probe state | Generic NODE error | Corrected head error | Point reduction |
|---|---|---:|---:|---:|
| 5 ms | cold | 4.27% | 2.11% | 50.7% |
| 5 ms | warm | 4.62% | 2.35% | 49.1% |
| 10 ms | cold | 2.39% | 1.91% | 20.2% |
| 10 ms | warm | 2.55% | 2.09% | 18.0% |

Errors are normalized Frobenius errors of the full position/velocity command-Jacobian matrices: each seed contributes the median of `(e_q+e_v)/2` over 256 probes in a stratum, then the five seed statistics are averaged. The table reports percentage error relative to the reference Jacobian; it does not report tracking error or control-cost improvement. Warm probes include completed preceding history.

The audit uses the fully observed smooth direct-torque two-link robot, augmented 20–60 ms training transitions, common one-plus-four-step supervised loss, and untouched confirmation probes. No 5/10 ms labels enter training. Generic NODE is a prespecified competitive time-aware model, rather than the deliberately low-order intervention.

Both 5 ms strictly positive reductions pass their preregistered uncertainty checks. All four required 50% reduction contrasts remain unresolved, and the 10 ms strictly positive reductions remain unresolved at the prescribed bounds. The 10 ms point estimates fall below the required floor, but conservative family-corrected rejection and paired Fieller precision do not yet establish a mandatory rejection. K=280 and look-one alpha=0.025 are unchanged; numerical uncertainty, bootstrap robustness and seed influence are retained in the source summary.

Independent reaggregation checked 180 model/stratum/horizon summaries from signed raw matrices against the cached metrics. The complete source, per-seed effects, uncertainty bounds and criterion reasons are in [the measurement summary](../evidence/MR/20261001T111511-ff500529/summary.json) and [the gate decision](../decisions/GR/0b79513e7f2cc7e8cc1eb8a416ba0e1330f54bab6e6f3d264977eb7eb5f1173d.json); their SHA-256 hashes are in [the report data](confirmation_action_audit.json).

The registered precision continuation adds seeds 5–9 while retaining all previous samples and alpha spending. The broader actuator-specific tests, complete competitor suite, causal control attribution, MPC, actuator-rate changes, noise/disturbance control and fresh-training replication are not established by this subset result.
