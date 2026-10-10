# Exploratory action-sensitivity progress result

**NONTERMINAL — one development seed; no scientific gate decision.**

Scope: direct-torque robot, augmented 20–60 ms training, one-step endpoint objective, training seed 9000, 128 cold and 128 warm development probes. These are completed standard E01 audits of already-qualified checkpoints, each frozen before viewing the result. The full M30 matrix, its shared multistep fits, lagged-actuator audits, confirmation seeds, and control tests are pending. No confirmation labels were opened and no statistical look was spent.

The corrected power head predicts the new command's short-time effect accurately in this subset. The interval-conditioned MLP and recurrent transition model have substantial fine-time errors. The generic NODE is also accurate: the power head lowers its 5 ms primary error by about one third, while at 10 ms the two are similar. This is support for a defect in some implementations and an unresolved competitive claim. It is not a C02 pass/failure: C02 uses augmented **shared-loss** fits, both plants, independent confirmation seeds, and all seven conventional competitors. No control advantage has been measured.

The percentages below are normalized **action-Jacobian matrix errors**, not position error, tracking error, or task success. Each value is the median over probes of the average of q and v normalized Frobenius errors. Cold and warm results remain separate. They have no training-seed confidence interval because there is only one exploratory seed.

| Model | Stratum | 5 ms primary Jacobian error | 10 ms primary Jacobian error | Evidence |
|---|---|---:|---:|---|
| power_correct | cold | 1.502% | 1.298% | [summary](../evidence/M30/20261001T082347-e1f2c0c0/summary.json) |
| power_correct | warm | 1.602% | 1.340% | [summary](../evidence/M30/20261001T082347-e1f2c0c0/summary.json) |
| mlp_dt | cold | 307.481% | 38.694% | [summary](../evidence/M30/20261001T082600-d4ce8ed0/summary.json) |
| mlp_dt | warm | 324.657% | 42.028% | [summary](../evidence/M30/20261001T082600-d4ce8ed0/summary.json) |
| gru_dt | cold | 218.032% | 35.001% | [summary](../evidence/M30/20261001T082612-0b04e9ad/summary.json) |
| gru_dt | warm | 91.092% | 18.406% | [summary](../evidence/M30/20261001T082612-0b04e9ad/summary.json) |
| node_generic | cold | 2.297% | 1.280% | [summary](../evidence/M30/20261001T082626-ed156418/summary.json) |
| node_generic | warm | 2.397% | 1.396% | [summary](../evidence/M30/20261001T082626-ed156418/summary.json) |
| node_structured | cold | 0.609% | 0.599% | [summary](../evidence/M30/20261001T082707-58d6befb/summary.json) |
| node_structured | warm | 0.782% | 0.775% | [summary](../evidence/M30/20261001T082707-58d6befb/summary.json) |

Finite-window log slopes use the qualified 2.5/5/10 ms window. The expected direct-torque position and velocity slopes are 2 and 1. These medians describe this window; they do not establish the exact asymptotic order at every smaller interval.

| Model | Stratum | Position slope | Velocity slope |
|---|---|---:|---:|
| power_correct | cold | 1.9967 | 0.9960 |
| power_correct | warm | 1.9962 | 0.9955 |
| mlp_dt | cold | -0.9025 | 0.3788 |
| mlp_dt | warm | -0.9485 | 0.3831 |
| gru_dt | cold | -0.4944 | 0.4081 |
| gru_dt | warm | 0.8836 | 0.2917 |
| node_generic | cold | 1.9981 | 0.9960 |
| node_generic | warm | 1.9991 | 0.9961 |
| node_structured | cold | 1.9973 | 0.9960 |
| node_structured | warm | 1.9973 | 0.9960 |

The corrected head and both NODE controls have nonzero gains close to the oracle in this subset. All 256 head probes have q/v gain ratios within the finite-library [0.5,1.5] band at 5 and 10 ms. This does not establish the protocol's population mean gain-coverage bound with one training seed. Raw matrix errors also check direction; a norm ratio alone is insufficient.

Reference measurements used the accepted 0.25 ms MuJoCo step, central differences at 0.01/0.02/0.04 N·m, a 0.125 ms physics comparison, and the independent rigid-body integrator. All measured rows resolved. Both NODEs were also audited with two half RK4 steps through the same weights. These refinements leave the reported qualitative comparison unchanged; their raw outcomes and numerical envelopes are in the seven audit manifests and [machine-readable progress record](exploratory_action_audit.json).

Scheduling deviation: this labelled progress subset was inspected before every development fit had finished, in response to the request for an early hint. The scheduler alone was paused, its live training block finished, and audits ran serially. The same scheduler and training rules then resumed. No architecture, threshold, training objective, optimizer grid, target distribution, or selection rule changed in response to these results. The complete G30 decision still requires its original full matrix.

Next: finish shared-loss and lagged fits, run the complete exploratory audit, then follow the preregistered mechanism/control cascade. A strong conventional model that already solves timing transfer is a meaningful negative possibility, not a baseline to remove.
