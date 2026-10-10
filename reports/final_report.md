# PTCO-1.0 decision report

Status: **NONTERMINAL**.

## 1. Verdict and scope

This report covers the selected prediction study. Robot-control confirmation, practical actuator-rate, noise/disturbance and fresh-training replication experiments were **not run within this study**. Separate control-development records are outside this report’s scientific decision scope. The decisive audit compares the discrete power head with the prespecified generic NODE on the fully observed smooth direct-torque two-link plant, augmented 20–60 ms training, common one-plus-four-step loss, and independent paired training seeds. Primary query horizons are 5 and 10 ms, with separate cold and warm probes. The available evidence does not yet support a terminal scientific verdict.

## 2. Claim-by-claim decisions

| Claim | Available decision |
|---|---|
| C01 | not established |
| C02 | partial / unresolved |
| C03 | not established |
| C04 | not established |
| C05 | not established |
| C06 | not established |

`claims.csv` contains all 294 atomic entries, including the 280 statistical criteria and explicit missingness. Estimates and bounds for non-fraction statistical rows concern the linear contrast; practical percentage thresholds are separate.

- `K02.relative/actuator=direct/stratum=cold/h_ns=5000000/competitor=node_generic/contrast=relative/training_condition=A/loss_mode=shared/method=power_correct`: **unresolved**; contrast -0.000269859396961254, bounds [-0.002268377533627518, 0.0017286587397049857]; gate GR.
- `K02.relative/actuator=direct/stratum=cold/h_ns=5000000/competitor=node_generic/contrast=strict_improvement/training_condition=A/loss_mode=shared/method=power_correct`: **pass**; contrast 0.021405506509495388, bounds [0.017733772678122085, 0.02507724034086869]; gate GR.
- `K02.relative/actuator=direct/stratum=cold/h_ns=10000000/competitor=node_generic/contrast=relative/training_condition=A/loss_mode=shared/method=power_correct`: **unresolved**; contrast -0.007807858431860201, bounds [-0.009167841412293418, -0.006447875451426987]; gate GR.
- `K02.relative/actuator=direct/stratum=cold/h_ns=10000000/competitor=node_generic/contrast=strict_improvement/training_condition=A/loss_mode=shared/method=power_correct`: **pass**; contrast 0.004139308107426404, bounds [0.0020170198996507303, 0.006261596315202079]; gate GR.
- `K02.relative/actuator=direct/stratum=warm/h_ns=5000000/competitor=node_generic/contrast=relative/training_condition=A/loss_mode=shared/method=power_correct`: **unresolved**; contrast -0.003083709540339108, bounds [-0.005494713952272543, -0.0006727051284056943]; gate GR.
- `K02.relative/actuator=direct/stratum=warm/h_ns=5000000/competitor=node_generic/contrast=strict_improvement/training_condition=A/loss_mode=shared/method=power_correct`: **pass**; contrast 0.018622962372583714, bounds [0.01452451392851322, 0.022721410816654214]; gate GR.
- `K02.relative/actuator=direct/stratum=warm/h_ns=10000000/competitor=node_generic/contrast=relative/training_condition=A/loss_mode=shared/method=power_correct`: **unresolved**; contrast -0.010147619921949774, bounds [-0.01184675372588661, -0.008448486118012941]; gate GR.
- `K02.relative/actuator=direct/stratum=warm/h_ns=10000000/competitor=node_generic/contrast=strict_improvement/training_condition=A/loss_mode=shared/method=power_correct`: **unresolved**; contrast 0.0018964393158314372, bounds [-0.0004968524134031571, 0.0042897310450660305]; gate GR.

## 3. Implementation and correctness

Implementation checks actually executed are recorded under M00/M10/M20 and the recorded unit runs. See `locks/environment.json`, each `validity.json`, plant XML/compiled manifests, and the independent rigid-body/analytic oracle evidence. A qualification GO does not establish all C01 requirements or a scientific PASS.

## 4. Data and selection

Generated corpus roles, dimensions, units, seeds and hashes are recorded in `data/*/*/*/manifest.json`; normalization uses training transitions only. Nominal learning curves, every attempted optimizer block and selected checkpoints are preserved under the training run manifests. Confirmation data are paired across methods and never used for nominal checkpoint selection.

## 5. Action-effect evidence

Independent primary-error reaggregation checked 1440 model/stratum/horizon summaries directly from the full signed q/v matrices and saved reference matrices. See `independent_primary_audit.json` and the decisive GR raw arrays. This checks sensitivity to command, not robot tracking error. Exploratory findings retain their discovery label in `exploratory_action_audit.md`.

| Interval | Stratum | Baseline error | Corrected error | Point reduction | Seeds | Probes per seed |
|---|---|---:|---:|---:|---:|---:|
| 5 ms | cold | 4.3351% | 2.1945% | 49.38% | 40 | 256 |
| 10 ms | cold | 2.3894% | 1.9755% | 17.32% | 40 | 256 |
| 5 ms | warm | 4.3413% | 2.4790% | 42.90% | 40 | 256 |
| 10 ms | warm | 2.4088% | 2.2192% | 7.87% | 40 | 256 |

Each error is the mean of seed-level median normalized Jacobian errors. Reductions are ratios of those means, not means of per-seed ratios; they are point estimates, not gate conclusions. Per-seed values remain in `paired_primary_results.json`.

## 6. Causal attribution

Full control attribution A, B and A−0.5B was not evaluated. Analytic nominal power-mode identity tests do not replace a measured control intervention. No attribution fraction is reported.

## 7. Prediction and control

No confirmatory MPC episodes, tracking improvements or task success rates are available. No claim of better control is made. Unfinished development fits and prediction jobs are recorded as such, rather than promoted to confirmation.

## 8. Robustness and computation

Reference finite-difference, physics-refinement and independent-ODE checks, and same-checkpoint NODE two-half-step audits, accompany each decisive sensitivity run. Numerical uncertainty is propagated into the contrasts. Practical rate/noise/disturbance control remains untested. Measured wall/CPU time and RSS are in `run_index.csv` and per-run resources; parent and child timings overlap and must not be summed blindly. The owner imposed no artificial resource ceilings.

## 9. Statistical integrity

The rejection family is K=280, including criteria for unexecuted later stages. Seed clusters remain paired, and probes are not independent training replications. Inference is conditional on the fixed training corpus and probe library.

- Look 4: alpha 0.003125; gate `GR`, outcome `INCONCLUSIVE`. Family-corrected rejection bounds, paired Fieller precision, bootstrap and influence checks are retained in its criterion evidence.

## 10. Replication and audit

The decisive raw sensitivity errors are independently reaggregated for this report. This is not the full M90 replay/retraining replication; C06 has not been established. All interrupted, superseded, stale and failed attempts remain visible in `run_index.csv`; only compatible completed decisive evidence supports the verdict.

## 11. Deviations and limitations

The registered rejection-only execution-order extension is in `config/revisions/MX-mechanism-negative-0004.json` and `rejection_shortcut.md`. It leaves scientific models, data budgets, losses, nominal selection, thresholds and K unchanged; a single qualified prespecified comparator can reject necessary best-baseline superiority. It cannot establish PASS. Other repairs and semantic invalidations are retained under `config/revisions/` and run provenance. The scope excludes contact, hidden delays/activation, unknown tau, physical robot safety, universal superiority and novelty.

## 12. Unresolved questions and resumption

Execution is nonterminal; continue only the exact action permitted by the validated gate, retaining the statistical look ledger and all paired samples.

## 13. Reproduction and artifact index

The exact study source, configuration, wheels and artifact hashes are retained with the evidence. `run_index.csv` lists every observed attempt. `reproduce.sh` defaults to verification/reaggregation; expensive reruns require an explicit argument and current authorization. Missing or altered archived artifacts fail verification.
