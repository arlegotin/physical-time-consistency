# PTCO-1.0 decision report

Status: **NONTERMINAL**.

## 1. Verdict and scope

Robot-control, practical actuator-rate, noise/disturbance and fresh-training replication experiments were **not run**. The decisive audit compares the discrete power head with the prespecified generic NODE on the fully observed smooth direct-torque two-link plant, augmented 20–60 ms training, common one-plus-four-step loss, and independent paired training seeds. Primary query horizons are 5 and 10 ms, with separate cold and warm probes. The available evidence does not yet support a terminal scientific verdict.

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

- `K02.relative/actuator=direct/stratum=cold/h_ns=5000000/competitor=node_generic/contrast=relative/training_condition=A/loss_mode=shared/method=power_correct`: **unresolved**; contrast -0.0004549040199720326, bounds [-0.0032087920323878557, 0.002298983992443763]; gate GR.
- `K02.relative/actuator=direct/stratum=cold/h_ns=5000000/competitor=node_generic/contrast=strict_improvement/training_condition=A/loss_mode=shared/method=power_correct`: **pass**; contrast 0.021608256597655313, bounds [0.016851940027669425, 0.0263645731676412]; gate GR.
- `K02.relative/actuator=direct/stratum=cold/h_ns=10000000/competitor=node_generic/contrast=relative/training_condition=A/loss_mode=shared/method=power_correct`: **unresolved**; contrast -0.008149264790448361, bounds [-0.010190228482930965, -0.00610830109796576]; gate GR.
- `K02.relative/actuator=direct/stratum=cold/h_ns=10000000/competitor=node_generic/contrast=strict_improvement/training_condition=A/loss_mode=shared/method=power_correct`: **pass**; contrast 0.0040558639227741295, bounds [0.0010688511967194606, 0.007042876648828799]; gate GR.
- `K02.relative/actuator=direct/stratum=warm/h_ns=5000000/competitor=node_generic/contrast=relative/training_condition=A/loss_mode=shared/method=power_correct`: **unresolved**; contrast -0.002911280378329552, bounds [-0.006744932966390156, 0.0009223722097310281]; gate GR.
- `K02.relative/actuator=direct/stratum=warm/h_ns=5000000/competitor=node_generic/contrast=strict_improvement/training_condition=A/loss_mode=shared/method=power_correct`: **pass**; contrast 0.019622904709291532, bounds [0.013567325520663007, 0.02567848389792006]; gate GR.
- `K02.relative/actuator=direct/stratum=warm/h_ns=10000000/competitor=node_generic/contrast=relative/training_condition=A/loss_mode=shared/method=power_correct`: **unresolved**; contrast -0.01032227022014898, bounds [-0.013188732412887226, -0.0074558080274107355]; gate GR.
- `K02.relative/actuator=direct/stratum=warm/h_ns=10000000/competitor=node_generic/contrast=strict_improvement/training_condition=A/loss_mode=shared/method=power_correct`: **unresolved**; contrast 0.0021567052787933904, bounds [-0.001651281918594232, 0.005964692476181014]; gate GR.

## 3. Implementation and correctness

Implementation checks actually executed are recorded under M00/M10/M20 and the recorded unit runs. See `locks/environment.json`, each `validity.json`, plant XML/compiled manifests, and the independent rigid-body/analytic oracle evidence. A qualification GO does not establish all C01 requirements or a scientific PASS.

## 4. Data and selection

Generated corpus roles, dimensions, units, seeds and hashes are recorded in `data/*/*/*/manifest.json`; normalization uses training transitions only. Nominal learning curves, every attempted optimizer block and selected checkpoints are preserved under the training run manifests. Confirmation data are paired across methods and never used for nominal checkpoint selection.

## 5. Action-effect evidence

Independent primary-error reaggregation checked 720 model/stratum/horizon summaries directly from the full signed q/v matrices and saved reference matrices. See `independent_primary_audit.json` and the decisive GR raw arrays. This checks sensitivity to command, not robot tracking error. Exploratory findings retain their discovery label in `exploratory_action_audit.md`.

| Interval | Stratum | Baseline error | Corrected error | Point reduction | Seeds | Probes per seed |
|---|---|---:|---:|---:|---:|---:|
| 5 ms | cold | 4.4126% | 2.2518% | 48.97% | 20 | 256 |
| 10 ms | cold | 2.4410% | 2.0354% | 16.62% | 20 | 256 |
| 5 ms | warm | 4.5068% | 2.5445% | 43.54% | 20 | 256 |
| 10 ms | warm | 2.4958% | 2.2801% | 8.64% | 20 | 256 |

Each error is the mean of seed-level median normalized Jacobian errors. Reductions are ratios of those means, not means of per-seed ratios; they are point estimates, not gate conclusions. Per-seed values remain in `paired_primary_results.json`.

## 6. Causal attribution

Full control attribution A, B and A−0.5B was not evaluated. Analytic nominal power-mode identity tests do not replace a measured control intervention. No attribution fraction is reported.

## 7. Prediction and control

No confirmatory MPC episodes, tracking improvements or task success rates are available. No claim of better control is made. Unfinished development fits and prediction jobs are recorded as such, rather than promoted to confirmation.

## 8. Robustness and computation

Reference finite-difference, physics-refinement and independent-ODE checks, and same-checkpoint NODE two-half-step audits, accompany each decisive sensitivity run. Numerical uncertainty is propagated into the contrasts. Practical rate/noise/disturbance control remains untested. Measured wall/CPU time and RSS are in `run_index.csv` and per-run resources; parent and child timings overlap and must not be summed blindly. The owner imposed no artificial resource ceilings.

## 9. Statistical integrity

The rejection family is K=280, including criteria for unexecuted later stages. Seed clusters remain paired, and probes are not independent training replications. Inference is conditional on the fixed training corpus and probe library.

- Look 3: alpha 0.00625; gate `GR`, outcome `INCONCLUSIVE`. Family-corrected rejection bounds, paired Fieller precision, bootstrap and influence checks are retained in its criterion evidence.

## 10. Replication and audit

The decisive raw sensitivity errors are independently reaggregated for this report. This is not the full M90 replay/retraining replication; C06 has not been established. All interrupted, superseded, stale and failed attempts remain visible in `run_index.csv`; only compatible completed decisive evidence supports the verdict.

## 11. Deviations and limitations

The registered rejection-only execution-order extension is in `config/revisions/MX-mechanism-negative-0004.json` and `rejection_shortcut.md`. It leaves scientific models, data budgets, losses, nominal selection, thresholds and K unchanged; a single qualified prespecified comparator can reject necessary best-baseline superiority. It cannot establish PASS. Other repairs and semantic invalidations are retained under `config/revisions/` and run provenance. The scope excludes contact, hidden delays/activation, unknown tau, physical robot safety, universal superiority and novelty.

## 12. Unresolved questions and resumption

Execution is nonterminal; continue only the exact action permitted by the validated gate, retaining the statistical look ledger and all paired samples.

## 13. Reproduction and artifact index

The exact study source, configuration, wheels and artifact hashes are retained with the evidence. `run_index.csv` lists every observed attempt. `reproduce.sh` defaults to verification/reaggregation; expensive reruns require an explicit argument and current authorization. Missing or altered archived artifacts fail verification.
