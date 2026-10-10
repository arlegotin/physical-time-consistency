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

- `K02.relative/actuator=direct/stratum=cold/h_ns=5000000/competitor=node_generic/contrast=relative/training_condition=A/loss_mode=shared/method=power_correct`: **unresolved**; contrast 0.00029662004164720325, bounds [-0.005866242615498538, 0.006459482698792898]; gate GR.
- `K02.relative/actuator=direct/stratum=cold/h_ns=5000000/competitor=node_generic/contrast=strict_improvement/training_condition=A/loss_mode=shared/method=power_correct`: **pass**; contrast 0.021671337986196453, bounds [0.010496351766374137, 0.03284632420601875]; gate GR.
- `K02.relative/actuator=direct/stratum=cold/h_ns=10000000/competitor=node_generic/contrast=relative/training_condition=A/loss_mode=shared/method=power_correct`: **unresolved**; contrast -0.0071332062742575615, bounds [-0.010857329170438567, -0.0034090833780765577]; gate GR.
- `K02.relative/actuator=direct/stratum=cold/h_ns=10000000/competitor=node_generic/contrast=strict_improvement/training_condition=A/loss_mode=shared/method=power_correct`: **unresolved**; contrast 0.004836606694252273, bounds [-0.0015706032839856402, 0.011243816672490187]; gate GR.
- `K02.relative/actuator=direct/stratum=warm/h_ns=5000000/competitor=node_generic/contrast=relative/training_condition=A/loss_mode=shared/method=power_correct`: **unresolved**; contrast -0.0003974936066863541, bounds [-0.007472201092387013, 0.006677213879014272]; gate GR.
- `K02.relative/actuator=direct/stratum=warm/h_ns=5000000/competitor=node_generic/contrast=strict_improvement/training_condition=A/loss_mode=shared/method=power_correct`: **pass**; contrast 0.022678812739630745, bounds [0.009860616535708472, 0.03549700894355301]; gate GR.
- `K02.relative/actuator=direct/stratum=warm/h_ns=10000000/competitor=node_generic/contrast=relative/training_condition=A/loss_mode=shared/method=power_correct`: **unresolved**; contrast -0.008139534590345021, bounds [-0.012915494037708867, -0.003363575142981176]; gate GR.
- `K02.relative/actuator=direct/stratum=warm/h_ns=10000000/competitor=node_generic/contrast=strict_improvement/training_condition=A/loss_mode=shared/method=power_correct`: **unresolved**; contrast 0.004588204470829861, bounds [-0.0031254262805580513, 0.012301835222217776]; gate GR.

## 3. Implementation and correctness

Implementation checks actually executed are recorded under M00/M10/M20 and the recorded unit runs. See `locks/environment.json`, each `validity.json`, plant XML/compiled manifests, and the independent rigid-body/analytic oracle evidence. A qualification GO does not establish all C01 requirements or a scientific PASS.

## 4. Data and selection

Generated corpus roles, dimensions, units, seeds and hashes are recorded in `data/*/*/*/manifest.json`; normalization uses training transitions only. Nominal learning curves, every attempted optimizer block and selected checkpoints are preserved under the training run manifests. Confirmation data are paired across methods and never used for nominal checkpoint selection.

## 5. Action-effect evidence

Independent primary-error reaggregation checked 180 model/stratum/horizon summaries directly from the full signed q/v matrices and saved reference matrices. See `independent_primary_audit.json` and the decisive GR raw arrays. This checks sensitivity to command, not robot tracking error. Exploratory findings retain their discovery label in `exploratory_action_audit.md`.

| Interval | Stratum | Baseline error | Corrected error | Point reduction | Seeds | Probes per seed |
|---|---|---:|---:|---:|---:|---:|
| 5 ms | cold | 4.2749% | 2.1078% | 50.69% | 5 | 256 |
| 10 ms | cold | 2.3940% | 1.9103% | 20.20% | 5 | 256 |
| 5 ms | warm | 4.6153% | 2.3474% | 49.14% | 5 | 256 |
| 10 ms | warm | 2.5455% | 2.0867% | 18.02% | 5 | 256 |

Each error is the mean of seed-level median normalized Jacobian errors. Reductions are ratios of those means, not means of per-seed ratios; they are point estimates, not gate conclusions. Per-seed values remain in `paired_primary_results.json`.

## 6. Causal attribution

Full control attribution A, B and A−0.5B was not evaluated. Analytic nominal power-mode identity tests do not replace a measured control intervention. No attribution fraction is reported.

## 7. Prediction and control

No confirmatory MPC episodes, tracking improvements or task success rates are available. No claim of better control is made. Unfinished development fits and prediction jobs are recorded as such, rather than promoted to confirmation.

## 8. Robustness and computation

Reference finite-difference, physics-refinement and independent-ODE checks, and same-checkpoint NODE two-half-step audits, accompany each decisive sensitivity run. Numerical uncertainty is propagated into the contrasts. Practical rate/noise/disturbance control remains untested. Measured wall/CPU time and RSS are in `run_index.csv` and per-run resources; parent and child timings overlap and must not be summed blindly. The owner imposed no artificial resource ceilings.

## 9. Statistical integrity

The rejection family is K=280, including criteria for unexecuted later stages. Seed clusters remain paired, and probes are not independent training replications. Inference is conditional on the fixed training corpus and probe library.

- Look 1: alpha 0.025; gate `GR`, outcome `INCONCLUSIVE`. Family-corrected rejection bounds, paired Fieller precision, bootstrap and influence checks are retained in its criterion evidence.

## 10. Replication and audit

The decisive raw sensitivity errors are independently reaggregated for this report. This is not the full M90 replay/retraining replication; C06 has not been established. All interrupted, superseded, stale and failed attempts remain visible in `run_index.csv`; only compatible completed decisive evidence supports the verdict.

## 11. Deviations and limitations

The registered rejection-only execution-order extension is in `config/revisions/MX-mechanism-negative-0004.json` and `rejection_shortcut.md`. It leaves scientific models, data budgets, losses, nominal selection, thresholds and K unchanged; a single qualified prespecified comparator can reject necessary best-baseline superiority. It cannot establish PASS. Other repairs and semantic invalidations are retained under `config/revisions/` and run provenance. The scope excludes contact, hidden delays/activation, unknown tau, physical robot safety, universal superiority and novelty.

## 12. Unresolved questions and resumption

Execution is nonterminal; continue only the exact action permitted by the validated gate, retaining the statistical look ledger and all paired samples.

## 13. Reproduction and artifact index

The exact study source, configuration, wheels and artifact hashes are retained with the evidence. `run_index.csv` lists every observed attempt. `reproduce.sh` defaults to verification/reaggregation; expensive reruns require an explicit argument and current authorization. Missing or altered archived artifacts fail verification.

