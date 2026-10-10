# Targeted control and efficiency pilot

Complete exploratory pilot on the two-joint robot. All results use one development training seed and ten paired tasks per main cell. These measurements do not establish population superiority, equivalence, real-time control, or a full PTCO PASS.

Actual commands update every 40 or 10 ms; sensing and replanning stay at 40 ms. Every predictor splits at physical command changes. Both coarse and 5 ms queries use the same CEM 128/4/16 budget and six torque knots.

## Quality and planning cost

Planning seconds are the mean of per-episode median latency. Lower realized cost is better. All successful and failed-policy episodes are retained.

| Plant | Command hold ms | Model | Maximum query ms | Mean cost | Planning s | Success / tasks | Fallbacks |
|---|---:|---|---:|---:|---:|---:|---:|
| direct | 40 | power_correct | 40 | 0.210674 | 0.645 | 3/10 | 0 |
| direct | 40 | power_correct | 5 | 0.213950 | 2.107 | 0/10 | 0 |
| direct | 10 | power_correct | 10 | 0.208965 | 1.522 | 4/10 | 0 |
| direct | 10 | power_correct | 5 | 0.212159 | 2.480 | 2/10 | 0 |
| direct | 40 | mlp_h | 40 | 0.211222 | 0.740 | 2/10 | 0 |
| direct | 40 | mlp_h | 5 | 0.213362 | 2.212 | 4/10 | 0 |
| direct | 10 | mlp_h | 10 | 0.211385 | 1.612 | 2/10 | 0 |
| direct | 10 | mlp_h | 5 | 0.211594 | 2.511 | 3/10 | 0 |
| direct | 40 | node_generic | 40 | 0.212047 | 0.671 | 2/10 | 0 |
| direct | 40 | node_generic | 5 | 0.212047 | 2.215 | 2/10 | 0 |
| direct | 10 | node_generic | 10 | 0.206823 | 1.522 | 2/10 | 0 |
| direct | 10 | node_generic | 5 | 0.206823 | 2.284 | 2/10 | 0 |
| direct | 40 | node_structured | 40 | 0.209514 | 0.669 | 2/10 | 0 |
| direct | 40 | node_structured | 5 | 0.209514 | 2.083 | 2/10 | 0 |
| direct | 10 | node_structured | 10 | 0.211315 | 1.452 | 1/10 | 0 |
| direct | 10 | node_structured | 5 | 0.211315 | 2.438 | 1/10 | 0 |
| direct | 40 | physics_fit | 40 | 0.210443 | 0.937 | 6/10 | 0 |
| direct | 40 | physics_fit | 5 | 0.210443 | 2.292 | 6/10 | 0 |
| direct | 10 | physics_fit | 10 | 0.208220 | 1.652 | 3/10 | 0 |
| direct | 10 | physics_fit | 5 | 0.208220 | 2.377 | 3/10 | 0 |
| lagged | 40 | power_correct | 40 | 0.241825 | 0.658 | 6/10 | 0 |
| lagged | 40 | power_correct | 5 | 0.241202 | 1.975 | 4/10 | 0 |
| lagged | 10 | power_correct | 10 | 0.244960 | 1.401 | 7/10 | 0 |
| lagged | 10 | power_correct | 5 | 0.246943 | 2.171 | 6/10 | 0 |
| lagged | 40 | mlp_h | 40 | 0.242336 | 0.660 | 6/10 | 0 |
| lagged | 40 | mlp_h | 5 | 0.248195 | 2.043 | 4/10 | 0 |
| lagged | 10 | mlp_h | 10 | 0.250321 | 1.446 | 7/10 | 0 |
| lagged | 10 | mlp_h | 5 | 0.253941 | 2.264 | 7/10 | 0 |
| lagged | 40 | node_generic | 40 | 0.243177 | 0.693 | 5/10 | 0 |
| lagged | 40 | node_generic | 5 | 0.243018 | 2.151 | 6/10 | 0 |
| lagged | 10 | node_generic | 10 | 0.243252 | 1.447 | 5/10 | 0 |
| lagged | 10 | node_generic | 5 | 0.243252 | 2.266 | 5/10 | 0 |
| lagged | 40 | node_structured | 40 | 0.242069 | 0.677 | 5/10 | 0 |
| lagged | 40 | node_structured | 5 | 0.241337 | 2.241 | 6/10 | 0 |
| lagged | 10 | node_structured | 10 | 0.242301 | 1.446 | 8/10 | 0 |
| lagged | 10 | node_structured | 5 | 0.242301 | 2.271 | 8/10 | 0 |
| lagged | 40 | physics_fit | 40 | 0.240240 | 0.946 | 6/10 | 0 |
| lagged | 40 | physics_fit | 5 | 0.240240 | 2.223 | 6/10 | 0 |
| lagged | 10 | physics_fit | 10 | 0.243857 | 1.662 | 6/10 | 0 |
| lagged | 10 | physics_fit | 5 | 0.243857 | 2.385 | 6/10 | 0 |

## Head versus each strong competitor

Positive percentages favor the head; negative percentages favor the competitor. Task counts show the number with lower head cost, not independent trained-model replications.

| Plant | Hold / query ms | Competitor | Cost reduction | Planning time reduction | Lower head cost / paired tasks |
|---|---|---|---:|---:|---:|
| direct | 40 / 40 | mlp_h | +0.26% | +12.75% | 6/10 |
| direct | 40 / 40 | node_generic | +0.65% | +3.78% | 6/10 |
| direct | 40 / 40 | node_structured | -0.55% | +3.49% | 5/10 |
| direct | 40 / 40 | physics_fit | -0.11% | +31.12% | 6/10 |
| direct | 40 / 5 | mlp_h | -0.28% | +4.75% | 6/10 |
| direct | 40 / 5 | node_generic | -0.90% | +4.90% | 5/10 |
| direct | 40 / 5 | node_structured | -2.12% | -1.16% | 3/10 |
| direct | 40 / 5 | physics_fit | -1.67% | +8.09% | 3/10 |
| direct | 10 / 10 | mlp_h | +1.14% | +5.63% | 5/10 |
| direct | 10 / 10 | node_generic | -1.04% | +0.06% | 2/10 |
| direct | 10 / 10 | node_structured | +1.11% | -4.77% | 5/10 |
| direct | 10 / 10 | physics_fit | -0.36% | +7.90% | 4/10 |
| direct | 10 / 5 | mlp_h | -0.27% | +1.26% | 4/10 |
| direct | 10 / 5 | node_generic | -2.58% | -8.56% | 2/10 |
| direct | 10 / 5 | node_structured | -0.40% | -1.72% | 4/10 |
| direct | 10 / 5 | physics_fit | -1.89% | -4.31% | 2/10 |
| lagged | 40 / 40 | mlp_h | +0.21% | +0.22% | 5/10 |
| lagged | 40 / 40 | node_generic | +0.56% | +5.03% | 8/10 |
| lagged | 40 / 40 | node_structured | +0.10% | +2.68% | 5/10 |
| lagged | 40 / 40 | physics_fit | -0.66% | +30.41% | 2/10 |
| lagged | 40 / 5 | mlp_h | +2.82% | +3.32% | 8/10 |
| lagged | 40 / 5 | node_generic | +0.75% | +8.16% | 7/10 |
| lagged | 40 / 5 | node_structured | +0.06% | +11.86% | 5/10 |
| lagged | 40 / 5 | physics_fit | -0.40% | +11.14% | 5/10 |
| lagged | 10 / 10 | mlp_h | +2.14% | +3.12% | 7/10 |
| lagged | 10 / 10 | node_generic | -0.70% | +3.20% | 2/10 |
| lagged | 10 / 10 | node_structured | -1.10% | +3.16% | 1/10 |
| lagged | 10 / 10 | physics_fit | -0.45% | +15.73% | 3/10 |
| lagged | 10 / 5 | mlp_h | +2.76% | +4.08% | 8/10 |
| lagged | 10 / 5 | node_generic | -1.52% | +4.21% | 5/10 |
| lagged | 10 / 5 | node_structured | -1.92% | +4.41% | 4/10 |
| lagged | 10 / 5 | physics_fit | -1.27% | +8.98% | 2/10 |

## Inference timing

Median CPU float64 prediction time at a 10 ms query. Batch one represents a single prediction; batch 128 represents one CEM candidate bank. The complete 5/10/40 ms measurements, p95 values and raw samples are archived in the profile artifact. These measurements do not measure training speed.

| Plant | Model / implementation | Batch one microseconds | Batch 128 microseconds | Neural rows / 128 candidates |
|---|---|---:|---:|---:|
| direct | power_correct / original | 152.2 | 451.3 | 256 |
| direct | power_correct / efficient | 97.2 | 264.0 | 128 |
| direct | mlp_h / original | 93.2 | 263.6 | 128 |
| direct | node_generic / original | 282.5 | 924.1 | 512 |
| direct | node_structured / original | 259.5 | 902.1 | 512 |
| direct | physics_fit / original | 2364.2 | 2818.4 | 0 |
| lagged | power_correct / original | 165.2 | 485.1 | 256 |
| lagged | power_correct / efficient | 146.0 | 428.9 | 256 |
| lagged | mlp_h / original | 103.5 | 277.9 | 128 |
| lagged | node_generic / original | 336.9 | 952.6 | 512 |
| lagged | node_structured / original | 297.9 | 942.1 | 512 |
| lagged | physics_fit / original | 2578.3 | 3081.5 | 0 |

## Interpretation and limits

The direct optimized head uses one neural pass. The lagged head batches two evaluations into one dispatch and still performs twice the neural row work. These are inference optimizations; no training-speed gain is claimed.

The quality/time frontier compares both prediction resolutions within each physical condition. Its membership is descriptive. A convenient forced fine-step comparison does not establish that a model is faster than a NODE using its coarsest valid step.

Eight true-plant audits cover only tasks 0–1 at each plant/rate. Their two-task means are excluded from the ten-task frontier. The calibrated mechanics reference has a stronger known-topology prior and one fitted model per plant. True-plant CEM is a reference, not a globally optimal controller.

Fresh independent confirmation must freeze the chosen practical comparison before new outcomes. The four-joint robot, reduced-data training, moving targets, and new perturbation tests have not been run in this pilot.

## Reproduce

Selected checkpoints, requests, source hashes and all complete M40 manifests are bound by `config/efficiency_pilot_freeze.json`, `config/efficiency_pilot_cells.json` and `reports/efficiency_pilot_batch.json`. Raw costs are recomputed from true trajectories and actual commands.

Run `.venv/bin/python -m poc.efficiency_result report` for verification/reaggregation. `profile` repeats the recorded timing procedure; `python -m poc.efficiency_pilot run` executes only missing compatible jobs. These commands do not spend a new statistical look.

See `efficiency_pilot_result.json` for all condition metrics, paired task differences and evidence hashes; its `profiles_artifact` points to immutable inference timing samples; `efficiency_pilot_quality_compute.svg` shows the descriptive comparison. Open `efficiency_pilot_demo.html` locally for side-by-side replay of every main pilot task.
