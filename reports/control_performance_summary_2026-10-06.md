# Measured control performance and speed

The correction is promising in the completed experiment: each of the five training-seed means favors it over the simple MLP predictor when planning in 5 ms steps. The average cost reduction is 18.4% for direct torque and 20.5% with motor lag. Statistical uncertainty about these control advantages remains unresolved; the additional-seed experiment is running under the registered signed-improvement rule, without a fixed percentage floor.

Each table cell below represents 100 episodes: five independently trained models, each evaluated on twenty paired tasks. Lower cost is better. The cost combines target tracking, velocity, command effort and final position error.

| Fine-step metric | Direct torque | Motor lag |
| --- | ---: | ---: |
| Head cost reduction versus MLP | 18.38% | 20.46% |
| Head cost change versus generic NODE | 1.00% higher | 0.13% lower |
| Head median planning time | 2.030 s | 1.983 s |
| MLP median planning time | 1.918 s | 1.960 s |
| Generic NODE median planning time | 2.058 s | 2.126 s |
| Head planning-time reduction versus NODE | 1.37% | 6.71% |
| Strict success: head / MLP / NODE | 25 / 12 / 26 of 100 | 51 / 12 / 62 of 100 |
| Head mean terminal position-error norm | 0.0186 rad | 0.0136 rad |
| MLP mean terminal position-error norm | 0.0658 rad | 0.0779 rad |
| NODE mean terminal position-error norm | 0.0162 rad | 0.0112 rad |

Planning time is the median of the 100 per-episode median end-to-end CPU planning times, using the same CEM budget. These are descriptive timing measurements, without timing uncertainty bounds. The runs are offline: the simulator waits for planning. Two network calls for the head versus four vector-field calls for NODE do not imply a twofold end-to-end speedup.

Relative to the MLP, the head's measured planning time is 5.84% longer with direct torque and 1.22% longer with motor lag. The MLP is the fastest of these three measured fine-step planners; the head improves its control-cost point estimates at a modest extra planning time.

Strict success requires both the position and velocity tolerances at every accepted physics step in the final 200 ms. The head's lagged success count is lower than NODE's despite their close mean costs. At nominal 40 ms model steps, the head's average cost is 0.69% higher than the MLP for direct torque and 0.41% lower with lag; its direct strict success count is 12/100 versus the MLP's 27/100. Mean-cost similarity therefore does not establish equal task success or nominal noninferiority.

No zero-command fallback occurred in the 1,200 original control episodes. Finer-physics checks preserved the observed differences, and the completed reproduction matches the original outcomes. This three-model clean fixed-clock study does not establish NODE equivalence, superiority over the full baseline suite, changed-actuator-rate benefit or perturbation robustness.

The [machine-readable summary](control_performance_summary_2026-10-06.json) records exact values, source hashes, aggregation definitions and reconciliation with all sixty cached seed-level cost summaries. [The original control result](control_confirmation_result.md) retains the signed bounds and unresolved classifications. [The separate prediction finding](mechanism_interpretation.md) concerns command-effect prediction on direct torque and is distinct from these control costs. No new statistical look was spent to generate this descriptive summary.
