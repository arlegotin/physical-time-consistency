# Practical findings from the completed pilot

The power head is a competitive, inexpensive predictor. This pilot does not establish better control than the strongest NODE models. Its clearest advantage is cheaper neural inference; its control gains depend on the actuator and timing condition.

All 408 planned episodes completed: 400 main comparisons and eight simulator audits. There were zero failed-search fallbacks. Each main condition uses the same ten tasks and one development training seed. The differences below are descriptive, without a new statistical look or a population superiority/equivalence claim.

## Where it wins and loses

The earlier, separately confirmed study found **22.58% lower tracking cost on direct torque and 26.04% lower cost with actuator lag versus the ordinary interval-conditioned MLP**, across ten training seeds. That result remains useful. It does not establish superiority over the stronger predictors in this new pilot. See the [confirmed utility report](utility_results_2026-10-07.md).

Against the stronger time-scaled MLP, the new direct-torque comparisons mostly tie: head cost differences range from a 0.28% loss to a 1.14% gain. With actuator lag, the head has lower mean cost in all four timing conditions, by **0.21–2.82%**; the two fine-query conditions favor it on eight of ten tasks.

Against NODE, the result is mixed. At 40 ms command holds on the lagged plant, the head has 0.56–0.75% lower cost than generic NODE and 0.06–0.10% lower cost than structured NODE. The structured comparison favors the head on only five of ten tasks. With 10 ms command holds, the head costs **1.10–1.92% more than structured NODE**. Direct-torque NODE comparisons also include gains and losses; the head costs 1.04–2.58% more than generic NODE at the faster command rate.

For example, with actuator lag, 40 ms commands and 5 ms queries:

| Predictor | Mean tracking cost | Mean episode median planning time |
|---|---:|---:|
| Power head | 0.241202 | 1.975 s |
| Time-scaled MLP | 0.248195 | 2.043 s |
| Generic NODE | 0.243018 | 2.151 s |
| Structured NODE | 0.241337 | 2.241 s |
| Fitted mechanics | 0.240240 | 2.223 s |

The head's 11.86% planning-time advantage over structured NODE in that table is a **matched fine-query comparison**. Structured NODE can use 40 ms queries, costing 0.242069 with 0.677 s planning. That alternative is almost three times faster than the fine-query head, with a small cost difference. The head can also use 40 ms queries: 0.241825 cost and 0.658 s planning. All resolutions appear in the [complete report](efficiency_pilot_result.md) and [quality/time plot](efficiency_pilot_quality_compute.svg).

## What the speed advantage means

At a 10 ms query, a single optimized head prediction takes approximately **97 microseconds on direct torque and 146 microseconds with lag**. The NODEs take 260–282 and 298–337 microseconds respectively: roughly **2–3 times as long**. A batch of 128 predictions favors the head by about 2.2–3.5 times. These are measured CPU float64 inference times on one representative input; full raw timing samples and p95 values are retained.

The direct optimization removes a redundant neural pass algebraically. The lagged optimization batches its two necessary passes into one dispatch; it still computes two neural rows per state. Twenty nominal closed-loop comparisons, ten per plant, matched the original implementation's commands, states, activation and costs exactly.

The time-scaled MLP remains about as cheap as the direct head and cheaper than the lagged head. Whole-controller timing also includes action-path handling, state/history bookkeeping and scoring. The pilot does not quantify each overhead component or establish that the 2–3-fold neural speedup becomes a controller speedup. No training-speed improvement was implemented or inferred; measured selected-fit training effort is in the [training budget](efficiency_pilot_training_budget.md).

## Is the comparison fair?

The learned competitors use the same training corpus, shared one-plus-four-step objective, full observed state, known actuator time constant, sensing/replanning clocks, torque limits, six-knot action representation, CEM budget and paired tasks. Networks have roughly 18,000 parameters. Every method has the same prespecified nominal optimizer opportunity; converged fits can take different numbers of epochs. Checkpoints are selected by nominal validation, without fine-query or control tuning. Both query resolutions are available to every model.

The fitted mechanics reference additionally knows the robot's mechanical topology. Its fitted parameters come from the same transition data, without privileged simulator masses or accelerations. It is reported as a reference with a stronger prior. The eight simulator audits use only tasks 0–1 at each plant/rate, so their means are excluded from the ten-task comparison frontier. True-simulator CEM is a finite search reference, not a globally optimal controller.

Fairness here concerns the specified implementations and budgets. This is not proof that every architecture has attained its best possible fit, nor a broad state-of-the-art robotics benchmark.

## A targeted clue about the remaining limitation

A post-pilot diagnostic checked the saved selected plans against their actual first 40 ms outcomes. With lag, 10 ms commands and 5 ms queries, mean task-level position prediction RMSE is about **0.000351 rad for the head**, **0.000111 rad for structured NODE**, and **0.001144 rad for the time-scaled MLP**. Velocity prediction RMSE is about 0.0176, 0.00549 and 0.0342 rad/s respectively.

The head improves substantially on the MLP's full-motion predictions there, but NODE remains more accurate. Correct short-time action order does not ensure accurate finite-time gain, drift or composition. Each policy visits different states and chooses different commands; this diagnostic does not isolate which of those mechanisms causes the gap. It compares only the actually executed first interval, not a six-step plan against a future trajectory changed by replanning. Every source file and endpoint match is checked in the [diagnostic receipt](efficiency_pilot_first_interval_diagnostic.json).

The next useful scientific test is a small **shared-state decomposition of drift, command response and composition**. Freeze the first 64 cold and warm development probes per plant before evaluating the selected head, time-scaled MLP and NODEs at 40, 10 and 5 ms. Compare zero-command predictions, command-induced increments, and held-command one-step versus two-half-step predictions against the qualified plant/oracle. Preserve initial activation when changing the command. Report vector errors and numerical envelopes without changing any success thresholds. This would distinguish specific model limitations before retraining or adding another robot. **It has not been run.**

Separately, instrument one representative controller decision to measure the shared overhead. Only then consider optimizing that common code and comparing quality at equal measured compute budgets. A four-joint study, reduced-data study or moving-target task should answer a stated new question; none has run in this pilot.

## Artifacts and verification

The strict consumer verified all 408 manifests and their raw artifacts and independently recomputed realized costs from true trajectories and executed command segments. The batch used approximately **7.91 CPU hours**, **7.94 producer wall hours**, **0.96 GiB of logical evidence**, and at most **403 MiB producer peak RSS**. These totals cover the completed episode producers, not training, interpreter startup, profiling, verification, interrupted attempts or the earlier study.

The complete source suite passed 952 tests after the final source change. The completion check passed all 25 focused head/pilot/report tests. The replay passed browser checks for all 80 task/plant/rate/resolution selections and 1,200 rendered model frames; the plot and replay were also inspected visually. See the [release record](efficiency_pilot_release_review.md) and [completion receipt](efficiency_pilot_completion.json).

Open the [standalone replay](efficiency_pilot_demo.html) locally to compare all five controllers on any of the ten tasks. It replays measured physical simulation time, while separately reporting offline planning time. The [full JSON](efficiency_pilot_result.json) retains every condition, paired task difference and evidence hash. `.venv/bin/python -m poc.efficiency_result report` verifies and regenerates the primary report from the complete local artifacts.
