# Completed primary control runtime

These are descriptive measurements from the completed 1,200-episode control cohort, with no statistical look or speed-superiority decision. All 48,000 raw decision timers were checked, and all 1,200 episode planner medians, resource records, manifest hashes and completion markers reconciled. See the [runtime receipt](control_primary_runtime_2026-10-04.json) for exact values and per-episode provenance.

The corrected head uses two network evaluations per predicted transition; the generic NODE uses four. Both evaluate the same number of candidate transitions. Half the network evaluations does not imply half the controller runtime: batched execution, physical simulation, planning and logging contribute.

| Plant | Model interval | MLP planner seconds | NODE planner seconds | Corrected-head planner seconds | Head time reduction versus NODE |
|---|---:|---:|---:|---:|---:|
| Direct torque | 5 ms | 1.918 | 2.058 | 2.030 | 1.37% |
| Actuator lag | 5 ms | 1.960 | 2.126 | 1.983 | 6.71% |
| Direct torque | 40 ms | 0.628 | 0.647 | 0.658 | -1.70% |
| Actuator lag | 40 ms | 0.644 | 0.664 | 0.636 | 4.27% |

Each table value is the median of 100 episode-level planner medians; each episode contains 40 planning decisions. These are actual elapsed planning times on the authorized CPU, with the same frozen search budget. At 5 ms, median total episode CPU time is 83.32 seconds for the head versus 83.40 for NODE on direct torque, and 80.73 versus 86.27 with actuator lag. End-to-end direct-torque CPU use is therefore effectively tied despite the head's lower evaluation count.

The [completed control point estimates](control_primary_descriptive_2026-10-04.md) show roughly similar tracking cost for the head and NODE, while both outperform the MLP at the finer interval. Their strict task-success results differ and remain visible in that report. The measured runtime advantage is modest and depends on the plant and interval; the head is slower on the nominal direct-torque condition. These serial study timings are not an isolated randomized speed benchmark. Training efficiency has not been compared, and the remaining reference, solver, uncertainty and reproduction checks are separate from these runtime measurements.
