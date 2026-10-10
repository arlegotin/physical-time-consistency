# Measured inference speed — PTCO-EFF-0.1

CPU float64, one thread, fixed representative input, 10 ms prediction. These are inference measurements; control quality and training speed are separate questions. All models use their frozen selected weights.

| Plant | Model | One state, µs | 128 states, µs |
|---|---|---:|---:|
| direct | Original head | 152.2 | 451.3 |
| direct | Optimized head | 97.2 | 264.0 |
| direct | Time-scaled MLP | 93.2 | 263.6 |
| direct | Generic NODE | 282.5 | 924.1 |
| direct | Structured NODE | 259.5 | 902.1 |
| direct | Fitted mechanics reference | 2364.2 | 2818.4 |
| lagged | Original head | 165.2 | 485.1 |
| lagged | Optimized head | 146.0 | 428.9 |
| lagged | Time-scaled MLP | 103.5 | 277.9 |
| lagged | Generic NODE | 336.9 | 952.6 |
| lagged | Structured NODE | 297.9 | 942.1 |
| lagged | Fitted mechanics reference | 2578.3 | 3081.5 |

Each cell retains 50 warm-up calls, then 500 batch-one or 100 batch-128 measurements. Raw samples, p95, physical inputs, parameter counts and work counters are archived. Direct-head simplification performs one neural row per state. The lagged head batches two rows per state; one dispatch does not halve its neural arithmetic.

The optimized head is faster than both NODEs for this matched single query. This does not establish a faster full controller: a NODE may use larger legal timesteps, and planner overhead is measured separately in the control pilot. The calibrated reference uses finer integration and a greater mechanics prior.

Frozen input SHA-256: `39bacb772bcc4af7e44879ea2d345c35e5247c7916f161c535c5a486c430d132`.
Selected uncontended measurement: [`raw profile`](efficiency_pilot_profile_runs/7b3742697218d5af15bc1a74ba2dce677c16e51dac46dc95f1acb1231082a3d7.json), 2026-10-07T11:59:48.400173Z through 2026-10-07T12:00:18.580657Z.

An earlier complete timing attempt overlapped the tail of a one-second verification test during profiler startup. It is preserved separately. The second complete run was selected prospectively because no test/training worker competed, without selecting timings by which run was faster.

Control outcomes are pending. No population advantage, equivalence, real-time robot result or training-speed result is inferred from these timings.
