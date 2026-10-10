# What the new same-state test tells us

The selected power head has clear command-response advantages, but it is not the most accurate model. On identical states and commands, the head often predicts the new command's effect on position better than the time-scaled MLP and generic NODE. Structured NODE remains more accurate at the decisive short intervals. The head also has a separate weakness in predicting motion already underway and in composing repeated steps. These findings are consistent with the timing hypothesis; this comparison uses different architectures and learned weights, so it does not isolate temporal powers as the causal reason for those advantages.

This diagnostic completed all **3,072 model/probe/horizon rows**, using the pilot's existing eight selected checkpoints. Each plant has 64 initial and 64 warm development states, evaluated at 5, 10 and 40 ms. There was no new training, MPC, checkpoint selection or statistical look. Three simulator resolutions and independent continuous equations agree: the largest empirical q/v endpoint envelopes are about 2.35e-13 rad and 2.52e-12 rad/s. The measurement took 25.1 seconds and produced about 1.1 MB of immutable evidence.

## Where the selected head wins

At **5 ms**, new-command position error is **73–80% lower than generic NODE**, across both plants and both state strata. It is **92–99% lower than the time-scaled MLP**. These are means of paired two-joint response-error norms in this fixed probe library, not control-cost reductions or independent-seed confidence statements. Against generic NODE, the head wins 58–60 of 64 position-response probes in each cell.

The velocity response is different. With lag at 5 ms, head command-response error is about **35% lower than generic NODE** and **94–97% lower than the time-scaled MLP**. With direct torque, both NODEs predict velocity response more accurately than the head. At 10 ms with lag, the velocity advantage over generic NODE also disappears in the warm stratum. The full table retains these losses.

Structured NODE is the strongest short-interval response comparator here. At 5/10 ms, its position and velocity command-response errors are lower in every plant/stratum/component cell. The head's good result against unrestricted generic dynamics is consistent with the value of physical structure, while the stronger structured comparator limits the superiority claim for this particular head.

## What limits the head

With actuator lag at 5 ms, averaging the equally sized cold/warm probe groups:

| Mean two-joint error norm | Power head | Generic NODE | Structured NODE |
|---|---:|---:|---:|
| New-command position response, rad | 6.85e-7 | 2.98e-6 | 1.48e-7 |
| New-command velocity response, rad/s | 2.22e-4 | 3.42e-4 | 8.57e-5 |
| Zero-new-command velocity prediction, rad/s | 5.29e-3 | 2.01e-3 | 2.29e-3 |
| Total velocity prediction, rad/s | 5.29e-3 | 2.01e-3 | 2.28e-3 |

The head's zero-new-command velocity error is about **24 times** its command-response error in that example. “Zero new command” retains the initial velocity and activation: the robot is still moving and the actuator can still exert torque. This identifies a substantial drift limitation even where the command-scaling mechanism beats generic NODE. It does not assign a causal percentage of closed-loop tracking cost to drift; signed drift and command errors can cancel, and their squared cross terms are retained.

Splitting the same 5 ms lagged command hold into two 2.5 ms predictions raises the head's mean velocity truth error from 0.00529 to 0.00564 rad/s. At 10 ms, splitting raises it from 0.00865 to 0.00985 rad/s. NODE predictions change far less under these refinements. Correct asymptotic powers do not guarantee a composable or accurate finite-time flow. Simply querying this head more often is not a general accuracy improvement.

The split prediction uses two transition queries instead of one for every model. For NODE, this refines one RK4 step to two half steps: **four versus eight vector-field evaluations**. For the power head and MLP it composes two learned maps. No new observations are supplied. This is an additional-computation diagnostic, not an equal-cost inference comparison.

On direct torque, the head's total position error at 5 ms beats generic NODE by 66–74%, but structured NODE is more accurate. The head's total velocity error is worse than both NODEs and also worse than the time-scaled MLP in each short-interval stratum. This prevents a position-only win from becoming a claim of overall dynamics superiority.

## Practical conclusion and next direction

The shareable evidence now has three distinct levels:

- **Confirmed control utility against the ordinary MLP:** the prior ten-seed study found 22.58% lower direct-torque tracking cost and 26.04% lower lagged cost. See [the utility report](utility_results_2026-10-07.md).
- **Compute and strong-baseline control tradeoff:** the completed pilot measured roughly 2–3 times cheaper individual neural predictions than NODE, with mixed, small control differences against the strongest models. That is inference speed, not training speed or a guaranteed whole-controller speedup. See [the pilot findings](efficiency_pilot_findings.md) and [saved simulation replay](efficiency_pilot_demo.html).
- **Command response and remaining limitation:** this new shared-state diagnostic establishes descriptive position-response wins against unrestricted dynamics and exposes drift/composition weaknesses against structured dynamics. These comparisons are consistent with the timing hypothesis and do not isolate its causal contribution. See [all measured conditions](shared_state_diagnostic_result.md) and [the raw-linked JSON](shared_state_diagnostic_result.json).

The next method change should target **consistent propagation of existing motion**, while preserving the cheap, correctly timed command response. A response-only gain adjustment or a more complex robot would leave the measured drift/composition problem unresolved. Any changed decoder/objective is a new exploratory method version, followed by fresh independent confirmation before a superiority claim. This diagnostic selects that direction; it does not claim that such a refinement has already been built or works.

The present method is a useful inexpensive alternative, with a demonstrated ordinary-baseline control gain and clear selected-head command-response advantages. Consistent superiority over structured NODE is **not established**. All short-interval components, losses and numerical bounds are available; no favorable subset substitutes for the complete result.

## Verification

```bash
.venv/bin/python -m poc.shared_state_diagnostic verify
```

The verifier checks the prospective freeze, original checkpoint/data identities, immutable completed run, full coordinate coverage, raw-to-summary values and both JSON/Markdown reports. The original qualified corpus is loaded in its preserved nominal source context, with manifest and array bytes required to match the original repository corpus. No stale historical CLI hash is waived. See [the verification receipt](shared_state_diagnostic_verification.json).

Full numerical verification requires the local data, checkpoints and evidence named by the freeze; those binary archives are excluded from Git. The reports and standalone simulation replay can be shared directly, while a Git clone alone does not restore the archived experiment inputs.
