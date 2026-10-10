# Targeted affine-flow refinement

Protocol: `PTCO-AFF-0.1` (exploratory). This is a new method version, not a relabeling of the completed PTCO power-head results.

## Purpose and finish line

The user wants reproducible, shareable simulation evidence of when a cheap physical-time-aware predictor excels or loses against strong models, including control quality and actual computation cost. They authorized autonomous local execution on the current branch, meaningful commits, and no artificial resource ceilings. A positive finding cannot be guaranteed. Completion means a measured comparison and an honest conclusion, including a negative result if this targeted refinement does not help. No further architecture search or larger robot is implicit in this plan.

The completed shared-state diagnosis found that the original head's fine-time command response can be good while its prediction of existing motion is inaccurate and poorly composable. The next experiment changes that specific weakness. It does not add another generic audit or tune the task to produce a win.

## Choice

Compare three approaches conceptually: adjusting command gains leaves the measured drift problem untouched; using full RK4 duplicates structured NODE and its four field calls; a two-evaluation affine-flow decoder couples the position/velocity updates and integrates known actuator lag explicitly. Implement only the last approach. Its mechanics prior is stronger than the original independent q/v head and is disclosed. Do not claim research novelty: acceleration affine in applied torque follows from the supplied smooth rigid-body equations.

## Predictor

Add `poc/affine_flow.py` without modifying any frozen model, training or evaluator source. The model exposes the existing `predict(state, command, h_s, history=None, power_mode=None)` API; `power_mode` must be absent. Input to a 128/128 tanh MLP is `[sin(q1),cos(q1),sin(q2),cos(q2),v1/vscale1,v2/vscale2]`. A linear six-output head predicts command-independent `f(q,v)` and a 2x2 torque response `B(q,v)`:

`f = (scale_v / 0.04) * z[:2]`

`B[i,j] = (scale_v[i] / (0.04 * 2)) * z_matrix[i,j]`.

Use default PyTorch initialization, biases, existing CPU float32 training and float64 evaluation. There are 18,182 trainable parameters for either actuator. The network receives no interval, new command, activation, privileged acceleration or simulator parameters. It learns endpoint losses only.

For frozen local coefficients define `step(q,v,f,B,u,w,h)`:

- direct: `dv = h*f + h*B*u`, `dq = h*v + 0.5*h*h*f + 0.5*h*h*B*u`;
- lagged: `dv = h*f + B*(old_v*w + new_v*u)`, `dq = h*v + 0.5*h*h*f + B*(old_q*w + new_q*u)`;
- `new_v = h - tau*(1-exp(-h/tau))`, `new_q = h*h/2 - tau*h + tau*tau*(1-exp(-h/tau))`, `old_v = h-new_v`, `old_q = h*h/2-new_q`, `tau=0.08`.

Evaluate coefficients once at the initial q/v. Apply this frozen-coefficient map for h/2 to obtain a predicted midpoint (including the exact midpoint activation). Evaluate coefficients at that q/v and apply the map over the full h from the original state, original activation and held new command. This is a two-stage approximation, not exact nonlinear integration. Return the same exact endpoint activation as every competitor. At h=0 return an exact no-op without a network evaluation. Negative/nonfinite intervals and invalid dimensions are errors; per-candidate divergence uses the existing failure statuses. Mixed zero/positive batch intervals must preserve zero rows exactly.

Use convergent power series for the small-r lag coefficients, with `expm1` for activation, and float64 tests against existing exact maps and independent high-precision coefficients. The stage activation is not fed into the field network; its torque integral is supplied explicitly. This architecture has the correct direct/lagged new-command minimum orders but does not guarantee fitted gains or accurate control. Each requested prediction performs two network dispatches/two neural rows per candidate; no work is hidden.

## Nominal fitting

Add `poc/affine_experiment.py`, using existing `TrainingSession`, `batch_objective`, nominal validation and checkpoint selection without changing their source. Start with the same qualified A train/validation corpora, train-only scales, seed 9000, shared one-plus-four-step loss, AdamW 1e-3, batch256 and 100-epoch blocks. Continue based only on the existing moving nominal convergence rule and the less-than-half-constant-velocity one-step criterion. The prespecified 3e-4 repair starts from the same initialization if a plateau fails nominal qualification. Preserve every attempted fit. Tiny exact analytic controls precede robot fitting. These old data and the selected architecture are development evidence; they cannot become fresh confirmation for a revision informed by prior results.

Freeze the method, sources, train/validation hashes, checkpoint-selection rule and existing comparator identities before training. Select only using nominal validation. Freeze each selected checkpoint before viewing new fine-time or control outputs. Resume optimizer/RNG/permutation state at complete updates; never choose a checkpoint by fine-time success.

## Measurements and continuation

Reuse the completed shared-state diagnostic's exact probe inputs and qualified reference endpoints. Measure the new predictor on both plants, all 128 probes each, 5/10/40ms, zero-new-command, held-command and two-half-step branches. Report q/v drift, command-response, total and split truth errors by cold/warm stratum beside the four old models. Reuse raw reference arrays only after their complete hash closure passes. This adds 768 model/probe/horizon rows, not new independent training clusters. Benchmark prediction latency at batch1 and batch128 using the pilot's warmup/repeat recipe, and retain actual training CPU/exposure.

Nominally qualified fits proceed to targeted control even if prediction errors disappoint: feedback can compensate. Run the candidate on the exact pilot task/stream IDs, both plants, actuator holds40/10ms and model partitions40/5ms: 80 new episodes, with unchanged CEM128x4/16elites, six knots, physical horizon and costs. Existing pilot comparator episodes are paired reusable observations; reverify raw true-state costs and immutable inputs. Preserve failed candidates and zero-command fallback episodes. Report all eight physical/timing cells against original power head, time-scaled MLP, generic/structured NODE and physics reference. Lower cost and faster measured planning are separate outcomes; no arbitrary superiority floor is imposed.

If the candidate is not a useful quality/compute improvement over the structured comparator in any measured condition, deliver that scoped negative result and the existing original-method wins; do not launch another architecture. If a particular condition improves descriptively, freeze that hypothesis before a separately registered independent confirmation using fresh version-specific data and training-seed namespaces. The full independent contract must name paired comparators, clocks, resource projection and uncertainty before execution; no new outcome-selected claim is described as confirmed. This conditional contract is required scientific work when development supports it, not an optional follow-up left for the user to request.

## Deliverables and limits

Write `reports/affine_refinement_findings.md`, raw-linked machine results, latency/training/control tables and a concise project completion status. Update the standalone replay with candidate trajectories only once they exist. Final conclusion distinguishes original method, new exploratory method and any independent confirmation. The two-link task is the only robot in this refinement; a larger benchmark is justified only by a future separately defined generalization question.

Minimum checks cover exact constant fields, fixed-initial-activation derivatives, physical units, zero/invalid intervals, autograd to every learned output, two-call accounting, interrupted training resumption, nominal-only selection, full raw coordinate coverage and report/hash closure. Existing historical producers and evidence stay unchanged. No paid services, additional dependencies or publication. Actual resource shortages are reported to the owner; no synthetic time/seed cap is a stopping rule.
