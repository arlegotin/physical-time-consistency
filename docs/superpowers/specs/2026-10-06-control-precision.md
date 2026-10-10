# Precision for the two fine-time MLP comparisons

This executes the already authorized precision branch in the owner-amended utility study. The question is whether the corrected head's mean realized control cost is lower than the qualified `mlp_dt` at 5 ms on each actuator chain. No percentage improvement floor is introduced.

The first control look preserves all 1,200 episodes and leaves all eight population contrasts unresolved. Each of the five seed means favors the head over the MLP at 5 ms. Their aggregate cost reductions are 18.38% direct and 20.46% lagged. These remain point estimates. The separate forty-seed prediction evidence, original PTCO ledger, control look 1, trained NODE comparator and its solver limitations remain unchanged.

## Intent and authorization

The owner requested execution on the current branch, continued monitoring, meaningful commits, no artificial resource caps, and useful measured improvements rather than a 50% floor. The existing control design's contract 9 permits prospectively sized additional clusters for live unresolved signs. S13.4 permits targeting the methods needed for a particular unresolved contrast; such a subset cannot establish a full all-claims PASS.

Use inline execution. This is a precision extension of the same experiment, not an architecture, task, objective, endpoint or model-selection revision. Preserve the original scientific implementations and all prior observations. Do not execute new scientific jobs alongside the existing reproduction queue.

## Sampling decision

`reports/control_precision_planning_2026-10-06.json` records the analysis before new outcomes. Estimated within-task contributions are 22.35% direct and 14.59% lagged, below the predeclared 70% episode-expansion rule. Doubling tasks under the same five fits is therefore not the selected continuation.

Increase training-seed clusters from 5 to 10, retaining 0–4 and adding the ascending unused control seeds 5–9. Use twenty paired episode IDs 0–19 per new seed. The t-width projection uses the larger empirical variance estimate and a 1.25 standard-deviation margin. It projects ten seeds for each sign; it does not promise that the new observations will support the sign.

The new workload is two plants × `mlp_dt,power_correct` × five new seeds × twenty episodes = **400 primary episodes**, all at `h_model=5 ms`. The six other control comparisons retain their original five-seed results and look IDs. Do not describe their unchanged intervals as ten-seed results, or claim nominal noninferiority, NODE equivalence, or full-suite superiority from this extension.

## Nominal fits and checkpoint freeze

The exact twenty required nominal coordinates are both actuator chains × the two named models × seeds 5–9, under the same A/shared objective and train/validation corpora. Evaluate eligibility before reusing any existing fit. Only the five direct `power_correct` fits at seeds 5–9 from the previously frozen forty-seed confirmation may be reused; all five must match the same model/data/scaler/nominal-selection semantics and pass independent score and hash checks. Their prediction outcomes were previously observed; this is disclosed. New control tasks for these seeds have not been opened. Reuse is by the complete ascending coordinate list, never by a favorable prediction score.

Every ineligible coordinate is fitted using the existing nominal-only training engine and S07 optimizer/adequacy procedure. Normally this means fifteen new fits and five reused fits; the registry records the actual verified count. A fit's qualification or selection uses only the original train/validation labels, never any old or new fine-time control score. Preserve optimizer extensions, failed fits and every seed.

Create one separate training context under `study/control-precision-look2`. Copy the union of the two original actuator train/validation inputs and early G00/G10/G20 validity evidence from `study/control-development`, using independent file copies and verified hashes. Copy no neural fits, confirmation arrays or development labels. Shared immutable corpus inputs and a common context are sufficient for independent RNG-seeded fits; twelve-context retraining remains a separate reproduction experiment. Run the unchanged plumbing controls for the families actually fitted. The coordinator and training engine source hashes must agree before using the engine.

Freeze all twenty selected manifests, archive and canonical tensor hashes, normalization/data hashes and nominal objectives before any new control outcome. Existing qualified direct-head fits remain external immutable references. New fit outputs belong to the new context. Do not mutate either source context or an old checkpoint.

## Control, raw evidence and numerical qualification

Reuse `poc.control_jobs` through a fresh subprocess for every concrete request. The additive coordinator must not import itself into the scientific producer's process or alter its sealed dependency list. Keep root 6101, candidate streams, 40 ms sensing/replanning/actuation, six knots, twelve variables, CEM 128 × 4 with sixteen elites, the 240 ms horizon, 1.6 s episode, physical cost, exact known activation and 250 µs plant tick unchanged.

Retain all candidates, actual actions, observations, dense truth, fallback events, partial attempts and raw resource records. Verify every manifest and independently reconstruct the realized cost, success and real clocks. New checkpoints must remain fixed throughout evaluation.

Run the existing next-finer-plant check at 125 µs on episode IDs 0–4 for every new coordinate: **100 matched finer episodes**. Combine its measured discrepancies with the already completed full original finer-physics evidence. Apply the same 2% cost, two-percentage-point effect and success/fallback convergence rules. A near-boundary affected cell requires full coverage of its new episode IDs; a changing classification or unconverged sign requires the named further numerical branch. Never use an absent discrepancy as zero.

The existing model-independent planner qualification and 450 same-rate reference runs remain evidence for the unchanged task and evaluator. This extension makes no new comparative claim about the calibrated physics model. The finer reruns independently audit the new physical paths. Do not silently describe a new-seed reference MPC comparison as executed. Any additional reference run must have its own exact registry and measured resource projection.

No NODE weights or inference solver are changed or evaluated in new-seed control. The existing NODE solver envelope and discretization-specific validation limitations remain attached to the original six unchanged comparisons.

## Inference and ledger

Combine all compatible initial MLP/head 5 ms episodes with all 400 new episodes; the two updated contrasts have ten independent training-seed clusters and twenty tasks per seed. Use paired seed means and the existing `contrast_bounds` implementation. Retain the original eight-comparison family and two directional tails: look 2 spends alpha `0.0125`; each tail uses `0.00078125`. The common 20,000 cluster-bootstrap indices use `SeedSequence([9102,2])`, quantile method `linear`. Rare-tail, central-bootstrap, numerical-envelope, zero-variance and leave-one-seed-out rules remain unchanged.

Use the existing utility look ledger with the original design hash and a new complete registry/batch/numerical evidence identity. Preserve look 1; never reset or duplicate spending. Reporting identical evidence is idempotent. No look may be spent from partial, stale, unqualified or missing evidence. Cache the six unchanged comparisons at look 1 without recomputing their inference from unchanged observations.

Lower/higher cost requires the same resolved signed bounds; otherwise report the exact unresolved status. A positive result is scoped to the tested MLP, 5 ms query and actuator chain. If uncertainty or influence remains, prospectively size the next ascending seed block from the live question. Ten is a planned first continuation, not a cap or stopping criterion.

## Implementation boundary

Add `poc/control_precision.py` for the request, nominal freeze and fixed serial dispatch, and `poc/control_precision_result.py` for raw/numerical qualification and the two-contrast report. Tests are `tests/test_control_precision.py` and `tests/test_control_precision_result.py`. Register the complete twelve-part nested contract `MX-control-precision-0001` in `config/revisions/` and `config/milestones.json` before new scientific execution. Do not edit `control_jobs`, the model/training/controller implementations, the original confirmation registry, `control_utility`, or the reproduction sources.

CLIs are `python -m poc.control_precision config/revisions/MX-control-precision-0001.json --dry-run`, the same command with `--prepare` or `--resume`, and `python -m poc.control_precision_result --report`. Prepare may create nominal inputs and registries but may not open a new control outcome. An existing live reproduction queue prevents scientific dispatch; read-only preparation, implementation and tests may continue only when their files are outside its semantic dependency closure.

## Twelve-part execution contract

1. Purpose: resolve the two existing fine-time MLP signs under the owner-amended goal.
2. Entry: valid look 1 and numerical qualification; completed original reproduction before new scientific dispatch; actual local authorization and resource availability.
3. Implementation: the two additive modules, tests, explicit request/registries and isolated nominal context; existing scientific engines remain sealed.
4. Protocol: freeze sample IDs and rules before new outcomes, qualify nominal fits, freeze checkpoints, execute the 400 paired episodes and at least 100 finer matches, then apply the unchanged inference.
5. Commands: the exact CLIs above, focused pytest modules, existing verification/reproduction interfaces for the actual produced artifacts.
6. Artifacts: request and context inventory; all nominal selections and attempts; 400 primary and 100 minimum finer manifests/raw arrays; combined ten-seed statistics; numerical qualification; appended utility look; resource ledger and report.
7. Validity: no fine-time selection, swapped seeds, omitted failures, mutable sources, changed clocks, partial coverage or unsupported numerical defaults. Verify original inputs and all new artifact hashes.
8. Rules: supported signed effect permits that limited claim; overlapping bounds require justified precision; invalid evidence requires repair; a concrete external shortage is BLOCKED. No full original PASS is possible from this subset.
9. Return: reproduce and review a supported limited result; a live unresolved contrast returns to a newly sized precision request; a numerical defect returns to its affected producer. No unchanged lucky retry.
10. Resources: the same-size measured initial primary cohort projects 32,354.43 CPU seconds and 1,005,141,786 artifact bytes for 400 episodes. Additional fitting, 100 finer episodes and audit/reproduction work are separately projected from measured jobs before dispatch. Record actual free bytes and preserve all evidence; no artificial time, epoch or memory cap and no paid service.
11. Recovery: content-addressed locks, complete optimizer/RNG resumption, per-episode deterministic recovery, immutable complete attempts and dependency-aware invalidation. Preserve all old outcomes and spending.
12. Completion: exact scope/coverage/hashes, an honestly evaluated look, reproduction of newly decisive behavior, whole-branch review and a reproducible report. Unresolved signs are never relabeled as wins.
