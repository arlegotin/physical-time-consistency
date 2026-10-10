# Targeted control and efficiency pilot

Approved direction: the owner's 2026-10-07 `proceed`, following the project reassessment. This is a new exploratory extension, `PTCO-EFF-0.1`, not a replacement for the completed PTCO evidence or its statistical ledger.

## Question and useful deliverable

Find whether the discrete power head offers lower realized control cost, lower computation for comparable observed quality, or neither against strong predictors. Report where, by how much, and at what inference and planning cost. A supported modest improvement is useful; no 50% or fivefold superiority floor applies. Development point estimates do not establish statistical superiority or equivalence.

The completed study supports control improvement over `mlp_dt`; it does not establish control superiority over NODE. Repeating that comparison cannot answer the strongest-baseline question. The next pilot tests actual faster command updates and an algebraically equivalent head implementation. A four-joint arm is conditional follow-up after identifying a useful effect or a specific limitation; it is not part of this pilot's implementation.

## Fixed experiment

- Existing smooth two-joint direct and lagged plants; known tau 0.08 s, full state observation, CPU, existing pinned environment.
- Models: optimized `power_correct`, time-scaled `mlp_h`, `node_generic`, `node_structured`, and `physics_fit`. All learned models use A data and the shared one-plus-four-step objective. Reuse the exact qualified seed-9000 development checkpoints; train only the missing two `mlp_h` fits with the original nominal recipe and learning controls.
- Ten paired development tasks, episode IDs 0–9, root 7101, seed 9000. No relabeling as confirmation. Calibrated physics has one fit per plant, not ten independent fitted models.
- Observations and replanning remain 40 ms. Actual command holds are 40 and 10 ms, using the existing six knots and twelve optimization variables. CEM remains the qualified 128 candidates, four iterations, sixteen elites.
- Prediction partitions are 40 ms and 5 ms. Each interval is split at every actual command boundary: a nominal 40 ms partition with 10 ms commands makes 10 ms queries. Report the actual work, not a misleading nominal timestep. Both resolutions are measured for every model, preventing a conveniently forced fine-step NODE comparison.
- Initially 400 complete exploratory episodes (5 models × 2 plants × 2 command rates × 2 partitions × 10 tasks). Measure a representative job before projecting the remaining batch. Execute serially, preserve failures and raw candidates, verify hashes and reuse completed compatible jobs. No arbitrary compute or seed caps.
- True-plant MPC audits use the first two development tasks at each actual command rate, with the coarsest legal partition: eight additional episodes. This calibrates interpretation, not a claim of globally optimal robot control.

## Equivalent head optimization

Leave `poc/models/power.py` and all historical producers unchanged. A new wrapper uses the same selected weights and scalers.

For direct actuation with correct powers `(2,1)`, the zero-command terms cancel exactly:

`q' = q + h*v + (h/h0)^2*scale_q*N(s,u,h)_q`

`v' = v + (h/h0)*scale_v*N(s,u,h)_v`.

This needs one network pass. Other direct power modes retain the original split formula. For lagged actuation, evaluate zero-command and selected-command features in one concatenated batch, then apply the original formula; initial activation is identical in both feature rows and endpoint activation remains exact. This is one dispatch with twice the neural row work, not a one-evaluation lagged model. Network dispatch counts and network row counts must be distinguished from the existing candidate-transition counter.

The wrapper accepts only the existing pointwise Linear/Tanh architecture in evaluation mode; reject stateful, dropout, batch-normalized or unknown networks. Require float64 equivalence of outputs, state/command Jacobians and weight gradients at 0, 2.5, 5, 10, 20, 40 and 60 ms, including mixed zero/nonzero batch intervals, nonzero initial activation and all three power modes. Float32 is a separate smoke check. Validate four-step rollout equality and exact known activation. Report measured selected-plan/trajectory differences if roundoff changes a tie; do not promise bitwise closed-loop identity after rearranging arithmetic.

## Measurement and evidence

Profile each selected model at batch 1 and 128 with 50 warm-up calls, then 500 and 100 measured calls respectively, at 5, 10 and 40 ms. Measure original and optimized heads using the same weights. Record median/p95, parameter count, calls, vector-field evaluations and neural row work. Do not infer training speed from inference timing; no optimized-training experiment is claimed.

Every control cell preserves the existing episode artifacts, workload and checkpoint identity, observation and action traces, numerical step, cost, success, RMSE, terminal error, fallback count, median/p95 planning latency and work counters. Bind new source/configuration hashes before outcomes. Original reports, code and statistical decisions remain available and unchanged.

The report shows every model/rate/partition, paired task differences, and quality versus planning time. Label the mean-cost/mean-time frontier descriptive and exploratory. Never call overlapping results equivalent, never report only favorable tasks or the best rate as an overall win. A baseline's development-selected resolution can become a future frozen confirmation choice, not a retrospective significance claim. References receive identical physical action parameterization and sensing.

If new effects are material relative to observed numerical discrepancies, prepare a fresh independent confirmation contract for the live comparison. If the strongest models remain comparable with no useful speed edge, report that limitation and choose a targeted data-efficiency or moving-target test only with a separately frozen, feasibility-qualified contract. Do not launch a large factorial or a second robot to search indiscriminately for a win.

## Execution and release

Use the current branch and make explicit commits. Preserve all prior evidence. Add narrow wrapper, pilot producer/coordinator, and report modules; reuse the validated plant, action compiler, CEM and nominal training engine. New requests reject changed checkpoint hashes, missing or duplicate task cells, incomplete runs, incompatible clocks and undeclared inference variants. Interrupted episodes restart their deterministic ID; interrupted fits use the original full optimizer/RNG resume semantics.

The pilot is complete when the optimized wrapper is verified, missing baselines qualify, all 408 episodes and model profiles are hash-verified, the honest quality/compute report exists, and the whole change has been reviewed. It is not a full PTCO PASS. Disk/RAM shortages are surfaced with a concrete projection, rather than deleting evidence or inventing caps. No cloud, paid services, physical robot, publication or push is authorized.

## Twelve-part extension contract

1. Purpose: identify a practical quality/compute edge or limitation against strong baselines, linked to C02/C04/C05 without replacing their completed decisions.
2. Entry: owner approval, intact qualified plants/planner, selected development fits, pinned local CPU environment, and readable existing evidence.
3. Implementation: additive equivalent head wrapper, explicit pilot requests/runner, and raw-to-summary/report consumer; original engines remain unchanged.
4. Protocol: the fixed experiment and profiling sections above; freeze selections before opening control outcomes; validate all paired identities.
5. Commands: `python -m poc.efficiency_pilot prepare`, `nominal`, `episodes`, `run`; `python -m poc.efficiency_result profile` and `report`. Run the plan's focused tests and final full suite.
6. Artifacts: prospective pretraining freeze, nominal selections, complete episode registry, immutable M40 manifests/raw files, batch/attempt receipts, profiles, JSON/Markdown results and SVG frontier.
7. Validity: algebraic/derivative equivalence, qualified fits, actual command-boundary splitting, fixed sensing, complete hash closure and raw-cost reconciliation. No discarded episodes or undeclared inference variant.
8. Decision: valid complete coverage finishes this exploratory pilot; signed point differences remain exploratory. Incomplete/stale evidence is invalid; a concrete missing resource is blocked. No population superiority/equivalence or full PTCO PASS is authorized.
9. Next: confirm a live useful comparison on fresh independent samples; otherwise state the specific limit and freeze a targeted follow-up. Four-joint validation follows a demonstrated reason, not an automatic rescue.
10. Resources: project from measured representative jobs; prior artifacts are about 2.5 MiB/episode, roughly 1 GiB total before fits/profiles. Runtime varies by model/query rate; no claimed measured pilot runtime until execution. No owner-imposed artificial caps.
11. Recovery: retain failed attempts, resume fits with optimizer/RNG and deterministic episode IDs, repair a reproduced defect and explicitly revise affected freezes if source semantics change. Never select a favorable duplicate.
12. Completion: all 408 episode cells, profiles, source/input closure and honest local report verified; meaningful commits and final whole-change review. Remaining confirmation/complexity work is labeled not run.
