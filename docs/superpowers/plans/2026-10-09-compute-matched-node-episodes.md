# Targeted episode precision, look 3

This is the existing task-expansion branch, with unchanged trained models,
robot, objective, clocks, timing-selected search budgets and eight contrasts.
The sixteen-seed second look is preserved with its report source at Git
`b84b2ec`. It measures a 0.9002% coarse AffineFlow advantage over coarse RK4,
but the Holm-bound interpretation changes under seed omission. The fine
AffineFlow/fine RK4 comparison also has an unstable unadjusted interpretation.

1. Purpose: distinguish a dependable small control improvement from a
   task-sensitive result. No improvement percentage floor is introduced.
2. Entry: all 960 primary episodes, 192 replay/refinement comparisons and
   current first/last comparator tensor checks passed. Look 2 remains live.
3. Implementation: additive `poc.compute_study_episodes`, an independent
   request/freeze/store, and look-3 joining in the report. Frozen producers
   and selected weights are unchanged; there is no additional training.
4. Protocol: retain all sixteen training seeds and tasks 0–9. Add tasks
   10–19 for every seed and all six model/query configurations, using the
   existing namespace and candidate-stream prefixes. There are 960 new
   primary episodes and 1920 combined primary episodes. Measured within-task
   variance accounts for 84.41% of coarse RK4-contrast variance and 71.98%
   of fine RK4-contrast variance, both above the prespecified 70% rule.
   Doubling independent tasks is therefore the appropriate precision source;
   these tasks do not count as new independent training seeds.
5. Commands: `python -m poc.compute_study_episodes prepare`, then `controls`
   and `reproduce` in separate serial processes; finally
   `python -m poc.compute_study_report aggregate --look 3`.
6. Artifacts: frozen request, complete new task receipts, 96 fresh-process
   replays and 96 finer-physics runs on task 10 of each seed/configuration,
   joined raw-linked result and immutable third spending record.
7. Validity: prior checks apply. Task/model pairing, file closure, physical
   cost arithmetic, explicit failures and numerical envelopes remain
   mandatory. Replay covers tasks 0 and 10 in the combined cohort. Existing
   first/last selected-tensor replications remain applicable because weights
   and training definitions have not changed.
8. Decisions: Holm family remains eight; look alpha is 0.00625. A supported
   query interval must beat all four NODE options, have stable influence,
   matching measured compute and zero intrinsic fallback. A stable, precise
   non-detection is reportable without equivalence. An unresolved live
   diagnostic requests a justified next block, not a favorable rerun.
9. Next: reportable result -> saved-motion demo, instructions, full software
   suite and one fresh whole-change review. Unresolved -> the named precision
   branch. A real missing resource -> an exact blocker; no artificial cap.
10. Resources: measured control jobs project about 3.13 CPU hours, 3.15 wall
    hours and 8.5 GiB additional artifacts including reproduction. Recorded
    process peak RSS is about 0.52 GiB. The authorized Data volume has about
    86 GiB free. These are projections, not scientific stopping limits.
11. Recovery: content-addressed jobs and original atomic boundary rules apply.
    Preserve interruptions and all old observations. Reject stale inputs,
    duplicated tasks or partial runs; do not select the better attempt.
12. Completion: all declared tasks and replays pass raw validation; joined
    sixteen-cluster inference resolves the live question or records its
    exact continuation. Presentation and review are required before release.

The source variance calculation is preserved in
`reports/provenance/compute_study_look_002_precision_analysis.json`.
