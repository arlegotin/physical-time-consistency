# Targeted training-seed precision, look 4

> **For agentic workers:** REQUIRED SUB-SKILL: Continue Task 3 under superpowers:executing-plans. This is its existing precision branch, not a new architecture or task search.

**Goal:** Resolve the training-seed sensitivity of the small coarse AffineFlow cost advantage against the strongest NODE.

**Architecture:** Add an immutable fourth request and producer/report modules; preserve all three prior looks and frozen sources. Keep scientific execution serial. Store mutable progress receipts on the authorized Data volume to avoid the repository-volume quota.

**Tech Stack:** Existing CPU PyTorch, NumPy, SciPy, MuJoCo and standard-library artifacts.

**Spec:** `docs/superpowers/specs/2026-10-09-compute-matched-node.md`.

## Global constraints

- Unchanged fixed data, train-only scales, all three models, shared objective, nominal-only optimizer opportunity and frozen timing budgets.
- Preserve sixteen seeds and tasks 0–19; add ascending seeds 2026100816–2026100837, giving 38 clusters and 760 tasks per configuration.
- All six configurations; 2,640 new primary episodes, 264 new fresh-process replays and 264 finer-physics runs on tasks 0 and 10.
- Eight contrasts, fourth-look alpha 0.003125, existing Holm/bootstrap/influence rules; no effect floor or artificial resource cap.
- CPU one-thread serial execution; no new dependencies, robot, paid service, publication or old-source edits.

## Evidence and twelve-part execution contract

1. **Purpose.** Distinguish a dependable small strongest-NODE gain from variation between learned checkpoints. Look 3 has 0.7946% coarse cost reduction, but its corrected bound and omission diagnostic remain unresolved. The cheaper-NODE contrast is supported and remains preserved.
2. **Entry.** All 1,920 primary episodes and 384 replay/refinement comparisons validate; first/current-last comparator tensors are exact. `reports/provenance/compute_study_look_003_precision_analysis.json` records within-task variance fractions 0.39177 coarse and 0.61580 fine, both below the prespecified 0.70 branch rule.
3. **Implementation.** Create `poc/compute_study_seed_precision.py`, `poc/compute_study_seed_report.py`, `tests/test_compute_study_seed_precision.py` and `config/compute_study_precision_004{,_freeze}.json`. Consume the existing model builder, training session, endpoint scorer and statistical helpers. Never edit a preceding producer/reporter that is bound to a spent look.
4. **Protocol.** Train all three models for the 22 new seeds using the same nominal-only recipe, measure existing 512 probes and isolated latency serially, then execute all tasks 0–19. New seeds are selected before their outcomes. The conservative t half-width calculation uses 1.25 times observed seed SD and fourth-look alpha/8; it projects 35 seeds for fine and 38 for coarse, so register 38 before execution. This projection is not a guarantee of a resolved result.
5. **Commands.** `python -m poc.compute_study_seed_precision prepare`, then `train`, `measure`, `controls`, `reproduce` in separate serial processes. `python -m poc.compute_study_seed_report aggregate` joins the four compatible cohorts; `demo` exports saved motion and `verify` rechecks unchanged evidence.
6. **Artifacts.** Request/freeze, 66 qualified fit receipts, prediction/profiling receipt, 2,640 primary receipts, 264 replay and 264 finer receipts, fresh final quadrature replication, immutable fourth result/spending record, full joined report/table/demo. Mutable receipt files live under `precision-004/receipts` in the explicit Data store; small repository pointers identify that location.
7. **Validity.** Same task pairing, command-stream prefixes, complete/hash-bound cells, independent raw cost arithmetic, explicit candidate failures, exact activation and converged truth. New-seed first/last inputs and parent result hashes must match. Timing measurements are isolated from tests and other scientific jobs. Training replications are not extra independent seeds.
8. **Rules.** Reuse the original reportable-result and interval-support rules without changing thresholds. A stable narrow non-detection is reportable without equivalence; an improved interval requires all four NODE contrasts, stable influence, matched actual compute and no fallback. Numerical/implementation failure requires diagnosis; a resource shortage is operational.
9. **Next.** Reportable result -> final saved-motion demo, full tests and one fresh whole-change review. Unresolved -> another justified, prospectively sized precision request, preserving the complete spending history; never choose a favorable subset.
10. **Resources.** Measured preceding jobs project 16.58 CPU hours, 18.24 wall hours including a 10% planning margin, and 24.76 GiB of new artifacts; these are projections, not caps. The owner-authorized Data volume has about 73 GiB available before this block. Maximum recorded process peak RSS is about 0.96 GiB.
11. **Recovery.** Existing content-addressed jobs, optimizer-boundary checkpoints and immutable completion markers apply. Progress-save failure must recover an already completed job rather than repeat it. Preserve attempts and disk-recovery records. Receipt placement changes operational storage only; model, clock, cost and selection semantics remain fixed.
12. **Completion.** All declared cohorts and replications pass closure and produce a reportable result or exact continuation. Task 3 remains active until presentation, verification and final review finish.

## Implementation steps

- [ ] Test a fourth request with exact next 22 seeds, twenty tasks and unchanged family; reject skipped seeds, altered task count, wrong look or an episode-variance branch. Expected RED: missing fourth producer.
- [ ] Implement request/context, external receipts and serial train/measure/control/reproduction commands. Expected GREEN: request and cohort-coordinate tests pass; original precision tests remain green.
- [ ] Test fourth design/coverage joining against a 16-seed/20-task parent, duplicate rows, wrong parent and incomplete reproduction. Expected RED: missing fourth report helpers.
- [ ] Implement the additive fourth reporter using existing arithmetic, uncertainty and presentation helpers. Expected GREEN: targeted report/precision tests pass; old report source hash remains unchanged.
- [ ] Freeze request and producer before new outcomes, commit the continuation and launch one serial pipeline with durable status. Measure rather than silently constrain actual consumption.
- [ ] After science, verify raw closure and unchanged-look idempotence; finish the original Task 3 completion/review contract.

## Review focus

- A disk-full progress save must reuse its sealed job, never rerun for a preferred outcome.
- New receipt roots must remain explicitly hash-bound without widening the original path validator.
- All prior tasks and seeds must remain in the fourth estimator and its spending identity.
- Final quadrature retraining must compare against the new last seed, while earlier replicas remain archived.
- A cheaper-NODE gain must not be described as supported strongest-NODE superiority until that contrast is resolved.

Self-review: the branch preserves the main spec's six configurations, inference family and serial timing contract. Old/new source and receipt interfaces are explicit; no change to the scientific hypothesis or acceptance floor is introduced.
