# Completed compute-matched NODE follow-up

The approved follow-up is complete on `main`: 4,560 paired control episodes,
38 training seeds, 912 replay/refinement checks, exact first/final comparator
retraining and a standalone saved-motion demo. No experiment queue remains
active and no further sample block is needed for the supported 40 ms result.

AffineFlow at 40 ms lowers the original weighted cost by **0.7066% against coarse
RK4 NODE and 1.0656% against coarse quadrature NODE**, at matched planning
allocation. Both gains pass the declared Holm and influence checks. This is a
small scoped gain: the 5 ms option does not beat every comparator, moving-state
prediction is worse than RK4, and scalar/cheap-NODE inference is slower.

See the [findings](compute_study_findings.md), [score explanation](compute_study_interpretation.md),
[machine result](compute_study_result.json), [demo](compute_study_demo.html) and
[reproduction instructions](compute_study_reproduce.md). All four immutable
results and earlier studies remain preserved.

## Verification

- Full suite: `.venv/bin/python -m pytest -q` — **1,053 passed in 169.59 s**.
- Fourth aggregate and repeated `poc.compute_study_seed_report verify` reproduce
  the unchanged result/spending, with all costs and reproduction checks valid.
- Independent cost discrepancy at most `2.220446049250313e-16`; empirical
  refined-physics cost envelope `2.1149748619109232e-13`.
- Browser: 4,560 clips, 38 seeds, 20 tasks, all six canvases painted on the final
  seed/task, 161 frames per clip. Demo/result/all four cohort bindings agree.
  Owned page and temporary loopback server closed.
- Original scientific pipeline exited. Terminal-registry recovery launched no
  duplicate. Failed launches, disk interruption and recovery records remain.

## Fresh whole-change review

One fresh `gpt-6-astra` reviewer with high reasoning reviewed
`fcdb61d..79e3c39` read-only, with no subagents or expensive experiment reruns.
Assessment: **Ready: Yes. No Critical or Important findings.**

The reviewer confirmed physical command-response orders, fixed initial
activation, mixed zero intervals, exact activation, controller parity,
infinity/status/fallback handling and paired task/RNG prefixes under unequal
budgets. All four freezes matched their bound sources. All 5,573 sealed jobs
had one start/complete pair and no remaining partial jobs, supporting the
receipt-only recovery. Every primary manifest/marker was checked; the complete
control summary and all eight statistical comparisons were independently
reconstructed. Forty-eight representative raw episodes across all four cohorts
were hash-checked and independently audited. First/final comparator artifacts,
canonical tensors and nominal selection matched. The reviewer ran **43 targeted
tests in 2.82 s**, using no cache/bytecode and temporary fixtures.

It confirmed that fine-query losses, warm-state prediction losses, cheap-NODE
speed, slightly worse tracking and fewer successes than coarse quadrature remain
visible. The sole minor comment concerned a pre-closeout pending-status sentence
in the reproduction guide. Planned release closeout now records completion;
no scientific/executable change or fix/re-review pass was required.

The reviewer declined to judge the five preexisting modified runtime files,
unrelated untracked artifacts and excluded deployment/generalization settings.
The executor retains each exclusion below. No deferred software minors remain.

## Executor rulings and costs

1. Work inline on the requested branch without repeated approval handoffs.
   Cost if wrong: another design might be preferred; local commits are reversible.
2. Use the authorized second-volume artifact store. Cost if wrong: transfer
   requires restoring its recorded paths and indexed artifacts, as well as code.
3. Validate full cell coverage at aggregation. Cost if wrong: partial producer
   output stays pending until the aggregator rejects it.
4. Prepare reporting while serial science runs. Cost if wrong: the report remains
   unusable until complete raw evidence validates.
5. Exclude the five preexisting modified runtime files. Cost if wrong: defects
   there remain outside this audit; this release makes no new claim about them.
6. Exclude unrelated untracked artifacts. Cost if wrong: their separate evidence
   is not certified by this review.
7. Exclude physical deployment, other robots, contact and unknown actuator
   dynamics. Cost if wrong: performance in those settings needs a new study.
8. Treat pending review-status prose as planned release closeout. Cost if wrong:
   another stale sentence could mislead; the final handoff and this record govern.

Frozen scientific sources and spent-look identities are unchanged. The user's
integration choice is retained: local commits on `main`, no push or publication.

The owned execution record is archived in
`provenance/compute_study_execution_2026-10-10.zip`, with per-file hashes in
`provenance/compute_study_execution_2026-10-10.json`. It preserves the ledger,
tests/review package, failed attempts, scripts, fourth execution logs and fourth
receipt metadata. Heavy indexed raw arrays and checkpoints remain on the Data
volume; source alone does not recreate the measured evidence.
