# Targeted precision continuation, look 2

This is the predeclared independent-seed branch of the existing compute-matched
study, not a new method or task. The first complete look is preserved. Eight
paired contrasts have central improvement intervals narrower than four points;
the coarse AffineFlow-versus-coarse RK4 contrast still changes its one-sided
interpretation when individual seeds are omitted. Its point improvement is
0.9695%, so it cannot yet establish superiority over every NODE option.

1. Purpose: resolve that influence diagnostic under the unchanged eight-contrast
   design. No positive effect floor is imposed.
2. Entry: first-look raw costs, replay/refinement, retraining, complete coverage
   and source closure passed. The raw-linked first look authorizes this branch.
3. Implementation: additive `poc.compute_study_precision`, a separate immutable
   freeze/store, and joined reporting. Original producers and freeze stay intact.
4. Protocol: add ascending seeds 2026100811–2026100815 for all three models, ten
   new independent paired tasks each, all six configurations, unchanged data,
   nominal selection and frozen timing budgets. Reuse the fixed 512-probe library
   for conditional training-seed inference. The within-episode variance estimate
   accounts for 67.57% of observed contrast variance, below the prespecified 70%
   rule for choosing episode expansion. Thus add independent training clusters.
   The existing four-point precision target already projects a smaller sample
   than the current cohort; use the minimum five additional clusters required
   by the influence diagnostic. This is a next information-gaining block, not
   a maximum sample size or a promise that the contrast will resolve positively.
5. Commands: `python -m poc.compute_study_precision prepare`, then `train`,
   `measure`, `controls`, and `reproduce` in separate serial processes; finally
   `python -m poc.compute_study_report aggregate --look 2`.
6. Artifacts: prospective request/freeze; fifteen qualified selected fits;
   new predictions/timings; 300 primary controls, 30 replay and 30 finer-physics
   runs; fresh last-seed quadrature retraining; combined sixteen-cluster result.
7. Validity: all previous checks apply, including hashes, complete paired cells,
   explicit failures, independent raw costs and same physical clocks. No old
   seed, task or unfavorable result is replaced. New raw artifacts live beneath
   the existing authorized Data-volume store, in `precision-002/`.
8. Decisions: a stable precise non-detection is reportable without claiming
   equivalence; a corrected, resolved positive contrast is reported at its scope.
   Full utility requires all four NODE comparisons for one Affine query interval,
   matching measured compute and zero intrinsic fallback. Otherwise a still-live
   influence or precision issue requests another prospectively justified block.
   Alpha for look 2 is 0.0125, with the same Holm family of eight comparisons.
9. Next: resolved result -> demonstration, reproduction instructions, full tests,
   one fresh whole-change review and final report. Unresolved -> named precision
   diagnostic. Actual resource absence -> exact blocker, without erasing look 1.
10. Resources: projection uses measured mean training CPU times and episode
    throughput: about three CPU hours and three GiB additional artifacts. The
    Data volume has about 94 GiB free. Owner imposed no artificial time/RAM/disk
    caps; runs remain serial. Projected time is not a scientific cutoff.
11. Recovery: optimizer-update resumption and content-addressed completed jobs
    reuse the existing tested store. Preserve interruptions and reject stale
    dependencies; repair a demonstrated defect rather than retrying lucky seeds.
12. Completion: combined raw-linked result, stable inference or an explicit next
    precision action, all required replay/training checks, and honest report.

The source request, measured variance and resource calculation are sealed in
`config/compute_study_precision_002.json` before opening new outcomes.
