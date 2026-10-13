# Research trajectory and claim boundaries

The repository preserves several completed studies. Their data namespaces,
methods and acceptance rules differ; they must not be pooled into one claim
of success for the original protocol.

| Stage | Question and outcome | Evidence |
|---|---|---|
| Original discrete power head | Changing temporal powers improves fine-time control against an ordinary residual MLP. Superiority over stronger models and the complete original PTCO-1.0 gates were not established. | [Control result](../reports/control_confirmation_result.md), [precision review](../reports/control_precision_release_review_2026-10-07.md) |
| Practical baseline pilot | Gains against time-scaled MLPs, NODEs and fitted mechanics are small and condition-dependent. One-seed development evidence. | [Pilot](../reports/efficiency_pilot_findings.md) |
| Shared-state diagnosis | Good command scaling can coexist with weak drift and composition; structured NODE is more accurate. | [Diagnosis](../reports/shared_state_diagnostic_findings.md) |
| AffineFlow refinement | A coupled acceleration model with analytical actuator integrals improves average fine-time prediction and batched speed, while moving-state errors and early control results favor or match NODE. | [Independent study](../reports/affine_refinement_findings.md) |
| Compute-matched NODE follow-up | At 40 ms queries, AffineFlow has a supported modest cost gain against both solvers at both query intervals. Fine 5 ms AffineFlow does not beat every option. | [38-seed study](../reports/compute_study_findings.md), [interpretation](../reports/compute_study_interpretation.md) |
| Short-calibration bounded pilot | The complete 21-fit, one-session simulator pilot resolves the frozen gate to `no_advantage`. Torque-affine models pass earlier than structured NODE, but fitted physics and the physical residual also pass at 30 seconds; residual improves quality and full scorer time versus AffineFlow at the same budgets and 20/40 ms partitions. Exact fresh numerical reproduction passed. | [Verified pilot](../reports/short_calibration_pilot/pilot_execution_findings.md), [scope amendment](superpowers/plans/2026-10-12-short-calibration-pilot-scope-amendment.md) |
| Final hybrid component test | The full 30-fit, three-session study resolves the fixed 30-second / 40-ms gate to `no_component_advantage`. Flow passes all conservative prediction bands, but loses paired decision quality in session 3 beyond numerical resolution; sessions 1 and 2 retain unresolved paired comparisons because of invalid anchor selections. Three fresh coefficient fits and their numerical products reproduce exactly. | [Component findings](../reports/hybrid_component/component_findings.md), [execution account](../reports/hybrid_component/task4_execution_report.md), [binding design](superpowers/specs/2026-10-12-hybrid-affine-component-design.md) |

The [README](../README.md) leads with the completed continuation and its negative
practical gates, while retaining the earlier compute-matched NODE result. That
earlier result supports the AffineFlow implementation on one known-lag arm. It does not retroactively
satisfy original power-head criteria, establish universal architecture
superiority or establish research novelty.

The private scientific archive retains the detailed execution journal.
The supplied original protocol is an archival specification; proposed checks
are not evidence that those checks ran.

The [short-calibration continuation](superpowers/specs/2026-10-12-short-calibration-affineflow-design.md)
tested whether torque-affine structure or AffineFlow propagation reduces the data
or computation needed for useful decisions. Its complete default pilot, full
scoring/truth/timing coverage and three fresh selected-fit numerical replicas
are retained in the [execution report](../reports/short_calibration_pilot/task9_execution_report.md).
The null supports a prospective stop at the pilot: the 195-fit development
matrix and conditional deadline/larger-arm stages remain unexecuted. Representation
evidence against structured NODE does not establish a unique Flow benefit against
the stronger physical alternatives.

This is one independent calibration session and one neural initialization per
method/budget on a noiseless known-lag simulator, with no cross-session confirmation,
hardware commissioning or real-time deadline claim. Passing the lowest tested
30-second budget is interval censored; benefits below it and in other draws are
untested. The prior pilot's interrupted resource-observation accounting defect remains
disclosed in its preserved reports; its primary ledgers completed cleanly, and no commissioning-compute
advantage is claimed. Resource headroom was adequate. Future practical-arm work
requires a consequential error remaining after simple physical calibration and
a separately declared command path, observable state, limits and acceptance rules.

The separately authorized final hybrid component study asks whether Flow propagation
adds useful savings when all five methods receive the same frozen nominal physics.
Its three 1,478-parameter coefficient networks share seeded initial tensors; the two
practical acceleration-residual references have 1,410 parameters. All methods are
newly trained on calibration sessions 1/2/3, initialization 0 and budgets 30/120
seconds, with fresh evaluation session 1. The reviewed source freeze includes the
focused resource-observation repair; it does not rewrite the historical pilot evidence.

The [complete component gate](../reports/hybrid_component/component_gate.json)
retains every method, budget, partition and session. At the primary configuration,
Flow's conservative q/v error ratios all satisfy the provisional 1.05 bands. In
session 3, however, 64 valid paired decisions have mean selected truth-cost loss
`4.764970410914889e-5` against joint numerical envelope `3.258027528318941e-15`;
62 identical selected commands cancel exactly. Sessions 1 and 2 preserve unresolved
comparisons because their anchors each have one failed decision among 64 states.
The demonstrated session-3 quality loss prevents all-session acceptance. There is
no fastest acceptable primary alternative and no declared median-saving or p95 ratio.

Pooled complete-scorer median/p95 timings are retained as diagnostics: Flow is
16.40/17.04 ms, coefficient RK4 28.84/29.87 ms, coefficient actuator-aware
16.11/16.70 ms, acceleration RK4 25.40/26.46 ms, and acceleration actuator-aware
14.39/14.92 ms. These measurements do not restore failed quality eligibility or
establish a physical deadline. Six finite optimization-limited fits, all secondary
curves and unresolved comparisons remain retained without changed caps or replacement
draws. Fresh-root selected tensors, validation and first/last prediction, complete
candidate-bank decision and three-grid truth products match exactly for all three
coefficient models.

This valid nonpositive result ends active investment in this line. The complete
scientific artifacts remain preserved; no new confirmation, hardware, deadline or
feedback-controller stage follows. The three noiseless known-lag simulator sessions
are exploratory, and the bounded optimizer result does not prove universal model
inferiority. Secondary prediction or runtime improvements do not change the primary
question or reopen the unexecuted standalone development matrix.
