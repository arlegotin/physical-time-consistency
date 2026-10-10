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

The last row is the main result in the [README](../README.md). It supports the
later AffineFlow implementation on one known-lag arm. It does not retroactively
satisfy original power-head criteria, establish universal architecture
superiority or establish research novelty.

The private scientific archive retains the detailed execution journal.
The supplied original protocol is an archival specification; proposed checks
are not evidence that those checks ran.
