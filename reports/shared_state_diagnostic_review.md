# Final diagnostic review and execution decisions

One fresh read-only reviewer inspected commits 69744dd..7691048, the source/spec/tests, frozen coordinates and completed measurements. It independently reconstructed all first-64 cold/warm coordinates from both original corpora, confirmed the eight pilot checkpoint selections, reaggregated all 3,072 rows, verified signed cancellation and reference uncertainty, and checked the 73–80% response-position gains, every short-interval loss against structured NODE and the drift/composition figures. Ten focused tests passed. It found no critical implementation issue.

## Reporting corrections

The review identified an important causal-interpretation issue: shared states and commands isolate prediction differences, but different architectures and learned weights do not isolate the temporal powers as their cause. The findings now describe selected-head command-response advantages, consistent with the timing hypothesis. No raw measurement or scientific estimator changed.

Two nominally minor findings were treated as reporting correctness requirements and corrected in the same prose-only pass. The actual MLP range rounds to 92–99%, because direct/warm is 92.388%. Split NODE queries use four versus eight field evaluations; the findings now distinguish RK4 refinement from learned-map composition and state that the split prediction costs two queries for all models. This matters to the user's accuracy/compute and fairness questions. No production code changed, so the existing green full-suite receipt remains applicable; direct numerical reconciliation and report verification were rerun instead of inventing prose tests.

The reviewer did not request another training, simulation or control run. All completed artifacts are retained. There are no remaining deferred findings from this review.

## Rulings made during execution

1. Existing current-branch authorization and the user's repeated instruction to continue apply to the already proposed small diagnostic. No new worktree, approval or merge menu is needed. Cost if wrong: an unintended workflow change; the actual branch remains main and only explicitly named files were staged.
2. The root corpus's historical G20 record references an earlier CLI hash. The unchanged loader validates the identical corpus in the pilot's preserved nominal source context. Both the manifest and NPZ must be byte-identical to the original, and both copies are hash-bound. This preserves strict provenance instead of waiving a stale check. Cost if wrong: invalid inherited provenance; the changed-copy regression fails closed.
3. Scientific identity binds the spec, sources, weights, data and explicit coordinates. Mutable plan checkbox/ledger progress is excluded from that semantic identity. Cost if wrong: a plan-only substantive change would require checking the bound scientific spec/source, rather than relying on the plan checksum.
4. Retain the prospective freeze created before the final reader-report verification helper as a premeasurement revision. No model outputs were opened under it; the final tested producer/freeze was committed before measurement. Cost if wrong: discovery could be mislabeled prospective; preserved timestamps, commits and immutable requests distinguish both versions.

## Reviewer matters set aside, and root decisions

- Prior ten-seed utility estimates: retain their clearly separate scope and existing measured report. They were not rerun in this diagnostic. Cost if wrong: inherited contextual control claims would need correction; the new matched measurements remain separate.
- Prior pilot timing/control: retain the completed pilot and its inference-only speed interpretation, all legal resolutions and losses. No speed or control rerun was performed here. Cost if wrong: an inherited tradeoff could be misstated; the linked complete pilot evidence remains the authority.
- Population superiority, seed robustness and other robots: make no such claim from a single development seed and fixed probe set. Cost if ignored: misleading generalization; fresh confirmation remains necessary for a revised method claim.
- Future drift/composition refinement: report it as a direction selected by diagnosis, with no assertion of implementation or efficacy. Cost if wrong: a proposed follow-up may not help; no positive result is attributed to it.
- Existing plant/oracle/model infrastructure: rely on its prior qualification plus the complete new three-resolution/independent-equation checks at these exact probes. Do not re-audit unrelated infrastructure. Cost if wrong: a shared implementation defect could remain; independent equations and analytic tests address the relevant interfaces.
- Fresh-clone binary distribution: explicitly state that full verification needs the archived local data/checkpoints/evidence. Reports and the prior standalone replay remain directly shareable. Cost if ignored: a reader could expect Git alone to reproduce nontracked binary artifacts.
- Pre-existing runtime edits/unrelated files: leave the five pre-existing tracked runtime changes and all historical evidence unstaged and intact. Cost if ignored: loss of user work or misleading commits.
- Additional independent arithmetic receipt: root independently recomputed 1,536 statistics from raw endpoints without the production decomposition/summary helper. Maximum discrepancy is 1.53e-16 and references pass. It is separately labeled supplemental verification, not reviewer evidence. Cost if wrong: supplemental assurance could be overstated; the reviewer independently checked the core result.

## Finish

The production-source full suite passed 962 tests in 132.44 seconds, before the measured run. Since then changes are reporting prose/receipts only. All task commands, the final review package, ledger and test/measurement logs are preserved. No push, publication, physical hardware action, new training, or new control rollout was performed.
