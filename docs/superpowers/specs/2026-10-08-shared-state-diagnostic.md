# Shared-state diagnosis of the power head

The user wants useful, shareable simulation evidence against strong baselines, including modest improvements and honest losses. The completed PTCO-EFF-0.1 pilot shows inexpensive neural inference but mixed control quality. Its saved first-interval errors compare different policies' visited states. The already proposed next action is therefore a small comparison on identical inputs, using existing weights and no additional training or MPC.

## Fixed question and scope

On the two qualified direct/lagged plants, which part of the selected power head's prediction error differs from the time-scaled MLP and generic/structured NODE: zero-new-command motion, command-induced motion, or composition? This is exploratory diagnosis, protocol PTCO-DIAG-0.1, training seed 9000, A/shared loss. It neither establishes population superiority nor revises any earlier statistical decision. All four architectures receive identical physical state, command, time and held-command paths. These pointwise models do not consume history; the warm states remain the actual recorded states after at least 320 ms. Initial activation remains fixed across command branches.

Freeze the first 64 trajectories in each existing A/development corpus, their initial state and first completed anchor at or after 320 ms, and the original independent channel-3 probe commands. Save explicit probe IDs/arrays and their hashes before evaluating models. Use the exact eight selected neural checkpoint identities from `config/efficiency_pilot_freeze.json`; preserve model source and normalization hashes. CPU float64, one thread. Horizons are 5, 10 and 40 ms; composition uses two equal half steps with the same held command and no new observations. Do not change any frozen pilot source or its measured artifacts.

For endpoint maps P (model) and T (truth), store the full vectors:

- drift error = P(s,0,h) - T(s,0,h);
- command error = [P(s,u,h)-P(s,0,h)] - [T(s,u,h)-T(s,0,h)];
- total error = P(s,u,h) - T(s,u,h), equal to the sum of the two preceding vectors;
- composition discrepancy = P(P(s,u,h/2),u,h/2) - P(s,u,h);
- composed truth error = P(P(s,u,h/2),u,h/2) - T(s,u,h).

Report q/v separately in rad and rad/s, mean/median/p90 probe vector norms, paired head-vs-each-baseline mean reductions and per-probe wins, and command errors normalized by the nonzero truth command increment. Report squared error contributions and their cross term; never add norms or attribute control causally from these decomposition results. Retain both cold/warm tables and raw arrays. A split prediction's extra evaluations are explicit and are diagnostic, not a hidden baseline advantage.

## Numerical and evidence contract

Reference endpoints for zero and nonzero new command use independent resets in MuJoCo at 250, 125 and 62.5 microsecond physics steps and the independent rigid-body DOP853 solver (rtol 1e-11, atol 1e-13, max step 250 microseconds). Tighten the independent solver on the first eight probes per stratum/plant to rtol 1e-12, atol 1e-14, max step 125 microseconds. Require every endpoint discrepancy <=1e-7 rad and <=1e-6 rad/s, exact activation agreement <=1e-9 N m, no warnings/nonfinite/missing rows. The largest observed discrepancy/roundoff estimate supplies an empirical per-probe q/v envelope. Drift/total error uncertainty uses the endpoint envelope; command-increment uncertainty sums both branch envelopes. Normalized errors require truth increments >100 times that envelope. This is empirical convergence, not a certified mathematical bound. Unresolved reference rows invalidate interpretation rather than being discarded.

Expected coverage: 2 plants x 4 models x 128 probes x 3 horizons = 3,072 rows, plus complete reference branches and tightened subset. A model output failure is saved with its status and prevents complete numeric aggregation; it is not silently dropped. Freeze code, data, weights, probe inputs and design before measurement; write immutable producer artifacts and a reproducible summary from them. Verification/reaggregation must reject changed hashes, missing or truncated arrays and changed coordinates. No statistical look or alpha spending.

## Outcome and next action

The result is a diagnosis, with supported differences and limits stated. If the head is mainly limited by drift, a response-only refinement is not justified. If command response or composition dominates, propose one minimal mechanism-specific refinement under a new protocol with fresh confirmation before asserting improvement. If differences are numerical or unresolved, diagnose that issue first. An additional robot or full control/training sweep is not automatically authorized by a favorable diagnostic. Preserve all prior negative and positive evidence.

Estimated new storage is below 20 MiB; runtime is measured on the first reference batch and recorded rather than capped. Existing local authorization applies. Remain on the current branch and make meaningful commits.

## Twelve-part extension contract

1. Purpose: explain the mixed strongest-baseline prediction/control result through shared-state decomposition; related C02/C03, without adding a mandatory success criterion.
2. Entry: completed, hash-bound PTCO-EFF-0.1 pilot; qualified plants and existing selected A/shared models; owner instruction to continue.
3. Minimal implementation: one additive diagnostic module and its tests; no frozen producer changes.
4. Protocol: exact fixed probes, four models, three horizons and reference sweeps above; zero new training/control episodes.
5. Commands: `.venv/bin/python -m poc.shared_state_diagnostic prepare`, then `run`, `report` and `verify`.
6. Artifacts: prospective freeze/probe arrays, immutable exploratory M40 manifest with raw vectors/reference diagnostics, JSON/Markdown report and verification receipt.
7. Validity: complete coordinates, hashes, finite outputs, fixed initial activation, two physics refinements and independent solver agreement; signed vector identity verified.
8. Rules: valid descriptive diagnosis if all measurements resolve; INVALID for missing/altered/incorrect evidence; BLOCKED only for concrete external prerequisites; no scientific PASS/FAIL from this one-seed diagnostic.
9. Next: select one targeted refinement only if errors support it; otherwise preserve an unresolved diagnosis and specify the discriminating test; do not run an automatic bigger benchmark.
10. Resources: existing CPU authorization, estimated <20 MiB extra storage, measured first-batch/total timing; no purchased compute or artificial cap.
11. Recovery: preserve any invalid attempt, reproduce its smallest defect, fix only additive diagnostic code and version/re-freeze affected inputs; never rewrite completed evidence.
12. Completion: all 3,072 rows and reference branches validated, raw-to-summary reaggregation matches, practical findings and reproduction commands committed, one fresh whole-change review.
