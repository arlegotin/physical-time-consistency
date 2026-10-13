# Hybrid component raw diagnostics

The component gate uses the fixed 30 seconds / 40 ms configuration and all-session eligibility. The metrics, plots and threshold curves below describe session-averaged provisional screens; they do not replace the component rule. See component_findings.md and component_gate.json for every session comparison and pooled actual scorer calls.
Gate: **no_component_advantage**. Reasons: primary_flow_not_acceptable_in_every_session.

Coverage complete: True. Expected fits: 30; nominal construction references: 6. Missing cells: 0; invalid: 0; retained failed fits: 0; optimization limited: 6.

This is a provisional simulator screen. Motion seconds are a calibration interaction proxy. There is no feedback deadline, deployment or hardware commissioning result. Held-response improvements do not establish decision improvements. Reduced unit fixtures are ineligible for empirical claims.

Prediction at 160 ms recorded motion must satisfy q RMSE ≤ .01 rad, v RMSE ≤ .10 rad/s, q p95 ≤ .03 rad and v p95 ≤ .30 rad/s, with zero prediction failures. RMSE is sqrt(mean joint-component squared error); p95 is the percentile of two-joint endpoint norms. Screens use absolute errors plus numerical envelopes. Sub-envelope differences cannot establish superiority. Decision failures must be ≤ 1%; conservative valid regret must be ≤ 1% of mean truth zero-command cost. Zero-command cost ≤ 1e-8 or an empty feasible truth bank leaves the screen unresolved. Unresolved sampled truth feasibility makes conditional regret undefined and the failure fraction a conservative upper bound. Prediction, decision and timing partitions are matched in every row.

Independent session clusters: 3. Initialization replicas are averaged within session; configuration screens use those session-averaged metrics. All paired model/budget/task rows use the same 10,000 fixed-namespace cluster draws. Bootstrap namespace: [20261012, 1, 1, 0, 0, 5]. Individual fitted-model screens and all session values remain in JSON. Passing an average does not mean every fitted model passes. Intervals are descriptive; three sessions are not powered confirmation or a customer reliability claim.

| Model | Motion proxy (s) | Main step (ms) | q RMSE (rad) | v RMSE (rad/s) | q p95 (rad) | v p95 (rad/s) | Failure fraction | Valid regret | Scorer median / p95 (s) | Screen | Limited |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| hybrid_accel_aware | 120 | 10.0 | 0.00091123 | 0.0104942 | 0.00260909 | 0.0295748 | 0 | 0.00025717 | 0.0213093 / 0.0217645 | pass | False |
| hybrid_accel_aware | 120 | 20.0 | 0.000849067 | 0.00971881 | 0.00243774 | 0.0278161 | 0 | 6.44693e-05 | 0.0166715 / 0.0171502 | pass | False |
| hybrid_accel_aware | 120 | 40.0 | 0.000811772 | 0.00943608 | 0.00229154 | 0.0257743 | 0 | 0.000121287 | 0.0144117 / 0.0150409 | pass | False |
| hybrid_accel_aware | 120 | 5.0 | 0.000929356 | 0.0107093 | 0.00265997 | 0.0301549 | 0 | 0.00025717 | 0.0282718 / 0.0291426 | pass | False |
| hybrid_accel_aware | 30 | 10.0 | 0.00122173 | 0.0139719 | 0.00355069 | 0.0398354 | 0.00520833 | 4.3496e-05 | 0.021299 / 0.0218247 | pass | True |
| hybrid_accel_aware | 30 | 20.0 | 0.00118152 | 0.0134622 | 0.003409 | 0.0379469 | 0.00520833 | 4.41385e-05 | 0.0166645 / 0.0171067 | pass | True |
| hybrid_accel_aware | 30 | 40.0 | 0.00116396 | 0.0133415 | 0.00321756 | 0.0369832 | 0.00520833 | 5.02281e-05 | 0.0143854 / 0.014908 | pass | True |
| hybrid_accel_aware | 30 | 5.0 | 0.00123396 | 0.0141191 | 0.0035841 | 0.0402211 | 0.00520833 | 5.53771e-05 | 0.0281978 / 0.0287619 | pass | True |
| hybrid_accel_rk4 | 120 | 10.0 | 0.000475476 | 0.0055436 | 0.00139918 | 0.0162975 | 0 | 5.36069e-05 | 0.0370452 / 0.0380203 | pass | False |
| hybrid_accel_rk4 | 120 | 20.0 | 0.000475437 | 0.00554339 | 0.00139911 | 0.016296 | 0 | 5.36069e-05 | 0.0290012 / 0.0297907 | pass | False |
| hybrid_accel_rk4 | 120 | 40.0 | 0.000474928 | 0.00553956 | 0.00139638 | 0.0162691 | 0 | 5.37823e-05 | 0.0254017 / 0.0261364 | pass | False |
| hybrid_accel_rk4 | 120 | 5.0 | 0.000475478 | 0.00554361 | 0.00139916 | 0.0162975 | 0 | 5.36069e-05 | 0.0484271 / 0.0494324 | pass | False |
| hybrid_accel_rk4 | 30 | 10.0 | 0.00102357 | 0.011693 | 0.00285458 | 0.0326501 | 0.0104167 | 4.44093e-05 | 0.0370176 / 0.0380603 | fail | True |
| hybrid_accel_rk4 | 30 | 20.0 | 0.00102356 | 0.0116929 | 0.00285444 | 0.0326511 | 0.0104167 | 4.44093e-05 | 0.0291142 / 0.0298973 | fail | True |
| hybrid_accel_rk4 | 30 | 40.0 | 0.00102342 | 0.0116927 | 0.0028534 | 0.0326666 | 0.0104167 | 4.44093e-05 | 0.0254169 / 0.0263872 | fail | True |
| hybrid_accel_rk4 | 30 | 5.0 | 0.00102358 | 0.011693 | 0.00285459 | 0.03265 | 0.0104167 | 4.44093e-05 | 0.0485214 / 0.0493517 | fail | True |
| hybrid_affine_aware | 120 | 10.0 | 0.000651712 | 0.00756495 | 0.00184084 | 0.0213862 | 0 | 6.529e-05 | 0.0228822 / 0.0233406 | pass | True |
| hybrid_affine_aware | 120 | 20.0 | 0.000574892 | 0.00661189 | 0.00162078 | 0.0189982 | 0 | 6.529e-05 | 0.0185761 / 0.0191333 | pass | True |
| hybrid_affine_aware | 120 | 40.0 | 0.000555721 | 0.00667249 | 0.00157977 | 0.0188645 | 0 | 6.49393e-05 | 0.0161459 / 0.0166654 | pass | True |
| hybrid_affine_aware | 120 | 5.0 | 0.000673959 | 0.0078274 | 0.00189876 | 0.0221215 | 0 | 0.00025717 | 0.0306312 / 0.031253 | pass | True |
| hybrid_affine_aware | 30 | 10.0 | 0.000995871 | 0.0113536 | 0.00298102 | 0.0341068 | 0 | 4.31549e-05 | 0.0228366 / 0.0233218 | pass | False |
| hybrid_affine_aware | 30 | 20.0 | 0.000942212 | 0.0106737 | 0.00286463 | 0.0317373 | 0 | 4.29795e-05 | 0.0185992 / 0.0191694 | pass | False |
| hybrid_affine_aware | 30 | 40.0 | 0.000908147 | 0.0103713 | 0.00263988 | 0.0296862 | 0 | 5.05221e-05 | 0.0160921 / 0.01667 | pass | False |
| hybrid_affine_aware | 30 | 5.0 | 0.00101177 | 0.0115449 | 0.00302897 | 0.0344588 | 0 | 4.31549e-05 | 0.0305713 / 0.0311991 | pass | False |
| hybrid_affine_flow | 120 | 10.0 | 0.000598025 | 0.00701583 | 0.00161471 | 0.0188729 | 0 | 3.23521e-05 | 0.0232362 / 0.0237028 | pass | True |
| hybrid_affine_flow | 120 | 20.0 | 0.000532561 | 0.00684265 | 0.00142588 | 0.0182802 | 0 | 3.23521e-05 | 0.0188522 / 0.0193658 | pass | True |
| hybrid_affine_flow | 120 | 40.0 | 0.000379749 | 0.00640323 | 0.00110977 | 0.0171233 | 0 | 4.84107e-05 | 0.0164067 / 0.0170269 | pass | True |
| hybrid_affine_flow | 120 | 5.0 | 0.000615235 | 0.00706183 | 0.00166598 | 0.0188809 | 0 | 3.23521e-05 | 0.0311453 / 0.0317609 | pass | True |
| hybrid_affine_flow | 30 | 10.0 | 0.000836446 | 0.00974096 | 0.00241423 | 0.027959 | 0 | 4.39756e-05 | 0.0232497 / 0.0238071 | pass | False |
| hybrid_affine_flow | 30 | 20.0 | 0.000791935 | 0.00960537 | 0.0023272 | 0.0274909 | 0 | 4.38002e-05 | 0.0188787 / 0.0195097 | pass | False |
| hybrid_affine_flow | 30 | 40.0 | 0.000701528 | 0.00924672 | 0.00209252 | 0.0264519 | 0 | 4.36249e-05 | 0.0164092 / 0.0169885 | pass | False |
| hybrid_affine_flow | 30 | 5.0 | 0.000848601 | 0.00977695 | 0.00245454 | 0.0280913 | 0 | 4.39756e-05 | 0.0311744 / 0.031918 | pass | False |
| hybrid_affine_rk4 | 120 | 10.0 | 0.000346004 | 0.00401163 | 0.00103638 | 0.0119477 | 0 | 4.82354e-05 | 0.0405444 / 0.0413684 | pass | True |
| hybrid_affine_rk4 | 120 | 20.0 | 0.000346005 | 0.00401148 | 0.00103639 | 0.0119463 | 0 | 4.82354e-05 | 0.0330325 / 0.0341264 | pass | True |
| hybrid_affine_rk4 | 120 | 40.0 | 0.000346164 | 0.00400878 | 0.001034 | 0.011922 | 0 | 3.21768e-05 | 0.0289156 / 0.0298668 | pass | True |
| hybrid_affine_rk4 | 120 | 5.0 | 0.000346004 | 0.00401163 | 0.00103638 | 0.0119478 | 0 | 4.82354e-05 | 0.0538713 / 0.0550682 | pass | True |
| hybrid_affine_rk4 | 30 | 10.0 | 0.000769819 | 0.00867673 | 0.00232594 | 0.0262914 | 0.015625 | 6.06308e-05 | 0.0404388 / 0.0415582 | fail | False |
| hybrid_affine_rk4 | 30 | 20.0 | 0.000769779 | 0.00867662 | 0.00232563 | 0.0262913 | 0.015625 | 6.06308e-05 | 0.0330313 / 0.0342223 | fail | False |
| hybrid_affine_rk4 | 30 | 40.0 | 0.000769201 | 0.00867494 | 0.00232302 | 0.0262892 | 0.015625 | 6.06308e-05 | 0.028849 / 0.0298017 | fail | False |
| hybrid_affine_rk4 | 30 | 5.0 | 0.000769821 | 0.00867673 | 0.00232597 | 0.0262914 | 0.015625 | 6.06308e-05 | 0.0536987 / 0.0548996 | fail | False |
| physics_nominal | 120 | 10.0 | 0.0178408 | 0.205395 | 0.04768 | 0.557586 | 0.25 | 0.00613284 | 0.0908195 / 0.0964551 | fail | False |
| physics_nominal | 120 | 20.0 | 0.0178408 | 0.205395 | 0.04768 | 0.557586 | 0.25 | 0.00613284 | 0.130618 / 0.135447 | fail | False |
| physics_nominal | 120 | 40.0 | 0.0178408 | 0.205395 | 0.04768 | 0.557586 | 0.25 | 0.00613284 | 0.211856 / 0.217475 | fail | False |
| physics_nominal | 120 | 5.0 | 0.0178408 | 0.205395 | 0.04768 | 0.557586 | 0.25 | 0.00613284 | 0.0730033 / 0.0764853 | fail | False |
| physics_nominal | 30 | 10.0 | 0.0178408 | 0.205395 | 0.04768 | 0.557586 | 0.25 | 0.00613284 | 0.0912736 / 0.0967451 | fail | False |
| physics_nominal | 30 | 20.0 | 0.0178408 | 0.205395 | 0.04768 | 0.557586 | 0.25 | 0.00613284 | 0.131461 / 0.138186 | fail | False |
| physics_nominal | 30 | 40.0 | 0.0178408 | 0.205395 | 0.04768 | 0.557586 | 0.25 | 0.00613284 | 0.213129 / 0.225049 | fail | False |
| physics_nominal | 30 | 5.0 | 0.0178408 | 0.205395 | 0.04768 | 0.557586 | 0.25 | 0.00613284 | 0.0732093 / 0.0764344 | fail | False |

Threshold brackets are interval-censored; no interpolated data requirement. All budgets, failures and nonmonotone curves remain visible.

| Model | Step (ms) | First pass (s) | Bracket (s) | Nonmonotone |
|---|---:|---|---|---|
| hybrid_affine_flow | 5 | 30 | [0, 30] | False |
| hybrid_affine_flow | 10 | 30 | [0, 30] | False |
| hybrid_affine_flow | 20 | 30 | [0, 30] | False |
| hybrid_affine_flow | 40 | 30 | [0, 30] | False |
| hybrid_affine_rk4 | 5 | 120 | [30, 120] | False |
| hybrid_affine_rk4 | 10 | 120 | [30, 120] | False |
| hybrid_affine_rk4 | 20 | 120 | [30, 120] | False |
| hybrid_affine_rk4 | 40 | 120 | [30, 120] | False |
| hybrid_affine_aware | 5 | 30 | [0, 30] | False |
| hybrid_affine_aware | 10 | 30 | [0, 30] | False |
| hybrid_affine_aware | 20 | 30 | [0, 30] | False |
| hybrid_affine_aware | 40 | 30 | [0, 30] | False |
| hybrid_accel_rk4 | 5 | 120 | [30, 120] | False |
| hybrid_accel_rk4 | 10 | 120 | [30, 120] | False |
| hybrid_accel_rk4 | 20 | 120 | [30, 120] | False |
| hybrid_accel_rk4 | 40 | 120 | [30, 120] | False |
| hybrid_accel_aware | 5 | 30 | [0, 30] | False |
| hybrid_accel_aware | 10 | 30 | [0, 30] | False |
| hybrid_accel_aware | 20 | 30 | [0, 30] | False |
| hybrid_accel_aware | 40 | 30 | [0, 30] | False |
| physics_nominal | 5 | None | [120, None] | False |
| physics_nominal | 10 | None | [120, None] | False |
| physics_nominal | 20 | None | [120, None] | False |
| physics_nominal | 40 | None | [120, None] | False |

Resources are separated in JSON: motion seconds proxy; fit execution including validation/selection; shared evaluation/truth replay/refinement; predictions; decision scoring; timing critical samples; audit/logging/serialization; original numeric store finalization. Producer spans exclude post-producer artifact hashing and manifest/marker/receipt/completed-event persistence. Timing measures one frozen earliest-primary trajectory-0 bank; decision records cover the complete fixed trajectory library. Network-forward diagnostics never replace complete scorer cost.

use nonoverlapping_producers for totals; inclusive producer_attempts and nested rate diagnostics overlap and must not be added; active and checkpoint persistence spans exclude inactive gaps, ledger observation writes and post-producer finalization. unknown-tail records are measured lower bounds, never complete throughput estimates; pilot projections must retain accounting_status and separately report actual compressed bytes and finalization overhead.

unique exact-receipt-bound original observational event; events.jsonl is outside immutable artifact hashes; resource-record write excluded. These observations validate exact receipt identity and schema but cannot prove the event log has never been edited. Reuse costs cannot replace original production costs.

Raw rows, decisions, numerical envelopes, all baselines, selected receipts, missing/failed/limited coordinates, session values, resource breakdown and bootstrap intervals are in short_calibration_result.json. Arrays and weights remain external in the selected ignored output store. See short_calibration_reproduce.md for exact portable acquisition and read-only verification commands.

Observed threshold budget ratios and their interval-censored ratio brackets are in JSON. Undefined ratios are not imputed.
