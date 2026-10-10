# What the measured control gain means

The completed fourth look covers 38 training clusters and twenty paired tasks each. The supported result is a modest reduction in the original weighted control score on the specified simulated arm, at roughly equal CPU planning time. These component averages explain that score; they introduce no new statistical decision.

Against the strongest coarse NODE, coarse AffineFlow has a 0.71% lower score. Most of the difference comes from lower joint-speed cost. Position tracking during motion is slightly worse; final position error and squared-torque effort are lower.

| Contribution to the original score | AffineFlow | RK4 NODE | Reduction (positive favors AffineFlow) |
|---|---:|---:|---:|
| tracking | 0.17099921 | 0.17070019 | -0.18% |
| velocity | 0.10951645 | 0.11171722 | 1.97% |
| effort | 0.00218004 | 0.00223257 | 2.35% |
| terminal | 0.00026897 | 0.00032836 | 18.09% |
| total | 0.28296468 | 0.28497834 | 0.71% |

All component rows are descriptive point estimates over the same 760 tasks. The terminal contribution is tiny in absolute score units, so its larger percentage does not imply a large overall gain. Squared torque is not measured energy. Lower joint-speed cost does not establish smoother motion or reduced wear.

The overall cost reduction is supported under the prospective eight-comparison correction and seed-influence checks. Its central 95% improvement interval is 0.45–0.96%; this descriptive interval is not a simultaneous confidence band. The cheaper coarse NODE comparison is also supported, at 1.07% lower cost. The 5 ms AffineFlow configuration does not beat every baseline, and NODE remains more accurate on moving-state prediction probes.

Strict final-200-ms success counts are 742/760 for coarse AffineFlow, 738/760 for coarse RK4 NODE and 746/760 for coarse cheaper NODE. These counts are reported separately; no statistically established success-rate improvement is claimed.

Inference speed depends on batching: at 5 ms, batch-128 AffineFlow is 30% faster than RK4 NODE and 29% slower than the cheaper NODE. A single query is 8% slower than RK4 NODE and 77% slower than the cheaper NODE. These are kernel timings, not real-time robot deployment. All measured planners take about 0.22–0.24 seconds per 40 ms feedback decision.

The [component record](provenance/compute_study_look4_cost_components.json) retains all six configurations and the final result hash. Every component sum was independently checked against all 4,560 saved episode costs. The [main findings](compute_study_findings.md) retain every comparison, uncertainty, prediction stratum, training quality and limitation. The [third-look breakdown](provenance/compute_study_look3_cost_components.json) is preserved as historical evidence.
