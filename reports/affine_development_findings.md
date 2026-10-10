# Affine-flow development finding

The revised model improves short-time prediction for the robot with a known actuator lag. It has not demonstrated better control than the strongest tested learned baseline, the structured neural differential equation (NODE).

This is PTCO-AFF-0.1 development at training seed9000, using existing augmented train/validation data. It motivates the prospectively registered [fresh-data confirmation](../docs/superpowers/specs/2026-10-08-affine-independent-confirmation.md). It is separate from the original power head and its independently supported22–26% control improvement over the ordinary MLP.

| At5 ms prediction interval | Affine q error,rad | NODE q error,rad | Affine v error,rad/s | NODE v error,rad/s |
|---|---:|---:|---:|---:|
| Direct torque | 0.00000483 | 0.00000407 | 0.001984 | 0.001629 |
| Known80 ms actuator lag | 0.00000270 | 0.00000589 | 0.001111 | 0.002275 |

These are equally weighted cold/warm mean two-joint norm errors on128 identical states per plant. The lagged improvement is54% for position and51% for velocity. Direct-torque errors are19–22% worse. Prediction error is a model measurement, not a realized tracking-cost gain.

On the lagged plant, batch128 inference at5 ms took median662 microseconds for affine-flow and966 microseconds for NODE (31% less time). A single query took336 versus307 microseconds (9% more time). The original power head remains cheaper at149/434 microseconds for batch1/128. All measurements use isolated CPU float64 queries; physical robot speed and real-time deployment are untested.

The complete80 new control episodes, paired with400 earlier baseline episodes, show small mixed tracking-cost differences. Against structured NODE at5 ms, affine-flow cost changes range from a1.33% improvement on direct torque with10 ms command holds to a0.97% loss on the lagged robot with40 ms holds. No new episode used the failed-search fallback. At the latter condition, its end-to-end planning time was6.3% longer despite cheaper batched queries; kernel latency does not establish controller latency. The coarsely queried NODE remains an essential lower-compute alternative.

Both models use the same supervised labels, nominal-only validation selection, four-step loss, observation/action clocks, CEM search and paired task streams. Their neural parameter counts are comparable:18,182 versus17,922. Affine-flow has an explicit affine-torque field and analytic actuator integrals, while NODE integrates a learned acceleration field with RK4. These different structural priors are part of the methods; this is not an isolated same-weight power-law intervention. Nominal convergence took100epochs on direct torque and300 on the lagged candidate fit. Inference speed and training cost must be reported separately.

See [all measured control gains and losses](affine_control_result.md), [raw prediction summaries](affine_prediction_result.json), [independent arithmetic](affine_prediction_independent_arithmetic.json), and the [standalone480-clip replay](affine_refinement_demo.html). Every replay is sourced from saved simulated trajectories. The fresh independent test will quantify prediction, inference and fine/coarse control uncertainty before any selected improvement is described as confirmed.
