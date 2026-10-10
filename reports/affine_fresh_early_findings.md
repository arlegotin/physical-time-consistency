# Initial independent affine-flow result: still under confirmation

The revised method predicts the known-lag robot more accurately in all five independent training pairs. This is encouraging, but the velocity bound still includes no gain and the position/velocity inference is sensitive to individual training seeds. The combined prediction-and-batched-speed claim is not yet confirmed. Six additional paired training seeds were registered before their outcomes, bringing the planned total to eleven.

All rows below query5 ms ahead on the same512 fresh held-out physical states and commands, equally weighting256 cold and256 warm probes. Position and velocity errors are mean two-joint norms in rad and rad/s. Inference is CPU-only float64, with the same current machine and isolated paired timings. Values are means of training-seed statistics; ratio intervals are central95% paired Fieller intervals, conditional on the fixed training/probe corpus.

| Metric | Affine-flow | Mechanics-aware NODE | Point estimate | Improvement interval |
|---|---:|---:|---:|---:|
| Position error | 3.0863298e-06 | 5.2930113e-06 | 41.7% lower | 8.0% to 63.6% |
| Velocity error | 0.0013108137 | 0.0020492382 | 36.0% lower | -2.7% to 60.4% |
| Inference, batch1 (µs) | 341.2 | 310.8 | -9.8% reduction | -16.1% to -3.9% |
| Inference, batch128 (µs) | 672.2 | 969.7 | 30.7% reduction | 25.2% to 35.6% |

Negative latency reduction means affine-flow is slower. Batch128 is representative of evaluating many planning candidates together; batch-one is a single query. These query timings do not establish a whole-controller speed advantage.

The strong comparator receives the same fully observed joint positions, velocities, activation, known80 ms actuator time constant and endpoint labels. Both use the same nominal convergence/repair policy and shared one-plus-four-step training objective. Affine-flow has18,182 parameters and two field-network calls; NODE has17,922 and four RK4 field calls. Affine-flow adds a torque-linearity prior and exact held-activation integrals, so this comparison is not an isolated same-weights action-power intervention.

The fresh independent control comparison has not yet run. The completed development cohort showed percent-level controller gains and losses, including a small lagged-plant loss. Therefore lower short-horizon prediction error does not yet establish better task performance. Both models will be compared at5 and40 ms internal queries, with identical40 ms hardware clocks and the same controller/search budget.

Raw inputs and the complete initial statistics are in `affine_fresh_measurement.json`, `affine_fresh_references.json` and `provenance/affine-look-001-result.json`. The preserved statistical receipt is `evidence/affine-fresh-looks/look-001.json`; the prospective continuation is `config/affine_precision_request_001.json`. No50% superiority floor applies, no weak seed was removed, and no physical robot, real-time or broader morphology claim is made.
