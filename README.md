# Physical-time consistency in learned robot dynamics

**AffineFlow is a learned transition model for smooth torque-controlled robots.** It learns acceleration drift and torque response, then propagates position and velocity using known actuator dynamics. The aim is reliable prediction when a planner queries a model at a different timestep from its training data, with less neural computation than an RK4 neural ODE.

**Status: the continuation is complete, and active investment in this research direction has ended.** The short-calibration pilot and final physics-prior component study did not establish a practical AffineFlow advantage over stronger alternatives. Better state prediction in the final study did not produce acceptable decision quality in every calibration session; cheap two-evaluation alternatives were also faster in diagnostic timing measurements.

An earlier simulation study found a **small, statistically supported control-cost advantage at matched CPU planning time**: AffineFlow with 40 ms queries reduced mean cost by **0.71% versus the strongest tested coarse NODE** and **1.07% versus a cheaper two-evaluation NODE**. That result covers **38 training seeds and 4,560 paired episodes** on one fully observed two-link arm with known actuator lag. It remains scoped historical evidence; the follow-ups below tested different questions and stronger physical alternatives.

[Final component findings](reports/hybrid_component/component_findings.md) · [Calibration pilot](reports/short_calibration_pilot/pilot_execution_findings.md) · [Research trajectory](docs/research-trajectory.md) · [Earlier control result](reports/compute_study_findings.md) · [Method](docs/method.md) · [Reproduction](docs/reproduction.md) · [Citation](CITATION.cff)

## Latest continuation results

The continuation asked whether AffineFlow needs less calibration data or computation to make useful decisions. It added two mechanism controls: the same torque-affine network propagated with RK4, and a two-evaluation actuator-aware NODE. Fitted physics and nominal physics plus a learned residual provided practical references. Every learned method trained through its own propagation rule.

- **Standalone calibration pilot: `no_advantage`.** All 21 fits completed. Torque-affine models passed the provisional simulator screen with 30 seconds of calibration, but fitted physics and the physics residual did too. The residual improved prediction, decision regret and complete-scorer runtime versus AffineFlow at the same budgets and 20/40 ms partitions. This was one calibration session; the larger development matrix was not run.
- **Final physics-prior component study: `no_component_advantage`.** All 30 neural fits across three fresh calibration sessions completed. At the fixed 30-second / 40-ms primary configuration, Flow reduced position RMSE by **20–38%** and velocity RMSE by **15–24%** versus the newly trained acceleration-residual RK4 reference. All prediction error bands passed, but realized selected-action cost was worse in session 3 beyond numerical uncertainty. Sessions 1 and 2 retained unresolved paired comparisons because the RK4 anchor had one invalid selected decision in each session.

The final study's complete-scorer median was **16.40 ms for Flow**, **16.11 ms for the coefficient actuator-aware control**, and **14.39 ms for the acceleration actuator-aware reference**. These are diagnostic timings: no primary method passed the quality requirements in all sessions, so there was no eligible comparator for the declared 25% runtime-saving claim. The scorer evaluates a shared finite candidate bank; these are not physical control-loop deadline measurements.

Fresh-process selected fits and numerical products reproduced exactly in both follow-ups. The studies use a noiseless, fully observed, two-joint simulator with known first-order actuator lag and provisional engineering thresholds. The negative results end this bounded research direction; they do not prove universal model inferiority. Code, complete outcomes, secondary improvements and unresolved comparisons are preserved. [Final execution account](reports/hybrid_component/task4_execution_report.md).

## Why physical time matters

A controller may observe a robot every 40 ms but simulate candidate commands with 5 ms model steps. Subdividing a prediction adds computation; it does not add observations or change the executed command path.

For direct torque, a new command affects velocity at order `h` and position at order `h²`. With a first-order motor lag, these become `h²` and `h³`, holding initial activation fixed. A predictor can fit training intervals while assigning the command an unrealistically immediate influence at shorter intervals. Correct response order is useful structure, but accurate drift, gain and integration still matter.

This repository measures those effects separately and tests whether they translate into realized model-predictive-control (MPC) performance.

## What AffineFlow does

The model learns a local acceleration law:

```text
q_dot = v
v_dot ≈ fθ(q, v) + Bθ(q, v) · effort
w_dot = (u − w) / τ                 [lagged actuator]
```

`u` is the new torque command; effort is `u` for direct torque and activation `w` for lagged actuation. A 128/128 tanh MLP predicts the two-dimensional drift `fθ` and a 2×2 response matrix `Bθ`. It receives joint sine/cosine and velocity features, with no simulator accelerations or true inertia parameters.

Each prediction evaluates the network at the initial state and a predicted midpoint, then propagates the full interval using the midpoint coefficients and analytical actuator integrals. This costs **two neural evaluations** and has **18,182 trainable parameters**. The nonlinear mechanical flow is approximated; exact integration, positive-definite inertia and stability are not guaranteed. [Equations and implementation](docs/method.md).

AffineFlow is a later refinement of the original **discrete power head**. Its results are reported as a separate method and protocol, rather than as a success of the original PTCO-1.0 acceptance criteria.

## Earlier compute-matched control result

The completed `PTCO-COMPUTE-0.1` study compares AffineFlow against two mechanics-structured neural ODE (NODE) predictors: a four-evaluation RK4 implementation and a competitive two-evaluation acceleration-quadrature implementation. Both NODEs enforce `q_dot = v` and correct actuator semantics.

All models use the same augmented training corpus, one-plus-four-step endpoint objective, nominal-only checkpoint selection and optimizer opportunity. The controller uses six torque knots over a 240 ms horizon. Sensing, replanning and executed command holds stay at **40 ms**. Each model can use **5 ms or 40 ms internal queries**. Candidate counts were selected from timing before new control outcomes, then frozen; measured planning times satisfy the declared 10% matching tolerance.

| Model, 40 ms queries | Mean realized cost ↓ | AffineFlow cost reduction | Position RMSE | Strict task successes | Mean episode median planning time |
|---|---:|---:|---:|---:|---:|
| **AffineFlow** | **0.282965** | — | 0.217797 rad | 742/760 | 218 ms |
| Structured NODE, RK4 | 0.284978 | **0.71%** | 0.217662 rad | 738/760 | 217 ms |
| Structured NODE, quadrature | 0.286012 | **1.07%** | 0.217627 rad | 746/760 | 215 ms |

Paired central 95% cost-improvement intervals are **0.45–0.96%** against coarse RK4 and **0.86–1.27%** against coarse quadrature. These descriptive intervals are not simultaneous confidence bands. Both gains pass the prospectively declared eight-comparison Holm correction, sequential alpha spending and seed-influence checks. All six model/query configurations have 760 episodes each. [All comparisons, including losses and unresolved results](reports/compute_study_findings.md).

The cost weights position error, joint speed, squared commanded torque and terminal position error. AffineFlow's gain over coarse RK4 comes mainly from lower joint-speed cost; position tracking during motion is slightly worse. Strict success-rate superiority is not established. **The 5 ms AffineFlow configuration does not beat every NODE option.** [Cost decomposition](reports/compute_study_interpretation.md).

## Accuracy and computation trade-offs

These are measurements from the earlier study at **5 ms prediction intervals**, separate from its matched-compute controller result and the calibration follow-ups:

- **Prediction:** average position/velocity endpoint errors are 29.5%/22.8% lower than RK4 across equally weighted cold and warm probes. On warm, already-moving states, RK4 is more accurate: AffineFlow errors are 5.4%/11.2% higher.
- **Batched inference:** at batch 128, AffineFlow takes 0.666 ms versus RK4's 0.956 ms, a 30.3% time reduction. Quadrature is faster at 0.516 ms.
- **Single-query inference:** AffineFlow takes 0.331 ms versus RK4's 0.308 ms and quadrature's 0.188 ms.
- **Training:** mean CPU fit times are 5.31 minutes for AffineFlow, 8.68 for RK4 and 5.01 for quadrature. Selected nominal objectives differ, so these are not training-speed measurements at equal fitting accuracy.

Measured planning takes about **0.22–0.24 seconds per 40 ms feedback decision**. The simulator waits for planning. This is an offline control experiment, not a real-time deployment demonstration.

## Inspect the result

Open [reports/compute_study_demo.html](reports/compute_study_demo.html) locally in a browser. It is a self-contained replay of saved MuJoCo trajectories, with all 38 seeds, twenty tasks per seed and six configurations. It performs no training or simulation and needs no server.

The [machine result](reports/compute_study_result.json), [metrics CSV](reports/compute_study_metrics.csv) and [interpretation](reports/compute_study_interpretation.md) retain the measurements. Raw candidate banks, checkpoints and datasets are external artifacts; a source checkout alone cannot validate every scientific result.

## Run the implementation

Use native **CPython 3.11**, CPU PyTorch and a virtual environment. Intentional pins are PyTorch 2.8.0, MuJoCo 3.14.0, NumPy 2.2.6, SciPy 1.15.3 and pytest 8.4.1. On macOS ARM64:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --only-binary=:all: -r requirements.in
python -m pip check
python -m pytest -q tests/test_affine_flow.py tests/test_quadrature_node.py tests/test_stateless_mpc.py
```

On Linux, install the official CPU build of `torch==2.8.0` from `https://download.pytorch.org/whl/cpu` first, then install the remaining pins. The original execution used native macOS ARM64, single-threaded CPU, float32 training and float64 numerical audits. No external dataset, robot hardware, GPU or paid service is required for the model implementation.

See [reproduction instructions](docs/reproduction.md) for the prediction API, verification boundaries, artifact requirements and historical experiment commands. The focused tests use synthetic fixtures; passing them is not a new empirical robot result.

## Evidence and limits

The earlier compute-matched study independently recomputes all 4,560 episode costs. **912 fresh-process replay and finer-physics comparisons pass**, first/last quadrature training replications reproduce selected tensors, and no controller episode uses an intrinsic failed-search fallback. Protocols and earlier statistical looks are retained. These are internal reproducibility checks, rather than independent laboratory replication.

The supported claim concerns **one smooth, fully observed, known-lag, two-link simulated arm**, the fixed corpus and specified MPC workload. Contact, unknown actuator dynamics, larger robots, physical hardware, energy consumption and general state-of-the-art superiority remain untested. The prior is motivated by established mechanics; research novelty is not asserted.

Earlier studies are indexed in [the research trajectory](docs/research-trajectory.md). The detailed executor journal and unmodified evidence remain in the private scientific archive.

## License and citation

Copyright © 2026 Artem Legotin. Original project code and documentation are licensed under **GNU AGPL-3.0-only**; see [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md). AGPL permits commercial use and requires source sharing in the circumstances specified by the license. Citation is requested separately from those legal obligations.

Use [CITATION.cff](CITATION.cff) to cite the software. Implementation, experiment execution and documentation were developed with AI assistance under the project owner's direction; the findings are scoped simulation evidence.

Maintainers preparing a public repository should use [the publication instructions](docs/PUBLICATION.md). The private execution history and original artifact store must remain intact for archival verification.
