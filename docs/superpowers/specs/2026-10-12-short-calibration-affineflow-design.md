# Short-calibration AffineFlow continuation

Status: proposed research design; no new method or experiment has run.

This continuation investigates whether AffineFlow can obtain a useful robot model with less calibration interaction or planning computation. The immediate investment is a bounded mechanism and data-efficiency study on the existing two-link simulator. A torque-controlled six- or seven-joint arm performing free-space transfers is the eventual application, conditional on those results and a documented practical modeling bottleneck.

The user requested a spec and plan, work on the current branch, and meaningful commits. The staging, numerical grids and provisional thresholds below are engineering choices made for this design, rather than requirements supplied by an application owner. Writing these documents does not start training, change the completed result, select hardware, or establish a benefit.

## 1. Evidence and research questions

The starting source revision is `638c061eab73053b549c2c0430b3da6fae85eab4`. The [completed compute study](../../../reports/compute_study_findings.md) reports a 0.71% control-cost reduction against coarse RK4 NODE and 1.07% against quadrature NODE. Both NODEs already propagate actuator activation analytically. RK4 is more accurate on naturally moving warm probes, and quadrature is faster in the isolated inference measurement. Planning takes approximately 218 ms per 40 ms decision while the simulator waits. These results concern one fixed training corpus, rather than independent calibration sessions.

Inspect `poc/affine_flow.py`, `poc/quadrature_node.py`, `poc/models/ode.py`, `poc/models/common.py`, `poc/train.py`, `poc/stateless_mpc.py` and `poc/physics_fit.py` when implementing. The first five contain two-joint assumptions; `stateless_mpc` additionally fixes six knots and 40 ms actuation. The public checkout omits historical binary corpora/checkpoints and local authorization files. The new study must run from freshly generated inputs without requiring that private archive.

Three questions have separate answers:

1. **Representation:** does learning `a = f(q,v) + B(q,v) effort` reduce the data needed compared with an unrestricted structured acceleration NODE?
2. **Propagation:** does AffineFlow add accuracy or speed beyond using that same representation with RK4, and beyond a cheap actuator-aware NODE?
3. **Usefulness:** do those differences improve command selection and, subsequently, accepted task performance under a real computation deadline?

Cold probes include nonzero velocity and activation. They are not evidence of an advantage when starting from rest. The hypothesized integration regime is an appreciable actuator transient with relatively slowly changing mechanical coefficients. Neither deliberately slowing actuators nor selecting only favorable state populations is part of the experiment.

## 2. Scope and gates

| Stage | Deliverable | Condition for the next stage |
|---|---|---|
| A: mechanism and learning curves | Two missing solver controls, fresh calibration sessions, prediction/data/runtime curves and fitted-physics/residual comparisons | Complete results identify a surviving representation or propagation advantage, or end with a bounded null finding |
| B: common decisions | Finite-bank decision regret and false-feasibility rates on independent moving states | Advantage affects decisions and survives the strongest appropriate alternatives; nominal/calibrated physics headroom is documented |
| C: deadline-constrained control | Continuous physical-time simulation and quality versus 5/10/20 ms compute budgets | A frozen task requirement is met with an operationally meaningful benefit and acceptable miss/fallback behavior |
| D: practical arm confirmation | Independent calibration sessions on a selected, qualified arm/configuration and processor | Confirm the prospectively chosen commissioning or compute claim; report failures and alternatives |

The accompanying [implementation plan](../plans/2026-10-12-short-calibration-affineflow.md) implements A and B, including timing needed to decide whether C is plausible. C and D require their own execution specs after their entry gates resolve platform, command-path and task choices. No larger arm benchmark is justified just to rescue a null result. A representation-only win is reported as such; it does not establish a unique advantage of the AffineFlow propagator.

Excluded from this iteration: contact, grasp impacts, online payload detection, unknown-context encoders, arbitrary actuator models, embedded deployment, a broad architecture search, a new optimizer, and modification of the historical study. Payload/tool configuration stays fixed within each future calibration-and-scoring block.

## 3. Global implementation constraints

- Work on the current branch and make meaningful local commits; no push or publication.
- Native CPython 3.11; CPU PyTorch 2.8.0, MuJoCo 3.14.0, NumPy 2.2.6, SciPy 1.15.3 and pytest 8.4.1; no new dependencies.
- Use additive modules and a new protocol `PTCO-CAL-0.1`; preserve historical source, freezes, checkpoints, reports and acceptance rules.
- Stage A/B use the existing two-joint lagged plant, common known `tau_s = 0.08`, integer physical time, and held torque commands in N*m.
- Float32 training; float64 numerical audits and scoring; single-threaded CPU with thread settings recorded.
- All methods receive the same available observations, command paths, nominal prior and data-budget partition. Train each learned solver through its own update rule.
- Generate all required inputs from the public checkout. Default mutable output is ignored `study/short-calibration`; allow an explicit `--output-root` without workstation-specific paths.
- Retain failures, unsuccessful fits and interrupted attempts. A completed job is reusable only when its input/source/model hashes match.
- Freeze recipes before their scored outcomes; no confirmation-based tuning, selected favorable seeds or post hoc tolerance changes.

## 4. Actuator-aware mechanism controls

All predictors retain `predict(state, command, h_s, history=None, power_mode=None)` and the existing `Prediction` shape, status and work counters. Inputs are immutable; negative/nonfinite intervals fail; zero intervals return the input without network evaluations. Mixed zero/nonzero batches preserve zero rows and finite gradients.

### Torque-affine RK4

Add `AffineRK4` using the identical `AffineFlow` coefficient network, feature order, output normalization and initialization. At every RK4 stage evaluate `q_dot = v`, `v_dot = f(q,v) + B(q,v) w(t)`, with exact stage activation. One step costs four coefficient-network evaluations; optional positive integer substeps cost `4 * substeps`. Endpoint activation remains analytic. On direct torque use `u` as effort, primarily for analytical tests.

Two comparisons must be labeled separately: (a) independent training through each solver for end-to-end performance; (b) identical frozen coefficient weights evaluated with both propagators to isolate numerical effects. Cross-solver prediction is a diagnostic and cannot replace the trained comparison.

### Two-evaluation actuator-aware NODE

Add `ActuatorAwareNODE` with the identical `NeuralODE(structured=True)` acceleration network, features and seeded initialization. Its mechanical midpoint is the current quadrature midpoint:

```text
a0 = a(q, v, w)
qm = q + h*v/2 + h²*a0/8
vm = v + h*a0/2
wm = w + (u-w)*(1-exp(-h/(2*tau)))
am = a(qm, vm, wm)
```

For positive h and one shared time constant:

```text
beta = 1 - exp(-h/(2*tau))
Kv = h - tau*(1-exp(-h/tau))
Kq = h²/2 - tau*h + tau²*(1-exp(-h/tau))
v_next = v + h*a0 + (Kv/beta)*(am-a0)
q_next = q + h*v + h²*a0/2 + (Kq/beta)*(am-a0)
```

Return exact endpoint activation and two acceleration evaluations. This is a proposed exponential-response quadrature, not a proven general nonlinear improvement. For direct torque retain the existing two-evaluation quadrature update; do not invent a tau for that path.

The identity is exact for arbitrary constant drift and a dense constant effort-to-acceleration matrix with shared tau, including nonzero initial activation. It does not extend automatically to different time constants per input: each output may mix multiple exponentials through a dense gain matrix. Stage A explicitly rejects nonuniform tau rather than applying output-wise ratios incorrectly.

Implement known coefficients in a new `poc/actuator_integrals.py`; do not change frozen historical helpers. Use `expm1` and stable small-ratio series for `Kv`, `Kq`, `Kv/beta` and `Kq/beta`, with `r = h/tau`. As r tends to zero, the last two ratios approach h and h²/3 respectively. Handle h=0 before division and maintain continuity and gradients across the arithmetic branch. Permit precomputation only for detached, known h/tau; keep differentiable intervals on the tensor path.

Verify float64 against independent high-precision scalar arithmetic over h/tau from `1e-8` to `4`, and random constant-coefficient dense systems against a separately derived closed form. Require ratio relative error <= `1e-11` and constant-system endpoints `atol=2e-10, rtol=2e-7`. Test float32 finiteness and correct leading command-response orders, `q:h³` and `v:h²` on the lagged plant. The supplied review's external check file is unavailable in this checkout; recreate the check independently instead of asserting it ran here.

Do not confuse the `.08` coefficient-output divisor in `AffineFlow.coefficients` with actuator tau. Stage A preserves that normalization. A future generalization must separately name `actuator_tau_s`, `drift_reference_s`, and `gain_reference_s`.

## 5. Comparator set

| Study ID | Representation and solver | Role |
|---|---|---|
| `affine_flow` | Existing f/B network; AffineFlow | Candidate |
| `affine_rk4` | Same f/B network; RK4 | Representation versus propagation control |
| `node_structured` | Existing unrestricted acceleration network; RK4 | Established strong learned comparator |
| `node_quadrature` | Same unrestricted acceleration network; existing quadrature | Existing cheap comparator |
| `node_actuator_aware` | Same unrestricted acceleration network; new two-evaluation rule | Cheap integration challenge |
| `physics_fit` | Existing five-parameter rigid dynamics; endpoint calibration and qualified RK4 | Simple identification gate |
| `physics_residual` | Frozen nominal rigid dynamics plus 32/32 tanh acceleration residual; RK4 | Small correction to an available prior |

For the residual, inputs are the same eight physical/periodic/normalized features as the structured NODE; its two outputs are acceleration corrections scaled by `scale_v/.04`. The initial nominal parameters are the declared historical reference initialization `[0.10, 0.02, 0.02, 0.05, 0.05]`, frozen without reading truth parameters. Use the same prior for unfitted `physics_nominal`, an additional nonlearning decision reference. Give that reference a hash-bound construction receipt with its budget's common scales, not a fictitious fit receipt. Measure its prediction screen as well as its decisions/timing so the practical headroom gate includes it; its receipts do not increase fit counts. A zero-residual fixture must reproduce nominal dynamics under the same integration resolution. Residual weights use ordinary seeded initialization; nominal physics remains frozen. Residual propagation uses one RK4 step per queried interval, with four residual-network calls correctly included in work counters; it does not inherit the rigid reference's zero-network-call accounting.

Fit `physics_fit` once per session/budget, rather than counting repeated deterministic fits as independent seeds. Use its existing endpoint fitting recipe and solver-resolution check. If its complete calibration suffices, report that result prominently; the current clean plant may leave no practical need for a large neural model. Do not inject a contrived mismatch in Stage A. Later mismatch must be supported by the selected real system or a separately justified simulator configuration.

Expose parameter counts, field calls, inference batch size, CPU/wall fitting time, failed fits, peak RSS and bytes. Architecture family initialization is paired: the two f/B models share initial network tensors; the three unrestricted NODEs share theirs. Different representations cannot be interpreted as identical-parameter causal interventions.

## 6. Calibration sessions and data accounting

Stage A is a simulation mechanism study. Its seconds measure observed calibration motion, not hardware commissioning time. Generate independent session roots in a new PCG64 namespace `[20261012, 1, role, session, trajectory, channel]`, where role codes are train-session=0, development-evaluation=1, confirmation=2. Never call historical root-seed helpers to generate new study inputs.

One session contains 600 separate one-second calibration segments, with 25 held commands at the declared 40 ms observation/command tick. Each segment starts from independently sampled `q in [-0.5,0.5]`, `v in [-0.25,0.25]`, `w in [-0.3,0.3]`. At each command tick use bounded excitation `u = clip(-q - .25*v + d, -1.5, 1.5)`; sample each component of d uniformly in `[-.75,.75]` every five ticks and hold d between draws. This common feedback/excitation collector uses observed state, not a learned model or privileged acceleration. Preserve any numerical failure with its coordinate; do not replace unsuccessful segments by new favorable draws.

Simulator resets make segments independent and are disclosed. Reset/settling overhead is not covered by the motion budget; real commissioning must use executable continuous trajectories and count that overhead. Calibration is not presented as hardware-safe merely because commands are bounded.

Use nested total motion budgets **30, 120 and 600 seconds**. Every fifth complete segment is validation; the other four are training. Thus train/validation seconds are 24/6, 96/24 and 480/120. Validation motion is counted, rather than furnished as a free fixed dataset. No windows cross segments or roles. Derive every model's common scales from that budget's training subset only; do not reuse full-session normalization or full-data weights for a smaller budget. Independent initialization at every budget is the primary learning curve; warm-start curves are deferred.

Calibration telemetry consists of the declared 40 ms observations and held-command log. Dense physics traces used to check truth are evaluator-only numerical records, not simulated extra robot sensors. Do not delete practically available fine-time labels on a future platform. Any common downsampling or telemetry choice must be frozen and justified as an acquisition choice before outcomes; all methods may use the same available high-rate records.

Start with one development session and one initialization at each budget: **18 neural fits plus 3 physical fits**. Only after this smoke/pilot matrix closes, expand to five independent development sessions and two paired initializations: **180 neural fits plus 15 physical fits**, including the pilot cells. Treat this as exploratory measurement. Record a throughput-based projection after the pilot before expansion, and provide explicit CLI phase/session/model/budget filters so expensive work is resumable and intentional.

Neural models use the existing train-only scales, AdamW recipe, effective batch 256 and `shared` objective (one step plus .25 times the four-step objective). Shuffle the existing eight-start windows, flatten that epoch's ordered start list, and batch up to 256 starts with a short final batch. This preserves every start once per epoch while handling the one-start tail in a 25-command segment; the historical trainer's unsplittable-window accumulator is incompatible. Use a study-local loader/trainer rather than changing the fixed historical schema or sampler. Allow learning rates `.001` and `.0003`, validation every five epochs, and 100-epoch blocks up to 1,000 epochs per learning rate. Stop a rate at the existing nominal plateau rule after at least 100 epochs. Select among all its valid checkpoints/rates by nominal validation objective with the existing deterministic tie rule. The ceiling bounds this pilot; a ceiling-hit fit is labeled optimization-limited, never data-insufficient by inference. Do not suppress a poor fit because it fails the historical constant-velocity qualification threshold. Numerical training failure is a recorded method failure; report the other allowed rate's result if valid and retain both attempts.

## 7. Prediction, decision and timing measurements

### Complete motion prediction

Generate 64 fresh evaluation trajectories of 1.6 s from the same collector law and independent role=1 roots. Use a common primary anchor population at 40 ms observations from 320 through 1,200 ms inclusive: 23 anchors per trajectory, 1,472 total. Every anchor has 400 ms of recorded future commands, so all horizons use the same population without extrapolating past the log. Initial states may be a secondary stratum, not half of an artificial primary mixture. Score every model/session/initialization/budget at h = 5/10/20/40 ms for independent held-command endpoint probes; use complete recorded command sequences for 80/160/400 ms open-loop rollouts. Internal query steps for rollouts are 5/10/20/40 ms, always split at actual command changes. Coarse steps must never bridge a change while assuming a held command. Add 80/160 ms single-held-command probes as a labeled transient stress test, not a replacement for natural motion.

Report physical q/v RMSE, p95 endpoint errors, failure fractions, drift/new-command response and composition errors separately. Compare independently trained models and the paired same-weight propagation diagnostic. The truth producer uses 250 us physics steps and verifies designated first/last trajectory references at 125 and 62.5 us; discrepancies form numerical envelopes. If an envelope exceeds 1% of a reported model error or unresolved candidate-cost separation, refine those truth cases before interpreting a difference. Analytical exactness alone cannot establish robot prediction accuracy.

### Common feasible candidate decisions

At the first anchor at/after 320 ms on each of the 64 trajectories, supply the same 256 six-knot, 240 ms candidate paths to every predictor. Knots are held for 40 ms. Include zero-command and the frozen collector's current PD command repeated across the horizon; remaining paths use common reproducible bounded draws in `[-1.5,1.5]`. Banks depend on evaluation coordinates, never model identity, data budget or initialization.

Use the existing controller's fixed physical-anchor cost: `.04*sum(position_error²) + .002*sum(velocity²) + .00004*sum(command²) + 5*terminal_position_error²`. Targets are independently drawn in `[-.5,.5]` per joint. Numerically resolve ties and select the lowest candidate index among predicted-cost ties. Independently simulate every candidate; the evaluator must not rank truth using any learned model.

For this synthetic decision test only, declare an engineering envelope `abs(q)<=1.5 rad`, `abs(v)<=3 rad/s`, and `abs(u)<=2 N*m`. It is a scoring envelope, not the existing plant's joint limits or a hardware safety certificate. Test true and predicted feasibility at fixed 5 ms anchors including the initial state; call this sampled feasibility, since between-anchor maxima are not certified. Separately retain any denser truth-envelope violations.

Score each 5/10/20/40 ms main query partition. To check intermediate anchors without silently replacing a coarse propagator by sequential 5 ms propagation, start each main step at its predicted state and query the same held command at offsets 5 ms, 10 ms, and so on through that step's endpoint. Only the endpoint advances the main trajectory; auxiliary states are used for constraint checks and then discarded. Reuse the already computed endpoint. Auxiliary queries within a main step may be packed in one batch; the construction and accounting are shared across methods. Include every auxiliary query, batching operation, feasibility check and selection in complete scorer timing/work counters. This adds actual work and may eliminate a coarse-step speed advantage; it supplies no free dense output. Costs remain attached only to the six 40 ms endpoints.

For the subset of truly feasible candidates, regret is `J_truth(selected) - min(J_truth(feasible))`. A model selects using its own predicted costs and sampled-feasibility checks. Also report false-feasible candidates, false-infeasible candidates, selected-unsafe fraction, and all-predicted-infeasible fraction. If its selection is truly infeasible, or it selects none, count a decision failure and do not assign a benign zero regret. Report regret conditional on valid decisions alongside the unconditional failure rate. If a state's true bank has no feasible candidate, record a bank-design failure shared by all methods; it cannot silently disappear from coverage or be replaced post hoc.

Reuse common scoring semantics, not `stateless_mpc.rollout_candidates` directly: its fixed 40 ms anchors cannot evaluate 5 ms feasibility or split all requested partitions. Add a dedicated pure study scorer while keeping the old optimizer unchanged.

### Inference and end-to-end scoring

Measure batches 1/128/256 across the query-step grid with inference mode, fixed CPU threads, warm-up and rotated method order. Use 20 warm-up calls and 50 measured repetitions per cell, reporting median/p95 and CPU/wall measurements. Time both the entire common bank scorer (including constraints and selection) and neural propagation alone. Report artifact/logging/verification work separately and include all decision-critical work in the end-to-end figure. Equivalent caching of fixed actuator coefficients is available to every applicable method.

Do not infer a 20 ms deployment from a component microbenchmark. If the simplest shared scorer cannot process useful banks within that time, record that outcome; Stage C may justify one shared compiled path through a new spec. Replacing a learned discrete transition by another solver during export is prohibited.

## 8. Provisional mechanism screen and interpretation

Learning-curve tables show each physical metric and all failures at every budget. For an explicitly provisional **simulator-useful model screen**, require at 160 ms recorded-command rollouts: q RMSE <= .01 rad, v RMSE <= .10 rad/s, q p95 <= .03 rad, v p95 <= .30 rad/s, and zero prediction failures. At common decisions require <=1% selected-unsafe/all-no-selection combined, and mean valid-decision regret <=1% of the mean truth zero-command cost over the fixed library. A zero-command cost <= `1e-8` makes that normalized regret screen undefined; show absolute regret and mark the screen unresolved. A no-feasible-bank case likewise makes the screen unresolved until a prospective bank-design amendment and fresh evaluation.

These tolerances are declared now to make budget thresholds reviewable. They are not customer requirements and cannot support a practical commissioning claim. Apply the screen to a complete model/query-step configuration: prediction, decisions and timing must use the same main partition. Do not combine fine-step accuracy with coarse-step timing. A model's first passing budget must be reported with the full curve; nonmonotone performance, optimization limits and never-passing curves remain visible. With only three budgets, threshold ratios are interval-censored: report the brackets and observed budget ratio without interpolating a precise data requirement. More budgets require a prospective amendment with new untouched scoring inputs.

Development aggregation treats calibration session as the independent cluster and averages initialization replicas within it. Paired resampling keeps all models/budgets/tasks from a session together (10,000 cluster bootstrap draws, fixed namespace). Show descriptive 95% intervals and all five session-level values; do not call five sessions a powered confirmatory demonstration. Repeated states or network seeds on the same corpus are not independent calibration sessions.

Proceed to a Stage C design only if all cells close and either (a) AffineFlow reaches the provisional screen at an earlier budget than the best learned alternative without worse decision reliability, or (b) it has lower end-to-end scoring time at the same passing screen without a decision-regret/reliability loss. Show physics and residual results alongside that decision. If simple physics already passes cheaply, record no practical neural-model headroom on this plant. Any subsequent application requires evidence of a consequential repeatable mismatch, not an artificially weakened reference. Ambiguous or null results finish A/B with a report; they do not automatically trigger more seeds or a larger robot.

If both f/B solvers win similarly, the surviving opportunity is torque-affine representation. If actuator-aware NODE removes the advantage, emphasize the cheaper solver. If only transient stress probes improve while natural-motion decisions worsen, the practical gate fails.

## 9. Deadline and arm roadmap

Stage C starts with a provisional 20 ms outer decision tick, 400 ms horizon and 5/10/20 ms compute budgets on the existing simulator. All methods may choose 5/10/20 ms internal steps using development data. These values require a new freeze, since they differ from the completed controller. Profile the whole pipeline first; remove nonessential candidate checksums/path-object construction from the critical loop, retaining deterministic replay evidence outside it. Compile/optimization choices and CPU placement must be shared across models.

Physical time advances during measured planning latency. A submitted job records observation time, intended activation tick, command-path version and expiry. A plan can activate only at its intended tick if ready before the deadline and still valid; late outputs are discarded. Continue the currently valid plan while a job runs; on plan exhaustion use the same qualified feedback fallback. Allow at most one outstanding plan job. Log observation age, queue/compute/activation latency, misses, stale discards, fallback intervals and actual commands. Use a deterministic injected-latency scheduler for correctness tests and a separate measured wall-clock run on a declared processor. Simulator lockstep that pauses for the optimizer is not the deadline experiment.

Freeze task success, tolerance, settling window, joint/torque/velocity/command-rate limits and fallback before independent scored runs. Include actual task success, settling time, overshoot, constraint violations, and deadline-miss consequences separately from the weighted cost. A compute frontier needs several budgets and configurations, not a single matched-time comparison. Candidate counts and any early stopping are tuned only on development data; unused/lost plans still consume compute.

Stage D first records the selected arm/processor, torque interface, inner feedback/compensation/filtering/rate limits, measurable actuator response, effort observability or common estimator, nominal calibration method and feasible free-space envelope. Do not impose 80 ms lag on a platform. If the shared-first-order assumption fails, document a new method extension and comparator protocol before using it. A future n-joint contract uses `q,v,w` each length n, a `3n -> 128 -> 128 -> n+n²` f/B network, explicit effort scales and separately named reference times. Heterogeneous actuators require a valid revised cheap comparator; they cannot be enabled by replacing scalar constants.

Include calibrated physics, nominal-plus-residual, and the robot's existing controller as the practical reference. Independently demonstrate where ordinary identification fails the intended requirement. Hardware candidate commands require an independently qualified supervisor; model-predicted feasibility alone is insufficient. Keep each configuration fixed per block and pair methods on the same data/task conditions. Count all physical calibration, validation, reset/settling, fitting, selection and acceptance-check time; distinguish interaction, compute, and total elapsed commissioning time.

Choose one primary claim before confirmation, preferably calibration interaction required to pass the task at a 20 ms deadline. A 2x interaction or compute reduction, or 15–20% accepted movement-time reduction, is a proposed meaningful target, not an expected result. Confirmation sample size is chosen prospectively from development session variability and the chosen claim; use fresh session/task roots, one fixed analysis, a family/multiplicity rule and numerical/reliability checks. No threshold or headline switching after results. Inaccessible hardware yields a stated simulation-only result rather than an implied deployment claim.

The practical setting is supported by [Context-Aware Deep Lagrangian Networks for Model Predictive Control](https://arxiv.org/html/2506.15249v3), which evaluates a nominal-plus-learned model on a seven-joint arm and uses 50 Hz MPC with a 1 kHz hardware feedback loop. It does not establish AffineFlow superiority. [acados discrete dynamics](https://docs.acados.org/python_interface/#acados_template.acados_model.AcadosModel.disc_dyn_expr) offers a possible later transition-export path; library choice and correct command semantics remain design work for Stage C/D.

## 10. Artifacts and acceptance of A/B

Provide `python -m poc.calibration_study {prepare,collect,train,predict,decisions,time,report,verify,status}` with required `--output-root` propagation and optional coordinate filters. `prepare` defaults to the pilot; expansion needs explicit `--matrix development`. No historical selected checkpoint is a required input.

Artifacts bind protocol, stage/matrix, source hashes, asset hashes, telemetry and actuator specification, session/budget/role identities, collector/recipe/seed, actual observation seconds, tensor/checkpoint hashes, solver/query partitions, candidate-bank identity and numerical truth envelopes. Job keys include those identities; commit mutable execution state at complete trajectory/optimizer-update/job boundaries. Changed inputs create a new job or fail resumption, never mutate a completed receipt. The new store must use `PTCO-CAL-0.1` rather than the hard-coded protocol in `compute_study.StudyStore`.

The report includes full coordinate coverage, missing/failed/optimization-limited cells, physical learning curves, timing/decision frontiers, session-level uncertainty, nominal/physics/residual results, and the A/B gate decision with its reason. Export tracked Markdown/JSON/CSV/SVG summaries and a portable reproduction command; raw arrays and weights stay in the selected ignored output store. A standalone learning-curve figure is sufficient; a new animation/demo is not required to decide this study.

Software acceptance requires analytical solver controls, data-role/budget isolation, own-solver training and deterministic resume fixtures, independent true-candidate scoring, sampled constraint/failure/tie behavior, rotated timing and complete raw-to-report recomputation. At least one synthetic end-to-end study must work without private files. Reproduce one fit for each new learned solver and one first/last prediction/decision coordinate in a fresh process; compare canonical selected tensors and numerical outputs. For empirical completion, both development matrices and truth audits must be complete; unsuccessful performance remains a valid result. Documentation-only completion of the present request does not satisfy those future experiment requirements.

### Pre-empirical evidence qualification and storage amendment (2026-10-12)

The continuation comparison retains every measured budget/partition and compares passing alternatives available at no greater calibration budget, including smaller-budget fits and other internally matched partitions. Unresolved required physical or learned references cannot support continuation. Matching f/B solvers supports `representation_only` only when an opportunity also survives comparison with unrestricted alternatives.

Independent candidate truth refines disagreement in sampled feasibility using the existing 31,250/15,625 ns bounded refinement grid, even when cost convergence plateaus. Keep all observed Boolean classifications and the historical disagreement flag. Qualification uses the latest two observed resolutions: agreement is resolved at those resolutions; disagreement at exhaustion is unresolved. This is numerical sampled-anchor qualification, not continuous safety certification. An unresolved bank produces no resolved oracle membership or conditional regret; its failure indicator is a conservative upper bound and its configuration/session screen remains unresolved.

The two large prediction and candidate record streams use `rows.jsonl.gz`: canonical UTF-8 JSONL inside standard-library gzip at fixed compression level 1, with zero timestamp and no original filename in the header. Values, types, record membership and decoded scientific meanings remain unchanged except for the explicit feasibility qualification above. Compression runs after scoring, outside critical timing. Outcomes and timing samples retain their existing JSONL names. Pilot storage/runtime projections use actual produced compressed bytes and runtime; unit-fixture compression ratios are not empirical evidence.

Resource records retain attempt identities, cumulative active CPU/wall spans and checkpoint persistence spans at committed boundaries. Caught interruptions close their measured spans; abandoned running attempts retain measured lower bounds and explicit unknown tails. Inactive gaps and resource-observation writes are excluded. Producer attempts retain parent-job/attempt provenance. Reports sum nonoverlapping producer attempts: nested store work is excluded when its parent is already represented, while inclusive store and rate spans remain diagnostics. Unknown-tail values cannot be interpreted as complete execution costs or point throughput estimates. Post-producer finalization remains separately measured for timing products.
