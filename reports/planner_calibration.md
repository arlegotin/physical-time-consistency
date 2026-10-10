# Analytic planner qualification

The first qualifying common search budget is **128 candidates, four iterations and sixteen elites**. On the fixed sixteen unit-mass tasks with five search streams each, it recovers about 80% of the available cost reduction on the median task in each stream. The weakest task/stream recovers 70.5%; the spread between stream mean reductions is 1.39 percentage points.

The earlier 64×2 and 128×2 budgets did not qualify and remain saved. Selection follows the original model-independent calibration sequence. No learned-model ranking was used.

The optimum comes from box-constrained linear least squares, with independently checked projected optimality and exact-map rollout cost. Raw sampled knots, candidate costs, selected commands and RNG namespaces are retained in the local calibration artifacts; their hashes and measured resource use are in [the calibration record](planner_calibration.json). [Planner configuration](../config/planner.json) marks analytic qualification separately from the pending robot-reference qualification.

This checks the planner on an exact double integrator. **It is not a learned-model robot-control result.** The robot task, physical-reference fit and control comparisons remain to be executed.

The selected analytic check can be rerun with `.venv/bin/python -m poc.mpc_calibration --candidates 128 --iterations 4`; it writes a new measured artifact and preserves the original.
