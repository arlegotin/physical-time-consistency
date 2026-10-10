# Mechanism and control utility: owner-directed scope amendment

**Date:** 2026-10-01. **Applies to:** the ongoing physical-time consistency project.

The user explicitly revised the goal after seeing the twenty-seed sensitivity results: “50% is artificial goal: if mechanism itself wins baseline it's already very good; don't be so harsh, we need positive results here without made-up goals.” This amendment governs subsequent work. The unchanged supplied PTCO-1.0 document and its completed decisions remain historical records.

The objective is to determine whether the specified power correction improves prediction of a new command's effects, and whether that improvement helps robot control against qualified time-aware baselines. A fixed percentage improvement is no longer a prerequisite for useful evidence or for proceeding to control. Report the measured size, uncertainty, nominal performance and computation cost. A positive finding can be limited to the conditions where it is supported; other conditions remain unresolved or negative.

The completed twenty-seed direct-torque comparison already passes the originally declared strict-improvement contrasts at 5 ms in both cold and warm probe strata and at 10 ms in the cold stratum. The warm 10 ms contrast remains unresolved. Those findings concern action-Jacobian error against `node_generic`; control, lagged-plant confirmation and superiority over the full competitor suite remain untested. The current forty-seed batch continues unchanged, preserving all observations and statistical looks.

Do not start further seed-doubling solely to decide the withdrawn 50% cutoff. After the current batch, summarize the original predeclared strict-improvement contrasts and proceed to a qualified controller. Collect further evidence only for an identified useful question, such as an unresolved improvement sign, actuator semantics, or control utility.

All physical units, plants, actuator observations, model definitions, train/validation boundaries, nominal checkpoint selection, clock separation, numerical qualification, pairing, failed-run preservation and honest resource accounting remain in force. No architecture, target distribution or loss is changed by this amendment. Numerical validity and planner qualification tolerances still serve measurement correctness; they are not claimed as scientific laws.

New control work uses the already specified six-knot MPC task, common physical cost anchors and shared CEM streams. Freeze the tested comparisons and uncertainty procedure before opening control outcomes. Start with the qualified core predictors and both actuator chains; expand the competitor scope when it answers a live question. A win over one tested baseline is reported at that scope, without a claim of superiority over untested models. Measure nominal and shifted performance together, including any tradeoff.

The owner revised the acceptance objective after prediction outcomes were observed. State that timing openly. Do not relabel the original 50% contrasts as passes, erase an original decision, reset its alpha ledger, or present an amended overall claim as untouched PTCO-1.0 confirmation. Existing predeclared strict-improvement results retain their original evidential status. Future control results must distinguish development from new confirmation.

Positive results are an outcome to investigate and document, never a requirement to manufacture. A supported mechanistic improvement is useful even when the original demanding all-claims PASS is not established. The delivered report will distinguish supported prediction benefit, observed control benefit, unresolved conditions and limitations.
