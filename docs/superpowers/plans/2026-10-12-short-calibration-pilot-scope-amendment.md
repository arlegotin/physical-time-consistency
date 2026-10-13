# Prospective scope amendment: close the short-calibration pilot

The continuation ends with the complete 21-fit exploratory pilot, its fresh-process reproduction and artifact review. The 195-fit development matrix, deadline controller and larger-arm experiment remain unexecuted. This is the controller's decision under the original plan's explicit permission to record a prospective scope amendment after a decisively null pilot; it is not a claim that the development study ran.

The decision was recorded on 2026-10-12 after primary raw verification and the prescribed independent numerical reproduction, before any development preparation or acquisition. The original spec, plan, frozen recipe, source revision, stopping rules, thresholds and scoring populations retain their original chronology. This amendment changes the remaining execution scope only.

## Evidence and decision

The [verified pilot](../../../reports/short_calibration_pilot/pilot_execution_findings.md) closes every declared coordinate: 21 finite fits, three nominal references, 24 prediction products, six same-weight propagation comparisons, 24 decision products and one complete pooled timing request. There are no missing, invalid or failed attempts; the capped 30-second actuator-aware fit remains labeled optimization-limited.

The frozen provisional gate is `no_advantage`, with reason `no_earlier_budget_or_faster_equal_screen_without_decision_loss`. All 96 configurations resolve. Both torque-affine models, fitted physics and the physics residual pass at the lowest tested 30-second budget. The torque-affine models pass earlier than the unrestricted structured NODE in this pilot, which supports investigating representation rather than claiming a unique AffineFlow propagator benefit.

At the same 30/120/600-second budgets and 20/40 ms main partitions, the residual model improves prediction errors, conservative decision regret and complete-scorer median/p95 time over AffineFlow, with equal zero decision failures. At 30 seconds and 40 ms, position RMSE is 0.00121 versus 0.00265 rad and scorer median is 29.4 versus 41.4 ms. Fitted physics is substantially more accurate but slower in its current implementation. Ordinary physical calibration therefore already resolves the current plant's model mismatch; a larger robot demonstration is not justified by these results.

The required separate-store affine-RK4, actuator-aware NODE and physics-residual fits reproduce selected tensors, canonical hashes and validation exactly. First/last prediction and candidate-bank coordinates reproduce numerical outputs, truth envelopes, failure statuses, selected actions, regret and work logs exactly. Independently replayed first/last truth at 250/125/62.5 microseconds also agrees exactly. Reproduction is not another independent calibration session.

The [resource projection](../../../reports/short_calibration_pilot/pilot_resource_projection.json) shows adequate disk headroom. Remaining development producer work is estimated at 9.708 serial hours, with approximately 11.411 cumulative pilot-plus-development hours before reporting, verification and finalization. These are workload estimates. The stop is a scientific investment decision, not a resource block.

## Remaining deliverables and limits

Finish the pilot execution checklist, portable reproduction evidence, research-trajectory update and independent artifact review; preserve all primary and replica artifacts. Keep the full development and conditional C/D steps explicitly unexecuted. No favorable metric or new threshold replaces the frozen gate, and no additional experiment is launched to rescue a positive headline.

This remains one independent calibration session and one neural initialization per method/budget, on a noiseless two-joint simulator with known common actuator lag. It establishes neither a cross-session null nor a hardware commissioning or real-time deadline result. Benefits below 30 seconds, different mismatches and other platforms are untested. Stopping here could miss an advantage in another draw or an untested regime; any renewed study needs its own prospective acquisition and scoring declaration.

The known interrupted-resource-observation accounting defect remains disclosed. All 132 primary ledgers completed without interrupted/abandoned attempts, unknown tails, failed events or error logs, so the primary run did not exercise that failure path. Affected future runs must qualify their resource totals and cannot use them to establish a commissioning-compute advantage.

A subsequent practical-arm proposal should first document a consequential repeatable error that remains after simple physical calibration, and then specify the real command path, observable actuator state, limits and externally meaningful acceptance requirements. The present result supplies no reason to launch a deadline or hardware implementation now.
