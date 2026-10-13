# Hybrid AffineFlow component findings

Gate: **no_component_advantage**. Reasons: primary_flow_not_acceptable_in_every_session.

The fixed primary configuration is 30 seconds / 40 ms. Budget 120 seconds and other partitions are diagnostics; they cannot switch the headline.

Every one of the three calibration sessions must pass the provisional simulator screen and all four conservative error bands at ≤ 1.05 times its newly trained hybrid_accel_rk4 anchor, with no worse prediction or decision failures. Paired selected truth costs must differ by no more than their mean joint numerical envelope. Identical selected commands cancel exactly. Ranking ties alone remain usable; no unique oracle optimum is claimed.

The component bar is ≥ 25% pooled complete-scorer median saving versus the fastest acceptable other method at this configuration, with no worse p95. Pooling uses 50 actual calls in each session (150 calls), never group medians. The two acceleration residual references have 1,410 parameters; the three coefficient networks have 1,478 and share the nominal physics and seeded initial tensors.

Fastest acceptable alternative: None. Median saving fraction: None; p95 ratio: None.

| Model | Budget (s) | Partition (ms) | Eligibility | Session statuses | Pooled median / p95 (s) |
|---|---:|---:|---|---|---|
| hybrid_affine_flow | 30 | 5 | fail | 1: unresolved, 2: unresolved, 3: fail | 0.031177208496956155 / 0.032110960445425 |
| hybrid_affine_rk4 | 30 | 5 | fail | 1: fail, 2: fail, 3: fail | 0.05371058300079312 / 0.05503338125272421 |
| hybrid_affine_aware | 30 | 5 | fail | 1: fail, 2: fail, 3: fail | 0.03057031250500586 / 0.03125241220186581 |
| hybrid_accel_rk4 | 30 | 5 | fail | 1: fail, 2: fail, 3: pass | 0.04856043749896344 / 0.04936998719640542 |
| hybrid_accel_aware | 30 | 5 | fail | 1: fail, 2: fail, 3: fail | 0.028228396498889197 / 0.02877716234215768 |
| hybrid_affine_flow | 30 | 10 | fail | 1: unresolved, 2: unresolved, 3: fail | 0.02325756249774713 / 0.02386789824522566 |
| hybrid_affine_rk4 | 30 | 10 | fail | 1: fail, 2: fail, 3: fail | 0.04043695900327293 / 0.04159332135532168 |
| hybrid_affine_aware | 30 | 10 | fail | 1: fail, 2: fail, 3: fail | 0.022837645999970846 / 0.023373060601443285 |
| hybrid_accel_rk4 | 30 | 10 | fail | 1: fail, 2: fail, 3: pass | 0.03700925000885036 / 0.03812028125539655 |
| hybrid_accel_aware | 30 | 10 | fail | 1: fail, 2: fail, 3: fail | 0.02126847948966315 / 0.021880027596489524 |
| hybrid_affine_flow | 30 | 20 | fail | 1: unresolved, 2: unresolved, 3: fail | 0.018906437493569683 / 0.019629091995739145 |
| hybrid_affine_rk4 | 30 | 20 | fail | 1: fail, 2: fail, 3: fail | 0.03307820850022836 / 0.034206325148261385 |
| hybrid_affine_aware | 30 | 20 | fail | 1: unresolved, 2: fail, 3: fail | 0.01861904200632125 / 0.019186312348028877 |
| hybrid_accel_rk4 | 30 | 20 | fail | 1: fail, 2: fail, 3: pass | 0.029131541501556057 / 0.029928710902458987 |
| hybrid_accel_aware | 30 | 20 | fail | 1: fail, 2: fail, 3: fail | 0.0166887295054039 / 0.017111272407782964 |
| hybrid_affine_flow | 30 | 40 | fail | 1: unresolved, 2: unresolved, 3: fail | 0.01640187499288004 / 0.017036166854813926 |
| hybrid_affine_rk4 | 30 | 40 | fail | 1: fail, 2: fail, 3: fail | 0.028842208499554545 / 0.02987325365975266 |
| hybrid_affine_aware | 30 | 40 | fail | 1: unresolved, 2: fail, 3: fail | 0.01611485399917001 / 0.016697204496449555 |
| hybrid_accel_rk4 | 30 | 40 | fail | 1: fail, 2: fail, 3: pass | 0.025397833000170067 / 0.026460520804539554 |
| hybrid_accel_aware | 30 | 40 | fail | 1: fail, 2: fail, 3: fail | 0.01438704149768455 / 0.01492227514318074 |
| hybrid_affine_flow | 120 | 5 | fail | 1: fail, 2: fail, 3: fail | 0.03113431249948917 / 0.031751642146264206 |
| hybrid_affine_rk4 | 120 | 5 | fail | 1: fail, 2: pass, 3: fail | 0.05389452150120633 / 0.054939333405491196 |
| hybrid_affine_aware | 120 | 5 | fail | 1: fail, 2: fail, 3: fail | 0.030636270996183157 / 0.031205668747861636 |
| hybrid_accel_rk4 | 120 | 5 | pass | 1: pass, 2: pass, 3: pass | 0.04843945850734599 / 0.049608179200731684 |
| hybrid_accel_aware | 120 | 5 | fail | 1: fail, 2: fail, 3: fail | 0.028246874499018304 / 0.02898602104978636 |
| hybrid_affine_flow | 120 | 10 | fail | 1: fail, 2: fail, 3: fail | 0.02323756250552833 / 0.023702876899187684 |
| hybrid_affine_rk4 | 120 | 10 | fail | 1: fail, 2: pass, 3: fail | 0.04046831250161631 / 0.04139284564735135 |
| hybrid_affine_aware | 120 | 10 | fail | 1: fail, 2: fail, 3: fail | 0.022863708501972724 / 0.023362110894959185 |
| hybrid_accel_rk4 | 120 | 10 | pass | 1: pass, 2: pass, 3: pass | 0.037075250009365845 / 0.03811785435755155 |
| hybrid_accel_aware | 120 | 10 | fail | 1: fail, 2: fail, 3: fail | 0.0213385829993058 / 0.021809727051731898 |
| hybrid_affine_flow | 120 | 20 | fail | 1: fail, 2: fail, 3: fail | 0.01886931250191992 / 0.01939472919621039 |
| hybrid_affine_rk4 | 120 | 20 | fail | 1: fail, 2: pass, 3: fail | 0.03305302049557213 / 0.03413339360995451 |
| hybrid_affine_aware | 120 | 20 | fail | 1: fail, 2: fail, 3: fail | 0.018596124995383434 / 0.01918170799399377 |
| hybrid_accel_rk4 | 120 | 20 | pass | 1: pass, 2: pass, 3: pass | 0.02895168749819277 / 0.029826198094087885 |
| hybrid_accel_aware | 120 | 20 | fail | 1: fail, 2: fail, 3: fail | 0.01665870850411011 / 0.017167816594883333 |
| hybrid_affine_flow | 120 | 40 | fail | 1: pass, 2: fail, 3: fail | 0.016394250000303145 / 0.01697395625305944 |
| hybrid_affine_rk4 | 120 | 40 | fail | 1: pass, 2: pass, 3: fail | 0.028902291000122204 / 0.02990091235769796 |
| hybrid_affine_aware | 120 | 40 | fail | 1: fail, 2: fail, 3: fail | 0.01611458299885271 / 0.016691710754093946 |
| hybrid_accel_rk4 | 120 | 40 | pass | 1: pass, 2: pass, 3: pass | 0.02538077099598013 / 0.026266612348990748 |
| hybrid_accel_aware | 120 | 40 | fail | 1: fail, 2: fail, 3: fail | 0.01437479200103553 / 0.014922537643724355 |

All eligible and ineligible alternatives, session reasons, paired state differences and retained ranking statuses are in component_gate.json. Raw prediction/decision outcomes, timing samples and rotated-order provenance, fit failures and optimization limits remain in short_calibration_result.json and the immutable external store.

These 5% margins are provisional engineering choices. Three independent sessions are exploratory, not powered confirmation. Unit fixtures are ineligible for empirical claims. A positive component result supports a later independent confirmation design; a valid nonpositive primary result ends active investment in this line. No physical deadline, hardware or total commissioning saving is established.
