# Control precision: corrected head versus MLP at 5 ms

The two updated comparisons use ten independent training seeds with twenty paired tasks each. Six other comparisons retain five seeds and their original look-1 evidence.

| Plant | Evidence | MLP cost | Head cost | MLP minus head | Lower bound | Upper bound |
| --- | --- | --- | --- | --- | --- | --- |
| direct | lower_cost | 0.317403845 | 0.245727704 | 0.0716761414 | 0.0291477648 | 0.114204518 |
| lagged | lower_cost | 0.379069483 | 0.280378317 | 0.098691166 | 0.0408691446 | 0.156513187 |

| Cached criterion | Evidence | Training seeds | Source look |
| --- | --- | --- | --- |
| utility_control/direct/mlp_dt/h40000000 | unresolved | 5 | 1 |
| utility_control/direct/node_generic/h40000000 | unresolved | 5 | 1 |
| utility_control/direct/node_generic/h5000000 | unresolved | 5 | 1 |
| utility_control/lagged/mlp_dt/h40000000 | unresolved | 5 | 1 |
| utility_control/lagged/node_generic/h40000000 | unresolved | 5 | 1 |
| utility_control/lagged/node_generic/h5000000 | unresolved | 5 | 1 |

Positive differences favor the head. Ratios are descriptive; signed bounds include paired training-seed uncertainty and measured numerical envelopes. Unresolved or influential clusters do not establish a win.

Control look 2 spends alpha 0.0125, with eight comparisons and two directional tails (0.00078125 per tail); common 20,000 training-seed bootstrap draws use [9102,2].

This extension does not establish nominal noninferiority, NODE equivalence, full-suite superiority, actuator-rate or noise robustness, or a physical-robot result. The original report and NODE solver limitations remain attached to the six cached comparisons.

All 800 paired primary episodes remain included, with 0 fallback intervals.
Resources below cover completed new primary/finer episode producers; training, prior experiments and failed/interrupted attempts are separate.
