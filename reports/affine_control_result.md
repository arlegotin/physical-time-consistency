# Affine-flow refinement: paired practical comparison

One development training seed; new architecture; fresh independent confirmation needed for outcome-selected improvement.

| Plant | Hold ms | Query partition ms | Baseline | Cost reduction | Planning time reduction | Paired wins | Fallbacks |
|---|---:|---:|---|---:|---:|---:|---:|
| direct | 40 | 40 | power_correct | -0.04% | -0.85% | 5/10 | 0 |
| direct | 40 | 40 | mlp_h | +0.22% | +12.01% | 6/10 | 0 |
| direct | 40 | 40 | node_generic | +0.61% | +2.96% | 6/10 | 0 |
| direct | 40 | 40 | node_structured | -0.60% | +2.67% | 4/10 | 0 |
| direct | 40 | 40 | physics_fit | -0.15% | +30.53% | 4/10 | 0 |
| direct | 40 | 5 | power_correct | +2.28% | +5.35% | 8/10 | 0 |
| direct | 40 | 5 | mlp_h | +2.01% | +9.85% | 7/10 | 0 |
| direct | 40 | 5 | node_generic | +1.40% | +10.00% | 7/10 | 0 |
| direct | 40 | 5 | node_structured | +0.21% | +4.26% | 4/10 | 0 |
| direct | 40 | 5 | physics_fit | +0.65% | +13.01% | 6/10 | 0 |
| direct | 10 | 40 | power_correct | +0.28% | +8.44% | 5/10 | 0 |
| direct | 10 | 40 | mlp_h | +1.42% | +13.60% | 6/10 | 0 |
| direct | 10 | 40 | node_generic | -0.76% | +8.49% | 2/10 | 0 |
| direct | 10 | 40 | node_structured | +1.39% | +4.07% | 6/10 | 0 |
| direct | 10 | 40 | physics_fit | -0.08% | +15.67% | 4/10 | 0 |
| direct | 10 | 5 | power_correct | +1.72% | +8.44% | 6/10 | 0 |
| direct | 10 | 5 | mlp_h | +1.46% | +9.59% | 6/10 | 0 |
| direct | 10 | 5 | node_generic | -0.81% | +0.60% | 3/10 | 0 |
| direct | 10 | 5 | node_structured | +1.33% | +6.86% | 7/10 | 0 |
| direct | 10 | 5 | physics_fit | -0.14% | +4.49% | 4/10 | 0 |
| lagged | 40 | 40 | power_correct | +0.53% | -12.47% | 6/10 | 0 |
| lagged | 40 | 40 | mlp_h | +0.74% | -12.23% | 7/10 | 0 |
| lagged | 40 | 40 | node_generic | +1.08% | -6.82% | 8/10 | 0 |
| lagged | 40 | 40 | node_structured | +0.63% | -9.46% | 7/10 | 0 |
| lagged | 40 | 40 | physics_fit | -0.13% | +21.73% | 6/10 | 0 |
| lagged | 40 | 5 | power_correct | -1.03% | -20.64% | 3/10 | 0 |
| lagged | 40 | 5 | mlp_h | +1.82% | -16.63% | 7/10 | 0 |
| lagged | 40 | 5 | node_generic | -0.28% | -10.79% | 5/10 | 0 |
| lagged | 40 | 5 | node_structured | -0.97% | -6.32% | 3/10 | 0 |
| lagged | 40 | 5 | physics_fit | -1.43% | -7.20% | 3/10 | 0 |
| lagged | 10 | 40 | power_correct | +0.88% | -14.36% | 8/10 | 0 |
| lagged | 10 | 40 | mlp_h | +3.01% | -10.79% | 8/10 | 0 |
| lagged | 10 | 40 | node_generic | +0.19% | -10.70% | 6/10 | 0 |
| lagged | 10 | 40 | node_structured | -0.20% | -10.74% | 6/10 | 0 |
| lagged | 10 | 40 | physics_fit | +0.44% | +3.64% | 4/10 | 0 |
| lagged | 10 | 5 | power_correct | +1.63% | -15.56% | 8/10 | 0 |
| lagged | 10 | 5 | mlp_h | +4.34% | -10.84% | 10/10 | 0 |
| lagged | 10 | 5 | node_generic | +0.13% | -10.70% | 6/10 | 0 |
| lagged | 10 | 5 | node_structured | -0.26% | -10.47% | 5/10 | 0 |
| lagged | 10 | 5 | physics_fit | +0.38% | -5.18% | 4/10 | 0 |
