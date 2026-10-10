# Shared-state diagnostic: measured results

All 3,072 planned model/probe/horizon rows are complete. Four selected models use identical states and held commands, with initial activation fixed. No new training or control episodes were run.

One development training seed, 64 cold and 64 warm probes per plant. These are descriptive prediction errors, not confirmed control gains. Numerical envelopes use three plant resolutions and independent continuous equations.

Values are means of two-joint vector norms, in radians for q and rad/s for v. Drift means zero **new** command; it retains initial velocity and activation. Command error compares the command-induced increment with truth. Split error uses two equal half steps with the same command.

| Plant | Stratum | ms | Component | Model | Drift error | Command error | Total error | Split truth error | Composition discrepancy |
|---|---|---:|---|---|---:|---:|---:|---:|---:|
| direct | cold | 5 | q | power_correct | 1.11736e-05 | 8.65034e-06 | 1.21475e-05 | 1.09439e-05 | 3.40957e-06 |
| direct | cold | 5 | q | mlp_h | 9.16165e-05 | 0.000121502 | 0.000139385 | 0.000160362 | 2.54467e-05 |
| direct | cold | 5 | q | node_generic | 5.80768e-05 | 3.67306e-05 | 4.73779e-05 | 4.73779e-05 | 1.67348e-11 |
| direct | cold | 5 | q | node_structured | 2.68322e-06 | 3.20709e-06 | 3.6898e-06 | 3.6898e-06 | 1.78238e-11 |
| direct | cold | 10 | q | power_correct | 3.74537e-05 | 2.95086e-05 | 4.08037e-05 | 3.99044e-05 | 1.26627e-05 |
| direct | cold | 10 | q | mlp_h | 0.000119178 | 0.000158475 | 0.000195346 | 0.000268089 | 8.63695e-05 |
| direct | cold | 10 | q | node_generic | 0.000112478 | 7.21151e-05 | 9.41048e-05 | 9.41047e-05 | 5.3306e-10 |
| direct | cold | 10 | q | node_structured | 1.07055e-05 | 1.28184e-05 | 1.4767e-05 | 1.4767e-05 | 5.67696e-10 |
| direct | cold | 40 | q | power_correct | 0.000237631 | 0.0001937 | 0.000290138 | 0.000351701 | 0.000255258 |
| direct | cold | 40 | q | mlp_h | 0.000296511 | 0.000228889 | 0.000316778 | 0.000416965 | 0.00034846 |
| direct | cold | 40 | q | node_generic | 0.000464688 | 0.000307678 | 0.000456107 | 0.000455958 | 5.33512e-07 |
| direct | cold | 40 | q | node_structured | 0.000168644 | 0.000203423 | 0.000236328 | 0.000236326 | 5.61644e-07 |
| direct | warm | 5 | q | power_correct | 1.31457e-05 | 9.43392e-06 | 1.58237e-05 | 1.45454e-05 | 4.05179e-06 |
| direct | warm | 5 | q | mlp_h | 0.000102843 | 0.000123934 | 0.000141218 | 0.000166117 | 2.91034e-05 |
| direct | warm | 5 | q | node_generic | 6.04464e-05 | 3.4937e-05 | 4.71204e-05 | 4.71204e-05 | 4.30734e-11 |
| direct | warm | 5 | q | node_structured | 4.58148e-06 | 3.00718e-06 | 4.45229e-06 | 4.4523e-06 | 4.84489e-11 |
| direct | warm | 10 | q | power_correct | 4.51891e-05 | 3.26584e-05 | 5.40976e-05 | 5.34347e-05 | 1.52605e-05 |
| direct | warm | 10 | q | mlp_h | 0.000137435 | 0.000162227 | 0.000192845 | 0.000269933 | 9.59134e-05 |
| direct | warm | 10 | q | node_generic | 0.000117893 | 6.76701e-05 | 9.33501e-05 | 9.33503e-05 | 1.38712e-09 |
| direct | warm | 10 | q | node_structured | 1.82961e-05 | 1.20262e-05 | 1.78137e-05 | 1.78141e-05 | 1.55498e-09 |
| direct | warm | 40 | q | power_correct | 0.000299192 | 0.000194426 | 0.000329033 | 0.000482797 | 0.000333864 |
| direct | warm | 40 | q | mlp_h | 0.000317314 | 0.000214428 | 0.00030937 | 0.000383534 | 0.000348508 |
| direct | warm | 40 | q | node_generic | 0.00047553 | 0.000269523 | 0.000484829 | 0.000485004 | 1.4229e-06 |
| direct | warm | 40 | q | node_structured | 0.000288575 | 0.000192499 | 0.000284914 | 0.00028542 | 1.59624e-06 |
| direct | cold | 5 | v | power_correct | 0.0027482 | 0.00254679 | 0.00348132 | 0.00383173 | 0.000416252 |
| direct | cold | 5 | v | mlp_h | 0.00258976 | 0.00318003 | 0.00335595 | 0.00385773 | 0.00069562 |
| direct | cold | 5 | v | node_generic | 0.00226409 | 0.00103952 | 0.00224306 | 0.00224306 | 1.61397e-10 |
| direct | cold | 5 | v | node_structured | 0.00107192 | 0.0012824 | 0.00147637 | 0.00147637 | 1.7734e-10 |
| direct | cold | 10 | v | power_correct | 0.00473346 | 0.00424115 | 0.00574845 | 0.00696991 | 0.00154065 |
| direct | cold | 10 | v | mlp_h | 0.00420967 | 0.00514614 | 0.00505759 | 0.00667671 | 0.00243955 |
| direct | cold | 10 | v | node_generic | 0.00451231 | 0.00206412 | 0.00448693 | 0.00448693 | 5.02074e-09 |
| direct | cold | 10 | v | node_structured | 0.00213566 | 0.00256171 | 0.00295503 | 0.00295503 | 5.5378e-09 |
| direct | cold | 40 | v | power_correct | 0.0134669 | 0.0091999 | 0.0146625 | 0.0169961 | 0.0121862 |
| direct | cold | 40 | v | mlp_h | 0.00704372 | 0.00815227 | 0.00672005 | 0.0116159 | 0.0118107 |
| direct | cold | 40 | v | node_generic | 0.0176684 | 0.00800611 | 0.0179414 | 0.0179422 | 4.28835e-06 |
| direct | cold | 40 | v | node_structured | 0.00834676 | 0.0101106 | 0.0118008 | 0.0118008 | 4.77411e-06 |
| direct | warm | 5 | v | power_correct | 0.00338296 | 0.00331489 | 0.00494954 | 0.00539802 | 0.000499984 |
| direct | warm | 5 | v | mlp_h | 0.00268566 | 0.0034464 | 0.00376545 | 0.00423426 | 0.000646426 |
| direct | warm | 5 | v | node_generic | 0.00277166 | 0.00120932 | 0.00269189 | 0.00269189 | 2.55436e-10 |
| direct | warm | 5 | v | node_structured | 0.00183121 | 0.00120277 | 0.00178118 | 0.00178118 | 4.20674e-10 |
| direct | warm | 10 | v | power_correct | 0.00562963 | 0.00561167 | 0.00822131 | 0.00988227 | 0.00187375 |
| direct | warm | 10 | v | mlp_h | 0.00398728 | 0.00557431 | 0.00594494 | 0.00747546 | 0.00224156 |
| direct | warm | 10 | v | node_generic | 0.00552478 | 0.00242845 | 0.00538551 | 0.0053855 | 8.15505e-09 |
| direct | warm | 10 | v | node_structured | 0.00365324 | 0.0024057 | 0.00356465 | 0.00356465 | 1.29891e-08 |
| direct | warm | 40 | v | power_correct | 0.0137788 | 0.00871327 | 0.0139283 | 0.0219674 | 0.0165517 |
| direct | warm | 40 | v | mlp_h | 0.00864483 | 0.00774769 | 0.00798393 | 0.0146844 | 0.0128331 |
| direct | warm | 40 | v | node_generic | 0.0215921 | 0.0100784 | 0.0215356 | 0.0215332 | 8.95913e-06 |
| direct | warm | 40 | v | node_structured | 0.0142659 | 0.00983699 | 0.0142885 | 0.0142858 | 1.14328e-05 |
| lagged | cold | 5 | q | power_correct | 1.60852e-05 | 6.86358e-07 | 1.60102e-05 | 1.78943e-05 | 5.72152e-06 |
| lagged | cold | 5 | q | mlp_h | 0.000615899 | 8.07672e-05 | 0.000639203 | 0.0007464 | 0.000114143 |
| lagged | cold | 5 | q | node_generic | 2.32445e-05 | 3.38407e-06 | 2.34467e-05 | 2.34471e-05 | 1.14592e-09 |
| lagged | cold | 5 | q | node_structured | 8.0108e-06 | 2.15515e-07 | 7.96611e-06 | 7.966e-06 | 6.8351e-10 |
| lagged | cold | 10 | q | power_correct | 5.40381e-05 | 4.79366e-06 | 5.36087e-05 | 6.34017e-05 | 2.29487e-05 |
| lagged | cold | 10 | q | mlp_h | 0.000757866 | 0.000131201 | 0.000805671 | 0.00114814 | 0.000362782 |
| lagged | cold | 10 | q | node_generic | 4.78615e-05 | 8.20516e-06 | 4.73125e-05 | 4.73207e-05 | 3.22257e-08 |
| lagged | cold | 10 | q | node_structured | 2.95388e-05 | 1.53171e-06 | 2.9204e-05 | 2.91989e-05 | 2.01243e-08 |
| lagged | cold | 40 | q | power_correct | 0.000344839 | 0.000131133 | 0.000374785 | 0.000486009 | 0.000384642 |
| lagged | cold | 40 | q | mlp_h | 0.000443371 | 0.00019288 | 0.000467161 | 0.000817708 | 0.000834951 |
| lagged | cold | 40 | q | node_generic | 0.000303501 | 9.02294e-05 | 0.00031476 | 0.000306253 | 2.1683e-05 |
| lagged | cold | 40 | q | node_structured | 0.000332807 | 5.17545e-05 | 0.000323778 | 0.000316158 | 1.49244e-05 |
| lagged | warm | 5 | q | power_correct | 9.30828e-06 | 6.83403e-07 | 9.34904e-06 | 8.67296e-06 | 3.47261e-06 |
| lagged | warm | 5 | q | mlp_h | 0.000256848 | 5.3185e-05 | 0.000261408 | 0.000317569 | 7.06119e-05 |
| lagged | warm | 5 | q | node_generic | 1.46478e-05 | 2.57512e-06 | 1.48501e-05 | 1.48502e-05 | 3.6693e-10 |
| lagged | warm | 5 | q | node_structured | 3.82277e-06 | 7.95768e-08 | 3.80599e-06 | 3.80592e-06 | 3.32165e-10 |
| lagged | warm | 10 | q | power_correct | 3.15913e-05 | 4.64384e-06 | 3.1838e-05 | 3.18144e-05 | 1.30058e-05 |
| lagged | warm | 10 | q | mlp_h | 0.000312351 | 8.43981e-05 | 0.000315269 | 0.000478202 | 0.000201593 |
| lagged | warm | 10 | q | node_generic | 2.88399e-05 | 5.30409e-06 | 2.89162e-05 | 2.89185e-05 | 1.05057e-08 |
| lagged | warm | 10 | q | node_structured | 1.50476e-05 | 6.09532e-07 | 1.49305e-05 | 1.49284e-05 | 9.94468e-09 |
| lagged | warm | 40 | q | power_correct | 0.000209858 | 8.46393e-05 | 0.000215237 | 0.000276294 | 0.000238917 |
| lagged | warm | 40 | q | mlp_h | 0.000282816 | 0.000147181 | 0.00028783 | 0.000460983 | 0.000424076 |
| lagged | warm | 40 | q | node_generic | 0.000141353 | 5.07903e-05 | 0.000142076 | 0.00014165 | 8.55738e-06 |
| lagged | warm | 40 | q | node_structured | 0.000222164 | 3.31922e-05 | 0.000218055 | 0.000216216 | 8.10872e-06 |
| lagged | cold | 5 | v | power_correct | 0.00747456 | 0.000297243 | 0.00748007 | 0.00790962 | 0.000496841 |
| lagged | cold | 5 | v | mlp_h | 0.0159247 | 0.00505803 | 0.0175486 | 0.0212163 | 0.00400266 |
| lagged | cold | 5 | v | node_generic | 0.00331487 | 0.000456739 | 0.00326461 | 0.00326462 | 3.01461e-08 |
| lagged | cold | 5 | v | node_structured | 0.00306963 | 0.00012439 | 0.00304321 | 0.00304321 | 1.94628e-08 |
| lagged | cold | 10 | v | power_correct | 0.0123338 | 0.000975472 | 0.012164 | 0.0135947 | 0.00181822 |
| lagged | cold | 10 | v | mlp_h | 0.0194949 | 0.00606283 | 0.0218913 | 0.0331075 | 0.0126173 |
| lagged | cold | 10 | v | node_generic | 0.00564173 | 0.00104245 | 0.00555628 | 0.00555663 | 8.73743e-07 |
| lagged | cold | 10 | v | node_structured | 0.00544552 | 0.000426427 | 0.00534884 | 0.00534897 | 4.61618e-07 |
| lagged | cold | 40 | v | power_correct | 0.0134199 | 0.00603334 | 0.0140205 | 0.0214953 | 0.0163746 |
| lagged | cold | 40 | v | mlp_h | 0.0164835 | 0.00870767 | 0.0178068 | 0.0266318 | 0.0243706 |
| lagged | cold | 40 | v | node_generic | 0.0107292 | 0.00473444 | 0.0115074 | 0.0114731 | 0.000301411 |
| lagged | cold | 40 | v | node_structured | 0.0134432 | 0.00320221 | 0.0131034 | 0.0130235 | 0.000177818 |
| lagged | warm | 5 | v | power_correct | 0.00310183 | 0.00014738 | 0.00310497 | 0.00337821 | 0.000354768 |
| lagged | warm | 5 | v | mlp_h | 0.00748654 | 0.0045682 | 0.00999279 | 0.0124761 | 0.00281121 |
| lagged | warm | 5 | v | node_generic | 0.0007111 | 0.000227335 | 0.000754627 | 0.000754626 | 9.40743e-09 |
| lagged | warm | 5 | v | node_structured | 0.00151676 | 4.70257e-05 | 0.00150721 | 0.00150721 | 5.68629e-09 |
| lagged | warm | 10 | v | power_correct | 0.00512039 | 0.000498137 | 0.00513773 | 0.00611304 | 0.00130999 |
| lagged | warm | 10 | v | mlp_h | 0.00935159 | 0.00448473 | 0.0114944 | 0.0188882 | 0.0086237 |
| lagged | warm | 10 | v | node_generic | 0.00142225 | 0.00046254 | 0.00150975 | 0.00150975 | 2.62314e-07 |
| lagged | warm | 10 | v | node_structured | 0.00296228 | 0.000179002 | 0.00292989 | 0.00292988 | 1.53176e-07 |
| lagged | warm | 40 | v | power_correct | 0.0102543 | 0.00345309 | 0.0106265 | 0.0132275 | 0.00978133 |
| lagged | warm | 40 | v | mlp_h | 0.0102049 | 0.00551011 | 0.0103433 | 0.0155939 | 0.0127296 |
| lagged | warm | 40 | v | node_generic | 0.00556867 | 0.00249337 | 0.0060015 | 0.00600086 | 0.000100344 |
| lagged | warm | 40 | v | node_structured | 0.0105897 | 0.00237136 | 0.0102467 | 0.0102351 | 8.29513e-05 |

## Paired head comparisons

Positive reductions favor the head. “Wins” counts paired probe norms; it is not an independent training replication count. Complete medians, p90 values, normalized command errors, numerical bounds and signed squared-error cross terms are in the JSON.

| Plant | Stratum | ms | Component | Baseline | Drift reduction | Command reduction | Total reduction | Total probe wins |
|---|---|---:|---|---|---:|---:|---:|---:|
| direct | cold | 5 | q | mlp_h | +87.80% | +92.88% | +91.28% | 64/64 |
| direct | cold | 5 | q | node_generic | +80.76% | +76.45% | +74.36% | 63/64 |
| direct | cold | 5 | q | node_structured | -316.43% | -169.73% | -229.22% | 5/64 |
| direct | cold | 10 | q | mlp_h | +68.57% | +81.38% | +79.11% | 60/64 |
| direct | cold | 10 | q | node_generic | +66.70% | +59.08% | +56.64% | 51/64 |
| direct | cold | 10 | q | node_structured | -249.86% | -130.21% | -176.32% | 8/64 |
| direct | cold | 40 | q | mlp_h | +19.86% | +15.37% | +8.41% | 40/64 |
| direct | cold | 40 | q | node_generic | +48.86% | +37.04% | +36.39% | 44/64 |
| direct | cold | 40 | q | node_structured | -40.91% | +4.78% | -22.77% | 28/64 |
| direct | warm | 5 | q | mlp_h | +87.22% | +92.39% | +88.79% | 63/64 |
| direct | warm | 5 | q | node_generic | +78.25% | +73.00% | +66.42% | 53/64 |
| direct | warm | 5 | q | node_structured | -186.93% | -213.71% | -255.41% | 4/64 |
| direct | warm | 10 | q | mlp_h | +67.12% | +79.87% | +71.95% | 59/64 |
| direct | warm | 10 | q | node_generic | +61.67% | +51.74% | +42.05% | 47/64 |
| direct | warm | 10 | q | node_structured | -146.99% | -171.56% | -203.69% | 5/64 |
| direct | warm | 40 | q | mlp_h | +5.71% | +9.33% | -6.36% | 28/64 |
| direct | warm | 40 | q | node_generic | +37.08% | +27.86% | +32.13% | 41/64 |
| direct | warm | 40 | q | node_structured | -3.68% | -1.00% | -15.49% | 25/64 |
| direct | cold | 5 | v | mlp_h | -6.12% | +19.91% | -3.74% | 29/64 |
| direct | cold | 5 | v | node_generic | -21.38% | -145.00% | -55.20% | 18/64 |
| direct | cold | 5 | v | node_structured | -156.38% | -98.60% | -135.80% | 12/64 |
| direct | cold | 10 | v | mlp_h | -12.44% | +17.59% | -13.66% | 25/64 |
| direct | cold | 10 | v | node_generic | -4.90% | -105.47% | -28.12% | 25/64 |
| direct | cold | 10 | v | node_structured | -121.64% | -65.56% | -94.53% | 18/64 |
| direct | cold | 40 | v | mlp_h | -91.19% | -12.85% | -118.19% | 10/64 |
| direct | cold | 40 | v | node_generic | +23.78% | -14.91% | +18.28% | 38/64 |
| direct | cold | 40 | v | node_structured | -61.34% | +9.01% | -24.25% | 28/64 |
| direct | warm | 5 | v | mlp_h | -25.96% | +3.82% | -31.45% | 22/64 |
| direct | warm | 5 | v | node_generic | -22.06% | -174.11% | -83.87% | 16/64 |
| direct | warm | 5 | v | node_structured | -84.74% | -175.60% | -177.88% | 7/64 |
| direct | warm | 10 | v | mlp_h | -41.19% | -0.67% | -38.29% | 18/64 |
| direct | warm | 10 | v | node_generic | -1.90% | -131.08% | -52.66% | 19/64 |
| direct | warm | 10 | v | node_structured | -54.10% | -133.27% | -130.63% | 12/64 |
| direct | warm | 40 | v | mlp_h | -59.39% | -12.46% | -74.45% | 18/64 |
| direct | warm | 40 | v | node_generic | +36.19% | +13.54% | +35.32% | 42/64 |
| direct | warm | 40 | v | node_structured | +3.41% | +11.42% | +2.52% | 30/64 |
| lagged | cold | 5 | q | mlp_h | +97.39% | +99.15% | +97.50% | 64/64 |
| lagged | cold | 5 | q | node_generic | +30.80% | +79.72% | +31.72% | 51/64 |
| lagged | cold | 5 | q | node_structured | -100.79% | -218.47% | -100.98% | 16/64 |
| lagged | cold | 10 | q | mlp_h | +92.87% | +96.35% | +93.35% | 64/64 |
| lagged | cold | 10 | q | node_generic | -12.91% | +41.58% | -13.31% | 34/64 |
| lagged | cold | 10 | q | node_structured | -82.94% | -212.96% | -83.57% | 18/64 |
| lagged | cold | 40 | q | mlp_h | +22.22% | +32.01% | +19.77% | 38/64 |
| lagged | cold | 40 | q | node_generic | -13.62% | -45.33% | -19.07% | 22/64 |
| lagged | cold | 40 | q | node_structured | -3.62% | -153.37% | -15.75% | 31/64 |
| lagged | warm | 5 | q | mlp_h | +96.38% | +98.72% | +96.42% | 64/64 |
| lagged | warm | 5 | q | node_generic | +36.45% | +73.46% | +37.04% | 47/64 |
| lagged | warm | 5 | q | node_structured | -143.50% | -758.80% | -145.64% | 10/64 |
| lagged | warm | 10 | q | mlp_h | +89.89% | +94.50% | +89.90% | 64/64 |
| lagged | warm | 10 | q | node_generic | -9.54% | +12.45% | -10.10% | 29/64 |
| lagged | warm | 10 | q | node_structured | -109.94% | -661.87% | -113.24% | 11/64 |
| lagged | warm | 40 | q | mlp_h | +25.80% | +42.49% | +25.22% | 35/64 |
| lagged | warm | 40 | q | node_generic | -48.46% | -66.64% | -51.49% | 21/64 |
| lagged | warm | 40 | q | node_structured | +5.54% | -155.00% | +1.29% | 28/64 |
| lagged | cold | 5 | v | mlp_h | +53.06% | +94.12% | +57.38% | 57/64 |
| lagged | cold | 5 | v | node_generic | -125.49% | +34.92% | -129.13% | 12/64 |
| lagged | cold | 5 | v | node_structured | -143.50% | -138.96% | -145.80% | 11/64 |
| lagged | cold | 10 | v | mlp_h | +36.73% | +83.91% | +44.43% | 52/64 |
| lagged | cold | 10 | v | node_generic | -118.62% | +6.43% | -118.92% | 11/64 |
| lagged | cold | 10 | v | node_structured | -126.49% | -128.75% | -127.41% | 15/64 |
| lagged | cold | 40 | v | mlp_h | +18.59% | +30.71% | +21.26% | 33/64 |
| lagged | cold | 40 | v | node_generic | -25.08% | -27.44% | -21.84% | 24/64 |
| lagged | cold | 40 | v | node_structured | +0.17% | -88.41% | -7.00% | 37/64 |
| lagged | warm | 5 | v | mlp_h | +58.57% | +96.77% | +68.93% | 55/64 |
| lagged | warm | 5 | v | node_generic | -336.20% | +35.17% | -311.46% | 5/64 |
| lagged | warm | 5 | v | node_structured | -104.50% | -213.40% | -106.01% | 14/64 |
| lagged | warm | 10 | v | mlp_h | +45.25% | +88.89% | +55.30% | 51/64 |
| lagged | warm | 10 | v | node_generic | -260.02% | -7.70% | -240.30% | 5/64 |
| lagged | warm | 10 | v | node_structured | -72.85% | -178.29% | -75.36% | 16/64 |
| lagged | warm | 40 | v | mlp_h | -0.48% | +37.33% | -2.74% | 27/64 |
| lagged | warm | 40 | v | node_generic | -84.14% | -38.49% | -77.06% | 17/64 |
| lagged | warm | 40 | v | node_structured | +3.17% | -45.62% | -3.71% | 28/64 |

## Reproduce

```bash
.venv/bin/python -m poc.shared_state_diagnostic verify
.venv/bin/python -m poc.shared_state_diagnostic report
```

These commands verify/reaggregate existing measured artifacts. `run` reuses a completed compatible job; changed inputs require a separately versioned design. The immutable manifest is `evidence/M40/20261007T230617-ff6297b2/manifest.json`.

No statistical look or alpha was spent. Correct action orders alone need not produce accurate drift, finite-time response or composition. Do not sum drift and command error norms: the signed vectors can cancel.
