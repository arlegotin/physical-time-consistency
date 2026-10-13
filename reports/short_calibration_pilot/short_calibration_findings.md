# Short calibration A/B (pilot, exploratory)

Gate: **no_advantage**. Reasons: no_earlier_budget_or_faster_equal_screen_without_decision_loss.

Coverage complete: True. Expected fits: 21; nominal construction references: 3. Missing cells: 0; invalid: 0; retained failed fits: 0; optimization limited: 1.

This is a provisional simulator screen. Motion seconds are a calibration interaction proxy. There is no feedback deadline, deployment or hardware commissioning result. Held-response improvements do not establish decision improvements. Reduced unit fixtures are ineligible for empirical claims.

Prediction at 160 ms recorded motion must satisfy q RMSE ≤ .01 rad, v RMSE ≤ .10 rad/s, q p95 ≤ .03 rad and v p95 ≤ .30 rad/s, with zero prediction failures. RMSE is sqrt(mean joint-component squared error); p95 is the percentile of two-joint endpoint norms. Screens use absolute errors plus numerical envelopes. Sub-envelope differences cannot establish superiority. Decision failures must be ≤ 1%; conservative valid regret must be ≤ 1% of mean truth zero-command cost. Zero-command cost ≤ 1e-8 or an empty feasible truth bank leaves the screen unresolved. Unresolved sampled truth feasibility makes conditional regret undefined and the failure fraction a conservative upper bound. Prediction, decision and timing partitions are matched in every row.

Independent session clusters: 1. Initialization replicas are averaged within session; configuration screens use those session-averaged metrics. All paired model/budget/task rows use the same 10,000 fixed-namespace cluster draws. Bootstrap namespace: [20261012, 1, 1, 0, 0, 5]. Individual fitted-model screens and all session values remain in JSON. Passing an average does not mean every fitted model passes. Intervals are descriptive; five sessions are not powered confirmation or a customer reliability claim.

| Model | Motion proxy (s) | Main step (ms) | q RMSE (rad) | v RMSE (rad/s) | q p95 (rad) | v p95 (rad/s) | Failure fraction | Valid regret | Scorer median / p95 (s) | Screen | Limited |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| affine_flow | 120 | 10.0 | 0.000760484 | 0.00875183 | 0.00210097 | 0.0240201 | 0 | 6.5427e-07 | 0.0403224 / 0.0425674 | pass | False |
| affine_flow | 120 | 20.0 | 0.000714749 | 0.00872765 | 0.00198579 | 0.0241009 | 0 | 6.5427e-07 | 0.0398099 / 0.0418563 | pass | False |
| affine_flow | 120 | 40.0 | 0.000644961 | 0.00890924 | 0.00188239 | 0.0242599 | 0 | 6.5427e-07 | 0.041435 / 0.0442041 | pass | False |
| affine_flow | 120 | 5.0 | 0.00077328 | 0.00876203 | 0.00212572 | 0.0239907 | 0 | 6.5427e-07 | 0.0475165 / 0.0514677 | pass | False |
| affine_flow | 30 | 10.0 | 0.00273124 | 0.0300112 | 0.00834496 | 0.088687 | 0 | 0.00184631 | 0.0399361 / 0.0428304 | pass | False |
| affine_flow | 30 | 20.0 | 0.00270782 | 0.0299305 | 0.00835501 | 0.0886442 | 0 | 0.00184631 | 0.0396103 / 0.0422532 | pass | False |
| affine_flow | 30 | 40.0 | 0.00264819 | 0.0296816 | 0.00807442 | 0.0896363 | 0 | 0.00184631 | 0.0413851 / 0.0449123 | pass | False |
| affine_flow | 30 | 5.0 | 0.00273759 | 0.0300322 | 0.00838283 | 0.0886157 | 0 | 0.00184631 | 0.047136 / 0.0488523 | pass | False |
| affine_flow | 600 | 10.0 | 0.000658509 | 0.0077216 | 0.00178527 | 0.020775 | 0 | 0.00148395 | 0.0400876 / 0.0427663 | pass | False |
| affine_flow | 600 | 20.0 | 0.000581158 | 0.0075891 | 0.00156742 | 0.0201962 | 0 | 0.00148395 | 0.0401444 / 0.043366 | pass | False |
| affine_flow | 600 | 40.0 | 0.000357166 | 0.00734852 | 0.000965529 | 0.0195271 | 0 | 0.00148395 | 0.042288 / 0.045197 | pass | False |
| affine_flow | 600 | 5.0 | 0.000678454 | 0.00775841 | 0.00183576 | 0.0208885 | 0 | 0.00148395 | 0.0477595 / 0.0492303 | pass | False |
| affine_rk4 | 120 | 10.0 | 0.000652292 | 0.00729761 | 0.00191516 | 0.0219922 | 0 | 0 | 0.0731658 / 0.0772363 | pass | False |
| affine_rk4 | 120 | 20.0 | 0.000652083 | 0.00729742 | 0.00191464 | 0.0219914 | 0 | 0 | 0.0739467 / 0.078159 | pass | False |
| affine_rk4 | 120 | 40.0 | 0.000648814 | 0.00729449 | 0.00190618 | 0.0219761 | 0 | 0 | 0.078576 / 0.08552 | pass | False |
| affine_rk4 | 120 | 5.0 | 0.000652305 | 0.00729763 | 0.0019152 | 0.0219922 | 0 | 0 | 0.0842782 / 0.0890769 | pass | False |
| affine_rk4 | 30 | 10.0 | 0.00248604 | 0.0272609 | 0.00805989 | 0.0877125 | 0 | 0.00152238 | 0.0723493 / 0.0754175 | pass | False |
| affine_rk4 | 30 | 20.0 | 0.00248608 | 0.0272612 | 0.00805912 | 0.0877121 | 0 | 0.00152238 | 0.0736625 / 0.0782143 | pass | False |
| affine_rk4 | 30 | 40.0 | 0.00248682 | 0.0272651 | 0.00805848 | 0.0877049 | 0 | 0.00152238 | 0.0784554 / 0.0840893 | pass | False |
| affine_rk4 | 30 | 5.0 | 0.00248603 | 0.0272609 | 0.00805994 | 0.0877125 | 0 | 0.00152238 | 0.0837315 / 0.0894989 | pass | False |
| affine_rk4 | 600 | 10.0 | 0.000380795 | 0.00418456 | 0.00109071 | 0.012422 | 0 | 0 | 0.0734735 / 0.0761082 | pass | False |
| affine_rk4 | 600 | 20.0 | 0.000380778 | 0.00418462 | 0.00109108 | 0.0124205 | 0 | 0 | 0.07441 / 0.0799544 | pass | False |
| affine_rk4 | 600 | 40.0 | 0.000380648 | 0.00418573 | 0.00109787 | 0.0123955 | 0 | 0 | 0.0794763 / 0.0831646 | pass | False |
| affine_rk4 | 600 | 5.0 | 0.000380796 | 0.00418455 | 0.00109068 | 0.0124221 | 0 | 0 | 0.0846329 / 0.0894667 | pass | False |
| node_actuator_aware | 120 | 10.0 | 0.00128578 | 0.0148359 | 0.00385236 | 0.0453511 | 0.015625 | 0.000116821 | 0.0411095 / 0.0429567 | fail | False |
| node_actuator_aware | 120 | 20.0 | 0.00126641 | 0.01456 | 0.00371877 | 0.0432816 | 0.015625 | 0 | 0.0394847 / 0.0414235 | fail | False |
| node_actuator_aware | 120 | 40.0 | 0.0013449 | 0.0156701 | 0.00374192 | 0.0441138 | 0.015625 | 3.32743e-05 | 0.0422309 / 0.0450515 | fail | False |
| node_actuator_aware | 120 | 5.0 | 0.00129321 | 0.0149327 | 0.00385913 | 0.0456572 | 0.015625 | 0.000116821 | 0.0479205 / 0.0522764 | fail | False |
| node_actuator_aware | 30 | 10.0 | 0.00344918 | 0.0388449 | 0.0103621 | 0.119846 | 0.046875 | 0.00159725 | 0.0404292 / 0.0436913 | fail | True |
| node_actuator_aware | 30 | 20.0 | 0.00345708 | 0.0388862 | 0.0103301 | 0.119105 | 0.046875 | 0.00159725 | 0.0390539 / 0.0418146 | fail | True |
| node_actuator_aware | 30 | 40.0 | 0.0035413 | 0.0398521 | 0.0104843 | 0.118969 | 0.03125 | 0.00159701 | 0.0426326 / 0.0468149 | fail | True |
| node_actuator_aware | 30 | 5.0 | 0.00344818 | 0.0388465 | 0.0103595 | 0.119932 | 0.046875 | 0.00159725 | 0.0475792 / 0.0502124 | fail | True |
| node_actuator_aware | 600 | 10.0 | 0.000888038 | 0.0102107 | 0.00241103 | 0.0273336 | 0 | 0 | 0.0409329 / 0.043159 | pass | False |
| node_actuator_aware | 600 | 20.0 | 0.000826765 | 0.00943084 | 0.00220195 | 0.0246852 | 0 | 0 | 0.03927 / 0.0432443 | pass | False |
| node_actuator_aware | 600 | 40.0 | 0.000811622 | 0.0094417 | 0.00208055 | 0.0237565 | 0 | 0 | 0.0427505 / 0.0465856 | pass | False |
| node_actuator_aware | 600 | 5.0 | 0.000906411 | 0.0104326 | 0.00243768 | 0.0278688 | 0 | 0 | 0.048007 / 0.0506684 | pass | False |
| node_quadrature | 120 | 10.0 | 0.00224809 | 0.0260125 | 0.00645063 | 0.0742314 | 0.015625 | 0.00150684 | 0.0401423 / 0.0430455 | fail | False |
| node_quadrature | 120 | 20.0 | 0.00219884 | 0.0253333 | 0.00625804 | 0.0730568 | 0.015625 | 0.00150684 | 0.0391333 / 0.0409594 | fail | False |
| node_quadrature | 120 | 40.0 | 0.00242458 | 0.0294527 | 0.0065888 | 0.0808659 | 0 | 0.00148329 | 0.0414366 / 0.0446689 | pass | False |
| node_quadrature | 120 | 5.0 | 0.00226861 | 0.0262725 | 0.0064098 | 0.0745535 | 0.015625 | 0.00150684 | 0.0467497 / 0.0494813 | fail | False |
| node_quadrature | 30 | 10.0 | 0.00402581 | 0.0449859 | 0.0110611 | 0.123167 | 0.046875 | 0.00166346 | 0.0404391 / 0.0424332 | fail | False |
| node_quadrature | 30 | 20.0 | 0.00405397 | 0.0452323 | 0.011264 | 0.124003 | 0.046875 | 0.00166346 | 0.0396517 / 0.0422342 | fail | False |
| node_quadrature | 30 | 40.0 | 0.00435187 | 0.0499696 | 0.0118576 | 0.134892 | 0.03125 | 0.00163663 | 0.0424905 / 0.0445457 | fail | False |
| node_quadrature | 30 | 5.0 | 0.0040226 | 0.0449818 | 0.0110977 | 0.124833 | 0.046875 | 0.00166346 | 0.0476045 / 0.051773 | fail | False |
| node_quadrature | 600 | 10.0 | 0.00155358 | 0.018041 | 0.00416373 | 0.0489119 | 0 | 0.000664884 | 0.0396684 / 0.0426626 | pass | False |
| node_quadrature | 600 | 20.0 | 0.00133146 | 0.0152157 | 0.00367248 | 0.0416577 | 0 | 0.000433297 | 0.0388443 / 0.0427315 | pass | False |
| node_quadrature | 600 | 40.0 | 0.00112671 | 0.0143412 | 0.00319087 | 0.0397865 | 0 | 0.00203946 | 0.0422281 / 0.0479726 | pass | False |
| node_quadrature | 600 | 5.0 | 0.00161735 | 0.0187999 | 0.00436659 | 0.050766 | 0 | 0.000346582 | 0.0466023 / 0.0489554 | pass | False |
| node_structured | 120 | 10.0 | 0.00159282 | 0.018229 | 0.00464216 | 0.05372 | 0.015625 | 0.000116821 | 0.0754564 / 0.0812169 | fail | False |
| node_structured | 120 | 20.0 | 0.00159281 | 0.0182289 | 0.00464165 | 0.0537195 | 0.015625 | 0.000116821 | 0.073971 / 0.0773132 | fail | False |
| node_structured | 120 | 40.0 | 0.00159271 | 0.0182286 | 0.00463602 | 0.053711 | 0.015625 | 0.000116821 | 0.0808508 / 0.0862339 | fail | False |
| node_structured | 120 | 5.0 | 0.00159282 | 0.018229 | 0.00464219 | 0.05372 | 0.015625 | 0.000116821 | 0.0863648 / 0.0917603 | fail | False |
| node_structured | 30 | 10.0 | 0.00410442 | 0.0453423 | 0.0122569 | 0.134437 | 0.046875 | 0.00159725 | 0.0749902 / 0.0791024 | fail | False |
| node_structured | 30 | 20.0 | 0.0041044 | 0.0453427 | 0.012258 | 0.134438 | 0.046875 | 0.00159725 | 0.0736759 / 0.0772938 | fail | False |
| node_structured | 30 | 40.0 | 0.00410408 | 0.0453499 | 0.0122561 | 0.134534 | 0.046875 | 0.00159725 | 0.0810092 / 0.0882485 | fail | False |
| node_structured | 30 | 5.0 | 0.00410442 | 0.0453423 | 0.0122568 | 0.134437 | 0.046875 | 0.00159725 | 0.0867432 / 0.091645 | fail | False |
| node_structured | 600 | 10.0 | 0.000372907 | 0.00429969 | 0.000946816 | 0.011113 | 0 | 0 | 0.0747841 / 0.0790339 | pass | False |
| node_structured | 600 | 20.0 | 0.000372943 | 0.00429995 | 0.000946964 | 0.0111142 | 0 | 0 | 0.0732227 / 0.0778493 | pass | False |
| node_structured | 600 | 40.0 | 0.000373545 | 0.00430391 | 0.00096129 | 0.0111256 | 0 | 0 | 0.0806417 / 0.0879627 | pass | False |
| node_structured | 600 | 5.0 | 0.000372905 | 0.00429967 | 0.000946802 | 0.0111129 | 0 | 0 | 0.0862222 / 0.0914205 | pass | False |
| physics_fit | 120 | 10.0 | 1.5507e-08 | 1.79432e-07 | 4.18092e-08 | 4.85561e-07 | 0 | 0 | 0.11122 / 0.114442 | pass | False |
| physics_fit | 120 | 20.0 | 1.5507e-08 | 1.79432e-07 | 4.18092e-08 | 4.85561e-07 | 0 | 0 | 0.154745 / 0.161701 | pass | False |
| physics_fit | 120 | 40.0 | 1.5507e-08 | 1.79432e-07 | 4.18092e-08 | 4.85561e-07 | 0 | 0 | 0.248779 / 0.255725 | pass | False |
| physics_fit | 120 | 5.0 | 1.5507e-08 | 1.79432e-07 | 4.18092e-08 | 4.85561e-07 | 0 | 0 | 0.0903411 / 0.0952734 | pass | False |
| physics_fit | 30 | 10.0 | 1.09423e-07 | 1.26109e-06 | 2.89157e-07 | 3.33453e-06 | 0 | 0 | 0.110258 / 0.115768 | pass | False |
| physics_fit | 30 | 20.0 | 1.09423e-07 | 1.26109e-06 | 2.89157e-07 | 3.33453e-06 | 0 | 0 | 0.153792 / 0.157512 | pass | False |
| physics_fit | 30 | 40.0 | 1.09423e-07 | 1.26109e-06 | 2.89157e-07 | 3.33453e-06 | 0 | 0 | 0.248827 / 0.258477 | pass | False |
| physics_fit | 30 | 5.0 | 1.09423e-07 | 1.26109e-06 | 2.89157e-07 | 3.33453e-06 | 0 | 0 | 0.0905904 / 0.0954987 | pass | False |
| physics_fit | 600 | 10.0 | 1.93613e-08 | 2.24284e-07 | 4.94781e-08 | 5.70606e-07 | 0 | 0 | 0.111019 / 0.116041 | pass | False |
| physics_fit | 600 | 20.0 | 1.93613e-08 | 2.24284e-07 | 4.94781e-08 | 5.70606e-07 | 0 | 0 | 0.15476 / 0.159197 | pass | False |
| physics_fit | 600 | 40.0 | 1.93613e-08 | 2.24284e-07 | 4.94781e-08 | 5.70606e-07 | 0 | 0 | 0.249956 / 0.259255 | pass | False |
| physics_fit | 600 | 5.0 | 1.93613e-08 | 2.24284e-07 | 4.94781e-08 | 5.70606e-07 | 0 | 0 | 0.0896463 / 0.0931613 | pass | False |
| physics_nominal | 120 | 10.0 | 0.0186585 | 0.214954 | 0.049736 | 0.568304 | 0.296875 | 0.00563714 | 0.111095 / 0.114042 | fail | False |
| physics_nominal | 120 | 20.0 | 0.0186585 | 0.214954 | 0.049736 | 0.568304 | 0.296875 | 0.00563714 | 0.153198 / 0.16043 | fail | False |
| physics_nominal | 120 | 40.0 | 0.0186585 | 0.214954 | 0.049736 | 0.568304 | 0.296875 | 0.00563714 | 0.250253 / 0.257985 | fail | False |
| physics_nominal | 120 | 5.0 | 0.0186585 | 0.214954 | 0.049736 | 0.568304 | 0.296875 | 0.00563714 | 0.0905213 / 0.0950287 | fail | False |
| physics_nominal | 30 | 10.0 | 0.0186585 | 0.214954 | 0.049736 | 0.568304 | 0.296875 | 0.00563714 | 0.110701 / 0.114888 | fail | False |
| physics_nominal | 30 | 20.0 | 0.0186585 | 0.214954 | 0.049736 | 0.568304 | 0.296875 | 0.00563714 | 0.154231 / 0.160558 | fail | False |
| physics_nominal | 30 | 40.0 | 0.0186585 | 0.214954 | 0.049736 | 0.568304 | 0.296875 | 0.00563714 | 0.249515 / 0.260295 | fail | False |
| physics_nominal | 30 | 5.0 | 0.0186585 | 0.214954 | 0.049736 | 0.568304 | 0.296875 | 0.00563714 | 0.0900414 / 0.0960336 | fail | False |
| physics_nominal | 600 | 10.0 | 0.0186585 | 0.214954 | 0.049736 | 0.568304 | 0.296875 | 0.00563714 | 0.11055 / 0.116799 | fail | False |
| physics_nominal | 600 | 20.0 | 0.0186585 | 0.214954 | 0.049736 | 0.568304 | 0.296875 | 0.00563714 | 0.153942 / 0.160811 | fail | False |
| physics_nominal | 600 | 40.0 | 0.0186585 | 0.214954 | 0.049736 | 0.568304 | 0.296875 | 0.00563714 | 0.248844 / 0.25957 | fail | False |
| physics_nominal | 600 | 5.0 | 0.0186585 | 0.214954 | 0.049736 | 0.568304 | 0.296875 | 0.00563714 | 0.0902492 / 0.0962351 | fail | False |
| physics_residual | 120 | 10.0 | 0.000402755 | 0.00459035 | 0.00120893 | 0.0137494 | 0 | 0 | 0.0409243 / 0.0431703 | pass | False |
| physics_residual | 120 | 20.0 | 0.00040271 | 0.00459014 | 0.00120825 | 0.0137482 | 0 | 0 | 0.0325615 / 0.0349815 | pass | False |
| physics_residual | 120 | 40.0 | 0.000402122 | 0.00458638 | 0.00119955 | 0.0137129 | 0 | 0 | 0.0294605 / 0.0316526 | pass | False |
| physics_residual | 120 | 5.0 | 0.000402758 | 0.00459036 | 0.00120897 | 0.0137494 | 0 | 0 | 0.055877 / 0.0585806 | pass | False |
| physics_residual | 30 | 10.0 | 0.00121111 | 0.0134325 | 0.00394077 | 0.0421964 | 0 | 0.00148329 | 0.0404666 / 0.0437539 | pass | False |
| physics_residual | 30 | 20.0 | 0.00121114 | 0.0134326 | 0.00394122 | 0.0421968 | 0 | 0.00148329 | 0.032525 / 0.034249 | pass | False |
| physics_residual | 30 | 40.0 | 0.00121163 | 0.0134351 | 0.00394846 | 0.0422056 | 0 | 0.00148329 | 0.029375 / 0.0310739 | pass | False |
| physics_residual | 30 | 5.0 | 0.00121111 | 0.0134325 | 0.00394074 | 0.0421964 | 0 | 0.00148329 | 0.05575 / 0.0603022 | pass | False |
| physics_residual | 600 | 10.0 | 0.000175075 | 0.00200197 | 0.00048898 | 0.00571259 | 0 | 6.5427e-07 | 0.0408456 / 0.0441128 | pass | False |
| physics_residual | 600 | 20.0 | 0.000174947 | 0.00200202 | 0.000487883 | 0.0057132 | 0 | 6.5427e-07 | 0.033008 / 0.036233 | pass | False |
| physics_residual | 600 | 40.0 | 0.000173118 | 0.00200334 | 0.00048787 | 0.00570467 | 0 | 6.5427e-07 | 0.0298907 / 0.0329869 | pass | False |
| physics_residual | 600 | 5.0 | 0.000175083 | 0.00200197 | 0.000489055 | 0.00571256 | 0 | 6.5427e-07 | 0.055999 / 0.0583138 | pass | False |

Threshold brackets are interval-censored; no interpolated data requirement. All budgets, failures and nonmonotone curves remain visible.

| Model | Step (ms) | First pass (s) | Bracket (s) | Nonmonotone |
|---|---:|---|---|---|
| affine_flow | 5 | 30 | [0, 30] | False |
| affine_flow | 10 | 30 | [0, 30] | False |
| affine_flow | 20 | 30 | [0, 30] | False |
| affine_flow | 40 | 30 | [0, 30] | False |
| affine_rk4 | 5 | 30 | [0, 30] | False |
| affine_rk4 | 10 | 30 | [0, 30] | False |
| affine_rk4 | 20 | 30 | [0, 30] | False |
| affine_rk4 | 40 | 30 | [0, 30] | False |
| node_structured | 5 | 600 | [120, 600] | False |
| node_structured | 10 | 600 | [120, 600] | False |
| node_structured | 20 | 600 | [120, 600] | False |
| node_structured | 40 | 600 | [120, 600] | False |
| node_quadrature | 5 | 600 | [120, 600] | False |
| node_quadrature | 10 | 600 | [120, 600] | False |
| node_quadrature | 20 | 600 | [120, 600] | False |
| node_quadrature | 40 | 120 | [30, 120] | False |
| node_actuator_aware | 5 | 600 | [120, 600] | False |
| node_actuator_aware | 10 | 600 | [120, 600] | False |
| node_actuator_aware | 20 | 600 | [120, 600] | False |
| node_actuator_aware | 40 | 600 | [120, 600] | False |
| physics_fit | 5 | 30 | [0, 30] | False |
| physics_fit | 10 | 30 | [0, 30] | False |
| physics_fit | 20 | 30 | [0, 30] | False |
| physics_fit | 40 | 30 | [0, 30] | False |
| physics_residual | 5 | 30 | [0, 30] | False |
| physics_residual | 10 | 30 | [0, 30] | False |
| physics_residual | 20 | 30 | [0, 30] | False |
| physics_residual | 40 | 30 | [0, 30] | False |
| physics_nominal | 5 | None | [600, None] | False |
| physics_nominal | 10 | None | [600, None] | False |
| physics_nominal | 20 | None | [600, None] | False |
| physics_nominal | 40 | None | [600, None] | False |

Resources are separated in JSON: motion seconds proxy; fit execution including validation/selection; shared evaluation/truth replay/refinement; predictions; decision scoring; timing critical samples; audit/logging/serialization; original numeric store finalization. Producer spans exclude post-producer artifact hashing and manifest/marker/receipt/completed-event persistence. Timing measures one frozen earliest-primary trajectory-0 bank; decision records cover the complete fixed trajectory library. Network-forward diagnostics never replace complete scorer cost.

use nonoverlapping_producers for totals; inclusive producer_attempts and nested rate diagnostics overlap and must not be added; active and checkpoint persistence spans exclude inactive gaps, ledger observation writes and post-producer finalization. unknown-tail records are measured lower bounds, never complete throughput estimates; pilot projections must retain accounting_status and separately report actual compressed bytes and finalization overhead.

unique exact-receipt-bound original observational event; events.jsonl is outside immutable artifact hashes; resource-record write excluded. These observations validate exact receipt identity and schema but cannot prove the event log has never been edited. Reuse costs cannot replace original production costs.

Raw rows, decisions, numerical envelopes, all baselines, selected receipts, missing/failed/limited coordinates, session values, resource breakdown and bootstrap intervals are in short_calibration_result.json. Arrays and weights remain external in the selected ignored output store. See short_calibration_reproduce.md for exact portable acquisition and read-only verification commands.

Observed threshold budget ratios and their interval-censored ratio brackets are in JSON. Undefined ratios are not imputed.
