# Algorithm comparison

Aggregates below are descriptive. Use per-scenario paired CIs for conclusions; action/representation/backend changes are distinct comparisons.

| ID | Method | Comparison | Complete / expected | Throughput/h | Lateness (s) | Safety | Training steps | Peak RAM (bytes) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V01 | heuristic | baseline | 55/55 | 104.691 | 14574.765 | 0.000 | — | — |
| V02 | fixed_cpsat | baseline | 55/55 | 103.327 | 19868.957 | 0.018 | — | — |
| V03 | qlearning_cpsat | representation | 165/165 | 102.873 | 22325.530 | 0.006 | 900000.000 | 839127040.000 |
| V04 | dqn_cpsat | learning | 165/165 | 103.679 | 20486.635 | 0.000 | 900000.000 | 1792499712.000 |
| V05 | a2c_cpsat | learning | 165/165 | 103.467 | 20519.001 | 0.018 | 900096.000 | 1648340992.000 |
| V06 | ppo_cpsat | learning | 165/165 | 102.703 | 21652.706 | 0.048 | 903168.000 | 1690939392.000 |
| V07 | qrdqn_cpsat | learning | 165/165 | 103.158 | 20648.552 | 0.012 | 900000.000 | 2069028864.000 |
| V08 | recurrent_ppo_cpsat | memory | 165/165 | 103.630 | 21103.058 | 0.018 | 900096.000 | 3927343104.000 |
| V09 | ppo_threshold_cpsat | charging | 165/165 | 101.933 | 22904.991 | 0.073 | 903168.000 | 1691959296.000 |
| V10 | sac_cpsat | continuous_action | 165/165 | 102.382 | 20860.137 | 0.030 | 900000.000 | 1815400448.000 |
| V11 | td3_cpsat | continuous_action | 165/165 | 103.248 | 21085.679 | 0.012 | 900000.000 | 1804718080.000 |
| V12 | mappo_dispatch | dispatch | 165/165 | 104.285 | 16099.208 | 0.012 | 900000.000 | 1880690688.000 |
| V13 | forecast_mpc_milp | system | 55/55 | 104.000 | 20477.319 | 0.000 | — | — |
| V14 | ppo_milp | backend | 165/165 | 103.297 | 20772.129 | 0.042 | 903168.000 | 1756397568.000 |
