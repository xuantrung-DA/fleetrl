# Nghiệm thu sửa hồi quy tồn đơn FleetRL

## Phạm vi và kết luận

Kết quả gate: **PASS**. S2/S4/S6 × 5 seed validation 1000–1004 × 3 method, horizon 1200 s. Ngưỡng đã khai báo trước: pending mean tăng ≤0.5 đơn, completion mean giảm ≤1 điểm phần trăm, safety không tăng ở từng cặp, lost_tasks=0. Không nới ngưỡng sau khi đo.

Baseline là commit 95fdcc414b8afcab0894867043c5d3aad5319cc1; parking_fix là bản đỗ ổn định trước sửa hồi quy. conservative_handoff là source cuối. Các cặp dùng cùng tape, robot ban đầu, config và checkpoint. Đối chứng causal dùng deterministic work 0.03, wall guard 4 s/decision 5 s; đối chứng vận hành dùng solver 0.1 s/decision 0.25 s. Một worker CP-SAT. Không train/finetune; MAPPO dùng checkpoint cũ. S4 chỉ quan sát 900–1200 s của burst 900–1500 s.

## Nguyên nhân và ablation trước khi sửa

Đọc [DIAGNOSIS_DISPATCH.md](<DIAGNOSIS_DISPATCH.md>) (360 episode trước production edit) và [DIAGNOSIS_CHARGE_HANDOFF.md](<DIAGNOSIS_CHARGE_HANDOFF.md>) (trước sửa handoff sạc). Cả hai giữ số liệu và file:dòng của source ở thời điểm chẩn đoán; source snapshots không bị thay.

- **a — xác nhận:** candidate/commit/MAPPO mask loại robot đi đỗ dù task-free; readiness lại đếm nó như worker. Cần thống nhất eligibility và tránh đếm hai lần.
- **b — không xác nhận là nguyên nhân chính:** S4/fixed pickup distance 27.15→26.59 m dù tồn tăng.
- **c — xác nhận lệch sạc qua placement/calendar:** S4/fixed sessions 7.6→8.8, SoC bắt đầu 31.16%→40.17%; seed1001 calendar rejection robot-task-cycle 923→3473. Charge/task cost không đổi. Bỏ correction làm pending 7.6→8.2. S6/heur1001 chứng minh parking chặn sạc cần thiết làm đảo thứ tự booking.
- **d — tác động gián tiếp nhỏ:** bỏ residence S4/fixed pending 7.6→7.2; guard chỉ lọc reposition, không lọc task.
- **e — hồi quy mean chỉ tái hiện ở S4/fixed:** 6.6±1.95→7.6±2.97 đơn; completion 82.77%→80.28%. S2/S6 và hai method khác không có cùng hồi quy mean. Năm seed không chứng minh mọi workload.
- **f — tương tác availability, reservation và calendar:** bỏ goal reservation có thể khôi phục pending nhưng đề xuất bay đã giữ; vì vậy giữ guard vật lý và sửa handoff.

S4/fixed ablation (5 seed): no_cost_correction 8.2 (+0.6); no_idle_return 8.6 (+1.0); no_residence 7.2 (-0.4); no_benefit_guard 6.2 (-1.4); no_goal_reservation 6.6 (-1.0); no_recovery_stability 7.6 (0). Delta so parking_fix 7.6. Không cộng các delta thành phân rã độc lập. Toàn bộ nhóm và SD trong aggregate.csv, paired delta trong ablation_vs_parking_fix.csv.

## Thay đổi cuối theo file

- parking.py: eligibility chung; chỉ ngắt parking task-free, không tải/booking/emergency.
- optimizer.py:249/346/483: task và sạc bắt buộc thay goal parking; không thêm sạc elective cho robot khỏe; dự đoán từ đích cạnh đang đi, cộng thời gian/auxiliary còn lại; giữ charge/task cost. Heuristic giữ idle khả thi trước khi dùng parking bổ sung.
- simulator.py:270/324/383: commit handoff qua cùng guard, giữ cạnh/reservation vật lý; tính năng lượng/lịch từ đích cạnh.
- methods/mappo_dispatch.py:36: active mask dùng cùng eligibility, giữ shape.
- optimization/milp.py:174/193 và CP-SAT optimizer.py:955/976: không đếm lại parking đã có candidate vào work/zone availability.
- tests: thêm next-cycle handoff cho cả 3 controller, immutable edge, energy guard, mandatory charge, no elective charge và bảo toàn idle assignment. Giữ nguyên assertions của test cũ/F05/no-demand.
- ABOUT.md đồng bộ hành vi. study_cleanup.py/study_status.py chỉ format để đạt gate Ruff của repo.

Bản task_handoff_v2 giải quyết S4 nhưng làm heuristic S6/1001 tồn 20→22 qua thứ tự sạc; mandatory charge handoff khắc phục. Thử ưu tiên pickup ETA vẫn khiến S2 completion giảm quá 1 pp; bản cuối giữ lựa chọn idle đang khả thi. Các trial được giữ trong run để kiểm tra, không chọn seed thuận lợi.

## Tồn đơn và completion — mean ± SD mẫu

Completion trong bảng là %. Mỗi ô có 5 seed.

| Nhóm | baseline: pending | baseline: completion_rate | parking_fix: pending | parking_fix: completion_rate | conservative_handoff: pending | conservative_handoff: completion_rate |
|---|---:|---:|---:|---:|---:|---:|
| S2/heuristic | 5.40 ± 7.64 | 85.57 ± 20.73 | 4.80 ± 5.59 | 87.20 ± 15.07 | 4.80 ± 5.45 | 87.23 ± 14.74 |
| S2/fixed_cpsat | 5.60 ± 4.62 | 84.97 ± 12.33 | 3.80 ± 2.68 | 90.05 ± 6.96 | 4.20 ± 1.92 | 89.04 ± 5.27 |
| S2/mappo_dispatch | 2.00 ± 1.22 | 94.74 ± 3.09 | 2.00 ± 1.00 | 94.82 ± 2.53 | 1.80 ± 0.84 | 95.33 ± 2.12 |
| S4/heuristic | 11.60 ± 8.62 | 69.89 ± 22.14 | 11.40 ± 8.68 | 70.40 ± 22.29 | 11.60 ± 8.50 | 69.96 ± 21.85 |
| S4/fixed_cpsat | 6.60 ± 1.95 | 82.77 ± 5.78 | 7.60 ± 2.97 | 80.28 ± 8.11 | 6.40 ± 3.85 | 83.33 ± 9.88 |
| S4/mappo_dispatch | 2.40 ± 0.55 | 93.75 ± 1.60 | 2.20 ± 0.45 | 94.26 ± 1.42 | 2.60 ± 0.55 | 93.30 ± 1.24 |
| S6/heuristic | 7.20 ± 7.56 | 80.97 ± 20.56 | 7.20 ± 7.66 | 80.90 ± 20.75 | 7.40 ± 7.47 | 80.46 ± 20.32 |
| S6/fixed_cpsat | 12.80 ± 5.02 | 67.27 ± 11.08 | 7.60 ± 3.05 | 80.09 ± 9.03 | 7.40 ± 3.51 | 80.40 ± 11.00 |
| S6/mappo_dispatch | 2.40 ± 0.55 | 93.82 ± 1.25 | 2.40 ± 0.55 | 93.75 ± 1.60 | 2.20 ± 0.84 | 94.19 ± 2.35 |

### Đối chứng ngân sách mặc định

| Nhóm | baseline: pending | baseline: completion_rate | conservative_handoff: pending | conservative_handoff: completion_rate |
|---|---:|---:|---:|---:|
| S2/heuristic | 5.40 ± 7.64 | 85.57 ± 20.73 | 4.80 ± 5.45 | 87.23 ± 14.74 |
| S2/fixed_cpsat | 6.00 ± 4.18 | 84.71 ± 10.60 | 4.00 ± 1.87 | 89.75 ± 4.63 |
| S2/mappo_dispatch | 2.00 ± 1.22 | 94.74 ± 3.09 | 1.80 ± 0.84 | 95.33 ± 2.12 |
| S4/heuristic | 11.60 ± 8.62 | 69.89 ± 22.14 | 11.60 ± 8.50 | 69.96 ± 21.85 |
| S4/fixed_cpsat | 5.80 ± 4.60 | 84.99 ± 11.78 | 6.00 ± 2.92 | 84.67 ± 6.92 |
| S4/mappo_dispatch | 2.40 ± 0.55 | 93.75 ± 1.60 | 2.60 ± 0.55 | 93.30 ± 1.24 |
| S6/heuristic | 7.20 ± 7.56 | 80.97 ± 20.56 | 7.40 ± 7.47 | 80.46 ± 20.32 |
| S6/fixed_cpsat | 9.60 ± 4.28 | 75.55 ± 9.09 | 9.00 ± 3.08 | 76.09 ± 10.00 |
| S6/mappo_dispatch | 2.40 ± 0.55 | 93.82 ± 1.25 | 2.20 ± 0.84 | 94.19 ± 2.35 |

## Năng lượng và quãng đường rỗng

consumed_wh là năng lượng robot tiêu thụ (Wh); empty_distance_ratio là % quãng đường rỗng/tổng. Không đồng nhất consumed với grid Wh. Không quy chênh năng lượng study14 cho lỗi này.

| Nhóm | baseline: consumed_wh | baseline: empty_distance_ratio | parking_fix: consumed_wh | parking_fix: empty_distance_ratio | conservative_handoff: consumed_wh | conservative_handoff: empty_distance_ratio |
|---|---:|---:|---:|---:|---:|---:|
| S2/heuristic | 177.71 ± 33.84 | 48.89 ± 6.77 | 182.66 ± 33.39 | 51.29 ± 5.52 | 183.72 ± 33.65 | 51.68 ± 6.16 |
| S2/fixed_cpsat | 372.95 ± 26.50 | 85.24 ± 2.09 | 304.86 ± 24.08 | 78.13 ± 0.89 | 299.11 ± 13.44 | 77.65 ± 0.89 |
| S2/mappo_dispatch | 296.97 ± 18.58 | 76.10 ± 0.74 | 262.09 ± 16.79 | 70.59 ± 2.04 | 259.28 ± 15.02 | 71.02 ± 2.17 |
| S4/heuristic | 156.62 ± 31.85 | 52.84 ± 7.86 | 163.69 ± 32.49 | 55.30 ± 6.25 | 161.69 ± 31.11 | 54.94 ± 5.95 |
| S4/fixed_cpsat | 375.06 ± 44.23 | 85.73 ± 0.74 | 291.26 ± 27.67 | 79.09 ± 3.55 | 291.34 ± 15.33 | 78.07 ± 3.57 |
| S4/mappo_dispatch | 303.29 ± 15.78 | 75.21 ± 2.84 | 269.73 ± 21.35 | 70.89 ± 1.34 | 262.13 ± 28.32 | 69.84 ± 1.98 |
| S6/heuristic | 169.25 ± 37.96 | 47.28 ± 5.40 | 175.36 ± 38.38 | 50.08 ± 4.29 | 175.06 ± 38.30 | 49.89 ± 4.24 |
| S6/fixed_cpsat | 338.74 ± 31.57 | 87.34 ± 1.73 | 276.65 ± 36.40 | 77.32 ± 2.86 | 282.55 ± 36.35 | 77.74 ± 2.27 |
| S6/mappo_dispatch | 308.75 ± 9.19 | 77.11 ± 2.26 | 263.02 ± 15.50 | 70.78 ± 1.19 | 256.68 ± 14.10 | 69.33 ± 2.08 |

## Dispatch và sạc

| Nhóm | baseline: mean_assignment_delay_s | baseline: mean_pickup_distance_m | baseline: charge_starts | parking_fix: mean_assignment_delay_s | parking_fix: mean_pickup_distance_m | parking_fix: charge_starts | conservative_handoff: mean_assignment_delay_s | conservative_handoff: mean_pickup_distance_m | conservative_handoff: charge_starts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S2/heuristic | 36.65 ± 48.26 | 15.40 ± 3.68 | 2.40 ± 1.52 | 39.75 ± 52.99 | 15.01 ± 3.14 | 2.40 ± 1.52 | 38.33 ± 53.75 | 15.66 ± 3.85 | 2.40 ± 1.52 |
| S2/fixed_cpsat | 75.15 ± 37.98 | 25.45 ± 2.42 | 8.40 ± 0.55 | 63.18 ± 18.28 | 26.67 ± 1.09 | 8.60 ± 1.14 | 51.31 ± 28.82 | 25.84 ± 1.68 | 9.00 ± 2.00 |
| S2/mappo_dispatch | 4.38 ± 1.67 | 22.77 ± 1.68 | 0.20 ± 0.45 | 4.46 ± 1.78 | 23.48 ± 2.22 | 3.20 ± 2.59 | 4.08 ± 1.50 | 22.53 ± 1.84 | 3.00 ± 2.24 |
| S4/heuristic | 50.37 ± 62.22 | 18.02 ± 5.81 | 4.40 ± 1.14 | 50.01 ± 61.52 | 17.82 ± 5.06 | 4.40 ± 1.14 | 48.93 ± 61.52 | 17.94 ± 4.96 | 4.40 ± 1.14 |
| S4/fixed_cpsat | 80.73 ± 48.77 | 27.15 ± 2.27 | 7.60 ± 1.14 | 75.86 ± 16.27 | 26.59 ± 1.76 | 8.80 ± 0.45 | 78.84 ± 34.44 | 26.28 ± 1.49 | 9.20 ± 0.45 |
| S4/mappo_dispatch | 6.48 ± 2.65 | 23.13 ± 1.64 | 0.00 ± 0.00 | 6.95 ± 2.16 | 23.03 ± 1.07 | 2.00 ± 1.22 | 6.41 ± 1.80 | 22.63 ± 2.02 | 2.00 ± 1.22 |
| S6/heuristic | 5.31 ± 3.59 | 14.11 ± 2.07 | 2.40 ± 1.52 | 5.51 ± 3.43 | 14.10 ± 1.81 | 2.40 ± 1.52 | 4.87 ± 2.88 | 14.25 ± 1.89 | 2.40 ± 1.52 |
| S6/fixed_cpsat | 119.58 ± 46.66 | 26.49 ± 3.18 | 5.40 ± 1.14 | 136.23 ± 43.64 | 26.53 ± 1.99 | 6.80 ± 1.92 | 109.89 ± 28.99 | 25.59 ± 2.13 | 6.20 ± 1.30 |
| S6/mappo_dispatch | 5.87 ± 3.49 | 23.19 ± 0.82 | 0.00 ± 0.00 | 5.78 ± 3.69 | 24.45 ± 1.23 | 0.80 ± 0.84 | 5.38 ± 3.23 | 22.96 ± 0.97 | 0.80 ± 0.84 |

Mean delay trên chỉ gồm task đã giao; delay censored của task chưa giao được báo riêng trong aggregate.csv. Các episode có nhiệm vụ còn chờ không được bỏ khỏi phép so sánh.

tasks.json có arrival/assignment, robot state/goal/position trước commit, pickup distance, counts lúc arrival. cycles.json có quyết định wait/reposition/charge/task theo từng chu kỳ và số robot charging/committed. charging.json có proposal/start/SoC/bookings/port_usage. charge_occupancy_sessions.csv ghi thời gian chiếm cổng port_arrived→port_released; session chưa kết thúc được censor tới 1200 s. charge_occupancy_episodes.csv báo tổng occupancy, mean duration và concurrency lúc burst 900–1200 s; charge_occupancy_aggregate.csv tổng hợp mean/SD 5 seed; đối chiếu occupancy với tick counter, dung sai tối đa một tick/session. aggregate.csv còn có mean/SD SoC lúc bắt đầu sạc.

## Safety, test và tái lập

Safety kiểm từng seed, không chỉ mean. Baseline S6/heur1004 có một route_wait_bound_exceeded lúc 951 s; bản cuối không tăng safety. Lost_tasks=0 ở mọi ca cuối. acceptance.json giữ các gate từng nhóm và paired SD; paired_vs_baseline.csv giữ từng seed/tape/source hash. duplicate_checks.json giữ đối chiếu các attempt trùng từ coordinator cũ; aggregate chọn attempt hoàn thành đầu tiên, không tính thành seed mới.

Production hash: `872ddf9c1be679ff602a776a4d97d19f7ca4e5e656aa061a40711736ebb0cff0`. Checkpoint SHA256: `7783db9387c017fb24a2edf54b2fc8212746733de5c3febe5a0d8e6106ec69bb`.

Pytest cuối: 193 tests, 0 failures, 0 errors. pytest_final.xml là bằng chứng trên source cuối. quality_checks.txt ghi Ruff/checksum/diff và protected paths.

Lệnh tái lập: dùng Python conda fleetrl-env đã có trên máy (repo không có .venv), scripts/diagnose_dispatch_regression.py --source runs/dispatch_regression_20261013/sources/conservative_handoff --variant conservative_handoff --control default --output <run_mới> --checkpoint runs/study14/jobs/train_mappo_dispatch_11/attempt_002/best_model.zip. Đối chứng baseline dùng sources/baseline; đổi --control deterministic cho đối chứng causal.
