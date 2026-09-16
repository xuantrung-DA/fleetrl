# Nghiệm thu FleetRL v2 — READY TO TRAIN

Ngày kiểm chứng: **11/09/2026**, Windows 11, Python **3.12.0**, PyTorch **2.6.0 CPU**, Stable-Baselines3/SB3-Contrib **2.7.0**, OR-Tools **9.14.6206**, SCIP **9.2.2**. `.venv` tách khỏi Python dùng chung; `pip check` trả **No broken requirements found**.

SHA-256 của source Python được test:

```text
09b0868080cb0f2bfb4d326a5f250a9c7c9da939a0b25452e3001616b60f7033
```

Hash khớp báo cáo ready-check và integration. Hash từng file giao nằm trong `verified_source_sha256_v2.json`. Báo cáo v1 được giữ nguyên làm lịch sử; các lần chạy trung gian trong `runs/acceptance_*` không thay thế bằng chứng cuối ở dưới.

## Kết quả kiểm chứng

| Kiểm tra | Kết quả | Bằng chứng |
| --- | --- | --- |
| Unit/regression, simulator, solver, metrics, registry, seed split, study resume | **133 passed**, 0 failed | [current-regression.xml](current-regression.xml) |
| Đường thực thi 14 phương án | **14/14 passed** | [ready-report-v2.json](ready-report-v2.json) |
| 11 learner: tham số đổi thật, finite, save/load, action sau load, resume, eval | **11/11 passed**; train 32 bước, resume tổng 48 | [ready-report-v2.json](ready-report-v2.json) |
| Hai worker Windows spawn + Recurrent PPO + TensorBoard | Passed | [integration-report-v2.json](integration-report-v2.json) |
| PPO compact, DQN compact, PPO continuous controls | Cả ba có update thật | [integration-report-v2.json](integration-report-v2.json) |
| 10/15/20 robot, mỗi ca 600 giây mô phỏng, có disturbances | Cả ba không ghi nhận safety incident; không mất task; cân bằng năng lượng sai số < 1e-5 Wh | [integration-report-v2.json](integration-report-v2.json) |
| GUI/replay headless, checksum artifact | Passed | [live-v2.png](live-v2.png), [replay-v2.png](replay-v2.png) |
| Study smoke + chạy tiếp sau limit | Từ 2/18 partial lên **18/18 complete**, 0 failed; chạy lại không tạo attempt mới | [study-smoke-v2.json](study-smoke-v2.json) |

Test bao gồm đối chiếu optimum CP-SAT/MILP trên bốn cấu hình nhỏ, lưới start và đuôi lịch sạc; terminal target của Q-learning; recurrent session reset; action liên tục và deadline chung; MAPPO mask/tranh chấp; MPC chọn chuỗi hai thao tác và forecast nhân quả; censoring/null, audit trạng thái, sạc một phần tick; missing-whole-replica CI và kiểm hash khi resume. MAPPO đã được đối chiếu riêng: cả actor và critic thay đổi tham số sau train.

17 warnings trong pytest là deprecation từ dependency, không có failed/skip test. Các checkpoint trong ready-check là model smoke để kiểm thực thi, không dùng kết luận chất lượng.

## Mười bốn phương án đã có

| ID | File trong `src/fleetrl/methods/` | Train / resume thực |
| --- | --- | --- |
| V01 | `heuristic.py` | Controller không RL, eval đạt |
| V02 | `fixed_cpsat.py` | Controller không RL, eval đạt |
| V03 | `qlearning_cpsat.py` | 32 → 48 fleet steps |
| V04 | `dqn_cpsat.py` | 32 → 48 fleet steps |
| V05 | `a2c_cpsat.py` | 32 → 48 fleet steps |
| V06 | `ppo_cpsat.py` | 32 → 48 fleet steps |
| V07 | `qrdqn_cpsat.py` | 32 → 48 fleet steps |
| V08 | `recurrent_ppo_cpsat.py` | 32 → 48 fleet steps |
| V09 | `ppo_threshold_cpsat.py` | 32 → 48 fleet steps |
| V10 | `sac_cpsat.py` | 32 → 48 fleet steps |
| V11 | `td3_cpsat.py` | 32 → 48 fleet steps |
| V12 | `mappo_dispatch.py` | 32 → 48 fleet steps |
| V13 | `forecast_mpc_milp.py` | Controller không RL, eval đạt |
| V14 | `ppo_milp.py` | 32 → 48 fleet steps |

14 file phương án dùng chung infrastructure; không nhân bản simulator/trainer thành 14 bản. M01–M16 có dictionary và công thức trong [metrics.py](../src/fleetrl/metrics.py), [metrics_dictionary.md](../docs/metrics_dictionary.md). M14 thuộc training manifest, M15/M16 thuộc study summary; không tạo số liệu giả cho một episode riêng lẻ.

## Lệnh đã chạy trên bản cuối

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q --tb=short --junitxml=reports/current-regression.xml
.\.venv\Scripts\python.exe -m fleetrl ready-check --output runs/ready_v2_verified --steps 32
.\.venv\Scripts\python.exe scripts/verify_v2.py --checkpoints runs/ready_v2_final --output runs/integration_v2_final
.\.venv\Scripts\python.exe -m fleetrl study test --config configs/study_smoke.yaml --output runs/study_v2_final --limit 2
.\.venv\Scripts\python.exe -m fleetrl study test --config configs/study_smoke.yaml --output runs/study_v2_final
.\.venv\Scripts\python.exe -m fleetrl study test --config configs/study_smoke.yaml --output runs/study_v2_final --limit 0
```

Integration dùng checkpoint Recurrent PPO đã đạt ở vòng trước; hash source của integration là hash bản cuối nêu trên. Vòng đầy đủ cuối có toàn bộ checkpoint/bundle và raw replay ở `runs/ready_v2_verified`; study smoke có bảng CSV/Markdown/plots ở `runs/study_v2_final`.

## Phạm vi bàn giao

**Đạt ready to train trên Windows CPU. Không cần nghiệm thu trung gian để code tiếp.** Có config cho từng phương án, tuning/search theo validation, train/resume, eval, study có expected grid, báo cáo và controls. Có xử lý khóa file tạm thời khi thay checkpoint trên Windows và kiểm nội dung checkpoint/replay khi nạp.

Chưa chạy study train dài 33 × 300.000 bước và 1.620 ca test chính; chưa có kết luận hội tụ, thứ hạng, cải thiện 5% hay bảo đảm điểm đồ án. CUDA, ROS/robot thật và cam kết hard real-time chưa được nghiệm thu. MPC giới hạn hai thao tác trong horizon; Q-learning/MAPPO hiện một environment. Bộ đếm update của từng thư viện có đơn vị khác nhau, được giải thích trong hướng dẫn; không so trực tiếp như số gradient step đồng nhất.

Hướng dẫn bắt đầu train: [READY_TO_TRAIN.md](../docs/READY_TO_TRAIN.md).
