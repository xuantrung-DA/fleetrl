# Rà soát trước train — Windows CPU

**Kết luận: ready to train trên môi trường Windows CPU đã kiểm chứng.**

Source Python SHA-256: `f365b0b65c653b5e44a39e4e2ac2cef7bd825d2066790854861bc9ec6ae3ef31`. Hash khớp cả bốn báo cáo thực thi; danh sách từng file ở [audit-source-sha256.json](audit-source-sha256.json).

## Lỗi đã sửa

- Validation/checkpoint định kỳ chạy sau cập nhật policy; final validation khớp checksum final model. Checkpoint selection dùng toàn bộ safety incidents, kể cả tải trọng/quyền sở hữu/mất task.
- Resume giữ giai đoạn curriculum, lịch exploration và checkpoint tốt nhất trước đó; kiểm source và toàn bộ env config. Study đủ ngân sách có thể hoàn tất artifact mà không học thêm.
- Model và metadata commit cùng ZIP; replay dùng thế hệ riêng và checksum. Đã giả lập ngắt trước/sau commit, mất sidecar, sửa payload. Retry Windows tăng từ 1,55 lên 6,5 giây sau khi tái hiện khóa tệp kéo dài vài giây cả ngoài sandbox.
- Study khóa tham số/defaults và checkpoint; report kiểm hash, từ chối artifact bị sửa. Episode thiếu không thành job complete; CLI báo lỗi cho study chưa đủ nếu không có --limit.
- MPC HOLD khi robot đều bận không còn bị tính fallback. MILP không ghi termination=optimal khi nghiệm chưa qua kiểm tài nguyên; MPC kiểm tính nguyên. CI không chấp nhận bản ghi baseline trùng.

## Bằng chứng trên source cuối

| Kiểm tra | Kết quả | Báo cáo |
| --- | --- | --- |
| Regression | 160 passed, 0 failed; 17 deprecation warnings từ dependency | [XML](audit-regression.xml) |
| 14 phương án | 14/14; 11 learner train 128 → resume 144 bước, tham số đổi, save/load khớp | [Ready](audit-ready.json) |
| Windows integration | 2 spawn workers, LSTM, TensorBoard, 3 controls, 10/15/20 robot × 600 s, live/replay | [Integration](audit-integration.json) |
| Full study workflow nhỏ | tune/search/train/test/report; 2 policy; 18/18 episode; chạy tiếp không lặp job | [Pipeline](audit-pipeline.json) |
| Chạy dài cả 14 phương án | 56/56 episode: 2 case × 2 seed × 14; mỗi episode 600 s | [Extended](audit-extended.json) |
| Curriculum thực | DQN 64 bước qua 3 giai đoạn, resume 16 bước; epsilon giữ đúng lịch | [Extended](audit-extended.json) |

Trong 56 episode: không ghi nhận safety incident, mất task hoặc policy exception; cân bằng năng lượng sai số tuyệt đối < 1e-5 Wh. Các kết quả này áp dụng cho những ca đã chạy.

## Kết quả vận hành dài — chưa phải bảng xếp hạng thuật toán

Case nominal10: 10 robot, map A, 60 task/giờ. Case stress20: 20 robot, map B, SOC ban đầu 25–45%, 120 task/giờ, burst, block/pause. Seeds 2000 và 2001; các phương án dùng cùng tape trong từng case/seed. Learner mới train smoke 128 bước, episode train chỉ 30 s. Hai case dài là kiểm vận hành và OOD, không dùng chọn model.

| Phương án | Completed/arrived nominal (2 seeds) | Completed/arrived stress (2 seeds) | Safety (4 ca) | Fallback cao nhất | P95 cao nhất (ms) |
| --- | --- | --- | ---: | ---: | ---: |
| heuristic | 19/19, 13/14 | 4/34, 4/26 | 0 | 0.0% | 13.2 |
| fixed_cpsat | 12/19, 13/14 | 8/34, 7/26 | 0 | 0.0% | 115.5 |
| qlearning_cpsat | 12/19, 13/14 | 8/34, 7/26 | 0 | 0.0% | 112.2 |
| dqn_cpsat | 17/19, 13/14 | 8/34, 4/26 | 0 | 0.8% | 112.5 |
| a2c_cpsat | 18/19, 13/14 | 8/34, 10/26 | 0 | 0.0% | 114.2 |
| ppo_cpsat | 18/19, 13/14 | 8/34, 21/26 | 0 | 0.0% | 113.6 |
| qrdqn_cpsat | 13/19, 13/14 | 8/34, 4/26 | 0 | 0.8% | 110.5 |
| recurrent_ppo_cpsat | 12/19, 13/14 | 8/34, 7/26 | 0 | 0.0% | 116.7 |
| ppo_threshold_cpsat | 19/19, 13/14 | 8/34, 8/26 | 0 | 0.0% | 16.4 |
| sac_cpsat | 15/19, 8/14 | 8/34, 7/26 | 0 | 0.0% | 115.8 |
| td3_cpsat | 17/19, 9/14 | 8/34, 7/26 | 0 | 0.0% | 113.5 |
| mappo_dispatch | 11/19, 11/14 | 31/34, 26/26 | 0 | 0.0% | 17.6 |
| forecast_mpc_milp | 12/19, 13/14 | 6/34, 5/26 | 0 | 0.0% | 111.1 |
| ppo_milp | 18/19, 13/14 | 25/34, 26/26 | 0 | 0.8% | 67.5 |

Throughput thấp, tồn đọng hoặc fallback cao trong một số ca vẫn là kết quả cần phân tích khi train dài. Không suy ra rằng cả 14 phương án đều tốt, đã hội tụ, hoặc phương án hiện tại tốt nhất. Chưa chạy study chính 33 × 300.000 bước; chưa nghiệm thu CUDA/robot thật/hard real-time.

## Tái chạy

```powershell
.\.venv\Scripts\python.exe -m pytest -q --tb=short --junitxml=reports/audit-regression.xml
.\.venv\Scripts\python.exe -m fleetrl ready-check --output runs/audit_ready_verified --steps 128
.\.venv\Scripts\python.exe scripts/verify_v2.py --checkpoints runs/audit_ready_final --output runs/audit_integration_final
.\.venv\Scripts\python.exe scripts/audit_study.py --output runs/audit_pipeline_final
.\.venv\Scripts\python.exe scripts/audit_pretrain.py --checkpoints runs/audit_ready_verified --output runs/audit_extended_final
```

Đổi tên các thư mục output khi tái chạy. UI/replay trong integration dùng checkpoint 128 bước từ vòng trước sửa làm tròn curriculum; các integration train/control và 56 episode dùng source cuối. Báo cáo VERIFICATION_V2.md cùng các vòng audit trung gian được giữ làm lịch sử, không thay thế kết quả cuối này.

Hướng dẫn bắt đầu: [READY_TO_TRAIN.md](../docs/READY_TO_TRAIN.md).
