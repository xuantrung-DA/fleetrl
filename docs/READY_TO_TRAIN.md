# FleetRL v2 — hướng dẫn train và nghiệm thu

FleetRL v2 có 14 phương án trong 14 file `src/fleetrl/methods/`, dùng chung simulator, planner, candidate guards và trainer. Mọi công thức KPI, thống kê và OOD nằm trong `src/fleetrl/metrics.py`; simulator/trainer chỉ ghi dữ liệu đo. M01–M13 thuộc episode, M14 thuộc lần train, M15–M16 cần dữ liệu study. Giá trị chưa đo hoặc mẫu số bằng 0 là `null`.

## Môi trường

Python 3.12 trên Windows, CPU; SB3/Contrib 2.7.0, PyTorch 2.6.0, OR-Tools 9.14.6206 với SCIP 9.2.2. `.venv` hiện tại đã tách khỏi site-packages của Python dùng chung. CUDA chưa được nghiệm thu. Source không cần dữ liệu hoặc API bên ngoài để train.

Chạy từ thư mục gốc repository:

```powershell
# Máy hiện tại đã có .venv. Máy mới dùng hai lệnh đầu để cài.
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock-windows-py312.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m fleetrl doctor
.\.venv\Scripts\python.exe -m fleetrl methods
```

`requirements.txt` khóa dependency trực tiếp; file `requirements-lock-windows-py312.txt` khóa cả dependency gián tiếp đã kiểm trên Windows. Bản source gốc có thể được giữ cục bộ trong `.baseline/fleetrl-v1.zip`; thư mục này không đưa lên Git. Báo cáo Linux v1 trong `reports/VERIFICATION.md` là lịch sử, không xác minh source v2.

Sau khi dọn format/import, source hash thay đổi. Các báo cáo cũ chỉ xác minh hash đã ghi trong báo cáo; xem [DEVELOPMENT.md](DEVELOPMENT.md) để kiểm tra phiên bản hiện tại và lưu ý khi resume.

## Train một phương án

```powershell
.\.venv\Scripts\python.exe -m fleetrl train --config configs/methods/dqn_cpsat.yaml --seed 11 --output runs/dqn/seed11
.\.venv\Scripts\python.exe -m fleetrl eval --config configs/methods/dqn_cpsat.yaml --method dqn_cpsat --checkpoint runs/dqn/seed11/best_model.zip --scenario S4 --seed 2000 --output runs/eval_dqn_S4
```

Đổi tên config để dùng phương án khác. V01 heuristic, V02 fixed CP-SAT, V13 forecast MPC không có policy RL để train; dùng tune/eval. Q-learning và MAPPO hiện chạy một environment; SB3 hỗ trợ nhiều worker bằng Windows spawn. `--steps` là số fleet transitions yêu cầu; thuật toán theo rollout có thể vượt số bước để hoàn tất rollout, số thực được ghi rõ.

Resume vào thư mục mới, cùng config/seed:

```powershell
.\.venv\Scripts\python.exe -m fleetrl train --config configs/methods/dqn_cpsat.yaml --seed 11 --steps 100000 --resume runs/dqn/seed11/final_model.zip --output runs/dqn/seed11_resume
```

Bundle gồm `*.zip`, bản metadata đọc được `*.zip.json` và `*.<generation>.replay.pkl` cho off-policy. Giữ nguyên thư mục bundle. Metadata và checksum payload nằm ngay trong ZIP; replay mới dùng tên riêng, sau đó ZIP được thay nguyên tử. Nếu bị ngắt trước khi cập nhật sidecar, loader dùng metadata trong ZIP để khôi phục đúng thế hệ. Không xóa file replay mà metadata đang tham chiếu. Windows có retry ghi file tối đa khoảng 6,5 giây cho khóa tạm thời; lỗi kéo dài vẫn được báo.

Model, optimizer, bộ đếm, replay hoặc Q-table/exploration được khôi phục; simulator và rollout đang dở bắt đầu episode mới. Resume phải giữ nguyên source, cấu hình môi trường và các tham số học; chỉ được đổi ngân sách bổ sung, tần suất ghi/validation, device, số thread và TensorBoard. Curriculum tiếp tục theo mốc tuyệt đối đã lưu; checkpoint tốt nhất từ lần train trước được giữ nếu policy mới kém hơn. Khi kéo dài ngân sách, exploration giữ lịch cũ. Dùng checkpoint tự tạo hoặc nguồn tin cậy.

Validation và checkpoint định kỳ chạy ở biên rollout sau cập nhật; lần validation cuối ghi checksum tham số để đối chiếu với final model. Ưu tiên checkpoint theo toàn bộ safety incidents, rồi completed, lateness và pending; nếu bằng nhau, chọn policy được đánh giá sau.

## Chạy study 14 phương án

```powershell
.\.venv\Scripts\python.exe -m fleetrl study plan --config configs/study.yaml --output runs/study14
.\.venv\Scripts\python.exe -m fleetrl study tune --config configs/study.yaml --output runs/study14
.\.venv\Scripts\python.exe -m fleetrl study search --config configs/study.yaml --output runs/study14
.\.venv\Scripts\python.exe -m fleetrl study train --config configs/study.yaml --output runs/study14
.\.venv\Scripts\python.exe -m fleetrl study test --config configs/study.yaml --output runs/study14
.\.venv\Scripts\python.exe -m fleetrl study report --config configs/study.yaml --output runs/study14
```

Hoặc chạy `scripts/train_study.ps1`. Mặc định job tuần tự. Có thể thêm `--limit 1` để chạy thử một job; chạy lại lệnh sẽ kiểm hash và tiếp tục job còn thiếu. Sau khi đổi source/config/protocol cần output study mới. Không gộp số liệu từ source khác vào cùng bảng.

Khi bắt đầu train, study khóa lựa chọn tham số học (kể cả lựa chọn dùng mặc định); khi test, study khóa baseline và checkpoint. Không chạy lại search/tune sau khi lựa chọn tương ứng đã khóa. Script tự bỏ qua các pha đã khóa khi chạy lại. Job thiếu hoặc trả episode chưa hoàn tất không được báo thành công; lệnh không có `--limit` trả exit code khác 0 nếu kết quả còn thiếu. Job đã hoàn thành phải qua kiểm hash cả khi xuất report. Nếu lần train bị ngắt sau khi đủ bước, study có thể hoàn tất artifact mà không học thêm một rollout.

Ngân sách chính: **33 lần train × 300.000 = 9.900.000 bước**, **1.620 ca test chính**, thêm **360 ca OOD có đối chứng**. Tuning, validation, 33 trial tìm tham số × 10.000 bước và các controls nằm ngoài tổng train chính. Không suy thời gian toàn study chỉ từ tốc độ PPO.

Train tape seeds 0–199; validation 1000–1002; test 2000–2004. Search cấp cùng số trial/budget: learning rate cho neural learners, alpha cho Q-learning; PPO threshold được train lại trong mỗi trial với ngưỡng tương ứng. Có thể bỏ search và train cấu hình mặc định đã khai báo, nhưng artifact sẽ ghi `predeclared_default`, không gọi đó là kết quả đã tune.

Report xuất CSV/JSON/Markdown, bảng 14 phương án, dictionary M01–M16, learning curves và biểu đồ trade-off. Job thất bại/thiếu vẫn nằm trong expected grid. CI bị giữ lại nếu thiếu cả training seed hoặc tape; không biến crash thành safety=0. OOD_A/OOD_B dùng cùng nhiệm vụ, deadline, thời điểm và khối lượng; S9 vẫn là thay đổi nhiều yếu tố đồng thời.

`configs/study_confirmatory.yaml` dùng checkpoint đã đóng băng từ `runs/study14`, test lại bằng seeds 3000–3004; không train lại. Các controls riêng: `configs/controls/ppo_compact.yaml`, `dqn_compact.yaml`, `ppo_continuous.yaml`. Chúng không làm tăng danh sách 14 phương án chính.

## Chi tiết cần biết khi bảo vệ

- V03 khác biểu diễn trạng thái; V10/V11 khác action space; V12 khác mô hình điều phối; V14 khác backend. So sánh không chỉ thay mỗi thuật toán học. Dùng controls để tách ảnh hưởng.
- MAPPO là actor dùng chung cho từng robot, critic tập trung; masks loại hành động không hợp lệ. Arbiter ưu tiên SOC rồi ID khi tranh chấp. PPO cập nhật xác suất hành động đã lấy mẫu, kể cả khi arbiter từ chối.
- MPC dùng EWMA từ lịch sử đã quan sát, tối ưu chuỗi tối đa **hai thao tác** trong horizon 300 s, chỉ commit thao tác đầu. Đây là MPC xấp xỉ có giới hạn số thao tác và thời gian dựng route; không có future-task oracle.
- MILP dùng start binaries trên cùng lưới 5 s/candidate/objective với CP-SAT. Ràng buộc tài nguyên được thêm khi phát hiện vi phạm; chỉ trả nghiệm sau khi kiểm toàn bộ port/queue và tính nguyên. Cách này tránh dựng hàng triệu hệ số ngay từ đầu.
- Deadline chung tính preparation/inference/model/solver/commit. Simulator vẫn phải kiểm và commit an toàn; quá ngân sách được ghi nhận, không có bảo đảm thời gian thực cứng. Thời gian `advance` được báo riêng.
- FEASIBLE hoặc UNKNOWN không tự đồng nghĩa timeout. Khi solver không trả nguyên nhân dừng đáng tin, timeout là `null`; raw status và số solver invocations vẫn được ghi.
- `training_updates` giữ bộ đếm của từng learner: PPO/Recurrent PPO dùng `_n_updates` theo epoch của SB3 (có thể bao gồm epoch cấu hình khi early-stop), A2C/DQN/QR-DQN/SAC/TD3 dùng counter của implementation, MAPPO dùng minibatch optimizer, Q-learning dùng TD update. Không xem các đơn vị này là số gradient step tương đương; so chi phí bằng fleet steps, wall time và RAM. Ready-check còn kiểm tham số thực sự thay đổi.
- Deadlock chỉ xác nhận hồi phục sau 30 s tiến triển liên tục hoặc hoàn thành thao tác; có đo riêng thời điểm tiến triển đầu tiên và ca chưa hồi phục.

## Kiểm tra nghiệm thu

Báo cáo rà soát mới nhất: [AUDIT_PRETRAIN.md](../reports/AUDIT_PRETRAIN.md), gồm 160 kiểm thử, 14 phương án train/eval, 56 episode dài và kiểm toàn bộ study workflow. Báo cáo ghi rõ source hash; các báo cáo v2 trước đó là lịch sử.

```powershell
.\.venv\Scripts\python.exe -m pytest -q --junitxml=reports/current-regression.xml
.\.venv\Scripts\python.exe -m fleetrl ready-check --output runs/ready_check --steps 32
.\.venv\Scripts\python.exe -m fleetrl study test --config configs/study_smoke.yaml --output runs/study_smoke
```

`ready-check` thực sự cập nhật tham số, kiểm finite, lưu/nạp, so action sau nạp, resume thêm update và eval cả 11 learner; ba baseline chạy controller/eval. Smoke chỉ xác nhận khả năng thực thi. Việc train dài, kiểm hội tụ và kết luận thuật toán nào tốt nhất là giai đoạn thí nghiệm sau bàn giao, không thể suy ra từ các checkpoint smoke.

Đối chứng dùng cùng PPO checkpoint trên backend khác: thêm `--frozen-backend-control` vào lệnh eval, chọn `--method ppo_milp` cho checkpoint PPO–CP-SAT hoặc ngược lại. Kết quả được gắn `comparison_role=frozen_backend_control`, không dùng làm run V06/V14 đã train độc lập.

GUI và replay:

```powershell
.\.venv\Scripts\python.exe -m fleetrl gui --method recurrent_ppo_cpsat --checkpoint runs/recurrent/seed11/best_model.zip
.\.venv\Scripts\python.exe -m fleetrl replay runs/eval_dqn_S4
```

Session LSTM được giữ trong từng episode và reset khi đổi episode/GUI reset. Hướng dẫn API v2 ở file này thay thế những lệnh dự kiến trong plan và phần PPO-only của tài liệu v1.
