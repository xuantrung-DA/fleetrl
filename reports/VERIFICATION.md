# Báo cáo kiểm chứng source FleetRL 1.0.0

Ngày kiểm chứng: 11/09/2026. **Đã kiểm được quy trình cài → giải nén source → test → train → lưu/nạp → resume.** Báo cáo này xác nhận khả năng thực thi của mã bàn giao; chưa xác nhận hiệu quả của policy sau train dài.

## 1. Môi trường và đúng bản source được kiểm

- Cài mới virtual environment Linux x86_64, Python 3.12.14.
- Cài `torch==2.14.0` từ `https://download.pytorch.org/whl/cpu`, nhận wheel **2.14.0+cpu**, rồi cài `requirements.txt`; `pip check` không có dependency hỏng.
- Giải nén ZIP ứng viên vào thư mục riêng, cài editable package từ thư mục đó và xác nhận `fleetrl.__file__` trỏ vào source vừa giải nén. Pytest, smoke và train 2 worker bên dưới chạy bằng venv CPU trên bản đã giải nén.
- `verified_source_sha256.json` lưu SHA-256 của 30 file code/config/test/build manifest; đã đối chiếu từng byte với bản giải nén và bản bàn giao. README, ABOUT và báo cáo được hoàn tất sau kiểm thử; code không thay đổi.
- `environment-linux-py312.txt` lưu full freeze của venv CPU cuối. `smoke_report.json` có `doctor` runtime. Các ca engine 1 giờ dùng venv phát triển cùng phiên bản simulator/solver/numpy; runtime riêng nằm trong `acceptance.json`.
- Chưa kiểm Windows, macOS, RTX 4050 hoặc thực thi CUDA. Không suy tốc độ laptop từ máy kiểm chứng này.

## 2. Pytest và smoke

| Phép kiểm | Kết quả thực đo |
| --- | --- |
| Pytest trên source giải nén | **111 passed**, 0 failed, 0 errors, 0 skipped; 8.19 s |
| CLI smoke CPU | **9/9 checks passed**; 8.48 s |
| PPO smoke | 128 bước thật; tham số hữu hạn; checkpoint nạp lại và inference được |
| Resume smoke | Học thêm 64 bước, tổng 192; output mới |
| Engine smoke | 10/15/20 robot × 120 giây; không mất task, va chạm hay energy emergency |
| Paired benchmark smoke | B0/B1 cùng một event tape; tạo CSV/JSON, manifest và replay |

Lệnh tái chạy sau khi cài từ README:

```bash
python -m pip check
python -m fleetrl doctor
python -m pytest -q --junitxml=runs/junit.xml
python -m fleetrl smoke --output runs/smoke_verification --train-steps 128
```

`junit.xml` lưu từng testcase. Các test bao gồm năng lượng/tick/đoạn sạc 80%, map/tape, task và booking, reservation/đường đi, snapshot trễ, giới hạn chờ, reward integral, optimizer thật, encoder masking/permutation, PPO cả entities/MLP, phân tách seed, lưu/nạp/resume, benchmark ghép cặp và GUI/replay headless. Pygame phát một deprecation warning của `pkg_resources`; không có test thất bại. SB3 checker cảnh báo observation dạng bảng 2D là dự kiến vì dùng feature extractor tùy chỉnh; shape/dtype/mask được kiểm riêng.

## 3. Ca vận hành đủ 1 giờ, có sạc thực tế

Cả bốn ca chạy đủ **3.600 giây mô phỏng / 720 quyết định**, controller fixed profile 0. S4 có burst nhu cầu, 40% robot khởi đầu ở dải pin thấp; ca này đặt rõ 20 robot và demand cơ sở 120 đơn/giờ.

| Case | Seed | Hoàn tất / đến | Tồn cuối | Phiên sạc | Decision p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| reference_10 | 3 | 70/70 | 0 | 28 | 15.06 |
| reference_15 | 3 | 90/92 | 2 | 21 | 39.28 |
| reference_20 | 3 | 119/121 | 2 | 23 | 50.32 |
| S4_20_charging_heavy | 2000 | 130/135 | 5 | 17 | 76.24 |

Mỗi ca có **0 collisions, 0 edge conflicts, 0 energy emergencies, 0 reserve violations và 0 deadlocks_unresolved**. Các ca tham chiếu cũng có 0 booking bị hủy. Đây là kết quả của những case đã chạy, không phải bảo đảm cho mọi cấu hình hay mọi sự cố.

Commit guard vẫn từ chối một số phương án: 137/420/580 lần ở ba ca tham chiếu; chủ yếu ô đỗ đã có robot, cùng một số lệnh vào robot tạm dừng hoặc không còn đủ điều kiện pin. Đây là lệnh bị chặn trước thực thi và được lưu vào log, không bị xóa để tạo số liệu đẹp. Solver có thể hết budget mà vẫn trả incumbent khả thi; tỷ lệ timeout và fallback được giữ trong JSON.

`acceptance.json` chứa đầy đủ config, metric, hash source/tape, command tái tạo và trạng thái từng ca. Để dùng command `PYTHONPATH=src ...` trong đó trên Windows, sau khi cài editable package chỉ cần bỏ tiền tố `PYTHONPATH=src`. CP-SAT có giới hạn wall-clock nên phương án, số phiên sạc và latency có thể khác giữa các lần chạy/máy; không tuyên bố deterministic từng bit.

Decision p95 đo controller/optimizer/commit cho fixed controller, không phải tốc độ train PPO và không bao gồm thời gian advance simulator. Đây không phải kết quả phần cứng RTX 4050.

## 4. Cấu hình train chính, 2 worker

Đã chạy đúng config chính, chỉ giảm budget để nghiệm thu đường chạy:

```bash
python -m fleetrl train --config configs/train.yaml --output runs/verification_default --steps 2048 --seed 11
```

- Status **completed**; 2,048 bước thực; 2 subprocess dùng spawn; CPU; horizon 3.600 giây.
- `entities` có 294,343 tham số; PPO ghi 10 update epochs; không có tham số/optimizer tensor NaN hoặc Inf.
- Target stage bật trộn episode theo 10/15/20 robot × các hệ số demand 0,7/1/1,3. Budget nhỏ hơn 3 rollout nên ca này không chạy warmup/intermediate; logic phân stage và seed/mask được kiểm bằng pytest.
- Validation đầu/cuối chạy đầy đủ 3 seed 1000/1001/1002; không dùng test để chọn checkpoint. TensorBoard/CSV, monitor, config, manifest, best/final checkpoint được tạo.
- Wall-clock 103.76 s, gồm validation và PPO. Chỉ là run kiểm đường chạy, không dùng để hứa thời gian train 300k bước.

**Kết quả ngắn hạn chưa tốt hơn:** completed mean trên validation giảm từ 96.67 trước update xuống 94.00 sau 2.048 bước; cả hai lượt có 0 safety events theo rank. `best_model.zip` giữ checkpoint ban đầu ở step 0; `final_model.zip` vẫn lưu bản cuối để phân tích. Cơ chế chọn best hoạt động đúng và không che kết quả xấu. Chi tiết nằm trong `default_train_report.json`.

Bỏ `--steps 2048` để dùng budget 300.000 của config. Chỉ đánh giá giả thuyết RL sau train đủ budget, tune trên validation và benchmark ghép cặp nhiều training seed. Số `ppo_updates` theo SB3 là tổng update epochs, không phải 10 rollout độc lập.

## 5. Phạm vi bàn giao và phần chưa xác minh

Gói chỉ gồm source Python, cấu hình, test, README/ABOUT, tài liệu kiến trúc/đối chiếu proposal và báo cáo kiểm chứng dạng văn bản. Không kèm venv, model binary, log run lớn hay dữ liệu ngoài. Đường dẫn tuyệt đối bên trong JSON là provenance của máy kiểm chứng, không phải đường dẫn cần dùng trên máy nhận; chạy lại các command sẽ tự sinh artifact mới.

Chưa chạy train 300k × nhiều seed, protocol 360 ca, benchmark convergence hay đo mức tăng 5%. Chưa chứng minh entity encoder hơn MLP, PPO hơn fixed optimizer, hoặc mọi layout/sự cố đều giải được deadlock. Planner commit từng cạnh với reservation/corridor guard; return-to-charge check dùng khả năng còn lịch ở thời điểm lập kế hoạch, chưa phải reservation cứng cho một nhiệm vụ còn chưa hoàn tất. Những giới hạn này được đối chiếu trong `docs/proposal_traceability.md` và giải thích trong ABOUT.

README hướng dẫn debug policy yếu theo metric/reward/action/solver; ABOUT chỉ rõ module, class/function và schema để agent tiếp theo bắt đầu sửa mà không cần đoán cấu trúc source.
