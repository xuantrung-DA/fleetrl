# Báo cáo sửa ổn định điểm đỗ FleetRL — 13/10/2026

**Trạng thái: đã sửa và kiểm chứng lỗi oscillation không đơn; CHƯA NGHIỆM THU TOÀN BỘ.**
Safety không tăng nhưng một số ca validation tăng pending. Không kết luận bản sửa đáp ứng đầy đủ mục 6 của prompt.

## 1. Phạm vi và phương pháp

- Đọc AGENTS.md và ABOUT.md trước; chẩn đoán trước thay đổi source trong `DIAGNOSIS_BEFORE.md`.
- Python 3.12.14 từ Conda `fleetrl-env`; repo không có `.venv`. Không cài dependency mới.
- Headless; seed mới chỉ 1000/1001. Checkpoint MAPPO seed train 11 đã đóng băng, dùng CPU và Torch một thread.
- Mỗi comparison: S1/S2/S4 × hai seed × heuristic/fixed_cpsat/mappo_dispatch = 18 ca, 1200 s/ca. S4 giữ burst 900–1500 s, quan sát 300 s đầu burst; đây là pilot ngắn.
- Các tập so sánh dùng cùng tape, khởi tạo, config chung và checkpoint hash. Có CSV kiểm tra pairing tự động.
- Fixed dùng profile mặc định 0, heuristic dùng charge_threshold mặc định 0.30; không phải cấu hình đã tune của study14 (profile 1 / threshold 0.20). Không chuyển bảng pilot này thành kết luận study14.
- Baseline source được sao chép bằng `git show HEAD:src/...` từ HEAD `95fdcc414b8afcab0894867043c5d3aad5319cc1`, vào `baseline_source/`; hash khớp lần chạy trước sửa.

## 2. Nguyên nhân đã xác nhận

`optimizer.py` trước sửa tính wait cho 5 s nhưng reposition một ô chỉ cho 0.5 s. Với robot 0 ở (29,1), cost wait=0.00017; reposition (30,1)=0.000024. Chu kỳ goal (30,1) → (29,1) → (30,1) ở 0/5/10 s đều từ `Simulator.submit_plan`. Lỗi xuất hiện khi demand=0 và không có yield/deadlock.
MAPPO cũng nhận các parking candidates này và chọn lại sau mỗi arrival. Arrival/route clearing đúng; không có bằng chứng renderer, wait cũ hoặc idle route gây ca zero. Cam kết khi đang đi đã có sẵn.
Các giả thuyết a–h, log cụ thể và line của source trước sửa nằm trong `DIAGNOSIS_BEFORE.md`. Không quy toàn bộ chênh energy study14 cho lỗi này.

## 3. Thay đổi source

| File | Thay đổi |
|---|---|
| src/fleetrl/parking.py:10 | Guard chung từ backlog/trailing demand đã quan sát: giảm lệch lực lượng trừ travel cost; bỏ đổi chỗ không có lợi; xét residency. Không đọc tape. |
| src/fleetrl/config.py:45 | `parking_min_improvement=0.02`, `parking_min_residence_s=60`; validate finite/range. YAML đặt trong `env`. |
| src/fleetrl/optimizer.py:484 | Giữ cả vị trí và goal đang cam kết; lọc parking candidates dùng chung mọi method. |
| src/fleetrl/optimizer.py:633 | Reposition trả duration/aux ít nhất bằng một decision interval; cùng chuẩn hóa /N như wait. |
| src/fleetrl/simulator.py:500 | Robot wait ở ngoài parking, không booking, về bay trống gần nhất; log `idle_parking`. Task/charge được chọn trước; guard vật lý vẫn áp dụng. |
| src/fleetrl/simulator.py:538,959,1034 | Ghi residence khi đặt/đến parking; deadlock diversion của reposition giữ bay mới. Task recovery vẫn khôi phục task goal. |
| tests/test_optimizer.py, tests/test_simulator.py | Thêm 15 test cases; không sửa assertion/tiêu chí của các test cũ. |
| scripts/diagnose_idle_motion.py | Trace từng tick, goal source/line, decision/profile, distance theo status/cause, goal intervals/cycles; CLI bảo vệ output cũ và seed validation. |
| ABOUT.md | Đồng bộ hợp đồng parking, module mới và hai field config. |

Không sửa metrics.py, reward, observation shape/cột, method implementation, CSV/report study14 hay checkpoint. Solver một worker; không thêm randomness. Simulator vẫn cho phép explicit reposition đã qua guard vật lý; hysteresis chiến lược nằm ở candidate layer dùng chung.

## 4. Ca không đơn và đơn thưa

10 robot, 600 s, initial SoC=99%, no disturbances; `before_diagnostic_v2/` và `final_diagnostic/`.

| Ca | Controller | Đường rảnh m trước → sau | ABA goal trước → sau | Wh trước → sau |
|---|---|---:|---:|---:|
| sparse | fixed_cpsat | 853 → 114 | 530 → 0 | 63.76 → 28.20 |
| sparse | heuristic | 10 → 23 | 0 → 0 | 23.11 → 23.57 |
| sparse | mappo_dispatch | 1169 → 71 | 190 → 0 | 76.13 → 26.13 |
| zero | fixed_cpsat | 1553 → 0 | 822 → 0 | 85.15 → 20.00 |
| zero | heuristic | 0 → 0 | 0 → 0 | 20.00 → 20.00 |
| zero | mappo_dispatch | 694 → 0 | 233 → 0 | 56.99 → 20.00 |

Zero: cả ba controller không đổi vị trí/status/goal trong 600 s, 0 parking goal assignments; energy đúng 10×12×600/3600 = 20 Wh. Test robot bắt đầu ngoài parking xác nhận sau arrival không đổi vị trí 300 s, một goal parking và đúng 1 Wh auxiliary cho một robot trong 300 s.
Sparse: max parking goal assignments/robot = heuristic 1, fixed 1, MAPPO 2; không có ABA. MAPPO robot 5 có hai goal parking tại 190 và 270 s, giữa đó có task; không phải goal bị đảo mỗi 5 s.
ABA là mẫu trong chuỗi goal rảnh, có thể gồm các lần quay về bay sau task khác nhau. Trong ca có đơn không coi mọi ABA là oscillation vô ích; đọc tick/task/booking và source cùng log.

## 5. So sánh validation với ngân sách mặc định

Solver wall limit=0.1 s, decision budget=0.25 s. Bảng đầy đủ: `comparison_default.csv`. Các cột đều trước → sau.

| Ca | Method | Seed | Throughput /h | Pending | Lateness s | Safety | Empty ratio | Wh |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| S1 | fixed_cpsat | 1000 | 75.0 → 75.0 | 2.0 → 2.0 | 296.5 → 676.2 | 0.0 → 0.0 | 0.817 → 0.753 | 226.9 → 185.4 |
| S1 | heuristic | 1000 | 75.0 → 75.0 | 2.0 → 2.0 | 113.5 → 113.5 | 0.0 → 0.0 | 0.514 → 0.556 | 130.4 → 138.9 |
| S1 | mappo_dispatch | 1000 | 75.0 → 75.0 | 2.0 → 2.0 | 51.0 → 68.9 | 0.0 → 0.0 | 0.823 → 0.711 | 224.0 → 162.7 |
| S1 | fixed_cpsat | 1001 | 69.0 → 69.0 | 3.0 → 3.0 | 2007.1 → 1748.6 | 0.0 → 0.0 | 0.865 → 0.772 | 241.7 → 179.7 |
| S1 | heuristic | 1001 | 39.0 → 39.0 | 13.0 → 13.0 | 5331.4 → 5331.4 | 0.0 → 0.0 | 0.587 → 0.615 | 81.3 → 79.1 |
| S1 | mappo_dispatch | 1001 | 72.0 → 72.0 | 2.0 → 2.0 | 78.8 → 60.9 | 0.0 → 0.0 | 0.744 → 0.700 | 182.5 → 162.3 |
| S2 | fixed_cpsat | 1000 | 129.0 → 126.0 | 0.0 → 1.0 | 1954.6 → 813.5 | 0.0 → 0.0 | 0.830 → 0.755 | 392.7 → 332.9 |
| S2 | heuristic | 1000 | 126.0 → 129.0 | 1.0 → 0.0 | 2236.4 → 2230.8 | 0.0 → 0.0 | 0.488 → 0.525 | 206.2 → 215.2 |
| S2 | mappo_dispatch | 1000 | 129.0 → 126.0 | 0.0 → 1.0 | 99.9 → 56.1 | 0.0 → 0.0 | 0.756 → 0.684 | 312.1 → 265.8 |
| S2 | fixed_cpsat | 1001 | 96.0 → 102.0 | 5.0 → 3.0 | 3004.2 → 1702.3 | 0.0 → 0.0 | 0.853 → 0.809 | 356.6 → 305.1 |
| S2 | heuristic | 1001 | 54.0 → 69.0 | 19.0 → 14.0 | 8402.4 → 8279.4 | 0.0 → 0.0 | 0.583 → 0.586 | 121.1 → 127.8 |
| S2 | mappo_dispatch | 1001 | 105.0 → 108.0 | 2.0 → 1.0 | 39.5 → 52.7 | 0.0 → 0.0 | 0.757 → 0.726 | 277.1 → 254.1 |
| S4 | fixed_cpsat | 1000 | 108.0 → 96.0 | 3.0 → 7.0 | 1805.7 → 1751.6 | 0.0 → 0.0 | 0.827 → 0.791 | 361.2 → 303.3 |
| S4 | heuristic | 1000 | 87.0 → 90.0 | 10.0 → 9.0 | 6522.7 → 6462.7 | 0.0 → 0.0 | 0.589 → 0.600 | 175.2 → 181.1 |
| S4 | mappo_dispatch | 1000 | 108.0 → 111.0 | 3.0 → 2.0 | 270.1 → 277.6 | 0.0 → 0.0 | 0.720 → 0.708 | 285.3 → 268.6 |
| S4 | fixed_cpsat | 1001 | 99.0 → 75.0 | 4.0 → 12.0 | 883.2 → 2857.8 | 0.0 → 0.0 | 0.848 → 0.829 | 316.5 → 259.9 |
| S4 | heuristic | 1001 | 75.0 → 75.0 | 12.0 → 12.0 | 4729.4 → 4730.9 | 0.0 → 0.0 | 0.601 → 0.619 | 145.0 → 148.9 |
| S4 | mappo_dispatch | 1001 | 102.0 → 102.0 | 3.0 → 3.0 | 34.9 → 49.8 | 0.0 → 0.0 | 0.797 → 0.712 | 290.2 → 235.0 |

**Không đạt pending gate theo từng ca: 4/18 ca tăng pending.** Safety = 0 ở tất cả ca.
Cùng source/config/tape, `readiness_probe/S1_fixed_cpsat_1000` và `final_comparison/S1_fixed_cpsat_1000` khác nghiệm FEASIBLE ngay t=0, cho pending 3 và 2. Bằng chứng `walltime_variation.json` và hai decisions.jsonl. Source cũ có sẵn wall-time cutoff (`optimizer.py:948`), nên không hứa tái hiện bitwise với deadline như vậy.

## 6. Đối chứng solver có ngân sách công việc xác định

Chỉ trong script chẩn đoán: `max_deterministic_time=0.03`, wall guard=4 s và decision budget=5 s cho cả trước/sau. Production solver không bị sửa. Đây là đối chứng tách ảnh hưởng wall deadline; không phải bằng chứng đạt latency 250 ms. Bảng đủ 18 ca: `comparison_controlled.csv`.

| Method | Mean throughput /h trước → sau | Mean pending trước → sau | Mean lateness s trước → sau | Mean empty ratio trước → sau | Mean Wh trước → sau |
|---|---:|---:|---:|---:|---:|
| heuristic | 76.000 → 79.500 | 9.500 → 8.333 | 4555.949 → 4524.770 | 0.560 → 0.584 | 143.195 → 148.485 |
| fixed_cpsat | 93.000 → 92.000 | 3.833 → 4.167 | 1477.325 → 1664.600 | 0.844 → 0.783 | 312.166 → 258.847 |
| mappo_dispatch | 98.500 → 99.000 | 2.000 → 1.833 | 95.699 → 94.316 | 0.766 → 0.707 | 261.875 → 224.743 |

**Pending vẫn tăng ở 3/18 ca đối chứng:**

- S2 / fixed_cpsat / seed 1000: 0 → 1.
- S2 / mappo_dispatch / seed 1000: 0 → 1.
- S4 / fixed_cpsat / seed 1001: 4 → 12.

Safety vẫn bằng 0. Vì vậy không dùng biến động deadline để bỏ qua hồi quy pending. Thử nghiệm `no_idle_return_probe/` cho S1 hồi phục pending khi tạm bỏ quay về bay; S2 vẫn còn hồi quy. Đây là probe chẩn đoán riêng, không phải source được bàn giao.
Tại S4/fixed/1001 sau đối chứng: 12 task pending, không lost/blocked/deadlock; có task 80 kg chờ từ 533/591/778 s, robot heavy đang sạc hoặc làm task khác. Điều này cần đối chiếu scheduling/charging và tune trên dynamics mới; không có bằng chứng chỉ một lỗi arrival hay một ngoại lệ guard sửa được toàn bộ trade-off. Không tự đổi reward, threshold sạc, profile hoặc thuật toán phân task để làm bảng qua.

## 7. Nghiệm thu và kiểm tra

| Tiêu chí | Kết quả |
|---|---|
| Không đơn: đứng yên ≥300 s, ≤1 parking goal | Đạt: 600 s ở ba controller; outside-parking test thêm 300 s sau arrival. |
| Near-tie / hysteresis | Đạt test: demand đổi rất nhỏ quanh điểm cân bằng vẫn chọn wait; vùng có lợi được đổi sau residence. |
| Đơn thưa | Đạt ca chẩn đoán: 0 ABA, max goals 1/1/2; đọc nguyên nhân task/parking trong trace. |
| F05 burst | Đạt unit test và S4 pilot: ví dụ robot 9 ở (19,28) chuyển (20,1) tại 1040 s; robot 8 (19,1) → (21,1) tại 1060 s; không tắt reposition. |
| Endpoint/deadlock yield | Đạt test riêng hai cơ chế: bay mới ổn định 300 s; parking diversion không khôi phục parking goal cũ. Task recovery test cũ vẫn qua. |
| Full regression | 184 tests, failures=0, errors=0; `pytest_final.xml`, khoảng 64.65 s. Source cuối khớp hash đã kiểm chứng sau khi bỏ probe tăng cost. |
| Safety/pending | Safety đạt; pending chưa đạt toàn bộ theo bảng default và controlled. |
| Ruff / diff | Ruff check toàn src/tests/scripts và diff --check qua; format của 7 file code thay đổi qua. Format toàn repo còn lỗi có sẵn ở study_cleanup.py / study_status.py; không tự sửa ngoài phạm vi. |
| Checksum | SHA256SUMS.txt được cập nhật sau thay đổi repo; chạy update_checksums.py --check. |

## 8. Giới hạn và phần cần chạy lại

Source trước: `dddadb0baf8ddfa3b370fa6dc15e8931693f313e88ad5acbfac057714acbfe34`.
Source sau: `744de7e00397e055a71096f0766177e357b216cbcc7740fe86ca6f9a0ea40a74`.
`runs/study14` và checkpoint cũ giữ nguyên; các con số pilot cũ/mới là frozen-policy transfer check, không phải study mới với learner đã thích nghi. Không so trực tiếp hay gộp study14 vào dữ liệu source mới.
Đề xuất sau khi xử lý pending gate: tune lại ba controller không học trên validation source mới; train/finetune riêng 11 learner, đặc biệt MAPPO vì mask/tập lựa chọn thay đổi; khóa config/checkpoint rồi test lại toàn ma trận + OOD. Những test seeds cũ đã từng được nhìn thấy cần ghi rõ development evidence; dùng confirmatory protocol mới nếu tuyên bố dữ liệu hoàn toàn mới. Không tự chạy cả study dài ngày trong đợt này.
Lỗi ngoài phạm vi được liệt kê: tính biến động nghiệm do wall cutoff có sẵn, hai file format có sẵn, warning dependency Pyparsing/SWIG. Không sửa những phần này.

Probe thêm `dispatch_cost_probe/` thử dùng decision_s thay cho 900 s khi đánh giá chi phí mất cơ hội nhận task. S2/seed1000 tăng pending lên 4 ở cả fixed và MAPPO; thử nghiệm không đạt nên đã bỏ. Source bàn giao khớp `after_controlled` theo SHA-256, không mang thay đổi probe này.

## 9. Tái chạy

```powershell
# Chạy từ repo root, dùng output mới để giữ evidence.
& 'python' scripts/diagnose_idle_motion.py --output runs/idle_parking_next --checkpoint runs/study14/jobs/train_mappo_dispatch_11/attempt_002/best_model.zip
# Đối chứng: thêm --suite comparison --solver-deterministic-time 0.03.
# Ghép tape: thêm --tapes-from runs/idle_parking_20261013/before_comparison.
```

Mỗi case có config/tape/summary, ticks.jsonl, goals.jsonl, decisions.jsonl; các run cuối còn có events.json và final_state.json. `acceptance.json` ghi các ca không đạt, không đổi số liệu để đạt gate.
