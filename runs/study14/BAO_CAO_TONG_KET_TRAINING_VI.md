# BÁO CÁO TỔNG KẾT TRAINING FLEETRL — STUDY14

**Ngày tổng hợp:** 02/10/2026  
**Thư mục kết quả:** `runs/study14`  
**Trạng thái:** Hoàn tất đầy đủ 1.980/1.980 episode kiểm thử; không có episode thiếu hoặc thất bại.

## 1. Tóm tắt dành cho người đọc nhanh

Study14 đã so sánh **14 phương án điều phối đội robot kho**, gồm 3 phương án không học và 11 phương án có học máy/RL. Các mô hình có học được train với 3 seed độc lập (11, 12, 13), mỗi seed có ngân sách khoảng 300.000 bước; tổng số bước train chính theo kế hoạch là 9.900.000 bước. Trước đó, hệ thống còn chạy 33 job tìm siêu tham số, mỗi phương pháp học thử 3 lựa chọn với tổng 30.000 bước tìm kiếm.

Mỗi phương án được kiểm tra trên 9 kịch bản S1–S9 và 2 cặp OOD_A/OOD_B. Mỗi episode mô phỏng một giờ vận hành. Mô hình học được đánh giá với 3 training seed × 5 task-tape seed = 15 episode/kịch bản; baseline xác định chỉ cần 5 episode/kịch bản. Tổng cộng có 1.620 episode lõi và 360 episode đối chứng OOD, bằng 1.980 episode.

### Kết luận chính

- **Kết quả chạy là đầy đủ:** 1.980/1.980 episode hoàn tất, đủ dữ liệu cho bảng so sánh mô tả và khoảng tin cậy ghép cặp.
- **Phương án có học tốt nhất về tổng thể là `mappo_dispatch`:** throughput trung bình 104,285 task/giờ, pending 11,279 task, lateness 16.099 giây và p95 quyết định khoảng 7,83 ms. Nó vượt rõ các mô hình RL khác về lateness và độ trễ quyết định.
- **Nếu chỉ nhìn hiệu quả vận hành tổng thể, `heuristic` vẫn rất mạnh:** throughput cao nhất toàn bộ bảng, 104,691 task/giờ; lateness thấp nhất, 14.575 giây; không có safety incident và dùng ít năng lượng hơn nhiều. Điều này có nghĩa là training RL chưa chứng minh được việc thay thế heuristic là có lợi.
- **Giả thuyết tăng throughput ít nhất 5% trên S4–S7 không đạt.** Trung bình S4–S7, `fixed_cpsat` đạt 114,6 task/giờ; MAPPO đạt 117,517 task/giờ, chỉ tăng khoảng **2,55%**. DQN tăng khoảng 1,34%. Không mô hình học nào đủ bằng chứng đạt mục tiêu ≥5% trên toàn nhóm này.
- **MAPPO vẫn cải thiện chất lượng phục vụ rất lớn:** trên S4–S7, lateness trung bình giảm từ 9.329,6 giây của `fixed_cpsat` xuống 501,6 giây; pending giảm từ 5,2 xuống 2,283 task và completion rate tăng từ 95,37% lên 98,03%.
- **S8 là lỗi/điểm yếu hệ thống nghiêm trọng:** khi snapshot trạng thái trễ 15 giây, đa số phương án chỉ hoàn thành khoảng 3–8 task/giờ, còn khoảng 96–101 task chưa xong. MAPPO tệ nhất trong nhóm được nêu: 3,2 task/giờ và 100,8 task pending. Không nên triển khai khi dữ liệu đầu vào có thể trễ tương tự mà chưa sửa fallback.
- **Safety nhìn chung tốt nhưng chưa bằng 0 tuyệt đối:** nhiều phương án có trung bình rất thấp, nhưng một số episode vẫn phát sinh incident. Vì safety là tiêu chí bắt buộc, không được bỏ qua chỉ vì giá trị trung bình nhỏ.
- **Kết quả chưa đủ để tuyên bố hội tụ hoặc “RL tốt hơn”.** Báo cáo gốc cũng cảnh báo rằng việc chạy đủ và có checkpoint không tự chứng minh mô hình đã hội tụ hoặc vượt baseline có ý nghĩa thống kê.

## 2. Bạn đã train những gì?

### 2.1 Ba phương án không cần training

| ID | Phương án | Ý nghĩa |
| --- | --- | --- |
| V01 | `heuristic` | Luật điều phối thủ công; baseline đơn giản, nhanh và dễ giải thích. |
| V02 | `fixed_cpsat` | Dùng CP-SAT với một profile trọng số cố định; đây là **reference chính** khi tính chênh lệch ghép cặp. |
| V13 | `forecast_mpc_milp` | Dự báo nhu cầu + MPC + MILP; là phương án tối ưu hệ thống nhưng không học policy. |

### 2.2 Mười một phương án đã train

| ID | Phương án | Thứ được học / điểm khác biệt | Siêu tham số được chọn |
| --- | --- | --- | --- |
| V03 | `qlearning_cpsat` | Q-learning dạng bảng chọn profile chiến lược cho CP-SAT. | `q_alpha=0.05` |
| V04 | `dqn_cpsat` | DQN chọn action rời rạc/profile cho CP-SAT. | `learning_rate=0.001` |
| V05 | `a2c_cpsat` | Actor-Critic đồng bộ chọn profile cho CP-SAT. | `learning_rate=0.0001` |
| V06 | `ppo_cpsat` | PPO trung tâm chọn profile cho CP-SAT. | `learning_rate=0.001` |
| V07 | `qrdqn_cpsat` | Distributional DQN, học phân phối giá trị thay vì chỉ kỳ vọng. | `learning_rate=0.001` |
| V08 | `recurrent_ppo_cpsat` | PPO có bộ nhớ tuần tự, nhằm xử lý quan sát thiếu/quá khứ. | `learning_rate=0.0001` |
| V09 | `ppo_threshold_cpsat` | PPO kết hợp ngưỡng sạc được tune. | `learning_rate=0.0003`, `charge_threshold=0.3` |
| V10 | `sac_cpsat` | SAC với không gian action liên tục rồi ánh xạ vào điều phối. | `learning_rate=0.0003` |
| V11 | `td3_cpsat` | TD3 với action liên tục, giảm over-estimation bằng hai critic. | `learning_rate=0.0003` |
| V12 | `mappo_dispatch` | Multi-agent PPO điều phối trực tiếp; khác cả biểu diễn/action so với nhóm chọn profile. | `learning_rate=0.001` |
| V14 | `ppo_milp` | PPO nhưng dùng MILP thay vì CP-SAT ở backend tối ưu. | `learning_rate=0.0003` |

Lưu ý: tên phương án cho biết cả thuật toán học lẫn backend. Vì vậy, ví dụ `ppo_cpsat` và `ppo_milp` không chỉ khác solver; kết quả cuối còn chịu ảnh hưởng của checkpoint, seed và cách backend xử lý bài toán.

### 2.3 Quy mô training

- 33 job train chính = 11 phương án học × 3 seed.
- Ngân sách kế hoạch: 300.000 bước/job, tổng 9.900.000 bước.
- Số bước thực tế trong bảng tổng hợp dao động quanh 900.000 bước/phương án khi cộng 3 seed. PPO/A2C có thể vượt nhẹ vì rollout phải kết thúc theo block cập nhật.
- 33 job search siêu tham số; mỗi phương án thử 3 lựa chọn, tổng 30.000 bước tìm kiếm/phương án.
- Training seed: 11, 12, 13.
- Validation seed dùng trong quy trình chọn/tune: 1000, 1001, 1002.
- Test tape seed cuối: 2000–2004.

## 3. Các kịch bản kiểm thử có ý nghĩa gì?

| Kịch bản | Nội dung | Điều muốn kiểm tra |
| --- | --- | --- |
| S1 | 10 robot, nhu cầu thường. | Ca nhẹ/cơ bản. |
| S2 | 15 robot, nhu cầu thường. | Ca chuẩn trung bình. |
| S3 | 20 robot, nhu cầu thường. | Khả năng mở rộng đội robot. |
| S4 | Nhu cầu burst 2×, 40% robot có pin đầu thấp. | Áp lực đồng thời về đơn hàng và sạc. |
| S5 | 20 robot, tỷ lệ robot tải nặng tăng lên 40%. | Đội không đồng nhất và nhiều tải nặng. |
| S6 | Trạm sạc bị đổi vị trí, công suất giảm còn 300 W/cổng. | Thiếu tài nguyên sạc và bố trí bất lợi. |
| S7 | Có chặn đường và robot tạm dừng. | Khả năng phục hồi trước nhiễu vận hành. |
| S8 | Snapshot/quan sát bị trễ 15 giây. | Kiểm tra conservative fallback khi dữ liệu cũ. |
| S9 | Map B chưa gặp + burst 2,5× ở vùng khác. | OOD kết hợp; không thể quy suy giảm cho riêng layout. |
| OOD_A/B | Cùng task tape, chỉ đổi layout. | Tách riêng ảnh hưởng của thay đổi bản đồ. |

## 4. Bảng kết quả tổng hợp 14 phương án

Các số dưới đây là trung bình trên toàn bộ 11 nhóm scenario (S1–S9, OOD_A, OOD_B). Vì S8 cực đoan, cột lateness tổng thể bị S8 chi phối mạnh; cần đọc cùng phần phân tích theo kịch bản.

| Hạng tham khảo | Phương án | Episode | Throughput/h ↑ | Lateness (s) ↓ | Pending ↓ | Safety ↓ | Bước train |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `heuristic` | 55/55 | **104,691** | **14.574,765** | **10,873** | **0,000** | Không train |
| 2 | `mappo_dispatch` | 165/165 | 104,285 | 16.099,208 | 11,279 | 0,012 | 900.000 |
| 3 | `forecast_mpc_milp` | 55/55 | 104,000 | 20.477,319 | 11,564 | **0,000** | Không train |
| 4 | `dqn_cpsat` | 165/165 | 103,679 | 20.486,635 | 11,885 | **0,000** | 900.000 |
| 5 | `recurrent_ppo_cpsat` | 165/165 | 103,630 | 21.103,058 | 11,933 | 0,018 | 900.096 |
| 6 | `a2c_cpsat` | 165/165 | 103,467 | 20.519,001 | 12,097 | 0,018 | 900.096 |
| 7 | `fixed_cpsat` | 55/55 | 103,327 | 19.868,957 | 12,236 | 0,018 | Không train |
| 8 | `ppo_milp` | 165/165 | 103,297 | 20.772,129 | 12,267 | 0,042 | 903.168 |
| 9 | `td3_cpsat` | 165/165 | 103,248 | 21.085,679 | 12,315 | 0,012 | 900.000 |
| 10 | `qrdqn_cpsat` | 165/165 | 103,158 | 20.648,552 | 12,406 | 0,012 | 900.000 |
| 11 | `qlearning_cpsat` | 165/165 | 102,873 | 22.325,530 | 12,691 | 0,006 | 900.000 |
| 12 | `ppo_cpsat` | 165/165 | 102,703 | 21.652,706 | 12,861 | 0,048 | 903.168 |
| 13 | `sac_cpsat` | 165/165 | 102,382 | 20.860,137 | 13,182 | 0,030 | 900.000 |
| 14 | `ppo_threshold_cpsat` | 165/165 | 101,933 | 22.904,991 | 13,630 | 0,073 | 903.168 |

“Hạng tham khảo” ưu tiên throughput rồi xem lateness/pending/safety, không phải một kiểm định thống kê chính thức. Không nên dùng thứ hạng này một mình để chọn mô hình.

## 5. Phân tích kết quả quan trọng

### 5.1 MAPPO là mô hình đã train đáng chú ý nhất

`mappo_dispatch` đứng đầu trong các phương án có học về throughput tổng thể và thấp hơn rất nhiều về lateness so với các mô hình RL còn lại. Trên nhóm stress chính S4–S7:

| Metric trung bình S4–S7 | `fixed_cpsat` | `dqn_cpsat` | `mappo_dispatch` | Cách hiểu |
| --- | ---: | ---: | ---: | --- |
| Throughput/h ↑ | 114,600 | 116,133 | **117,517** | MAPPO tăng 2,55% so với reference, chưa đạt mục tiêu 5%. |
| Total lateness (s) ↓ | 9.329,6 | 9.370,4 | **501,6** | MAPPO giảm trễ rất lớn. |
| Pending ↓ | 5,200 | 3,667 | **2,283** | MAPPO để lại ít đơn chưa hoàn thành hơn. |
| Completion rate ↑ | 95,37% | 96,82% | **98,03%** | MAPPO hoàn thành tỷ lệ đơn cao hơn. |
| Deadline violation rate ↓ | 40,88% | 41,27% | **9,01%** | MAPPO bảo vệ deadline tốt hơn rõ rệt. |
| Decision p95 (ms) ↓ | 36,57 | 39,84 | **8,22** | MAPPO đáp ứng online nhanh hơn. |
| Safety incident ↓ | 0,050 | **0,000** | 0,033 | MAPPO tốt nhưng chưa bằng 0 tuyệt đối. |

Khoảng tin cậy ghép cặp cho throughput của MAPPO so với `fixed_cpsat`:

- S4: +0,533 task/h, CI95 [-2,200; 3,667] → chưa chắc chắn khác biệt.
- S5: +1,067 task/h, CI95 [-0,533; 2,867] → chưa chắc chắn khác biệt.
- S6: +8,067 task/h, CI95 [3,465; 13,800] → cải thiện rõ ở ca sạc khó.
- S7: +2,000 task/h, CI95 [0,200; 4,402] → cải thiện nhỏ nhưng CI không cắt 0.

Do S4 và S5 có CI chứa 0, không thể tuyên bố MAPPO luôn tăng throughput trên mọi ca stress.

### 5.2 Heuristic chưa bị RL vượt qua

`heuristic` đạt kết quả rất cạnh tranh dù không train:

- Throughput tổng thể cao nhất: 104,691 task/h.
- Lateness tổng thể thấp nhất: 14.574,765 giây.
- Safety incident trung bình bằng 0.
- Decision p95 khoảng 6,94 ms, nhanh hơn nhóm solver/RL.
- Năng lượng trung bình 571,23 Wh/episode, thấp hơn gần một nửa so với nhiều phương án CP-SAT quanh 1.086 Wh.

Trên S4–S7, heuristic đạt 117,1 task/h và lateness chỉ 337,7 giây. MAPPO có throughput cao hơn một chút (117,517) nhưng lateness lớn hơn (501,6), safety không bằng 0 và phức tạp hơn đáng kể. Vì vậy, bằng chứng hiện tại không ủng hộ tuyên bố “RL vượt heuristic một cách toàn diện”.

### 5.3 DQN là mô hình CP-SAT học ổn định nhất trong bảng tổng thể

Trong nhóm mô hình học chọn profile cho CP-SAT, `dqn_cpsat` có throughput tổng thể cao nhất (103,679), safety trung bình 0 và peak RAM khoảng 1,79 GB. Tuy nhiên lateness 20.486,6 giây vẫn cao hơn MAPPO/heuristic rất nhiều, và trên S4–S7 nó chỉ tăng throughput 1,34% so với `fixed_cpsat`.

### 5.4 PPO tiêu chuẩn không phải mô hình thắng

`ppo_cpsat` đạt 102,703 task/h, thấp hơn `fixed_cpsat` và có safety incident trung bình 0,048. Trên S4–S7, throughput có lúc tăng nhẹ nhưng cũng giảm ở S5 và S7; lateness thường cao hơn reference. Việc train PPO đủ bước không tự động tạo ra chính sách tốt hơn baseline.

`ppo_threshold_cpsat` là phương án yếu nhất trong bảng tổng thể: throughput 101,933, lateness 22.905 giây và safety incident 0,073. Đặc biệt S4 chỉ đạt 92,467 task/h so với 100,6 của `fixed_cpsat`; chênh lệch ghép cặp -8,133 task/h với CI95 [-17,000; -0,998], cho thấy cấu hình ngưỡng sạc này làm xấu rõ ca burst + pin thấp.

### 5.5 MILP không tạo lợi thế rõ ràng cho PPO

`ppo_milp` đạt 103,297 task/h, nhỉnh hơn `ppo_cpsat` 102,703 nhưng vẫn không vượt rõ `fixed_cpsat`/DQN/MAPPO. Lateness và safety cũng không cho thấy lợi thế nhất quán. Đây chỉ là bằng chứng mô tả; không được quy toàn bộ chênh lệch cho backend nếu chưa có ablation kiểm soát chặt checkpoint/action tương đương.

### 5.6 S6 cho thấy điều phối thông minh có ích khi sạc khó

S6 đổi vị trí trạm và giảm công suất sạc. Đây là ca mà:

- `fixed_cpsat`: 93,8 task/h; 10,2 pending; 20.308 giây lateness.
- `dqn_cpsat`: 99,0 task/h; 5,0 pending; 16.574 giây lateness.
- `mappo_dispatch`: 101,867 task/h; 2,133 pending; 163 giây lateness.
- `heuristic`: 102,2 task/h; 1,8 pending; 81 giây lateness.

MAPPO cải thiện rất mạnh so với fixed CP-SAT, nhưng heuristic vẫn nhỉnh hơn một chút.

### 5.7 S8 là cảnh báo triển khai quan trọng nhất

S8 cố tình làm snapshot trạng thái trễ 15 giây. Kết quả:

| Phương án | Throughput/h ↑ | Pending ↓ | Completion rate ↑ | Lateness (s) ↓ |
| --- | ---: | ---: | ---: | ---: |
| `heuristic` | 7,800 | 96,200 | 7,51% | 158.380 |
| `fixed_cpsat` | 7,600 | 96,400 | 7,31% | 159.095 |
| `dqn_cpsat` | 7,600 | 96,400 | 7,31% | 159.090 |
| `ppo_milp` | **7,933** | **96,067** | **7,64%** | **157.922** |
| `mappo_dispatch` | 3,200 | 100,800 | 3,07% | 174.283 |

Ngay cả phương án tốt nhất cũng chỉ hoàn thành dưới 8% khối lượng. Điều này cho thấy fallback với dữ liệu trễ chưa hoạt động ở mức chấp nhận được. MAPPO đặc biệt nhạy với stale observation. Trước khi demo/triển khai, cần coi freshness của snapshot là điều kiện bắt buộc hoặc thiết kế lại fallback S8.

## 6. Cách đọc từng nhóm metric M01–M16

### M01 — Throughput (`throughput_per_hour`)

- Đơn vị: task/giờ.
- Công thức: số task hoàn thành chia cho số giờ mô phỏng.
- Càng cao càng tốt, nhưng không được tối ưu riêng metric này vì có thể hy sinh deadline, safety hoặc năng lượng.

### M02 — Completion và backlog (`pending`, `completion_rate`)

- `pending`: số task đã đến nhưng chưa hoàn thành khi episode kết thúc; càng thấp càng tốt.
- `completion_rate`: completed/arrived; càng gần 1 (100%) càng tốt.
- Task pending ở cuối ca bị right-censored: chưa biết cần thêm bao lâu để hoàn tất.

### M03 — Total lateness (`total_lateness_s`)

- Tổng số giây trễ của tất cả task đã đến.
- Với task chưa xong, độ trễ tính tới thời điểm kết thúc ca T.
- Càng thấp càng tốt. Metric này phản ánh mức độ nghiêm trọng tổng cộng, không chỉ số task bị trễ.

### M04 — Deadline violation (`deadline_violation_rate`)

- Tỷ lệ task đã đến hạn nhưng bị trễ.
- Càng thấp càng tốt.
- Khác M03: hai phương án có cùng tỷ lệ vi phạm nhưng một phương án có thể trễ lâu hơn nhiều.

### M05 — Task wait (`mean_wait_s`)

- Thời gian trung bình từ lúc task được tạo đến khi robot bắt đầu pickup.
- Càng thấp càng tốt; task chưa được xử lý được tính chờ tới cuối ca.

### M06 — Energy (`consumed_wh`, `energy_per_completed_wh`)

- `consumed_wh`: tổng điện năng robot tiêu thụ trong episode.
- `energy_per_completed_wh`: Wh cho mỗi task hoàn thành; phù hợp hơn khi throughput khác nhau.
- Càng thấp càng tiết kiệm, nhưng phải đọc cùng completion rate để tránh trường hợp “ít tốn điện vì làm rất ít task”.

### M07 — Travel (`empty_distance_ratio`)

- Tỷ lệ quãng đường robot chạy không tải trên tổng quãng đường.
- Càng thấp thường càng hiệu quả.
- Trong kết quả, heuristic khoảng 0,46–0,48 ở nhiều ca; nhiều mô hình CP-SAT khoảng 0,82–0,84. Đây là một nguyên nhân hợp lý khiến nhóm CP-SAT tốn năng lượng hơn, nhưng cần trace tuyến đường để xác nhận quan hệ nhân quả.

### M08 — Charge wait (`mean_charge_wait_s`)

- Thời gian trung bình một yêu cầu sạc phải chờ trước khi bắt đầu/cancel/kết thúc ca.
- Càng thấp càng tốt. Cần xem cùng số yêu cầu sạc và số quan sát bị censored.

### M09 — Charger utilization

- `charger_utilization`: tỷ lệ thời gian cổng đang sạc thật.
- `port_occupied_utilization`: tỷ lệ thời gian cổng bị chiếm dụng.
- Cao không luôn tốt: gần 100% có thể là sử dụng hiệu quả hoặc dấu hiệu nghẽn cổ chai. Phải đọc cùng charge wait và pending.

### M10 — Safety (`safety_incidents`)

- Tổng incident duy nhất được audit từ trạng thái đã commit và các bộ đếm vật lý.
- Mục tiêu là 0. Trung bình 0,01 vẫn có nghĩa là đã từng có episode phát sinh incident.
- Không được đánh đổi safety để lấy throughput.

### M11 — Deadlock recovery

- `deadlock_recovery_rate`: số deadlock được xác nhận đã giải quyết / số deadlock phát hiện.
- `deadlock_recovery_s`: thời gian cần để phục hồi.
- Giá trị null thường có nghĩa không có deadlock nên mẫu số không tồn tại, không đồng nghĩa thất bại.

### M12 — Online decision latency (`decision_p95_ms`)

- p95 là mức mà 95% quyết định nhanh hơn hoặc bằng giá trị này.
- Mục tiêu thiết kế là ≤250 ms. Các trung bình báo cáo đều dưới ngưỡng này.
- p95 hữu ích hơn mean vì phản ánh các lần quyết định chậm ở đuôi phân phối.

### M13 — Solver và fallback

- `solver_nonoptimal_rate`: tỷ lệ lần solver không báo optimal.
- `solver_timeout_rate`: tỷ lệ timeout đã xác nhận.
- `fallback_rate`: tỷ lệ dispatch phải dùng phương án dự phòng.
- Trạng thái FEASIBLE/UNKNOWN không tự động chứng minh timeout; lý do dừng không được báo thì giữ null.

### M14 — Tài nguyên training

- Gồm bước train, số lần update, wall time và peak RAM của toàn cây process.
- Peak RAM lớn nhất trong bảng là `recurrent_ppo_cpsat`, khoảng 3,93 GB.
- Q-learning dạng bảng thấp nhất trong nhóm học, khoảng 0,84 GB.
- Đây là chi phí huấn luyện, không phải trực tiếp là RAM inference production.

### M15 — Statistical stability

- Báo mean, standard deviation và bootstrap CI95 trên các task tape ghép cặp và checkpoint độc lập.
- Nếu CI95 của chênh lệch throughput chứa 0, dữ liệu chưa đủ để nói phương án chắc chắn tốt hơn reference.
- “Có giá trị trung bình cao hơn” và “cải thiện có ý nghĩa/ổn định” là hai câu khác nhau.

### M16 — Generalization / OOD

- So sánh OOD_A và OOD_B với cùng workload để cô lập ảnh hưởng layout.
- Báo chênh lệch tuyệt đối và tỷ lệ thay đổi so với control.
- S9 thay đổi nhiều thứ cùng lúc (map + burst/vùng nhu cầu), nên không dùng S9 để kết luận riêng về khả năng đổi map.

## 7. Safety, latency và tài nguyên

### Safety

- `heuristic`, `dqn_cpsat` và `forecast_mpc_milp` có safety trung bình tổng thể bằng 0.
- Các phương án còn lại có incident thấp nhưng không bằng 0; cao nhất là `ppo_threshold_cpsat` 0,073 incident/episode.
- Nên truy ngược đúng episode/seed có incident trước khi chọn checkpoint triển khai.

### Độ trễ online

- Mục tiêu p95 ≤250 ms đã được đáp ứng trong kết quả mô phỏng.
- Heuristic và MAPPO nhanh vì ít phụ thuộc vòng giải tối ưu nặng ở mỗi quyết định.
- Kết quả này là trên máy và mô phỏng hiện tại; chưa chứng minh latency khi nối robot/hạ tầng thật.

### RAM training

| Phương án | Peak RAM xấp xỉ |
| --- | ---: |
| `qlearning_cpsat` | 0,84 GB |
| `a2c_cpsat` | 1,65 GB |
| `ppo_cpsat` | 1,69 GB |
| `ppo_milp` | 1,76 GB |
| `dqn_cpsat` | 1,79 GB |
| `sac_cpsat` | 1,82 GB |
| `td3_cpsat` | 1,80 GB |
| `mappo_dispatch` | 1,88 GB |
| `qrdqn_cpsat` | 2,07 GB |
| `recurrent_ppo_cpsat` | **3,93 GB** |

## 8. Giới hạn và điều không được kết luận quá mức

1. **Study hoàn tất không đồng nghĩa mô hình hội tụ.** Cần nhìn learning curve, reward/metric theo bước và chạy confirmatory seed mới.
2. **Các test seed 2000–2004 đã nằm trong protocol phát triển.** Nếu đã dùng kết quả này để chỉnh model/reward, cần một protocol xác nhận mới với seed chưa từng xem, ví dụ 3000+.
3. **Có source migration giữa quá trình train.** Study đã được tiếp tục sau khi sửa lỗi nested deadlock recovery làm mất `recovery_goal`. Checkpoint trước và sau sửa có thể đến từ hai source hash khác nhau; job resume đầu tiên là `train_mappo_dispatch_11`. Điều này làm giảm độ “sạch” của so sánh nếu thay đổi ảnh hưởng hành vi.
4. **Baseline có 5 episode/kịch bản, mô hình học có 15.** Đây là thiết kế do baseline xác định và RL có 3 checkpoint seed, nhưng độ bất định phải được đọc từ paired CI thay vì chỉ so mean.
5. **S8 chi phối số trung bình tổng thể.** Khi chọn mô hình cho vận hành bình thường, phải xem bảng từng scenario; khi đánh giá độ bền, không được loại S8 chỉ vì kết quả xấu.
6. **Đây là kết quả mô phỏng.** Chưa chứng minh hiệu quả trên robot thật, mạng thật, cảm biến trễ thật hoặc hệ quản lý kho thật.

## 9. Đề xuất chọn mô hình theo mục tiêu

- **Cần phương án an toàn, đơn giản, dễ demo và hiệu quả ngay:** chọn `heuristic` làm baseline vận hành chính.
- **Cần trình bày đóng góp RL tốt nhất:** chọn `mappo_dispatch`, nhưng nói chính xác rằng nó cải thiện mạnh lateness/completion trên S4–S7, chưa đạt mục tiêu +5% throughput và rất yếu ở S8.
- **Cần kiến trúc RL + CP-SAT truyền thống:** `dqn_cpsat` là lựa chọn cân bằng nhất trong nhóm đó vì throughput cao nhất nhóm và safety trung bình 0.
- **Không nên chọn bản hiện tại:** `ppo_threshold_cpsat`, do kết quả tổng thể thấp và suy giảm có bằng chứng ở S4.
- **Trước triển khai:** bắt buộc sửa/đánh giá lại stale-observation fallback của S8 và điều tra từng safety incident.

## 10. Việc nên làm tiếp theo

1. Chọn 3 ứng viên cho vòng xác nhận: `heuristic`, `mappo_dispatch`, `dqn_cpsat`.
2. Chạy confirmatory protocol với seed mới chưa dùng để tuning (ví dụ 3000–3004), khóa source hash và config trước khi chạy.
3. Tách riêng S8 thành bài kiểm thử độ trễ quan sát 0/1/5/10/15 giây để tìm ngưỡng hệ thống bắt đầu sụp đổ.
4. Truy vết các episode có `safety_incidents > 0` và phân loại nguyên nhân thay vì chỉ nhìn trung bình.
5. Kiểm tra learning curves của MAPPO/DQN để xác định reward còn tăng, đã plateau hay mất ổn định cuối training.
6. Nếu mục tiêu luận văn là chứng minh ≥5% throughput, cần cải tiến thuật toán hoặc sửa giả thuyết; kết quả hiện tại không đạt tiêu chí đó.

## 11. Các file bằng chứng đi kèm

- `report.md`: bảng kết quả theo từng method × scenario.
- `algorithm_comparison.csv/.md`: bảng tổng hợp 14 phương án.
- `summary.json`: mean, standard deviation, paired CI95 và generalization.
- `episodes.csv/.jsonl`: dữ liệu gốc từng episode.
- `metrics_dictionary.csv/.md`: định nghĩa metric M01–M16.
- `learning_curves.png`: đường học của các mô hình.
- `benchmark.png`: biểu đồ benchmark.
- `quality_tradeoffs.png`: biểu đồ đánh đổi chất lượng.
- `plan.json`: protocol, seed, số job và ngân sách bước.
- `selected_learning.json`: siêu tham số học được chọn.
- `source_migration.json`: bằng chứng thay đổi source giữa quá trình train.

## 12. Kết luận cuối

Study14 đã hoàn thành đúng lưới đánh giá và tạo được bộ bằng chứng lớn, có thể tái kiểm tra. Kết quả tốt nhất của phần RL là `mappo_dispatch`: nó xử lý deadline, backlog và ca thiếu sạc tốt hơn phần lớn mô hình học khác. Tuy nhiên, `heuristic` vẫn là đối thủ mạnh nhất về hiệu quả tổng thể, độ đơn giản, năng lượng và safety; mục tiêu tăng throughput ≥5% chưa đạt. Điểm cần ưu tiên cao nhất không phải train thêm một cách mù quáng mà là sửa độ bền khi observation trễ (S8), điều tra safety incident, rồi chạy một vòng confirmatory với seed mới và source hoàn toàn cố định.
