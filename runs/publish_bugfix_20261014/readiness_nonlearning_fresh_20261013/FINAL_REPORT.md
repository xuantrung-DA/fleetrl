# Kết quả cuối: 300 ca test-only với 3 controller không học

## 1. Kết luận

**Hoàn tất phạm vi 300/300 ca được cho phép. Hệ thống dùng được có điều kiện; chưa nghiệm thu toàn bộ khả năng phục vụ.**

Giữ source hiện tại, không train/tune, không chạy 33 checkpoint học. Kết quả này tách riêng khỏi Study14. Tập xác nhận 3005–3009 được người dùng duyệt trước khi đo. Sau ca 32, người dùng cho phép tiếp tục riêng các lỗi completion; ngưỡng 50% và nhãn không đạt vẫn giữ nguyên.

| Nội dung | Kết quả |
|---|---:|
| Ca đã chạy / kế hoạch | 300 / 300 |
| Đạt mọi gate đã yêu cầu | 100 |
| Chưa kiểm đủ gate | 199 |
| Không đạt | 1 |
| Safety incident / mất task / crash | 0 / 0 / 0 |
| Deadlock phát hiện / xử lý / còn lại | 6 / 6 / 0 |
| p95 quyết định cao nhất giữa các ca | 46,01 ms, ngưỡng 250 ms |
| CP-SAT/MPC thiếu tỷ lệ timeout | 200 ca, gồm cả ca completion không đạt |
| Tổng thời gian chạy các episode | 40,25 phút, không gồm thời gian chờ phê duyệt |
| Dữ liệu tại thời điểm audit | khoảng 1,09 GiB |

100 ca đạt đều là heuristic. 199 ca CP-SAT/MPC còn nhãn chưa kiểm vì timeout chưa được xác minh; ca S6/fixed/3005 vừa thiếu bằng chứng timeout vừa không đạt completion, nên mang nhãn không đạt. Không được hiểu bảng này là 299 ca đạt đầy đủ.

### Năm điểm mạnh

- Không có safety incident, mất task hoặc crash trong 300 ca xác nhận.
- Cả 6 deadlock được giải quyết; không còn deadlock chưa xử lý.
- Cả 30 ca không có đơn đạt đỗ ổn định ít nhất 600 giây và năng lượng phụ tải đúng kỳ vọng.
- Cả 15 ca có đơn đến khi robot đang đi đỗ đạt kiểm tra nhận task ở chu kỳ kế tiếp.
- Source/config/plan/tape khớp bản khóa; 213 test và kiểm tra chất lượng của source hiện tại đã được lưu ở pha trước.

### Năm điểm yếu

- S6/fixed/3005 chỉ hoàn thành 17/39 đơn, dưới ngưỡng 50%.
- Tỷ lệ timeout của CP-SAT/MPC vẫn chưa xác minh, không được suy từ p95 thấp.
- Heavy80 với pin thấp có nhiều ca hoàn thành 0 đơn; miễn gate completion đã khai báo trước không chứng minh phục vụ tốt.
- Delay=15 chủ yếu dùng heuristic cục bộ với telemetry vật lý mới; chưa chứng minh controller xử lý telemetry hoàn toàn trễ.
- 11 learner/33 checkpoint và giao diện demo 3D chưa được kiểm trong pha này; 5 tape mỗi nhóm còn là mẫu nhỏ.

### Ba bước tiếp theo

1. Giữ báo cáo này như bằng chứng riêng cho 3 controller, với ca âm và gate thiếu được công bố.
2. Người dùng chọn phạm vi tiếp theo: kiểm checkpoint cũ hoặc phương án train lại; chưa tự khởi động.
3. Nếu tiếp tục sửa source, tái hiện/chọn sửa trên validation và duyệt một tập xác nhận mới vì 3005–3009 đã được quan sát.

## 2. Ca completion không đạt

**S6 / fixed_cpsat / seed 3005:** 17/39 đơn (43,59%), còn 22 đơn; safety/lost=0. Heuristic cùng tape hoàn thành 23/39; đây là một tape, chưa đủ để kết luận ưu thế tổng quát.

Đã xác nhận nút thắt ở snapshot cuối t=1195s: lịch sạc thật không có ứng viên task; probe chẩn đoán bỏ booking có 48 ứng viên. Booking kéo dài qua horizon 1200s, trong khi một số robot H còn idle và có pin. Probe không thay guard hoặc booking trong episode thực. Đây là bằng chứng calendar chặn phục vụ ở snapshot đó, chưa chứng minh nguyên nhân duy nhất của cả episode hoặc một giải pháp sửa cụ thể.

Trace và chẩn đoán: `S6_COMPLETION_GATE_DIAGNOSIS.json`, `S6_CALENDAR_WITNESS.json`, `pilot/S6_replay1200_fixed_cpsat_3005/issue.json`. Không thêm ca completion không đạt nào sau khi resume. Lỗi này khác lỗi safety endpoint đã xử lý ở validation trước.

## 3. Đèn giao thông theo controller và kịch bản

Xanh: đủ 5 tape và mọi gate đạt. Vàng: gate chưa xác minh. Đỏ: ít nhất một ca không đạt. Heavy80 miễn completion từ plan trước đo; safety vẫn bắt buộc.

| Kịch bản | Heuristic | Fixed CP-SAT | Forecast MPC |
|---|---|---|---|
|S4_replay1200|🟢 Đạt|🟡 Chưa kiểm timeout|🟡 Chưa kiểm timeout|
|S2_replay1200|🟢 Đạt|🟡 Chưa kiểm timeout|🟡 Chưa kiểm timeout|
|S6_replay1200|🟢 Đạt|🔴 Không đạt|🟡 Chưa kiểm timeout|
|zero_inside|🟢 Đạt|🟡 Chưa kiểm timeout|🟡 Chưa kiểm timeout|
|zero_outside|🟢 Đạt|🟡 Chưa kiểm timeout|🟡 Chưa kiểm timeout|
|sparse|🟢 Đạt|🟡 Chưa kiểm timeout|🟡 Chưa kiểm timeout|
|arrival_during_parking|🟢 Đạt|🟡 Chưa kiểm timeout|🟡 Chưa kiểm timeout|
|delay_0|🟢 Đạt|🟡 Chưa kiểm timeout|🟡 Chưa kiểm timeout|
|delay_1|🟢 Đạt|🟡 Chưa kiểm timeout|🟡 Chưa kiểm timeout|
|delay_5|🟢 Đạt|🟡 Chưa kiểm timeout|🟡 Chưa kiểm timeout|
|delay_10|🟢 Đạt|🟡 Chưa kiểm timeout|🟡 Chưa kiểm timeout|
|delay_15|🟢 Đạt|🟡 Chưa kiểm timeout|🟡 Chưa kiểm timeout|
|full_burst_recovery|🟢 Đạt|🟡 Chưa kiểm timeout|🟡 Chưa kiểm timeout|
|heavy80_low_soc|🟢 Đạt|🟡 Chưa kiểm timeout|🟡 Chưa kiểm timeout|
|urgent_region_shift|🟢 Đạt|🟡 Chưa kiểm timeout|🟡 Chưa kiểm timeout|
|blocked_pause_yield|🟢 Đạt|🟡 Chưa kiểm timeout|🟡 Chưa kiểm timeout|
|layout_A_burst_1|🟢 Đạt|🟡 Chưa kiểm timeout|🟡 Chưa kiểm timeout|
|layout_B_burst_1|🟢 Đạt|🟡 Chưa kiểm timeout|🟡 Chưa kiểm timeout|
|layout_A_burst_2.5|🟢 Đạt|🟡 Chưa kiểm timeout|🟡 Chưa kiểm timeout|
|layout_B_burst_2.5|🟢 Đạt|🟡 Chưa kiểm timeout|🟡 Chưa kiểm timeout|

## 4. Chỉ số và cách đọc so sánh

| Chỉ số | Ý nghĩa |
|---|---|
| Completed / pending | Đơn đã giao / đơn chưa giao cuối horizon |
| Completion | Completed chia số đơn đến |
| Throughput | Đơn hoàn thành mỗi giờ mô phỏng |
| Lateness / deadline violation | Tổng giây trễ / tỷ lệ vi phạm hạn giao |
| Energy per completed | Wh tiêu thụ mỗi đơn hoàn thành; không xác định khi completed=0 |
| Empty distance ratio | Tỷ lệ quãng đường robot đi không chở hàng |
| Charge wait | Thời gian chờ sạc trung bình |
| p95 | Thời gian quyết định mà 95% chu kỳ không vượt |
| Timeout null | Chưa xác minh tỷ lệ timeout, không có nghĩa bằng 0 |
| Local fallback rate | Tỷ lệ chu kỳ dùng heuristic cục bộ thay controller |

`nonlearning_metrics.csv` có đủ 300 ca và nguồn đo; `nonlearning_aggregate.csv` có mean/SD theo kịch bản/controller. `nonlearning_paired_CI95.csv` có 292 hàng so sánh chỉ số theo cùng tape, 2000 bootstrap. Số hàng thiếu do chỉ số không xác định khi không hoàn thành đơn. Controller không học không có training seed; CI chỉ resample task tape. Nếu CI chứa 0, không kết luận khác biệt; không tuyên bố một phương án tốt hơn trên mọi kịch bản.

Delay=15 giữ danh sách task từ snapshot trễ nhưng dùng telemetry vật lý mới để thực thi fallback. Không có task ngoài allowlist được nhận. Đọc tỷ lệ fallback từng ca trong CSV trước khi diễn giải kết quả như khả năng chịu trễ của controller.

## 5. Provenance và giới hạn source

Source hash: `8bb96d90d4c7ac92595fdb4805c33ce5ffb958736237611a42f5b59948319c95`. Live source và frozen source khớp; config, plan và tất cả task tape đã đối chiếu lại sau khi kết thúc.

Source endpoint fix đã qua 213 test, Ruff và 193 checksum ở `../endpoint_wait_fix_20261013/QUALITY_FINAL.json`; 198 validation ở `../readiness_endpoint_validation_20261013/VALIDATION_AUDIT.json` là dữ liệu phát triển riêng, không gộp vào 300 ca xác nhận.

Fixture chu kỳ endpoint trên validation 1000 tái hiện được cơ chế lỗi và đã giải quyết 2 task không có safety incident. 40 ca S4 tự nhiên trước/sau không tái hiện safety; không được nói rằng lỗi tự nhiên được tái hiện trên 40 ca đó. So sánh validation fixed có giảm mean 0,4 completed và tăng lateness trong lượt đo; nguyên nhân chênh lệch chưa được xác nhận, không khẳng định mọi hiệu năng đã cải thiện.

`CONTINUE_COMPLETION_APPROVAL.json` lưu phê duyệt sau ca 32. `STOP_REQUIRED_first_completion.json` và `STOP_REQUIRED.json` giữ lịch sử ca âm; chúng không còn là blocker sau khi hoàn tất theo phê duyệt. `PHASE_COMPLETE.json`, `pilot_status.json` và `PHASE_AUDIT.json` xác nhận pha đã xong. Các kết quả Study14 giữ riêng. Khi đo checkpoint cũ trên source này, nhãn bắt buộc là **checkpoint train trên source cũ**; lượt này chưa chạy checkpoint nào.

Demo 3D chưa được kiểm trực tiếp; kết luận ở đây dựa trên simulator, replay và audit vật lý. Báo cáo học phần có thể dùng để trình bày kiểm chứng có điều kiện; chưa gọi hệ thống nghiệm thu toàn bộ.

## 6. Ước lượng train lại — chưa thực hiện

Ước lượng từ thời gian các job lịch sử, quy đổi 300.000 bước × 3 training seed; không phải benchmark train trên source hiện tại hay khoảng tin cậy.

| Phương án | Danh nghĩa | Khoảng dự trù | Dung lượng dự trù |
|---|---:|---:|---:|
| Đủ 11 learner × 3 seed | 48,7 giờ | 36,5–85,2 giờ | 2,6–4,9 GiB |
| MAPPO dispatch + DQN CP-SAT | 12,0 giờ | 9–21 giờ | 1–2 GiB |
| Hai ứng viên trên + PPO CP-SAT | 14,8 giờ | 11,1–25,9 giờ | 1–2 GiB |

Chưa gồm tune, giai đoạn xác nhận, checkpoint tạm và mọi replay mở rộng. Cả ba phương án dự kiến vượt 6 giờ, phải có lựa chọn/phê duyệt trước khi chạy. Không có train/tune hoặc inference từ checkpoint học được khởi động ở lượt này.

## 7. Lệnh và tệp bằng chứng

Chạy từ repository root trong environment hiện tại:

```powershell
python runs/readiness_nonlearning_fresh_20261013/run_nonlearning.py
python runs/readiness_nonlearning_fresh_20261013/report_nonlearning.py
```

Runner bỏ qua 300 job complete; không đo lại tự động. Report generator tạo CSV và báo cáo chi tiết `REPORT.md`. Kết luận cuối nằm ở `FINAL_REPORT.md`; `COMPLETION_AUDIT.md` là bản tóm tắt trạng thái. Không xoá ca âm hoặc dùng kết quả cũ để thay số liệu của source mới.
