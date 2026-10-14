# Cập nhật hiện tại — 13/10/2026

Phần dưới là hồ sơ bước đầu source 53ed, được giữ làm lịch sử. Bản sạc cuối trước sửa pause là `a5697e0bd8be567caa1cf3878d268da1be402b3193ed82940f50f85de155ce5b`: dùng min target cho heuristic/advanced khi có backlog hoặc demand cấu hình dương và đã quan sát task, kể cả khoảng trống giữa các đơn. Không đọc trước tape; ablation threshold vẫn 95%; không nới guard.

Shared backlog source 055ac giữ ca âm S6/fixed1008 18 completed/19 pending. Quy tắc ongoing demand sau đó đạt 35 completed/2 pending cùng ca; heuristic1001 đạt32/5. 200 test đầy đủ đạt trên bản sạc cuối. Pilot tiếp theo dừng tại73/198 vì lỗi pause độc lập ở handoff, không phải lỗi sạc.

Người dùng cho phép sửa lọc pause. Source mới `ed607d948dea6e37bf2718c469e603476811b2eec8819f0c1dca4837beda6242`; bằng chứng riêng tại `../paused_candidate_20261013/REPORT.md`, pilot mới `../readiness_pause_final_20261013`. Không train/tune, không sửa Study14, tất cả ca âm cũ giữ nguyên.

---

# Sửa nút thắt sạc — kiểm chứng bước đầu

Người dùng đã chọn cho phép xử lý nút thắt sạc trước. Không train/tune learner.

## Nguyên nhân và các thử nghiệm được giữ lại

Source trước sửa: `872ddf9c1be679ff602a776a4d97d19f7ca4e5e656aa061a40711736ebb0cff0`.
Trace S6/heuristic/1001 cho thấy task bị return-calendar từ chối khi hai cổng H bị lịch sạc dài chiếm. Robot H13 còn 58% pin; không phải thiếu pin vật lý để làm task hiện tại. Tuy nhiên guard cũng bảo vệ khả năng về sạc nên không được bỏ guard chỉ để tăng completed.

Thử thứ tự cổng ít linh hoạt hơn (source `26eaf998ee0f0401310bb392619519defa1c8738a5034c5e0f45fb0b89cc51de`) chỉ làm completed 17→15, pending 20→22. Đã hoàn tác. Snapshot, ca âm và test thử còn trong run này.

Bản hiện tại sửa riêng controller heuristic: khi có task waiting/blocked đã quan sát, dùng min(charge_targets), mặc định 60%; khi không có backlog vẫn 95%. Không đổi charge_threshold, cấu hình target, guard pin/reserve, return-calendar, road/queue wait bound, reward/observation hoặc ngưỡng nghiệm thu. PPO threshold ablation vẫn 95%; các controller khác giữ tập target cũ. Vì heuristic thay đổi, phải gắn source/version mới khi so sánh, không gọi số liệu này là Study14 gốc.

Source sau sửa: `53ed06ec4184a50a12606a07c1b908e97796880a98adeb596f0734ba67bf1315`.

## Cùng tape/config, S6/heuristic/1001 ở 1.200 s

| Chỉ số | Trước | Sau |
|---|---:|---:|
| Completed | 17 | 32 |
| Pending | 20 | 5 |
| Completion | 45,95% | 86,49% |
| Lateness tổng, s | 6334,22 | 3806,35 |
| Safety / lost | 0 / 0 | 0 / 0 |
| Năng lượng mỗi đơn, Wh | 6,83 | 5,82 |

Tổng consumed tăng 116,12→186,37 Wh do phục vụ thêm đơn; không gọi đó là tiết kiệm tổng năng lượng. Empty ratio tăng 55,70%→60,95%; charge wait tăng 30,63→89,75 s. Booking robot4 port1 đổi đích 95→60%, kết thúc 1128 s thay vì dự kiến 2425,5 s. Slot có thể về sạc được mở lại; task27 được robot12 giao lúc1085 s, hoàn thành1154 s. Vẫn có booking95% được tạo khi không có backlog (robot11 lúc135 s); không tuyên bố đã giải quyết mọi nút thắt.

## Kiểm tra và phần còn lại

76 test optimizer/simulator pass, gồm ba test mới về partial-charge khi có backlog, full-charge khi không có backlog và giữ calendar rejection. Các assertion cũ không nới. Full pytest đang chạy; chưa coi bước này nghiệm thu toàn hệ thống.

Kế hoạch mới khóa trước chạy ở `runs/readiness_after_charge_20261013/plan.json`: 198 episode, gồm 5 seed validation cho S2/S4/S6 và các ca mới. Ngưỡng/holdout giữ nguyên; checkpoint có nhãn train trên source cũ. Đo latency tuần tự sau khi full pytest kết thúc. Gặp lỗi mới khác phạm vi đang xử lý thì dừng báo theo mục D. Giai đoạn >6 giờ vẫn phải hỏi trước. Chưa chạy seed xác nhận3000–3004, chưa mở rộng full14.
