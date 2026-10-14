# Bản xuất bằng chứng sửa lỗi FleetRL

57 file được chọn, giữ cả ca không đạt, hồi quy và timeout chưa đủ bằng chứng. Thư mục gốc không bị sửa. `manifest.json` ghi từng đường dẫn nguồn/xuất, hash gốc/hash xuất và dung lượng. Chỉ chuẩn hóa LF, đường dẫn cá nhân, hostname và cách serialize JSON/CSV; chỉ số và source hash lịch sử được giữ.

- `idle_parking_20261013`: đỗ, năng lượng, đối chứng mặc định/controlled.
- `dispatch_regression_20261013`: tồn đơn, handoff sạc, ablation và gate.
- `charging_access_20261013`, `paused_candidate_20261013`, `stale_fallback_20261013`: cơ chế lỗi và kiểm chứng.
- `endpoint_wait_fix_20261013`, `endpoint_wait_validation_20261013`: endpoint wait, plan/result/map và CI95.
- `readiness_nonlearning_fresh_20261013`: kết quả mới, traffic light, ca từng phải dừng và lịch sạc S6.
- `study14_current_source_testonly`: identity, CSV 270 episode, summary và giao thức đã dùng.
- `demo3d_recorded_validation`: kiểm chứng demo trước bước đóng gói; không thay cho kiểm chứng source đóng gói mới.

`study14_current_source_testonly/run.py` là bản lưu giao thức lịch sử, không chạy từ thư mục xuất: nó cần dữ liệu study gốc và source hash tương ứng. Không đóng gói toàn bộ model, replay, quyết định, log hay SQLite. Liên kết đến dữ liệu không được chọn cần bộ dữ liệu gốc; không coi chúng là file có sẵn sau clone.

Notebook tại `notebooks/fleetrl_before_after_fixes_vi.ipynb` chứa sẵn bảng/hình. Script tạo notebook cần 273 artifact gốc, không hỗ trợ tái tạo toàn bộ chỉ từ gói chọn lọc này. Không so sánh trực tiếp những hàng khác kịch bản/điều kiện; thiếu số liệu hoặc timeout chưa tái hiện không đồng nghĩa lỗi đã hết.
