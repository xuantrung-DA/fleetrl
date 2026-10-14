# Sửa admission cho robot đang pause

Đã được người dùng cho phép. Source `ed607d948dea6e37bf2718c469e603476811b2eec8819f0c1dca4837beda6242`. Robot còn pause không được lập ứng viên hoặc chặn dispatch bằng ưu tiên khẩn năng lượng. Hết pause được xét lại; guard simulator giữ nguyên.

82 test liên quan đạt. Cả heuristic, fixed_cpsat và MAPPO nhận task lúc t=5, hoàn thành 1/1; safety=0. Fixed_cpsat chưa xác minh tỷ lệ timeout (null). MAPPO dùng **checkpoint train trên source cũ**, không train/tune.

Ca lỗi cũ vẫn lưu tại `../readiness_charging_final_20261013/HANDOFF_PAUSE_DIAGNOSIS.json`. Pilot mới ở `../readiness_pause_final_20261013`; không trộn với Study14.

Toàn bộ 202 test đạt (206,43 s), Ruff check/format đạt, 192 checksum xác minh và git diff --check đạt.
