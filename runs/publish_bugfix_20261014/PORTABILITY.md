# Kiểm chứng gói demo đã đóng gói

Ngày 14/10/2026, Windows/Python 3.12.14, thư mục sạch khác với repo root, chỉ sao chép các file dự định xuất bản; không có .git, node_modules cũ, study gốc hoặc model ngoài gói. Venv mới có include-system-site-packages=false; kiểm tra source và các dependency được import bên trong gói. Đây là kiểm chứng ở thư mục khác trên cùng máy, chưa phải máy vật lý thứ hai.

- Cài requirements.txt vào venv sạch. PyTorch 2.6.0 dùng wheel PyPI chính thức đã xác minh SHA-256 do tải trực tiếp bị chậm; không sao chép site-packages từ Conda. Các dependency còn lại cài bằng pip theo requirements.
- npm ci và npm run build: TypeScript/Vite đạt; Node 24.12.0, Vite 7.3.6. Có cảnh báo bundle >500 kB, không phải lỗi build.
- 101 test đạt, 6 test integration/slow không chọn, 0 failure/error. Các file: tests/test_demo3d.py, test_demo3d_recordings.py, test_optimizer.py, test_simulator.py, test_stale_fallback.py. Thời gian và dependency trong portability.json.
- Heuristic/MAPPO/DQN record-only: mỗi phương pháp chạy S4/seed1001 với horizon 5 s, disturbances=false, 11 frame, complete, không có fallback. Đây là smoke test ngắn, không đối chiếu KPI với study dài.
- Heuristic LIVE chạy 5 s; không cần checkpoint. Hai model thật tải từ lock tương đối bên trong demo3d/models.
- Frontend được HTTP server phục vụ. API thấy mẫu complete; đọc 120 frame ở 0 và 1800 s, 3 frame cuối ở 3599 s. Tổng 7.201 frame, cuối 3.600 s. Lặp lại sau shutdown/restart server vẫn đạt; gzip giải nén vào cache ignored.
- Hash gốc/xuất của hai model và 57 báo cáo khớp provenance/manifest; nguồn gốc không đổi. Gzip giải nén khớp SHA-256 raw. Quét đường dẫn cá nhân/tên máy trong mẫu, metadata, báo cáo và text notebook không thấy thông tin còn sót. C:/file trong XML là đầu vào test traversal cố ý, không phải đường dẫn cá nhân.

Chưa kiểm chứng trên máy vật lý thứ hai, chưa đo FPS/render bằng trình duyệt của bản build đóng gói mới, không train hoặc chạy lại study dài. Replay đầy đủ là phiên lịch sử, source hash ở header giữ nguyên; source hash hiện tại xem portability.json. Chỉ có kiểm tra ngắn mới áp dụng cho source đóng gói hiện tại; các báo cáo cũ giữ điều kiện/source lịch sử.

Các lệnh thực hiện từ root bản sao sạch:

```powershell
npm ci # trong demo3d/web
npm run build # trong demo3d/web
.\.venv\Scripts\python.exe -X utf8 scripts/verify_publication.py
.\.venv\Scripts\python.exe -m pytest tests/test_demo3d.py tests/test_demo3d_recordings.py tests/test_optimizer.py tests/test_simulator.py tests/test_stale_fallback.py -q -m 'not slow and not integration'
```

Không đưa venv, node_modules, dist, wheel tải về, cache giải nén, smoke replay hoặc log thô của thư mục kiểm chứng lên Git. Bằng chứng chọn lọc là PORTABILITY.md và portability.json.
