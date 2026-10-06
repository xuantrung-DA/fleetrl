# FleetRL Live 3D

Viewer Three.js chạy trên máy cục bộ và nhận trạng thái trực tiếp từ simulator FleetRL qua SSE.
Viewer không tính planner, năng lượng, reward hay KPI.

LIVE sử dụng bộ đệm 1 giây mô phỏng và cập nhật vị trí theo từng khung hình render.
Chuyển động chỉ tiếp tục đến thời điểm đã nhận từ simulator; khi thiếu dữ liệu,
viewer chờ bộ đệm thay vì đoán đường đi. Tạm dừng và tốc độ áp dụng cho cả phát hình.

Giao diện, kệ, robot, ánh sáng, camera và nhãn được lấy từ
`C:\Users\phath\Desktop\FleetRL_3D-view\fleetrl_viewer.html`.
Toàn bộ file đầu vào được copy nguyên bản vào `demo3d/original/` để đối chiếu.
`web/src/original-viewer.js` chứa mã viewer gốc, chuyển Three.js từ CDN sang import
cục bộ và bổ sung giới hạn bộ nhớ LIVE. `web/src/main.ts` nối viewer với API/SSE
và thêm bảng điều khiển phiên ở sidebar. CSS và bố cục gốc nằm trong `web/index.html`.

Khi mở trang, kho hiển thị ở trạng thái chờ. Chọn phương pháp/kịch bản rồi bấm **Bắt đầu** để chạy LIVE. Map và số robot theo kịch bản. Các nút nạp file và demo MOCK đã được bỏ. Thanh phát/tua chỉ hiện khi bấm **Xem lại phiên**.

## Cài và build frontend

```powershell
cd .\demo3d\web
npm install
npm run build
cd ..\..
```

## Chạy demo

Từ thư mục gốc repository và môi trường Python đã dùng để train:

```powershell
python -m fleetrl demo3d --study runs/study14 --scenario S4 --seed 2000 --port 8765
```

Mở `http://127.0.0.1:8765`. Server chỉ lắng nghe loopback. Các replay được lưu vào
`runs/demo3d/<session-id>/replay.jsonl`.

Ba phương pháp trình diễn là `heuristic`, `mappo_dispatch` và `dqn_cpsat`. MAPPO/DQN chỉ
được bật khi checkpoint đã khóa trong study còn tồn tại và checksum khớp. Đổi phương pháp
tạo phiên mới với cùng scenario/seed. Sau khi thêm đơn hoặc chặn/mở ô, giao diện đánh dấu
phiên đã can thiệp; KPI của phiên đó chỉ dùng để quan sát.
