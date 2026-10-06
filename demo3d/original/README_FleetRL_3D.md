# FleetRL — lớp hiển thị 3D và replay

## (a) Kế hoạch, giả định và câu hỏi mở

Đọc map/snapshot → kiểm tra hợp đồng → hiển thị Three.js → phát lại theo giây mô phỏng. Không gọi hoặc viết lại planner, năng lượng, reward, KPI. Các fixture demo chỉ là dữ liệu minh họa cố định.

- Map A/B lấy **đúng `make_map` trong maps.py đính kèm**, gồm 40×30 ô, 12 pickup, 4 dropoff, 24 parking, hai trạm/bốn cổng. Không cần map A giả định vì source đã có. Map B đổi bands, một số pickup và vị trí cổng theo source.
- Demo dùng cấu hình cổng 600 W. Muốn S6, nạp map được export với `power_w=300`; viewer không tự suy tên scenario.
- Mapping `(x,y)` → `(x+0.5, height, y+0.5)`, mỗi cell là 1 m. Biên sàn là x=0…40, z=0…30. Zone = int(x≥20)+2*int(y≥15).
- Kích thước robot, màu, kệ nhiều tầng, chiều cao khay là lựa chọn hiển thị; không thay collision geometry của engine.
- Fixture giữ pin 70%/35%, KPI trống, solver ghi MOCK — NO SOLVER. Không phải kết quả vận hành hoặc mô hình năng lượng. Nạp replay thật bỏ nhãn mock chỉ khi `mock` không true.
- Muốn chuyển động chính xác cần `move_to`, `move_remaining_s`, `edge_duration_s`, và frame mỗi tick 0.5 s. Adapter lấy edge duration từ `1/speed_mps` của robot nguồn. Nếu engine có thời gian cạnh thực khác do rounding/slowdown, exporter phải truyền `edge_duration_s` thực thay vì suy từ speed. Đây là điểm cần xác nhận trong `_move`, chưa được cung cấp.
- Thiếu motion fields: hiển thị cell snapshot, không nội suy từ tương lai và không tự tìm đường. Replay nhảy nhiều cạnh giữa hai frame bị từ chối; cần xuất dày hơn.
- `route` là đường dự kiến đã biết, **không khẳng định đã đặt chỗ**. `reservations` riêng từ planner. Không đoán quyền hành lang từ màu hoặc route.
- Heatmap, danh sách corridor, hướng đang được phép và ô tránh phải do dữ liệu quan sát/exporter cung cấp. Snapshot hiện tại chưa có các trường này; thiếu thì không vẽ. Không tự tạo hành lang hẹp giả trên map thật.
- Lịch start/end là **kế hoạch** đã biết tại frame, không phải kết quả thực hiện tương lai. KPI không có hiện `—`, tuyệt đối không tính bằng animation hoặc lấy KPI tổng kết episode.
- LIVE là API nhận frame trực tiếp từ host (`FleetRL.pushLive`), không có websocket/server sẵn. File replay luôn hiện REPLAY. Demo đồng thời hiện REPLAY + MOCK DATA.

Câu hỏi mở, không chặn việc sử dụng: hàm `evaluation.snapshot_record` thực tế serialize ra sao? `_move` xử lý phần lẻ tick thế nào? API export reservations/corridor grants đang có format nào? Cần `evaluation.py`, `simulator.py`/`planner.py` hoặc một JSONL thật để kiểm chứng tích hợp toàn phần; không khẳng định đã tương thích với format chưa được cung cấp.

## (b) Mã hoàn chỉnh

- `fleetrl_viewer.html`: toàn bộ CSS/JS, procedural geometry, UI, fixture demo A/B.
- `export_fleetrl_replay.py`: adapter Python thuần stdlib, dùng được Python 3.12.
- `map_A.json`, `map_B.json`: map trích đúng logic maps.py, cấu hình 600 W.
- `mock_replay_20.json`: fixture 20 robot, 40 giây/81 frame; metadata mock=true.
- `test_contract.cjs`, `test_adapter.py`: kiểm tra hợp đồng dữ liệu và adapter.

Three.js cố định **0.160.1**, chỉ tải một script từ cdnjs:
https://cdnjs.cloudflare.com/ajax/libs/three.js/0.160.1/three.min.js
Không texture/asset/font ngoài, không bundler, không server. Một file HTML chứa toàn bộ mã ứng dụng nhưng cần Internet để tải Three.js. Không phải file offline tuyệt đối.

Kệ, carton và vạch dùng InstancedMesh; robot AMR dùng geometry gộp và dùng chung theo loại. Chất lượng Thấp/Vừa/Cao có pixelRatio tối đa 1/1.5/2; bóng DirectionalLight tắt/1024/2048 tương ứng. Có contact shadow và texture CanvasTexture, không tải asset ngoài. Camera orbit tự viết bằng phép xoay camera (không có vật lý robot).
Tài liệu API tham khảo: https://threejs.org/docs/pages/WebGLRenderer.html và https://archive.threejs.org/docs/api/en/objects/InstancedMesh.html .

## (c) Cách chạy

1. Giải nén; mở `fleetrl_viewer.html` bằng Chrome hoặc Edge, có Internet.
2. Nhấn Phát. Mặc định map A, 15 robot, MOCK DATA. Chọn A/B và 10/15/20 rồi nhấn Demo mới để áp dụng.
3. Kéo để orbit, lăn chuột để zoom, Shift+kéo để pan. Chọn top-down hoặc theo robot.
4. Click robot hoặc chọn từ danh sách. Nhãn hiển thị ID, L/M/H, IDLE/WORK/CHARGE/HOLD, %, thanh pin và kg nếu có tải.
5. Nạp replay có `{map,frames,mock}` để đổi cả map/dữ liệu; hoặc nạp map trước rồi JSONL mỗi dòng một frame.
6. Kéo thanh tua; đổi 0.5×/1×/2×/4×/8× chỉ đổi tốc độ đọc dữ liệu. Pause dừng đồng hồ mô phỏng; camera vẫn dùng được.
7. Bật grid/labels/routes/reservations/heatmap. Liên kết P→D của task được chọn nằm cao hơn kệ và chỉ biểu thị hai endpoint, không phải đường thực thi.

Cổng NẠP màu xanh; BỊ CHIẾM màu vàng; TRỐNG màu xám. Cả ba có chữ trong panel. LED và vòng sáng sạc chỉ là hiệu ứng dựa trên t.
Các trạng thái idle→IDLE; to_pickup/pickup/to_dropoff/dropoff/reposition→WORK; to_charge_queue…charge_egress→CHARGE; recovering/hold→HOLD. Mã gốc luôn có trong panel.

## (d) Nối với FleetRL

CLI, tại thư mục gốc project:

```powershell
.\.venv\Scripts\python.exe path\export_fleetrl_replay.py runs\raw.jsonl --map runs\map.json -o runs\replay.json
```

Đầu vào hỗ trợ: list record, `{frames:[...]}`, JSONL hoặc một Snapshot dict. Record có thể là `{snapshot: {...}, kpi: {...}, decision: {...}}`. Tham số `--map` nhận JSON từ `save_map`. Không tự nhận bất kỳ wrapper lạ nào; thiếu trường báo record số mấy và dừng.

Tích hợp trực tiếp tại chỗ recorder có Snapshot dataclass:

```python
import json
from export_fleetrl_replay import map_to_viewer, snapshot_to_frame

# snapshot, kpi_now, decision_plan do engine/controller hiện có cung cấp.
# Lưu mỗi tick 0.5 s, SAU khi engine tạo trạng thái nhất quán.
frame = snapshot_to_frame(snapshot, kpi=kpi_now, decision=decision_plan)
replay_stream.write(json.dumps(frame, ensure_ascii=False, allow_nan=False) + "\n")
# Xuất một lần (hoặc khi thay map cho session mới):
with open("map.json", "w", encoding="utf-8") as f:
    json.dump(map_to_viewer(snapshot.warehouse), f, ensure_ascii=False)
```

Tên `snapshot`, `kpi_now`, `decision_plan`, `replay_stream` là biến tích hợp minh họa, không phải API mới của simulator. Chỉ ghi decision ở tick có quyết định hoặc để null; không lặp cùng quyết định mỗi tick nếu muốn log gọn.

Mapping adapter:

| Source | Viewer |
|---|---|
| sim_time_s / observed_at_s | t / observed_at |
| Robot.kind / position / path | type / cell / route |
| battery_wh / capacity_wh | soc (đổi đơn vị) |
| move_to / move_remaining_s | giữ nguyên |
| speed_mps | edge_duration_s=1/speed_mps nếu chưa có duration thực |
| Task.weight_kg / created_s / deadline_s / status | kg / created / deadline / state |
| Port.position / station_id / compatible_kinds | cell / station / compatible |
| ChargeBooking.start_s / end_s / requested_at_s | start / end / known_at |
| DecisionPlan.profile_id | decision.profile |
| metrics tại tick | kpi nguyên bản |

`Snapshot.ports` chỉ chứa config; adapter xác định occupancy từ robot.position tại ô cổng, và NẠP chỉ khi status=charging. Nếu recorder đã có occupied_by/charging_robot thì giữ nguyên. Booking không chứng minh đang nạp. `queue_robot_ids` được liệt kê theo occupancy ô chờ và trạng thái waiting_charge, không bịa thứ tự phục vụ. Các ô chờ dùng chung được giữ theo station.

Các extension đều tùy chọn, chỉ chứa thông tin đã biết tại t:

```json
{
  "reservations": [{"robot_id": 3, "cell": [4, 5]}],
  "heatmap": [{"cell": [4, 5], "value": 0.7}],
  "corridors": [{
    "cells": [[4, 5], [5, 5], [6, 5]],
    "direction": [[4, 5], [5, 5]],
    "passing_cells": [[3, 5], [7, 5]]
  }]
}
```

`heatmap.value` phải được upstream cung cấp chuẩn hóa 0–1, viewer chỉ ánh xạ màu. Không tự tính số lần kẹt. `direction=null` nếu chưa biết/không cấp hướng. Không điền future event tape vào extras. Danh sách reservation chỉ là snapshot hiện tại; không đưa các reservation chưa biết tại thời điểm frame.

LIVE, gọi trong cùng trang từ integration tin cậy:

```javascript
FleetRL.pushLive(firstFrame, mapJson);
FleetRL.pushLive(nextFrame);
```

LIVE không dùng tốc độ UI, không ngoại suy sau frame mới nhất; tua/play bị khóa. Nạp replay/demo trở lại sẽ mở khóa. Không cài transport/network listener tự động.

## (e) Nâng cấp hình ảnh và kết quả kiểm tra

Bản cập nhật 30/09/2026 gồm kệ pallet xanh–cam, carton với seed cố định, sàn epoxy, bốn zone nhạt, ba mẫu AMR, đèn sạc, nhãn gọn có chống chồng, đường nối nhãn bị dịch, toolbar riêng, chú giải thu gọn và reset camera bằng R. Bật Nhãn đầy đủ hoặc Xuyên kệ theo nhu cầu. Điều chỉnh Thấp/Vừa/Cao chỉ đổi renderer, không sửa frame.

Đã sửa lỗi JSONL: khi dữ liệu là một mảng frame, lấy map đã nạp thay vì nhầm Array.prototype.map với map JSON. Schema không đổi.

| Hạng mục | Kết quả |
|---|---|
| Vị trí nội suy trước/sau nâng cấp | 21.870 vị trí đối chiếu khớp |
| Kệ và carton trong ô walls | Đã kiểm vertex của instance, không có điểm lấn ô trống |
| Robot xoay trong footprint | Geometry L/M/H có bán kính ngang tối đa 0.360/0.440/0.472 m |
| 20 nhãn, góc mặc định | Không chồng nhau ở 1600×1000 và 1366×768 |
| Nạp map JSON rồi replay JSONL | Kiểm thử trình duyệt đạt |
| LIVE nhận liên tiếp hai frame, khóa play/seek | Kiểm thử trình duyệt đạt |
| Chuyển LIVE sang demo/REPLAY | Kiểm thử trình duyệt đạt |
| Đổi chất lượng | Frame, KPI và giờ mô phỏng không thay đổi khi đang pause |
| File sai JSON/JSONL | Có thông báo lỗi, không có uncaught browser error |
| Pause, đổi tốc độ | Đã kiểm: pause giữ t, tốc độ không sửa dữ liệu |
| Draw calls demo 20 robot mức Vừa | 173; demo có block ghi nhận 177 |
| Mục tiêu 60 FPS RTX 4050 | Chưa đo trên phần cứng đó; không khẳng định đã đạt |
| Replay thật của project | Vẫn cần evaluation.py hoặc replay mẫu để kiểm tích hợp đầy đủ |

Kiểm thử trình duyệt dùng Chromium headless với SwiftShader (render phần mềm). Không dùng FPS của môi trường này để suy ra FPS RTX 4050. Mock vẫn không cung cấp kết quả KPI thực nghiệm.

Chạy kiểm tra hợp đồng và adapter:

```powershell
node test_contract.cjs
python test_adapter.py
```

Nghiệm thu FPS trên máy: chọn 20 robot, chất lượng Vừa; mở Chú giải & hiệu năng, phát replay ở tab đang hoạt động. Ghi FPS sau khi shader tải xong, độ phân giải và tốc độ phát. Thử cả A/B, mọi overlay, pause/tua và nạp replay thật.
