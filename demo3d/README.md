# Demo 3D FleetRL — LIVE và replay

Demo dùng simulator, planner, năng lượng và KPI của FleetRL. Three.js hiển thị trạng thái. LIVE cũ vẫn được giữ.

## Clone, cài và chạy trên Windows

Cần Git, Python 3.12 và Node.js 22.12 trở lên (khuyến nghị LTS). Không cần Conda, ổ D: hoặc study của tác giả. Chọn thư mục bất kỳ, chạy PowerShell:

```powershell
git clone --branch run-and-demo https://github.com/xuantrung-DA/fleetrl.git
cd .\fleetrl
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
cd .\demo3d\web
npm ci
npm run build
cd ..\..
.\.venv\Scripts\python.exe -X utf8 -m fleetrl demo3d --study demo3d/models --scenario S4 --seed 1001 --port 8765
```

Nếu không có `py`, dùng đường dẫn Python 3.12 đã cài thay cho `py -3.12`. Mở [http://127.0.0.1:8765](http://127.0.0.1:8765). Server chỉ lắng nghe loopback. Hướng dẫn clone áp dụng sau khi gói này đã được commit và push.

## Ba cách xem

- **LIVE:** chọn phương pháp, kịch bản, seed; bấm **Bắt đầu**. Simulator chạy đồng thời với viewer. Có thể thêm đơn/chặn đường; phiên can thiệp được đánh dấu, KPI chỉ dùng quan sát. Heuristic không cần checkpoint.
- **Chạy trước và lưu:** simulator đánh giá controller/policy đã có, ghi mỗi tick 0,5 giây mô phỏng, không chờ thời gian thực. Đây không phải train. Khi xong, replay mở tạm dừng; bấm **Phát**.
- **Replay có sẵn:** chọn `mappo_s4_1001_full` trong **Phiên đã lưu**, bấm **Mở replay**. Không cần chạy mô phỏng trước. Có phát/tạm dừng/tốc độ/tua. Mở replay dừng phiên LIVE đang chạy; replay không hỗ trợ can thiệp kho.

## Replay mẫu đầy đủ dạng gzip

[Nguồn và checksum](samples/mappo_s4_1001_full/provenance.json): MAPPO dispatch, S4, seed tape 1001, 3.600 giây mô phỏng (60 phút), 7.201 frame, trạng thái `complete`, không can thiệp. Bản ghi thật từ phiên `748f38c95f4e`. Source hash trong header là phiên bản lúc ghi. Header giữ trạng thái lúc bắt đầu; trạng thái kết thúc ở `recording.json`.

`replay.jsonl.gz` khoảng 30,4 MiB, giữ toàn bộ byte JSONL gốc. Lần mở đầu server kiểm tra hash, giải nén vào `runs/demo3d/_replay_cache/` (khoảng 254 MiB đĩa) rồi lập chỉ mục thời gian; giao diện có thể phải chờ. Lần sau dùng cache; restart vẫn tìm được bản ghi và lập lại chỉ mục. Cache/phiên mới không đưa lên Git. Không sửa cache thủ công; nếu cache hỏng, dừng server rồi tự xóa cache để tạo lại.

Viewer tải 120 frame mỗi đoạn (khoảng 60 giây mô phỏng), API giới hạn 240 frame. Server giữ chỉ mục tối đa hai bản ghi, đọc frame bằng byte-offset, không nạp toàn bộ replay vào RAM. Replay tách thuật toán khỏi render; FPS vẫn phụ thuộc máy và cảnh.

## Hai model được đóng gói

`demo3d/models/` chứa MAPPO và DQN CP-SAT, train seed 11, mỗi model có ZIP, sidecar metadata và lock SHA-256. Lock dùng đường dẫn tương đối với thư mục chứa lock. [Provenance](models/provenance.json) ghi hash gốc/hash xuất và payload không đổi. MAPPO được chuẩn hóa đường dẫn lineage trong metadata: hash ZIP thay đổi, trọng số và payload nguyên vẹn. Không có replay buffer; gói phục vụ suy luận, không phải resume train.

Ba phương pháp `heuristic`, `mappo_dispatch`, `dqn_cpsat` dùng được với lệnh trên. Thiếu model/lock, sai checksum hoặc metadata sẽ tắt phương pháp tương ứng và hiển thị lý do. Heuristic vẫn dùng được. Có thể dùng study cục bộ bằng `--study runs/study14`; không đóng gói toàn bộ study.

## Dữ liệu và lỗi thường gặp

- Thiếu frontend: chạy `npm ci`, `npm run build` trong `demo3d/web`.
- Cổng bận: đổi `--port 8765` sang cổng khác.
- Gzip hỏng/sai hash: API báo lỗi; lấy lại file đúng từ repository.
- Cache cần ít nhất 254 MiB đĩa; đổi nơi ghi bằng `--output <thư-mục>` nếu cần.
- Phiên mới ở `runs/demo3d/<session-id>/replay.jsonl` và `recording.json`. Chỉ `complete` hoặc `stopped` được mở; `stopped` không phải hoàn tất. Bản ghi cũ thiếu metadata có nhãn legacy, không phải bằng chứng test hoàn chỉnh.
- Replay study có schema khác dù cùng tên `.jsonl.gz`, không đọc trực tiếp với viewer `fleetrl-viewer/1`, không tự suy diễn tick thiếu.

API `/api/recordings` liệt kê mẫu và output; `/api/recording-window?id=<id>&start=<giây>&limit=120` đọc một đoạn. Viewer gốc trong `demo3d/original/` vẫn được giữ.

## Báo cáo và notebook

[Bản xuất báo cáo sạch](../runs/publish_bugfix_20261014/README.md) giữ cải thiện, ca không đạt, hồi quy và giới hạn timeout. Mỗi báo cáo là bằng chứng của source/config lịch sử trong đó, không tự chứng minh source mới. [Notebook tiếng Việt](../notebooks/fleetrl_before_after_fixes_vi.ipynb) chứa sẵn bảng/hình để đọc mà không chạy lại.

Script tạo lại notebook cần 273 artifact gốc; nhiều file không nằm trong gói. Các cell đọc `runs/` lịch sử chỉ chạy khi có bộ dữ liệu gốc tương ứng. Bản xuất đã chuẩn hóa có hash riêng; không thay hash lịch sử của notebook.

`complete` nghĩa là phiên chạy hết horizon; không có nghĩa mọi đơn hàng đều hoàn thành. MiB = 1.048.576 byte; giây ở đây là thời gian mô phỏng.
