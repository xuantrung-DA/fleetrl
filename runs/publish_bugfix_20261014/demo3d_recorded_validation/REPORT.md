# Kiểm chứng chế độ chạy trước và replay — 14/10/2026

## Thay đổi

- Giữ LIVE, các điều khiển và bản ghi cũ.
- Thêm nút Chạy trước và lưu: một episode dùng controller/checkpoint hiện có,
  lưu từng tick 0,5 s, bỏ chờ thời gian thực. Không train/tune.
- Thư viện phiên đã lưu đọc trên đĩa, hỗ trợ restart server.
- Replay tải cửa sổ 120 frame, API giới hạn 240; server cache tối đa hai chỉ mục
  thời gian/byte-offset. Nút Xem lại phiên cũng dùng reader theo đoạn.
- Trạng thái mới lưu ở recording.json. Bản ghi cũ có nhãn legacy,
  không tự nhận là episode hoàn tất. Bản ghi đang chạy/lỗi không được liệt kê như complete.

## Kiểm tra

- tests/test_demo3d_recordings.py và tests/test_demo3d.py: 12 passed,
  junit tại pytest_final.xml. Test record-only dùng episode 5 s với heuristic;
  không ngủ theo wall-clock, ghi đủ 11 frame từ 0 tới 5 s, status complete.
  Kiểm tra thêm nhánh LIVE vẫn có pacing.
- Lần chạy pytest đầu lỗi thư mục Temp của sandbox; chuyển basetemp vào workspace,
  giữ XML ca lỗi tại pytest.xml và kết quả cuối tại pytest_final.xml.
- Ruff check/format của demo3d và file test mới qua.
- npm run build: TypeScript và Vite qua; còn warning bundle >500 kB.
- Trình duyệt mở bản ghi MAPPO/S4/1001 cũ 264,2 MiB, 7.201 frame,
  tua tới 1.200 s, phát 8x qua nhiều cửa sổ, pause và tua tới 3.599 s.
  Không thấy lỗi dữ liệu/hiển thị trong các thao tác này.
- API đọc đầu/giữa/cuối: api_evidence.json. Mỗi request tối đa 120 frame mặc định;
  cuối phiên có ít frame hơn, end_s=3600. Latency trong evidence là cache đã lập,
  không phải benchmark lập chỉ mục lần đầu.

## Giới hạn

- Chưa chạy episode 3.600 s mới bằng nút record-only; ca toàn horizon trong UI
  là bản ghi có sẵn, không phải kết quả mô phỏng mới.
- Không tuyên bố hết giật trên mọi máy; chi phí đồ họa của viewer vẫn tồn tại.
- Chỉ hỗ trợ schema fleetrl-viewer/1; replay bài test gzip theo bước quyết định
  không được nội suy thành dữ liệu tick vật lý giả.
- Source hash của package thay đổi vì thêm tính năng demo; không thay physics,
  reward, metrics, optimizer hoặc kết quả test cũ trong đợt này.
