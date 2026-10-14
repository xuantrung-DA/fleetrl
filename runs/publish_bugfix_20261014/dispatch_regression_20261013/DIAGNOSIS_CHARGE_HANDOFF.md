# Chẩn đoán bổ sung trước khi sửa handoff sạc

Bản task_handoff_v2 đã cho task preempt parking, giữ cạnh và qua test mới.
S4/fixed mean pending 6.4 (baseline 6.6), nhưng heuristic S2/S6 còn lệch
nhẹ hơn ngưỡng completion/pending đã đặt; không nới ngưỡng.

## Ca S6/heuristic/1001, cùng tape

`baseline/`, `parking_fix/`, `task_handoff_v2/`: tasks.json, parking.json,
charging.json và eligibility_replay.json. Replay đã khớp commit và các
counter physical, không dựa vào suy đoán renderer.

- Baseline/parking_fix: robot 11 nhận task 32 lúc 80 s, hoàn thành 142.5 s,
  được giao sạc port 2 lúc 145 s; robot 4 vẫn nhận task 28 lúc 315 s và
  robot 10 nhận task 16 lúc 320 s. Tổng pending=20.
- Task_handoff_v2: robot 11 nhận task 32 ngay lúc 65 s khi đang reposition,
  hoàn thành **sớm hơn**, 129 s. Nhưng tại 130 s nó bị endpoint clearance
  đưa về parking (21,28); sạc vẫn bị status reposition chặn.
- Robot 14 xong task lúc 130.5 s, tới parking (11,1) và chiếm port 2 lúc
  150 s với SoC 29.9%. Robot 11 chỉ sạc lúc 155 s, phải lấy port 1, SoC
  29.6%. Hai cổng còn lại đã có booking từ t=0. Do port 1 bị chiếm sớm,
  task 28/16 không còn được giao; tổng pending=22.

Đây là cơ chế cụ thể gây lệch sạc: **parking/yield không cho sạc theo nhu
cầu năng lượng preempt**, mặc dù task đã được phép. Không phải hệ số cost
charge bị đổi. Task làm xong sớm hơn vẫn có thể làm thứ tự admission cổng
xấu đi nếu phải hoàn thành chuyến về parking trước khi nhận sạc.

## Hướng sửa tối thiểu

Cho sạc bắt buộc theo threshold/energy urgency được thay goal parking,
giữ nguyên target, giá cost/gain và thứ tự ưu tiên sạc hiện có. Không cung
cấp charge elective cho mọi robot đang parking khỏe mạnh. Cạnh đang đi
vẫn được giữ và hoàn tất; candidate/guard tính từ destination của cạnh,
cộng thời gian và auxiliary còn lại. Dùng guard lịch cổng/queue hiện có.
Không thay reward/observation/map, không bỏ no-demand stability.

Sau sửa phải chạy lại toàn bộ 5 seed × S2/S4/S6 × ba method; không kết
luận chỉ từ ca trên.
