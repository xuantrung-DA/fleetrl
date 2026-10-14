# Chẩn đoán trước khi sửa source

Ngày 13/10/2026. Evidence chính: `before_diagnostic_v2/`, seed 1000, 600 s,
10 robot, SoC đầu 99%, không disturbances. Không sử dụng seed test.
`before_comparison/` có 18 ca S1/S2/S4 × seed 1000/1001 × ba controller,
mỗi ca 1200 s; S4 giữ burst 900–1500 s nên chỉ quan sát 300 s đầu burst.

| Tải | Controller | Đường rảnh m | Chu kỳ goal A→B→A | consumed Wh | Yield/deadlock events |
|---|---|---:|---:|---:|---:|
| Không đơn | heuristic | 0 | 0 | 20.00 | 0 |
| Không đơn | fixed_cpsat | 1553 | 822 | 85.15 | 0 |
| Không đơn | mappo_dispatch | 694 | 233 | 56.99 | 0 |
| 12 đơn/giờ | heuristic | 10 | 0 | 23.11 | 1 |
| 12 đơn/giờ | fixed_cpsat | 853 | 530 | 63.76 | 1 |
| 12 đơn/giờ | mappo_dispatch | 1169 | 190 | 76.13 | 1 |

## Nguyên nhân xác nhận

1. `optimizer.py:535-545` tính wait cho 5 s; `optimizer.py:572-604`
   tính reposition chỉ theo thời gian hành trình; `optimizer.py:623-633`
   tính duration/energy trực tiếp rồi chia N cho cả hai. Chia N nhất quán;
   lỗi là khoảng thời gian so sánh khác nhau. Robot 0, tick 0, zero fixed:
   wait tại (29,1) cost 0.00017; reposition (30,1) cost 0.000024.
   Không đơn, nhu cầu bốn vùng bằng 0, đi một cạnh vẫn rẻ hơn đứng yên.
2. `optimizer.py:546-567` tiếp tục cung cấp các điểm đỗ khác sau mỗi lần
   robot trở về idle, không có kiểm tra lợi ích/cooldown. `goals.jsonl` ghi
   robot 0 của fixed: (30,1) tại 0 s, (29,1) tại 5 s, (30,1) tại 10 s.
   Nguồn luôn là `Simulator.submit_plan`, trước sửa `simulator.py:498`.
3. MAPPO sử dụng cùng candidate generator (`mappo_dispatch.py:37-75`)
   và chọn ứng viên parking không có guard lợi ích. Ví dụ robot 6:
   (30,28) tại 0 s, (28,28) tại 5 s, (30,28) tại 10 s.
4. Ca không đơn có năng lượng nền chính xác 20 Wh = 10×12×600/3600.
   Phần vượt ở fixed/MAPPO là chuyển động thật, không phải renderer.

## Các giả thuyết

- a: xác nhận tái phát ứng viên sau arrival; bác bỏ ý rằng goal đang đi
  bị đổi mỗi 5 s: candidate generator chỉ nhận idle và submit guard từ chối
  busy. Không cần sửa cam kết hành trình hiện có.
- b: xác nhận wait thua reposition do duration khác nhau. Dao động nhu cầu
  không phải điều kiện cần: lỗi xuất hiện khi demand=0; trailing demand vẫn
  có thể gây đổi vùng trong ca có đơn, cần hysteresis dùng dữ liệu đã đến.
- c/d: loại trừ như nguyên nhân cần của lỗi không đơn (0 yield/deadlock).
  Ca đơn thưa có một endpoint yield; không thể giải thích hàng trăm chu kỳ.
- e: arrival xóa goal/path và trả idle đúng; việc đặt goal tiếp theo do plan.
  Không có bằng chứng lỗi detour/arrival gây các chu kỳ parking trong ca zero.
- f: wait không đặt goal, idle không có path/move_to trong ca zero.
- g: heuristic không oscillate ở hai ca chẩn đoán; fixed và MAPPO bị ảnh hưởng.
  Fallback dùng heuristic; kiểm thêm bằng test ép solver UNKNOWN.
- h: không tìm thấy nguyên nhân ngoài các giả thuyết yêu cầu phải dừng.

## Hướng sửa đã xác định

So wait/reposition trên ít nhất một decision interval; lọc parking không
có lợi tại tầng candidate dùng chung và revalidate tại simulator. Cho phép
đổi vùng theo giảm lệch đội có ngưỡng/cooldown; giữ parking destinations
đang cam kết trong occupied set. Khi nhường xong giữ bay mới; không quay về
goal parking cũ sau deadlock diversion. Khi không có việc ngoài parking,
đưa robot về bay theo khoảng cách rồi tọa độ, giữ task/charge ưu tiên.
Không đổi observation shape, reward, metrics hay các báo cáo cũ.

Các số trên không chứng minh toàn bộ chênh lệch energy trong study14 do lỗi
này: study khác horizon, seed, initial state và có task/charging. Đối chiếu
validation trước/sau riêng trong thư mục mới.
