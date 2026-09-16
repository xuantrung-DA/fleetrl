# Đối chiếu proposal với source bàn giao

Tài liệu gốc: `docs/PROPOSAL.md`, phiên bản 11/09/2026. Bảng dưới phân biệt **đường code đã có** với **bằng chứng hiệu quả**. “Có triển khai” không có nghĩa mọi mục tiêu định lượng trong proposal đã đạt; trạng thái kiểm thử chạy thực tế nằm trong báo cáo bàn giao. Đặc biệt, smoke train không chứng minh +5% throughput, p95≤250 ms hay chất lượng của 300.000 bước học.

| ID | Yêu cầu proposal | Điểm triển khai / bằng chứng cần đọc | Giới hạn cần giữ rõ |
| --- | --- | --- | --- |
| F01 | Cấu hình kho, đội, trạm; lưu/tải và từ chối dữ liệu lỗi | `config.py`, `maps.py`, `types.py`; validation config/map và hash trong manifest | Cấu hình experiment dùng YAML; map/tape dùng JSON. Mô hình là lưới 2D, không CAD/ROS. |
| F02 | Sinh kịch bản cùng seed, gồm đơn và sự cố | `scenario.py`, `EventTape`; seed + config sinh tape có hash | Tái lập tape khác với bit-identical solver dưới giới hạn wall-clock. Không cho policy thấy sự kiện tương lai. |
| F03 | Theo dõi robot, tải, pin, nhiệm vụ | `simulator.py`, `Robot`, log/frame trong `evaluation.py` | Trạng thái mô phỏng; không phải telemetry robot thật. |
| F04 | Giao việc đủ tải/pin, không trùng | `optimizer.py` candidate/model; `Simulator.submit_plan`; tests optimizer/simulator | Deadline là chi phí mềm. Đơn không khả thi giữ trong backlog và KPI. |
| F05 | Reposition robot rảnh tới điểm đỗ | `optimizer.py`, `maps.py`, planner và commit | Kế hoạch bảo thủ có thể HOLD khi route/ô đỗ bị chiếm. Không bảo đảm mọi robot đến được mọi mục tiêu trong giao thông dày. |
| F06 | Ưu tiên động theo deadline/tuổi | `optimizer.py`, observation shortlist | Profile là trọng số; không phải policy tự viết lại task đang mang hàng. |
| F07 | Đặt chỗ ô/cạnh theo thời gian | `planner.py`: `GridPlanner.timed_path`, `ReservationTable`, phát hiện narrow corridor và kiểm chiều; tests planner | Triển khai rolling route với giữ cả hai đầu cạnh khi đi và kiểm từng chuyển động; không phải bộ MAPF đầy đủ. Đường dự kiến không bảo đảm tồn tại khi mọi robot đều đang giữ chỗ. |
| F08 | Sự cố, đường vòng, deadlock/recovery | `simulator.py`, `planner.py`, event tape | Có thể phát hiện và ghi unresolved; không được tính HOLD vô thời hạn là hoàn tất nhiệm vụ. Không hỗ trợ chuyển hàng sang robot khác. |
| F09 | Sạc chủ động theo pin/việc/nhu cầu | `optimizer.py` advanced charging candidates và mục tiêu năng lượng/readiness | Là surrogate online; cần benchmark chứng minh lợi ích. |
| F10 | Cổng tương thích, lịch không chồng, hàng đợi và đến trễ | `optimizer.py` optional intervals/NoOverlap/Cumulative; simulator booking lifecycle | Cổng cố định công suất; không chia điện động. Route và tình trạng pin được kiểm lại khi commit/thực thi. |
| F11 | Partial charging đích 60/80/95% | `energy.py`, `ChargeBooking`, simulator nạp/nhả cổng | Mô hình tuyến tính hai đoạn, không nhiệt hay lão hóa pin. |
| F12 | Phối hợp sạc với lực lượng phục vụ | `optimizer.py` readiness/regional deficit objective | Mức lực lượng tối thiểu là chi phí mềm. Khả năng đáp ứng thực phụ thuộc route và demand. |
| F13 | Năng lượng cho task + tới cổng + dự phòng | `energy.py`, optimizer candidate checks, simulator execution | 300 s chờ đường và 300 s chờ sạc là giả định cấu hình; vi phạm phải thành cảnh báo/counter, không thành bảo đảm vật lý tuyệt đối. |
| F14 | Curriculum, PPO rollout/update, reward components | `rl.py`, `env.py`, configs; smoke train và tests RL | Đủ pipeline train không đồng nghĩa đã có policy tốt; chưa thay train dài bằng một smoke run. |
| F15 | Save/load, chọn checkpoint validation | `rl.py`: training/evaluation callbacks và SB3 serialization | Resume tiếp tục weights/optimizer, không phục hồi chính xác simulator, rollout buffer và mọi RNG. |
| F16 | Start/pause/reset, tốc độ phát | `ui.py` Pygame | Cửa sổ cần desktop. Headless không xác minh trải nghiệm tương tác trên mọi máy. |
| F17 | Dashboard map/robot/task/charge/KPI | `ui.py`, frame snapshots | Giao diện demo cục bộ; không web, đăng nhập, CRUD hay production WMS. |
| F18 | Tăng đơn/chặn đường/dừng robot có timestamp | `ui.py`, `Simulator.inject_event`, event tape | Đối chiếu baseline bằng tape đã lưu, không bấm sự cố tại các thời điểm khác nhau rồi so kết quả. |
| F19 | Log action/profile/cost/solver/fallback/reason | `DecisionPlan`, environment decision info, evaluation JSONL | Reason giải thích ứng viên/ràng buộc quan sát được; không tuyên bố giải thích nhân quả bên trong PPO. |
| F20 | Cùng tape, CSV/KPI/CI, baseline và ablation | `evaluation.py`, configs benchmark; B0/B1/H/A | Full protocol cần 3 seed H + 3 seed A, 9 scenario × 5 test tape = 360 ca 1 giờ. Chạy vài ca smoke không thay full protocol. |
| F21 | Replay snapshot/event với nhãn rõ | `ui.py` replay, run frames/log | Replay là bản ghi; không giả đó là suy luận LIVE hoặc dùng tốc độ phát làm throughput. |
| F22 | Policy/solver/stale fallback | `validate_profile`, optimizer conservative fallback, environment và simulator validation | Fallback giữ tính khả thi hoặc HOLD; không hứa giải được mọi congestion/deadlock. |
| F23 | Gói bằng chứng truy config/version/hash/policy/tape | `evaluation.py` manifest/export; reports và docs | Số liệu chưa chạy phải để chưa xác minh; source ZIP không tự tạo video bảo vệ hay bằng chứng robot thật. |

## Thay đổi kiến trúc có chủ đích

1. **Giữ PPO MLP của proposal và thêm entity encoder.** Default entity pooling có mask và global-query attention, actor/critic vẫn 2×128. Có switch MLP để ablation; không tuyên bố mạng mới chắc chắn hơn MLP về KPI.
2. **Tách tính khả thi khỏi RL.** Sáu action đều hợp lệ, nên không áp action masking lên sáu profile. Mask của observation chỉ loại padding; tải/pin/cổng/route do optimizer và simulator kiểm.
3. **Điều phối đường đi bảo thủ.** Các đầu cạnh bị giữ trong suốt chuyển động, corridor đã phát hiện giữ một chiều theo occupancy/pending moves. Route lập trước 60 s và kiểm/commit từng cạnh, chưa commit nguyên tử route 5 s của toàn đội; đây không phải tối ưu MAPF hay bằng chứng thông hành toàn cục.
4. **Tái lập thực dụng.** Config, seed, source/policy/tape hashes và phiên bản được lưu; solver có giới hạn wall-clock và resume mở lại episode nên không tuyên bố mọi run sẽ giống từng bit trên hai máy.
5. **Sửa chuẩn hóa mục tiêu solver.** Tất cả thành phần cộng chia số robot, cost scale 1e6; dùng incremental lateness + pressure tuổi/deadline để không bỏ đói đơn trễ. Readiness dùng completed tasks 60 s gần nhất và backlog; fallback service 120 s khi chưa có dữ liệu. Charger-return check là calendar witness hiện tại, không phải reservation tương lai bất biến. Thêm minimum charge gain 5% và energy urgency ưu tiên chung B1/H để tránh microcharge và tránh robot khỏe chiếm lịch trước robot pin khẩn; không đổi reward RL.
6. **Phạm vi vẫn là research simulator.** Không có camera/3D/ROS/fleet adapter production, robot thật, chuyển giao hàng, thermal battery hay dynamic charger power allocation. Các mục đó vốn nằm ngoài phạm vi cuối môn hoặc cần dữ liệu vận hành riêng.

## Tiêu chí kết luận thực nghiệm

- Hiệu quả RL chỉ được kết luận sau validation/tuning đã đóng băng rồi chạy test ghép cặp; báo throughput, pending, trễ, energy, charge wait, safety/deadlock và latency cùng nhau.
- Mọi phương pháp dùng cùng candidate/feasibility/planner/budget phù hợp với định nghĩa baseline. B1 là profile cố định chọn trên validation; ablation A cần train riêng với advanced charging tắt.
- Báo đầy đủ training seed, event tape, CI và số mẫu. Đơn chưa hoàn tất vẫn tính; không loại scenario xấu, không kéo dài horizon để hoàn tất backlog.
- Nếu RL không hơn B1, kết luận giả thuyết hiệu quả chưa đạt và debug theo `README.md`; không sửa baseline để tạo chiến thắng.
