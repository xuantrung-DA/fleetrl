# Chẩn đoán hồi quy dispatch trước khi sửa production

## Phạm vi và bằng chứng

Đã hoàn thành **360 episode**: 8 phiên bản × S2/S4/S6 × seed validation
1000–1004 × heuristic/fixed_cpsat/mappo_dispatch. Mỗi episode 1200 s,
cùng tape, khởi tạo robot, checkpoint MAPPO và cấu hình chung. S4 quan sát
900–1200 s của burst 900–1500 s. Source baseline lấy từ commit HEAD
95fdcc414b8afcab0894867043c5d3aad5319cc1; parking_fix là source đang làm việc
744de7e00397e055a71096f0766177e357b216cbcc7740fe86ca6f9a0ea40a74.

Đối chứng dùng CP-SAT deterministic work=0.03, wall guard=4 s, decision
guard=5 s, một worker; **không sửa production solver**. Đây là đối chứng
causal của source, không chứng minh latency 250 ms. Ngưỡng nghiệm thu đã
ghi trước trong protocol.json: mean pending tăng tối đa 0.5 đơn/episode,
mean completion giảm tối đa 1 điểm phần trăm; safety không tăng và không
mất task. Báo SD mẫu, không suy kết luận tổng quát từ một seed.

`aggregate.csv/json`, `paired_vs_baseline.csv`, `ablation_vs_parking_fix.csv`
chứa toàn bộ mean/SD, delay giao đơn, khoảng cách pickup, sạc, energy và
empty ratio. Mỗi case có tasks.json/csv, cycles.json, charging.json,
parking.json, events.json và final_state.json. Tasks.json giữ vị trí,
status, goal và robot được giao ngay trước commit; số idle/parked/charging
ở thời điểm task xuất hiện. Assignment delay của task chưa giao được giữ
dưới dạng censored tới horizon, tách khỏi mean của nhóm đã giao.

## Hồi quy trên nhiều seed

| Scenario / method | Pending trước mean ± SD | Pending parking_fix mean ± SD | Completion trước → sau |
|---|---:|---:|---:|
| S2 / heuristic | 5.4 ± 7.64 | 4.8 ± 5.59 | 85.57% → 87.20% |
| S2 / fixed_cpsat | 5.6 ± 4.62 | 3.8 ± 2.68 | 84.97% → 90.05% |
| S2 / mappo_dispatch | 2.0 ± 1.22 | 2.0 ± 1.00 | 94.74% → 94.82% |
| S4 / heuristic | 11.6 ± 8.62 | 11.4 ± 8.68 | 69.89% → 70.40% |
| S4 / fixed_cpsat | 6.6 ± 1.95 | 7.6 ± 2.97 | 82.77% → 80.28% |
| S4 / mappo_dispatch | 2.4 ± 0.55 | 2.2 ± 0.45 | 93.75% → 94.26% |
| S6 / heuristic | 7.2 ± 7.56 | 7.2 ± 7.66 | 80.97% → 80.90% |
| S6 / fixed_cpsat | 12.8 ± 5.02 | 7.6 ± 3.05 | 67.27% → 80.09% |
| S6 / mappo_dispatch | 2.4 ± 0.55 | 2.4 ± 0.55 | 93.82% → 93.75% |

Chỉ S4/fixed vượt hai ngưỡng thực nghiệm đã đặt. Không kết luận hồi quy
chung cho cả ba method hoặc cả ba scenario. Với 5 seed, độ biến động còn
lớn; không coi đây là chứng minh thống kê trên mọi workload.

## Giả thuyết và nguyên nhân đã xác nhận

### a. Cam kết parking chặn task — xác nhận điều kiện sai

`optimizer.py:488` chỉ sinh ứng viên cho status idle; `simulator.py:399`
từ chối robot không idle hoặc còn move_to; `mappo_dispatch.py:57` chỉ bật
active_agents cho idle. Robot task-free đang reposition vì vậy không nhận
task dù còn đủ tải/pin/đường và lịch cổng. Replay xác nhận task khả thi bị
loại chỉ vì parking commit. S4/fixed có trung bình 18.8 robot-cycle loại
theo cơ chế này sau sửa (bản cũ 23.6); MAPPO 23.4 so với 20.0.

Đây là thiếu khả năng preempt parking đã có ở baseline, trở nên không phù
hợp với các hành trình quay về/đổi parking dài của bản ổn định. Số lượng
robot-cycle không chứng minh nó là nguyên nhân duy nhất của ca +8 pending.

Đồng thời `optimizer.py:884` và `:906` đếm reposition vào work/available
của objective, trong khi candidate và commit lại cấm robot này nhận task.
Mô hình availability và admission không thống nhất. Khi cho parking nhận
task cần tránh đếm robot đó hai lần trong work_count và candidate variables.

### b. Parking xa pickup — không xác nhận như nguyên nhân chính

S4/fixed mean khoảng cách tại giao đơn **27.15 → 26.59 m**, không tăng.
S2/fixed tăng 25.45 → 26.67 m nhưng pending lại giảm. S4/1001 riêng tăng
24.94 → 26.44 m; đây là một phần trạng thái của ca xấu, không đủ để kết
luận phải thêm ô đỗ hoặc sửa map. Không sửa map theo phỏng đoán này.

### c. Sạc lệch — xác nhận kết quả, không phải đổi hệ số charge

S4/fixed mean số session **7.6 → 8.8**, SoC lúc bắt đầu **31.16% → 40.17%**.
Riêng seed1001: 7 → 9 session, SoC 29.35% → 39.78%; delay giao đơn đã giao
38.98 → 71.20 s, delay có censored 39.56 → 114.56 s.

Replay cùng các plan ghi nhận **923 → 3473 robot-task-cycle** bị lịch cổng
chặn return-calendar ở S4/1001. Nếu bỏ riêng calendar trong phép kiểm
counterfactual thì ứng viên task tồn tại: không phải payload hay thiếu
pin thuần túy. Các task 80 kg tại 533/591/778 s không được giao ở parking_fix.
Đây là tác động qua placement/availability và bookings, không phải chỉ
thời gian đi tới pickup hay tình trạng deadlock.

Các hệ số của task/charge và profile không bị đổi trong bản sửa; phần
`_candidate_cost` mới chỉ áp dụng với kind reposition. Ablation bỏ correction
không khắc phục S4 mà làm mean pending **7.6 → 8.2**. Vì vậy không sửa lại
charge reward/threshold hoặc quy toàn bộ hồi quy cho correction.

### d. Residence/hysteresis — tác động nhỏ qua placement

Residence chỉ được kiểm ở `parking.py:18` sau khi task candidates đã sinh;
nó không trực tiếp cấm robot idle nhận task. Tắt riêng residence giảm mean
pending S4/fixed 7.6 → 7.2, vẫn vượt baseline 6.6 và không khôi phục toàn bộ
ca xấu. Không bỏ cơ chế giữ ổn định parking chỉ từ kết quả này.

### e. Nhiễu/seed — đã đo, phạm vi hồi quy hẹp hơn ví dụ ban đầu

5 seed cho mỗi nhóm. S2 và S6 không tái hiện hồi quy mean như S4/fixed.
Các lần chạy trùng do hai coordinator resume được giữ nguyên; 44 cặp
đã kiểm có cùng kết quả physical/diagnostics (latency wall khác nhau).
`duplicate_checks.json` ghi đối chiếu; aggregate chỉ chọn lần hoàn thành
đầu tiên cho một seed, không coi các lần lặp này là seed độc lập.

### f. Tương tác candidate/availability/reservation

Sạc và lịch cổng thay đổi theo tập ứng viên, không chỉ theo một hệ số.
Bỏ việc giữ goal parking đang cam kết làm S4/fixed về mean 6.6, nhưng đây
không phải giải pháp an toàn: có thể đề xuất bay đã được robot khác giữ,
rồi simulator từ chối. Cần giữ reservation và sửa task handoff/availability
nhất quán, thay vì bỏ guard vật lý hoặc tạo lại oscillation.

## Ablation S4/fixed, cùng 5 seed

| Phiên bản | Mean pending | Delta so parking_fix |
|---|---:|---:|
| baseline trước ổn định parking | 6.6 | -1.0 |
| parking_fix | 7.6 | 0 |
| no_cost_correction | 8.2 | +0.6 |
| no_idle_return | 8.6 | +1.0 |
| no_residence | 7.2 | -0.4 |
| no_benefit_guard | 6.2 | -1.4 |
| no_goal_reservation | 6.6 | -1.0 |
| no_recovery_stability | 7.6 | 0 |

Các delta là đối chứng từng cơ chế trong một hệ thống tương tác; không
cộng chúng để tuyên bố một phân rã nhân quả độc lập. Bảng đầy đủ cho S2/S6
và ba method trong CSV. Return parking không phải nguyên nhân duy nhất;
recovery stability không đóng góp trong các ca này.

## Hướng sửa theo bằng chứng, trước khi sửa production

1. Cho task-free reposition nhận task; giữ goal parking nếu plan chỉ wait.
   Không cho nhận lại reposition/charge tùy tiện khi đang đi parking.
2. Nếu còn một cạnh đang thực thi, giữ nguyên reservation/cạnh đó đến
   arrival; chuyển route từ ô đích của cạnh, tính cả phụ tải/thời gian còn
   lại vào eligibility và commit. Không teleport hoặc cancel giữa cạnh.
3. Dùng cùng eligibility ở candidate generator, simulator guard và MAPPO
   active mask. Không đổi shape/cột observation hay reward.
4. Sửa work/zone readiness để robot parking có candidate không bị đếm
   thêm như một busy worker. Giữ charge cost/threshold/profile, reservation,
   no-demand stability và F05. Chạy kiểm tra nhiều seed trước khi kết luận.

## Ngoài phạm vi và giới hạn

Baseline S6/heuristic/1004 có safety=1 (`route_wait_bound_exceeded` lúc 951 s);
parking_fix ca này safety=0. Không giả sử toàn bộ baseline luôn safety=0.
Không mất task. Mọi safety mới phải so từng cặp và không được tăng.
Study14/checkpoint không bị sửa; đây là frozen-policy validation, không
quy cả chênh energy của study14 cho lỗi parking.
