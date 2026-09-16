# Kế hoạch mở rộng FleetRL: 14 phương án và 16 nhóm metrics

Ngày lập: 11/09/2026. Trạng thái: kế hoạch triển khai, chưa phải chức năng đã hoàn thành.

Phạm vi do người dùng chốt: triển khai đủ 14 phương án, đo đủ 16 nhóm metrics, có thể train/eval/so sánh/replay bằng một quy trình tái lập. Thứ tự bên dưới là thứ tự thực hiện; mọi phương án đều nằm trong phạm vi cuối cùng. Mục tiêu là có bằng chứng lựa chọn thuật toán cho đồ án và nền tảng mở rộng thực tế. Không đặt điều kiện PPO phải thắng.

Tài liệu này bổ sung cho proposal gốc; giữ nguyên `PROPOSAL.md` và PDF để truy vết yêu cầu ban đầu. Các tên module, API, CLI và config mới dưới đây là thiết kế dự kiến.

## 1. Điểm xuất phát đã kiểm tra

| Thành phần | Hiện trạng | Việc cần làm |
| --- | --- | --- |
| Simulator, năng lượng, planner | Có robot không đồng nhất, task, đặt chỗ ô/cạnh, lịch sạc, sự cố và guard thực thi | Dùng chung cho các phương án; bổ sung instrumentation và kiểm invariant |
| `rl.py` | Trainer, callback, checkpoint/load và báo cáo gắn với PPO | Tách encoder, trainer, policy adapter và checkpoint metadata |
| `env.py` | Gymnasium Dict observation, `Discrete(6)`, gọi trực tiếp `FleetOptimizer` | Tách lõi bước mô phỏng và các adapter action/observation |
| `optimizer.py` | Candidate, objective, heuristic và CP-SAT nằm trong một file | Tách eligibility/candidate, trọng số, backend solver và fallback |
| `evaluation.py` | Chỉ nhận B0/B1/H/A; `predict()` không giữ recurrent state | Registry 14 phương án, inference session, benchmark theo protocol |
| `metrics.py` | Có phần lớn metric phục vụ, năng lượng, sạc và latency | Schema có đơn vị/mẫu số, thêm metric thiếu, sửa nghĩa timeout và xử lý run lỗi |
| CLI/config/script | Config và train-seed runner chủ yếu dành cho PPO | Config theo thuật toán, study runner, resume từng job, export bảng |
| Bằng chứng cũ | 30 file khớp SHA-256; báo cáo đi kèm ghi 111 test đạt; đã chạy lại 18 test planner trong phiên làm việc | Giữ báo cáo cũ; tạo kết quả xác minh mới cho source mới |
| Môi trường hiện tại | Python 3.12; thiếu `stable_baselines3`, `sb3_contrib`, `ortools` | Kiểm dependency và khóa phiên bản trên môi trường triển khai trước khi chạy tích hợp |

Một số giới hạn cần giữ rõ: planner hiện giữ/commit từng cạnh, không phải MAPF đầy đủ; simulator cập nhật planner trong `advance`; kiểm đường về sạc hiện là bằng chứng khả thi theo lịch đang biết, không phải đặt chỗ tương lai bất biến.

## 2. Danh mục phương án có ID cố định

`variant_id` nhận diện bộ thuật toán; `run_id` nhận diện một lần chạy. Không dùng tên `hybrid` cho mọi model khác PPO. Alias cũ B0/B1/H/A được ánh xạ tương ứng V01/V02/V06/V09.

| ID | Tên cấu hình | Phương án | Triển khai chính | Nhóm diễn giải |
| --- | --- | --- | --- | --- |
| V01 | `heuristic` | Heuristic | Ưu tiên/tuổi đơn, robot đủ điều kiện gần nhất, sạc theo ngưỡng đã tune | Baseline không học |
| V02 | `fixed_cpsat` | CP-SAT profile cố định | Chọn profile tốt nhất trên validation; sạc nâng cao | Baseline không học; đối chiếu chính |
| V03 | `qlearning_cpsat` | Q-learning bảng + CP-SAT | Trạng thái rút gọn/rời rạc, Q-table, epsilon-greedy, sáu profile | Khác biểu diễn trạng thái |
| V04 | `dqn_cpsat` | DQN + CP-SAT | SB3 DQN, Dict observation, encoder chung | So sánh thuật toán học |
| V05 | `a2c_cpsat` | A2C + CP-SAT | SB3 A2C, encoder chung | So sánh thuật toán học |
| V06 | `ppo_cpsat` | PPO + CP-SAT hiện tại | SB3 PPO; mặc định entity encoder hiện tại | Mốc tương thích và phương án chính |
| V07 | `qrdqn_cpsat` | QR-DQN + CP-SAT | SB3-Contrib QR-DQN, encoder chung | So sánh thuật toán học |
| V08 | `recurrent_ppo_cpsat` | Recurrent PPO/LSTM + CP-SAT | SB3-Contrib; entity encoder + LSTM; state riêng từng environment | Ablation bộ nhớ |
| V09 | `ppo_threshold_cpsat` | PPO + CP-SAT, sạc theo ngưỡng | Train riêng; đích 95%; ngưỡng khóa sau validation | Ablation sạc nâng cao |
| V10 | `sac_cpsat` | SAC + CP-SAT | SB3 SAC; action liên tục sinh năm trọng số | Thay action space |
| V11 | `td3_cpsat` | TD3 + CP-SAT | SB3 TD3; cùng miền trọng số của V10 | Thay action space |
| V12 | `mappo_dispatch` | MAPPO + bộ xử lý tranh chấp + planner | Actor cho từng robot, critic tập trung, action mask, arbiter lịch/tài nguyên | Thay mô hình điều phối |
| V13 | `forecast_mpc_milp` | Dự báo + MPC + MILP | Dự báo nhân quả, tối ưu nhiều thao tác kế tiếp; chỉ commit thao tác đầu | Hệ thống không RL |
| V14 | `ppo_milp` | PPO + MILP | PPO sáu profile; backend MILP trên cùng candidate/objective với CP-SAT | Thay backend tối ưu |

V13 chọn MILP làm backend mặc định để phạm vi triển khai cụ thể và dùng lại nền tảng V14. CP-SAT cho MPC là tùy chọn mở rộng sau khi V13 đạt; không cần cả hai backend MPC để hoàn thành bảng 14 phương án.

V01/V02/V13 không có policy RL cần train. V13 dùng bộ dự báo xác định, fit/tune trên train/validation. Mười một phương án còn lại cần các lần học độc lập. CBS/ECBS/EECBS không thuộc bảng 14 đã chốt; giữ planner chung, không thêm nghiên cứu planner trong kế hoạch này.

## 3. Kiến trúc source dự kiến

Giữ các module công khai đang dùng để tương thích; tách phần triển khai bên trong theo trách nhiệm. Đặc biệt giữ import path cũ của `MaskedEntityEncoder` cho việc nạp checkpoint cũ nếu schema thực sự tương thích.

```text
src/fleetrl/
  config.py                 # đọc/validate config phiên bản cũ và mới
  types.py                  # Snapshot, DecisionPlan và metadata phiên bản
  env.py                    # FleetEnv tương thích API Discrete(6) hiện tại
  optimizer.py              # FleetOptimizer tương thích API hiện tại
  rl.py                     # API train/load/evaluate cũ, chuyển tới trainer chung
  simulator.py, planner.py, energy.py, maps.py, scenario.py
  observations.py           # encoder observation chuẩn, có schema ID
  metrics.py                # tính metric từ counters/events, schema ID
  evaluation.py             # runner một ca + lưu kết quả
  cli.py, ui.py
  methods/
    registry.py             # 14 MethodSpec cố định
    contracts.py            # DecisionRequest, PolicyAdapter, Controller, Trainer
    baselines.py            # heuristic và fixed profile
    policies.py             # adapter SB3; quản lý inference session
    tabular.py              # Q-learning và Q-table checkpoint
    mappo.py                # learner MAPPO và policy/critic
    arbitration.py          # giải tranh chấp MAPPO và đặt lịch khả thi
    mpc.py                  # controller tối ưu nhiều bước
    forecasting.py          # dự báo chỉ đọc lịch sử được quan sát
  learning/
    encoders.py             # entity/MLP dùng chung
    training.py             # vòng đời train, callbacks, tài nguyên, checkpoint
    environments.py         # lõi dispatch + discrete/continuous/multi-agent adapters
    compact_state.py        # đặc trưng/bins cho Q-learning và controls
  optimization/
    candidates.py           # sinh candidate và kiểm điều kiện vật lý chung
    objectives.py           # ObjectiveWeights và sáu profile
    cpsat.py                # backend CP-SAT hiện tại
    milp.py                 # backend MILP tương đương bài toán một thao tác
    contracts.py            # budget, candidate set, solver result
  experiments/
    protocol.py             # train/validation/test split và study definitions
    runner.py               # danh sách job, trạng thái, chạy tiếp
    statistics.py           # paired bootstrap, kiểm đủ grid, CI
    reporting.py            # CSV/JSON/Markdown/plots
    resources.py            # timing, CPU, RAM cả process tree, GPU nếu có
```

Hợp đồng tối thiểu:

- `MethodSpec`: variant ID, thuật toán, backend, action/observation schema, chế độ sạc, loại trainer, nhóm so sánh, dependency và khả năng resume.
- `PolicyAdapter.predict(observation, state, episode_start, deterministic)`: trả action, state mới, diagnostics. Không hard-code ép action thành một scalar integer.
- `DecisionRequest`: một trong `profile_id`, `objective_weights`, `robot_candidate_choices`; không cho các kiểu xung đột cùng tồn tại.
- `Controller.propose(snapshot, request, deadline) -> DecisionPlan`: chỉ dùng snapshot được cấp; cả baseline/MPC đều đi qua interface này.
- `SolverBackend.solve(snapshot, candidates, weights, remaining_budget)`: trả nghiệm, objective/bound, raw status, lý do dừng nếu xác định được và thời gian dùng.
- `Trainer`: train/save/load/resume/evaluate; metadata dùng tên chung `training_updates` kèm nghĩa theo thuật toán, không gắn mọi update với PPO.
- Lõi dispatch dùng chung: lấy snapshot, suy luận, đề xuất, kiểm/commit, advance simulator, tính reward/metrics. Mọi trainer, eval và GUI đều đi qua cùng logic thực thi.
- Một đồng hồ deadline xuyên suốt snapshot/inference/model/solve/commit. Fallback dùng phần ngân sách còn lại, không được khởi động thêm một budget 250 ms mới.
- `DecisionPlan`/log phiên bản mới có kiểu action và trọng số thật. `profile_id` có thể vắng mặt khi action liên tục/multi-agent; không ghi giả profile 0.

Config mới tách `env`, `method`, `train`, `optimizer`, `evaluation`, `study`; vẫn đọc được YAML cũ bằng migration có phiên bản. Mỗi thuật toán có validator riêng: không bắt DQN/Q-learning/MAPPO thỏa điều kiện minibatch của PPO.

## 4. Các quyết định triển khai từng nhóm

### 4.1. Các thuật toán dùng sáu profile

- Giữ V06 làm mốc hồi quy. Tách refactor khỏi thay đổi reward, candidate và vật lý; dùng fixed snapshot/unit tests và run có manifest để phát hiện lệch hành vi.
- V04/V05/V06/V07 dùng cùng thông tin, preprocessing và entity encoder family. Ghi số tham số, critic/target network, optimizer updates và tỷ lệ cập nhật trên mẫu; kiến trúc head phải phù hợp từng thuật toán, không ép mọi hyperparameter giống PPO.
- Thêm các trường riêng cho off-policy: replay capacity, warmup, train frequency, gradient steps, target update và exploration. Dự toán RAM từ shape observation trước khi cấp buffer.
- Q-learning dùng bảng thưa với schema bins lưu trong artifact. Đặc trưng khởi đầu: thời gian còn lại, tồn/đơn quá hạn, số robot rảnh, pin thấp, cổng bận và nhu cầu vùng. Bins chỉ chọn trên train/validation; log số state gặp, tần suất visit, state chưa gặp lúc test và tie-break xác định.
- Q-learning cập nhật đúng terminal; không bootstrap sau kết thúc ca. Chạy bài toy có nghiệm biết trước để xác minh quá trình học trước khi đưa vào FleetRL.
- Có control DQN/PPO nhận cùng đặc trưng rút gọn và cùng bins với Q-learning để phân biệt ảnh hưởng biểu diễn với thuật toán. Đây là run hỗ trợ, không thêm tên vào bảng 14.
- V08 giữ và reset LSTM state riêng từng worker, từng episode, từng lần GUI reset; save/load và runner eval phải thống nhất. Sửa riêng `choose_action`, `evaluate_model`, `run_ui` hiện bỏ recurrent state. Bộ nhớ không được bỏ qua guard snapshot quá cũ.
- V09 train độc lập, không chỉ lấy checkpoint V06 đổi cờ khi eval. Tune ngưỡng V01 và V09 theo đúng phương pháp; nếu V09 so nhiều ngưỡng thì mỗi ngưỡng cần được train tương ứng. Không lấy ngưỡng tốt cho heuristic làm bằng chứng nó cũng tốt cho PPO.
- Resume: policy/optimizer và bộ đếm được khôi phục; off-policy cần lưu replay buffer, tabular cần Q-table/visit/exploration state. Nếu cố ý không lưu replay phải ghi chế độ resume thiếu trạng thái. Không hứa khôi phục chính xác simulator/rollout/RNG khi chưa triển khai điều đó.

### 4.2. SAC và TD3 sinh trọng số liên tục

- Action chuẩn hóa `Box(-1,1,shape=(5,))`, ánh xạ sang `(wT,wG,wA,wZ,wE)`.
- Miền khởi đầu: `wT,wG,wA,wZ` trong `[0.5,3.0]`, `wE` trong `[0.2,1.0]`; bao phủ sáu profile hiện tại. Miền được ghi vào config và khóa trước test.
- Hệ số thưởng nhận task, hệ số thời lượng, chuẩn hóa theo số robot và guard năng lượng giữ như bộ hiện tại; RL không điều khiển ràng buộc cứng.
- Ghi action thô/trọng số đã giải mã; NaN, sai shape hoặc ngoài miền hợp lệ đi qua fallback có reason. Không âm thầm cho action ngoài chuẩn thành một mẫu suy luận thành công.
- Giữ cùng miền và encoder giữa SAC/TD3; exploration và target update theo từng thuật toán.
- Thêm control PPO với action liên tục khi cần kết luận lợi ích do thuật toán hay do bỏ giới hạn sáu profile. Control được báo riêng và tính vào dự toán bổ sung.

### 4.3. MILP thay CP-SAT

- Backend dự kiến: OR-Tools `MPSolver` với SCIP; dependency probe phải xác nhận backend tồn tại và giải được bài scheduling nhỏ trên môi trường khóa phiên bản trước khi chọn chính thức.
- Dùng chung task/charge/reposition/wait candidates, energy guards, objective, cost scale và wait bounds của CP-SAT. Không bỏ readiness, cân bằng vùng hoặc queue capacity để làm MILP nhẹ hơn rồi gọi là cùng bài toán.
- Bản đầu dùng biến chọn start trên lưới 5 s cho mỗi charge candidate; start không quá cửa sổ cho phép. Đây là cách mã hóa MILP khác với interval của CP-SAT, nhưng phải biểu diễn cùng tập lịch khả thi.
- Ràng buộc port và queue phủ các đoạn thời gian giữa mọi mốc thay đổi occupancy, gồm booking đang giữ và toàn bộ đuôi lịch. Không cắt interval khi vượt cửa sổ bắt đầu 900 s. Giới hạn 500 là số operation candidates; log riêng số biến/ràng buộc MILP sau khai triển start.
- Kiểm chứng các case nhỏ bằng enumeration/optimum độc lập: objective và tính khả thi tương đương CP-SAT khi cả hai giải tối ưu; không yêu cầu nghiệm tie phải giống hệt.
- Nghiệm MILP phải được kiểm tích phân trong tolerance và kiểm lịch độc lập trước commit. Không làm tròn một nghiệm phân số rồi thực thi.
- Với V14, train PPO riêng trên backend MILP cho so sánh hệ thống. Thêm phép thử một policy đóng băng chạy cả hai backend để đo tác động riêng của solver; lưu rõ checkpoint lineage, không coi checkpoint dùng lại là seed học độc lập.

### 4.4. MAPPO

- Đây là triển khai riêng có actor theo robot và critic tập trung; không chạy PPO sáu profile toàn đội rồi đổi tên thành MAPPO.
- Actor chia sẻ tham số, nhận trạng thái robot, loại/năng lực và đặc trưng các candidate hợp lệ của robot đó; critic nhận thông tin toàn đội đã quan sát. Dùng masks cho robot padding và candidate padding.
- Action chọn một candidate task/charge/reposition/wait. Robot bận có action giữ cam kết; không hủy hàng đang mang. Giữ cùng giới hạn shortlist/candidate và thông tin nhân quả như các controller khác.
- Arbiter xử lý task trùng, cổng/hàng đợi/đỗ tranh chấp, chọn lịch sạc khả thi và từ chối thành WAIT có lý do. Quy tắc xác định trước, không dùng thông tin tương lai; simulator vẫn kiểm lại.
- Ghi cả action đề xuất và quyết định thực thi; log `arbitration_override_rate`. Trong PPO update, log probability gắn với action robot đã lấy mẫu; arbiter là một phần transition, không sửa action trong buffer thành lệnh đã bị thay thế.
- Reward chung của đội giữ như FleetRL, không nhân reward với số robot. Chuẩn hóa actor loss theo số agent đang có quyền quyết định; critic vẫn học từ thời gian vật lý của ca.
- Một timestep ngân sách là một lần quyết định toàn đội/5 s, không phải một action của một robot. Log riêng agent actions để không tạo lợi thế ngân sách giả.
- Tham chiếu MAPPO chính thức, nhưng viết adapter/trainer có test phù hợp FleetRL và dependency khóa riêng; không kéo nguyên các môi trường benchmark không liên quan vào package.

### 4.5. MPC và dự báo

- Bộ dự báo đầu tiên: EWMA theo vùng từ task đã quan sát; các thông số chỉ fit/tune trên train/validation. Reset trạng thái đúng episode; snapshot trễ chỉ cho cập nhật lịch sử đã thực sự nhận được.
- MPC dùng đồ thị thao tác theo thời gian với vị trí/pin/robot available time; tối ưu được chuỗi nhiều thao tác, gồm giao việc và sạc. Giữ các booking/cam kết đang thực thi, sức chứa queue, tương thích cổng, tải và dự phòng.
- Mốc khởi đầu cho horizon: 300 s, start grid 5 s, giới hạn task/candidate như giao thức chung; độ dài horizon/giới hạn mở rộng được đo và tune trên validation theo cùng trần online. Khoảng sạc vượt horizon vẫn được giữ đến hết.
- Forecast đi vào nhu cầu phục vụ/readiness và terminal value; không biến forecast thành task thật, không đánh dấu hoàn thành một đơn chưa xuất hiện. Tối ưu task đã biết và khả năng sẵn sàng trước nhu cầu dự báo.
- Chỉ commit thao tác đầu; quyết định sau cập nhật từ snapshot mới. Bài test phải có tình huống chuỗi hai thao tác cho kết quả khác tối ưu chỉ một thao tác, xác nhận MPC thực sự nhìn nhiều bước.
- Ghi prediction, horizon, số node/biến, solve status/gap và fallback. Forecast và dựng mô hình nằm trong latency online.
- V13 mặc định có một bộ tham số dự báo xác định đã fit/tune, không có ba training seed RL. Nếu sau này thay bằng dự báo neural, phải sửa protocol và số replicate thay vì âm thầm giữ thống kê cũ.

## 5. Hợp đồng 16 nhóm metrics

Đặt `T` là thời gian mô phỏng thực đã chạy, `N` là số task duy nhất đã tới, `F` là số hoàn thành, `U=N-F`. Metric episode đi kèm trạng thái complete/partial/failed, schema version, đơn vị, mẫu số và lý do thiếu. Mẫu số bằng 0 trả `null`, không dùng 0 giả. Partial/failed run không được gộp như ca hoàn chỉnh.

| ID | Nhóm metric / key dự kiến | Định nghĩa và việc chỉnh source |
| --- | --- | --- |
| M01 | Throughput: `throughput_per_hour` | `F/(T/3600)`; chỉ so ca cùng horizon trong bảng chính; đã có |
| M02 | Backlog/completion: `pending`, `pending_rate`, `completion_rate`, `blocked_tasks` | U, U/N, F/N và số blocked; giữ cả đơn ngoài shortlist; thêm pending rate |
| M03 | Trễ: `total_lateness_s` | Tổng `max(0,c*-deadline)` với c* là completed time hoặc T; cận dưới cho đơn chưa xong; giữ riêng khỏi integral có priority dùng trong reward |
| M04 | Deadline: `deadline_due`, `deadline_violated`, `deadline_violation_rate` | Chỉ mẫu số là các đơn có deadline <= T; hoàn thành muộn hoặc chưa xong đều vi phạm; báo pending chưa đến hạn |
| M05 | Chờ lấy hàng: `mean_wait_s`, `wait_censored` | Từ tạo đến bắt đầu lấy hoặc T; báo số chưa được lấy; N=0 -> null trong schema mới |
| M06 | Năng lượng: `consumed_wh`, `energy_per_completed_wh`, `grid_wh`, pin/SoC đầu-cuối | Bao gồm mọi robot, đi rỗng/chờ/công việc chưa xong; F=0 -> null. Lưu snapshot pin đầu, theo loại robot và kiểm cân bằng năng lượng |
| M07 | Đi rỗng: `empty_distance_m`, `loaded_distance_m`, `empty_distance_ratio` | Đếm cạnh thực sự hoàn tất, không lấy quãng đường kế hoạch; tỷ lệ đi rỗng trên tổng đường thực |
| M08 | Chờ sạc: `mean_charge_wait_s`, `charging_requests`, `charge_wait_censored`, `charge_cancelled` | Từ accepted request tới actual start; pending tới T, hủy tới cancellation time. Tách nhóm bắt đầu/chưa bắt đầu/hủy, không loại yêu cầu khó |
| M09 | Cổng sạc: `charging_active_s_by_port`, `port_occupied_s_by_port`, các utilization | Tích phân tại mỗi tick. Active time dùng phần tick thực nạp; occupied gồm vào/ra/docking khi thực chiếm cổng. Chia cho T từng cổng; fleet aggregate chia tổng port-time |
| M10 | Đúng/an toàn: duplicate assignment, payload, lost task, collision, negative battery, reserve, emergency | Kiểm trạng thái thực và trace task IDs. Tách proposed-rejected khỏi executed violation; assertion/error phải lưu run failed và diagnostics, không để zero counter che crash |
| M11 | Deadlock/phục hồi: incident counts, `recovery_time_*`, unresolved/censored | Incident ID và thời điểm detect/attempt/first progress/resolved. First move chỉ là tiến triển; resolved khi tiếp tục tiến tới đích trong cửa sổ 30 s hoặc hoàn tất thao tác, không tái kẹt cùng incident. Tới T chưa resolved là censored |
| M12 | Online latency: p50/p95/max, `online_budget_exceeded_rate` | Từ lấy snapshot qua encode/inference/forecast, dựng/giải và guard commit; tính đúng một lần. Ghi breakdown, model-load/warmup riêng. Planner chạy tiếp trong tick/advance có timing riêng, không tuyên bố đã nằm trong online latency |
| M13 | Solver/fallback/rejection: raw status, nonoptimal, timeout reason, fallback và rejected rate | Lưu số solve thật, dispatch thật, đề xuất thật làm mẫu số tương ứng. FEASIBLE không đồng nghĩa timeout; thêm `solver_nonoptimal_rate`. Không biết lý do dừng thì ghi unknown/null. Báo theo lý do và theo scenario |
| M14 | Chi phí học/tài nguyên: steps, wall time, updates, CPU/RAM/VRAM | Tách rollout/update/validation/checkpoint và tổng. Đo RAM đồng thời của parent+workers, không cộng peak ở thời điểm khác nhau; GPU allocated/reserved và sampling scope rõ; không có GPU -> N/A |
| M15 | Độ biến động/CI: mean, SD, paired differences, CI95 | Bootstrap theo scenario/tape và training replicate; kiểm đủ grid bằng manifest; không coi tick hoặc nhiều checkpoint cùng seed là mẫu độc lập |
| M16 | OOD/generalization: KPI theo domain và degradation | Giữ S9 map B + burst làm stress test kết hợp; thêm cặp ID/OOD chỉ đổi layout hoặc burst để đo suy giảm riêng. Mẫu số 0 thì báo chênh lệch tuyệt đối; không gộp các workload khác nhau thành một tỷ lệ khó diễn giải |

Các thay đổi đo cần thực hiện trước khi train chính thức:

- Bổ sung counters/events ngay trong simulator cho M09/M10/M11; không suy sử dụng cổng từ số lần sạc hoặc booking dự kiến.
- `charge_step` cần cung cấp thời lượng nạp thực hoặc equivalent accounting để xử lý nạp xong giữa tick; đối chiếu grid energy và active time bằng test tính tay.
- Hiện `deadlocks_resolved` tăng sau một chuyển động phục hồi. Giữ trường legacy có nhãn, thêm incident schema mới; không hồi tố báo cáo cũ theo định nghĩa mới.
- Chuẩn hóa nghĩa `solver_timeout_rate`: hiện flag đánh dấu FEASIBLE/UNKNOWN là `timed_out`; schema mới phải tách trạng thái chưa chứng minh optimal, lý do dừng và vượt budget. Mọi ngưỡng nghiệm thu ghi rõ trường nào được dùng.
- Luồng lỗi phải finalize partial metrics, error và replay cuối có thể lưu được. Trường chưa đo có `missing_reason`; không dùng thiếu metric để coi một phương án an toàn hoặc nhanh hơn.
- Theo dõi dropped operational events: runner hiện dùng chỉ số list trong khi simulator có thể cắt bớt list khi `log_decisions=False`. Thay bằng event sequence/cursor hoặc sink drain ổn định, giữ bounded memory và không mất log.
- Ghi metric theo nhóm tải 5/15/40/80 kg và ưu tiên 1/3: arrived/completed/pending/late/wait. Pin và năng lượng có breakdown L/M/H; report không xóa nhóm ít mẫu.
- Metric definition changes dùng `metrics_schema_version=2`; replay/manifest nâng schema có reader tương thích v1. Không gộp số liệu v1/v2 khi nghĩa khác nhau.

## 6. Protocol so sánh và cách chọn model

Các mặc định dưới đây để triển khai cấu hình study; có thể tune trên validation, sau đó khóa trước final test.

| Hạng mục | Mặc định |
| --- | --- |
| Train scenarios | Map A, curriculum 5 -> 10 -> trộn 10/15/20 robot; cùng phân phối episode cho nhóm so trực tiếp |
| Train environment seeds | 0-199 |
| Validation | Map A, seed 1000-1019; periodic checkpoint trên tập con cố định 1000-1002, chọn cấu hình cuối trên tập đã khai báo đầy đủ |
| Final test | S1-S9, seed 2000-2004, mỗi ca 3600 s |
| Training replicas | 11, 12, 13 cho mỗi phương án học; baseline deterministic không nhân bản giả thành ba seed học |
| Nominal training budget | 300000 fleet decision steps/replica; log actual steps và overshoot do rollout; không tính agent action của MAPPO như fleet step |
| Online limit | 250 ms tổng dispatch; solver tối đa 100 ms và bị chặn bởi phần budget còn lại; cùng phần cứng, không chạy job train khác lúc đo latency |
| Tuning | Ghi trước search space, số trial, trần tuning compute và dữ liệu dùng; mỗi thuật toán được cấu hình riêng trong ngân sách công bằng |
| Reference | V02 profile cố định được tune trên validation |
| Primary efficacy test | S4-S7: throughput paired improvement mục tiêu >=5%, trễ tăng không quá 5%, không tăng vi phạm; báo từng ca và CI |

Các tập test seeds 2000/3 và số liệu đã xuất hiện trong báo cáo nghiệm thu cũ là development evidence đã biết. Manifest đánh dấu điều này. Để tuyên bố xác nhận trên dữ liệu hoàn toàn mới sau vòng sửa/tuning, tạo thêm confirmatory protocol với seed từ 3000 và lưu tape/hash trước khi chạy final; số ca đó là budget bổ sung. Không dùng kết quả S9/final test để chỉnh reward, bins, forecast hoặc checkpoint.

Chia kết quả theo câu hỏi nghiên cứu:

1. **Có lợi ích của học chọn profile không?** V02 so với V04/V05/V06/V07 trên observation và phần còn lại tương đương.
2. **Biểu diễn/bộ nhớ/sạc có ích không?** V03 với compact-state controls; V08 với V06; V09 với V06 được train riêng theo từng chế độ.
3. **Đổi thiết kế hệ thống có ích không?** V10/V11/V12/V13; báo action, thông tin, tuning và công sức triển khai khác biệt.
4. **Backend tối ưu có ảnh hưởng gì?** V14 với V06; vừa frozen-policy backend test, vừa independently trained system test.

Không đổi tất cả hyperparameter thành giá trị PPO để gọi là công bằng. Dùng cùng thông tin/reward/action khi câu hỏi là thuật toán học, công bố kiến trúc và số tham số, so cả performance-vs-steps và performance-vs-wall-time.

Checkpoint selection mặc định giữ thứ tự vận hành: tối thiểu vi phạm tính đúng/an toàn -> tối đa completed -> tối thiểu total lateness -> pending; thêm schema metric mới và log tie-break. Trước final test, chọn cấu hình bằng validation theo tiêu chí đã ghi. Final report gồm constraint pass/fail và các trade-off, không có một weighted score tùy tiện để đổi người thắng.

CI theo từng scenario và cặp tape/initial state hợp lệ. Khi so nhiều phương án, công bố trước đối chiếu chính; có thể thêm simultaneous bootstrap intervals hoặc hiệu chỉnh Holm cho một family kiểm định đã định nghĩa. Không chọn phương pháp thống kê sau khi nhìn người thắng. Nếu thiếu replicate/case hoặc run failed, hiển thị coverage, failure rate và phần paired analysis còn hợp lệ; không chỉ báo các run sống sót.

OOD bổ sung có cặp chỉ đổi map hoặc nhu cầu; đây là phép đo riêng với seed và workload specification được khóa. Không gọi S9 tốt/xấu hơn S1 là ảnh hưởng riêng của map vì nhiều yếu tố cùng đổi.

## 7. Thứ tự triển khai và tiêu chí qua từng mốc

| Mốc | Nội dung | Source/config chính | Điều kiện nghiệm thu |
| --- | --- | --- | --- |
| P0 | Chụp mốc source, dependency và baseline chạy được | packaging, reports, tests, `configs/smoke.yaml` | Hash inventory; dependency lock thực tế; smoke/test hiện có được chạy trên máy mục tiêu hoặc ghi rõ lỗi môi trường; không thay báo cáo cũ |
| P1 | Tách registry/API, config schema, action adapters và checkpoint bundle; V01/V02/V06/V09 | `config.py`, `types.py`, `env.py`, `optimizer.py`, `rl.py`, `methods/`, `learning/` | V06 tương thích; alias cũ hoạt động; action/schema/backend mismatch bị chặn; mọi đường chạy chung guard |
| P2 | Hoàn thiện schema và instrumentation 16 nhóm metrics | `simulator.py`, `energy.py`, `metrics.py`, `evaluation.py`, `experiments/` | Test tính tay, null/censoring, incident lifecycle, latency không đếm đôi, event stream không mất dữ liệu; raw -> summary khớp |
| P3 | Q-learning, DQN, A2C, QR-DQN, Recurrent PPO | `methods/tabular.py`, `policies.py`, `learning/`, configs V03/V04/V05/V07/V08 | Mỗi thuật toán có update thật; checkpoint load/inference/resume đúng; LSTM reset đúng; finite states/rewards; test compact-state |
| P4 | Trọng số liên tục và SAC/TD3 | `optimization/objectives.py`, env continuous adapter, configs V10/V11 | Action mapping, finite/bounds/fallback, shared physical constraints; train/load/eval cả hai; kiểm buffer tài nguyên |
| P5 | Backend MILP và MPC/dự báo | `optimization/milp.py`, `methods/mpc.py`, `forecasting.py`, configs V13/V14 | MILP cùng bài toán trên toy oracle; status/tolerance/time budget đúng; forecast không đọc tương lai; MPC có test nhiều thao tác; V14 train riêng |
| P6 | MAPPO và arbiter | `methods/mappo.py`, `arbitration.py`, multi-agent adapter, config V12 | Test credit/action masks/terminal; tranh task/cổng không thực thi sai; log proposed/executed/override; budget tính fleet steps |
| P7 | Study runner, CLI/UI, export và nghiệm thu tích hợp 14 phương án | `cli.py`, `ui.py`, `scripts/`, `experiments/`, docs | Liệt kê đủ 14 IDs; chạy job có trạng thái và resume đúng; smoke toàn bộ; đủ ca 10/15/20 robot và lỗi tiêm; bảng/plots/replay truy được artifact |
| P8 | Profile máy, tuning, train đủ budget, benchmark và báo cáo | configs studies, runs, reports | Đủ checkpoint độc lập, protocol đã khóa, KPI/CI và failure coverage đầy đủ; báo cả kết quả âm; kết luận trong phạm vi đã đo |

P0-P7 là hoàn thiện phần mềm sẵn sàng chạy nghiên cứu. P8 là thực nghiệm có chi phí lớn. Chỉ hoàn thành P8 mới được ghi đủ bảng kết quả thực đo; có đủ entry trong registry không đồng nghĩa đủ 14 implementation.

Việc này có nhiều thay đổi nối tiếp; thực hiện và nghiệm thu từng mốc. Lỗi ở một backend không được che bằng chạy V02 rồi ghi kết quả dưới tên backend bị lỗi; fallback trong episode được gắn cờ, dependency thiếu khiến job not_ready/failed.

## 8. Bộ test cần bổ sung

Giữ các test hiện tại và bổ sung kiểm tra hành vi có ý nghĩa:

- **Contracts/regression:** registry đúng 14 IDs, config sai bị từ chối, YAML legacy migration, checkpoint mismatch, old replay đọc được; candidate guards/profile invariance không đổi sau refactor.
- **Algorithm correctness:** Q-learning toy known optimum; replay/target update DQN; finite learning update cho A2C/PPO/QR-DQN/SAC/TD3; recurrent sequence/reset; MAPPO mask/log-prob/shared reward/critic update. Không chỉ kiểm save/load một model chưa update.
- **Optimization:** CP-SAT/MILP so với enumeration case nhỏ; queue/port/long tails, tải, pin, fractional/invalid incumbent; infeasible/timeout/exception fallback có budget còn lại.
- **Causality:** hai tape khác phần tương lai nhưng chung lịch sử cho cùng policy/forecast input; agent/critic không có future tape hoặc seed làm feature; snapshot trễ vẫn giữ guard.
- **Metrics:** task 0/completed 0, task chưa due, overdue pending, charge cancelled/pending, mid-tick charging, collision/assertion failures, lost-task detection, deadlock recurring/censored, percentile đơn vị và mẫu số đúng.
- **Evaluation:** ghép cả initial state hash và tape hash; metadata thiếu training seed không tự thành replica độc lập; duplicated checkpoint content/lineage; missing whole replica/tape phát hiện từ protocol manifest; failed run không biến mất khỏi bảng.
- **Runtime/integration:** multiprocessing spawn Windows, one-worker CPU, 10/15/20 robot; chặn đường, pause robot, pin thấp, cổng bận, policy NaN, inference quá budget; 1-hour validation không bị thay bằng short smoke.
- **UI/replay:** chọn variant/checkpoint, hiển thị scalar/vector/per-robot action đúng loại, reset LSTM/MAPPO state, nhãn LIVE/REPLAY, status và lý do fallback; render headless chỉ là một phần kiểm tra GUI.

Timing tests dùng clock giả hoặc fixture latency để kiểm accounting; tốc độ thật được profile riêng, tránh test flaky phụ thuộc tải máy. Recurrent và multi-agent phải kiểm chuỗi nhiều bước, không chỉ snapshot đơn lẻ.

## 9. CLI, cấu hình và artifact giao cuối

Các lệnh sau là API dự kiến, chưa tồn tại cho tới khi triển khai:

```powershell
python -m fleetrl methods
python -m fleetrl doctor --method qrdqn_cpsat
python -m fleetrl train --config configs/methods/dqn_cpsat.yaml --seed 11 --output runs/dqn/seed_11
python -m fleetrl eval --config configs/methods/dqn_cpsat.yaml --method dqn_cpsat --checkpoint runs/dqn/seed_11/best_model.zip --scenario S4 --seed 2000 --output runs/eval_dqn_S4_2000
python -m fleetrl study --config configs/studies/full_14.yaml --phase plan --output runs/study14
python -m fleetrl study --config configs/studies/full_14.yaml --phase smoke --output runs/study14
python -m fleetrl study --config configs/studies/full_14.yaml --phase train --output runs/study14
python -m fleetrl study --config configs/studies/full_14.yaml --phase test --output runs/study14
python -m fleetrl report --study runs/study14 --output reports/study14
```

`phase plan` chỉ validate dependency/config/protocol, liệt kê job và dự toán từ profile đã đo; không tự train. Runner resume bằng job ID + source/config/protocol/checkpoint hash, không chỉ kiểm thư mục tồn tại. Job lỗi vẫn lưu traceback/partial evidence; có thể chạy tiếp các job độc lập, study tổng có trạng thái incomplete/failed cho tới khi xử lý xong.

Cấu hình giao: 14 file `configs/methods/`, study `smoke_14`, `validation`, `full_14`, confirmatory, compact-state và continuous-PPO controls. Không dùng cờ rời rạc không được ghi vào manifest để thay mô hình.

Artifact tối thiểu:

- `study_manifest.json`, `jobs.jsonl` hoặc SQLite job index: expected grid, trạng thái, nguồn, config, split, trial/budget và phần cứng.
- Từng checkpoint bundle: `variant_id`, algorithm, action/observation/metric schema, encoder, backend, chế độ sạc, source/config/weights hashes, seed học, parent checkpoint, số bước và optimizer/replay state thích hợp.
- `episodes.csv/jsonl`, `summary.json`: tất cả ca, metric thiếu, denominator, complete/failed status và CI; raw events/decisions/replay đủ để truy lại số liệu.
- `algorithm_comparison.csv/md`: 14 dòng chính, nhóm so sánh, trạng thái kiểm chứng, throughput/trễ/tồn/an toàn/latency/tài nguyên và liên kết chi tiết.
- `metrics_dictionary.csv/md`: đủ M01-M16, key/unit/formula/direction/denominator/censoring/scope/schema.
- Plots: learning-vs-steps, learning-vs-wall-time, CI theo scenario, throughput-lateness và quality-latency trade-off, OOD, năng lượng/sạc và các case lỗi.
- README/ABOUT/CONTRACTS/traceability cập nhật theo API thật; `reports/VERIFICATION_...` mới có hash source và phạm vi kiểm chứng. Report cũ vẫn là lịch sử, không được dùng như bằng chứng source mới đã đạt.

## 10. Ngân sách và điều kiện kết thúc

Giả định 3 training seed cho mỗi một trong 11 phương án học:

- **33 lần train**, mỗi lần 300000 fleet decision steps danh nghĩa: **9.900.000 bước**, chưa gồm tuning, controls, hoặc overshoot rollout.
- **33 checkpoint + 3 controller không RL = 36 controller instance**.
- Test chính **36 x 9 scenario x 5 tape = 1620 ca**, mỗi ca 3600 s, tương đương **1.166.400 fleet decisions**.
- Compact-state controls, frozen-policy backend tests, continuous-PPO control, ID/OOD paired tests và confirmatory tape chưa gặp là budget bổ sung; study planner phải liệt kê riêng.
- Tuning/smoke/validation trong train không nằm trong các tổng trên. Không chuyển kết quả nhiều checkpoint của cùng lần train thành nhiều seed học.
- Ước lượng thời gian từng method từ rollout + update + validation + logging đo trên máy thật, kèm RAM/VRAM. Không ngoại suy toàn bộ 14 bộ từ tốc độ PPO hoặc solver p95. Chạy seed tuần tự mặc định; chỉ tăng số job cùng lúc sau khi đo tài nguyên, không tranh CPU trong latency benchmark.

Hoàn thành phần chỉnh source khi đủ 14 đường thực thi thật, đủ schema/instrumentation M01-M16, tất cả checks bắt buộc phù hợp đã đạt, CLI/GUI/study runner và docs chạy theo hướng dẫn. Hoàn thành nghiên cứu khi có đủ dữ liệu theo protocol, coverage/failure được công bố và kết luận dựa trên validation/test đã định nghĩa. Một phương án không đạt latency hoặc hiệu quả vẫn phải hiện trong bảng với trạng thái đúng; không bắt buộc mọi phương án đạt ngưỡng để báo cáo nghiên cứu trung thực.

## 11. Tài liệu kỹ thuật đã đối chiếu khi lập kế hoạch

- SB3 xác định action space hỗ trợ và lưu ý số bước thực có thể vượt budget do rollout; dùng để thiết kế adapter/step accounting, không suy ra thứ hạng trên FleetRL: [RL Algorithms](https://stable-baselines3.readthedocs.io/en/master/guide/algos.html).
- Recurrent PPO yêu cầu truyền LSTM state và episode-start flags, gồm MultiInput policy: [Recurrent PPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html).
- QR-DQN có MultiInput policy và học phân phối returns: [QR-DQN](https://sb3-contrib.readthedocs.io/en/master/modules/qrdqn.html).
- Backend MILP dự kiến dựa trên API MPSolver/SCIP; vẫn cần probe bản cài thật: [OR-Tools MIP example](https://developers.google.com/optimization/mip/mip_example).
- MAPPO có code tham chiếu chính thức; adapter FleetRL và bài toán task/charging cần tự kiểm chứng: [marlbenchmark/on-policy](https://github.com/marlbenchmark/on-policy).
- Quy trình báo cáo nhiều run và độ bất định tham khảo [Deep RL that Matters](https://arxiv.org/abs/1709.06560) và [Statistical Precipice](https://arxiv.org/abs/2108.13264); đây không phải bằng chứng phương án nào sẽ thắng trên FleetRL.
