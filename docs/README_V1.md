# FleetRL — điều phối công việc và sạc cho đội robot kho

Source Python cho đề tài **Joint Task and Charging Coordination for Heterogeneous Warehouse Robot Fleets using Reinforcement Learning and Optimization**. Mục tiêu là cài môi trường, chạy smoke test rồi huấn luyện PPO trên simulator 2D; sau đó phân tích và sửa các run có kết quả kém bằng log/benchmark đã chuẩn bị.

Một PPO tập trung chọn **6 profile chi phí**; OR-Tools CP-SAT chọn task/chờ/reposition/sạc, cổng và đích SoC; planner và simulator thực thi chuyển động, pin và sự cố. Hỗ trợ tối đa 20 robot, 3 loại năng lực, 4 cổng, sạc 60/80/95%, event tape tái lập, baseline/ablation và dashboard/replay cục bộ. Bản `entities` dùng masked entity encoder + attention pooling; bản `mlp` giữ kiến trúc 2×128 của proposal để đối chiếu.

**Đây là code sẵn để train, không phải cam kết policy đã tối ưu.** Một smoke run xác minh rollout/update/save/load/eval hoạt động. Chỉ benchmark sau train dài và nhiều seed mới cho biết PPO có hơn optimizer hay không. Không có mô hình robot thật, ROS, camera, API trả phí hay dịch vụ cloud bắt buộc.

## Đọc hai file nào?

- **README.md:** cài, kiểm tra, train/resume, đánh giá, benchmark, GUI/replay, tuning và xử lý lỗi.
- **ABOUT.md:** sơ đồ luồng, schema, observation/action/reward, năng lượng, cơ chế checkpoint, file nào phụ trách gì và inventory class/function để agent sửa đúng chỗ.

`docs/PROPOSAL.md` giữ proposal gốc; `docs/proposal_traceability.md` đối chiếu F01–F23 và giới hạn; `docs/rl_architecture.md` giải thích lựa chọn PPO/entity encoder, nguồn và ablation. `reports/VERIFICATION.md` ghi rõ lệnh đã chạy, số liệu và phạm vi chưa đo; dữ liệu kèm gồm `smoke_report.json`, `acceptance.json`, `junit.xml` và environment freeze.

## 1. Yêu cầu máy và cài đặt

Dùng **Python 3.12** trong virtual environment. Môi trường kiểm chứng là Linux/Python 3.12.14; chưa kiểm cài/chạy trên Windows hoặc CUDA. Cần Internet ở lần đầu tải các wheel. Chạy simulator và PPO nhỏ bằng CPU trước; GPU RTX 4050 là tùy chọn và không thay thế CPU cho solver/planner. Không cần dữ liệu tải bên ngoài: map và task tape được sinh cục bộ theo seed.

Giải nén ZIP rồi mở terminal **ngay trong thư mục có `pyproject.toml`, `README.md` và `src/`**. Không chạy trực tiếp từng file trong `src/fleetrl/`.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Có thể gọi Python trong `.venv` bằng đường dẫn đầy đủ như trên nếu máy chặn activation script. Nếu muốn dùng các lệnh `python ...` trong tài liệu:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Lệnh trên cài wheel PyTorch CPU từ index chính thức để tránh tải các gói CUDA không cần cho lần train đầu. Đã kiểm cài mới thành công trên Linux/Python 3.12 với `torch==2.14.0+cpu` từ index này, sau đó cài `requirements.txt`. Windows và thực thi trên GPU/CUDA chưa được kiểm; các phép thử cụ thể theo từng môi trường nằm trong `reports/VERIFICATION.md`. Nếu index CPU không truy cập được, có thể bỏ lệnh index riêng và chạy `python -m pip install -r requirements.txt` từ index mặc định; bản wheel có thư viện CUDA vẫn có thể chạy bằng CPU nhưng tải lớn hơn. Nếu đã chủ động cài PyTorch CUDA phù hợp máy, bỏ bước cài CPU rồi chạy requirements/doctor.

`requirements.txt` khóa các dependency trực tiếp đã kiểm và cài editable package + pytest. `pyproject.toml` khai báo package; `reports/environment-linux-py312.txt` giữ full freeze của môi trường đã kiểm chứng. Freeze ghi phiên bản trên máy kiểm chứng, không phải bộ wheel offline dùng chung mọi hệ điều hành/CUDA. Để tái chạy cùng môi trường, ưu tiên Python và package versions đó; để cài CUDA, chọn bản PyTorch phù hợp máy rồi kiểm `doctor` trước khi train.

## 2. Nguyên tắc vận hành

- Tất cả command chạy từ thư mục gốc đã cài editable package. Xem `python -m fleetrl --help` và help của subcommand khi thay tham số.
- Mỗi experiment có output directory riêng. **Không dùng lại thư mục train cũ**, kể cả khi resume; tạo tên mới để giữ lịch sử và chứng cứ.
- `train.total_timesteps` là số quyết định môi trường, không phải tick vật lý. Một decision mặc định tiến 5 s, một tick 0,5 s. PPO làm tròn budget theo rollout đầy đủ.
- Full horizon mặc định 3.600 s. Horizon ngắn trong smoke chỉ kiểm đường chạy; các kết quả đó không đủ đánh giá lợi ích sạc dài hạn.
- Không mở GUI khi đo tốc độ headless. Tốc độ phát GUI không thay giờ mô phỏng dùng tính KPI.

## 3. Kiểm máy, smoke và bắt đầu train

Sau khi activate venv (trên Windows, nếu không activate được, thay `python` bằng `.\.venv\Scripts\python.exe` trong các lệnh dưới):

```bash
python -m fleetrl doctor
python -m pytest -q
python -m fleetrl smoke --output runs/smoke_001 --train-steps 128
```

Smoke tạo `runs/smoke_001/smoke_report.json`. Thành công phải có `status: "passed"`, từng mục `[PASS]` và process exit code 0. Nếu fail, report giữ exception; sửa lỗi trước train dài. Smoke chạy thật: Gymnasium/SB3 env checker, reset/tape/mask, rollout 120 s ở 10/15/20 robot, PPO update trên 5 robot, kiểm tham số finite, save/load, inference seed giữ riêng, resume và paired benchmark nhỏ. Nó không thay regression suite `pytest` và không đo convergence.

Lệnh train chính:

```bash
python -m fleetrl train --config configs/train.yaml --output runs/H_seed11 --seed 11
```

Lệnh dùng `entities`, CPU, 2 env, curriculum và 300.000 quyết định theo config. Số bước thực được làm tròn theo rollout. Không cần chuẩn bị dataset hoặc điền token/API key. Dừng bằng Ctrl+C sẽ cố lưu checkpoint gián đoạn nếu model còn hữu hạn; kiểm manifest trước resume.

| Cấu hình có sẵn | Mục đích |
| --- | --- |
| `configs/train.yaml` | Bản chính, đội tham chiếu 15 robot, horizon 3.600 s, entities, 300k bước. |
| `configs/train_20.yaml` | Đội đích 20 robot, 120 đơn/h, entities. |
| `configs/mlp.yaml` | Actor/critic MLP tham chiếu proposal, cùng budget chuẩn. |
| `configs/smoke.yaml` | 5 robot, 120 s, 1 env, rollout 64, batch 32, 128 bước; chỉ kiểm pipeline. |

Một vài biến thể hợp lệ:

```bash
python -m fleetrl train --config configs/train_20.yaml --output runs/H20_seed11 --seed 11
python -m fleetrl train --config configs/mlp.yaml --output runs/MLP_seed11 --seed 11
python -m fleetrl train --config configs/train.yaml --output runs/H_cpu1_seed12 --seed 12 --n-envs 1
python -m fleetrl train --config configs/train.yaml --output runs/H_cuda_seed11 --device cuda --seed 11
```

Chỉ chạy CUDA sau khi `doctor` xác nhận. `--robots N` đồng thời đặt demand=N×6 đơn/h; nếu muốn đổi số robot nhưng giữ demand khác, chỉnh YAML thay vì dùng override đó. Muốn sửa PPO sâu hơn, chỉnh nhóm `train:` trong YAML; tham số chưa có flag CLI vẫn đọc được từ YAML theo bảng đầy đủ ở `ABOUT.md`.

Xem TensorBoard trong một terminal cùng venv:

```bash
python -m tensorboard.main --logdir runs/H_seed11/logs
```

Mở địa chỉ local mà TensorBoard in ra. Thư mục có thể chứa nhiều stage; đọc cả `logs/progress.csv` và `validation.json`.

## 4. Resume và đánh giá model

Resume vào **thư mục mới**, giữ encoder và environment/reward cùng experiment. `--steps` là **số bước học thêm**, không phải tổng muốn đạt:

```bash
python -m fleetrl train --config runs/H_seed11/config.yaml --resume runs/H_seed11/final_model.zip --output runs/H_seed11_more --steps 300000 --no-curriculum
```

Checkpoint định kỳ và `interrupted_model.zip` cũng có thể dùng ở `--resume` nếu kiểm tra nạp được và finite. Resume A phải giữ thêm `--ablation-charging`; flag đó quyết định chế độ sạc của lệnh train. Encoder khác sẽ bị từ chối. Resume không tái hiện bit-identical simulator/rollout/RNG của tiến trình cũ.

Đánh giá model trên một scenario:

```bash
python -m fleetrl eval --config runs/H_seed11/config.yaml --method hybrid --checkpoint runs/H_seed11/best_model.zip --scenario S4 --seed 2000 --output runs/eval_H_S4_2000
python -m fleetrl eval --config configs/train.yaml --method fixed --scenario S4 --seed 2000 --output runs/eval_B1_S4_2000
```

**Checkpoint không tự áp lại environment config.** Luôn truyền đúng `--config`, rồi chỉ đổi scenario/horizon có chủ đích. `--method ablation` tắt sạc nâng cao; dùng model A đã train riêng. Không đưa model H vào chế độ A rồi gọi đó là ablation đã retrain.

Lưu một case cố định để debug và chạy nhiều controller cùng input:

```bash
python -m fleetrl scenario --config configs/train.yaml --name S4 --seed 2000 --output cases/S4_2000
python -m fleetrl eval --config cases/S4_2000/config.yaml --tape cases/S4_2000/tape.json --method fixed --seed 2000 --output runs/case_B1
python -m fleetrl eval --config cases/S4_2000/config.yaml --tape cases/S4_2000/tape.json --method hybrid --checkpoint runs/H_seed11/best_model.zip --seed 2000 --output runs/case_H
```

Giữ cả seed khởi tạo robot, config và tape; chỉ chung tape mà khác pin/robot ban đầu vẫn chưa là so sánh công bằng. `scenario` xuất thêm `map.json`; để dùng map tự sửa, đặt `env.map_path` tới JSON đó và giữ hash ở hồ sơ. Đường dẫn tương đối được tính từ thư mục terminal hiện tại (gốc project), ví dụ `cases/S4_2000/map.json`, không từ vị trí file YAML.

## 5. Tuning baseline và benchmark

`tune` hiện quét **fixed profile và ngưỡng sạc heuristic** trên validation; không tự huấn luyện/tune hyperparameter PPO:

```bash
python -m fleetrl tune --config configs/train.yaml --seeds 1000,1001,1002 --profiles 0,1,2,3,4,5 --thresholds 0.2,0.3,0.4 --output runs/baseline_tune
```

Đọc `tuning.json` và `selected.yaml`. B1 dùng fixed profile được chọn. Ngưỡng được tune bằng B0; muốn kết luận ngưỡng tốt cho A phải so các **model A được train theo từng ngưỡng** trên validation với cùng ngân sách, không suy B0/A có cùng ngưỡng tốt nhất. Giữ ngưỡng/config tương ứng suốt train và eval.

Benchmark B0/B1 nhỏ để kiểm luồng trên máy:

```bash
python -m fleetrl benchmark --config runs/baseline_tune/selected.yaml --methods heuristic,fixed --scenarios S1,S4 --seeds 2000,2001 --horizon 300 --output runs/bench_short
```

Benchmark H sau một seed (chưa là full protocol):

```bash
python -m fleetrl benchmark --config runs/baseline_tune/selected.yaml --methods heuristic,fixed,hybrid --scenarios S1,S4 --seeds 2000,2001 --checkpoint hybrid=runs/H_seed11/best_model.zip --output runs/bench_H_single_seed
```

Để chạy full protocol bằng các seed độc lập, helper dưới đây chạy tuần tự ba seed, có dừng khi subprocess thất bại. Mỗi seed vẫn có cùng full budget; đây không phải ba checkpoint từ một lần train:

```bash
python scripts/train_seeds.py --config runs/baseline_tune/selected.yaml --seeds 11,12,13 --output runs/full_H
python scripts/train_seeds.py --config runs/baseline_tune/selected.yaml --seeds 11,12,13 --output runs/full_A --ablation-charging
```

Sau khi đã có đủ sáu checkpoint, lệnh dưới chạy protocol 360 ca mặc định S1–S9, test seeds 2000–2004, horizon 3.600 s. Lệnh viết một dòng để chạy được cả Bash lẫn PowerShell:

```bash
python -m fleetrl benchmark --config runs/baseline_tune/selected.yaml --methods heuristic,fixed,hybrid,ablation --checkpoint hybrid=runs/full_H/seed_11/best_model.zip --checkpoint hybrid=runs/full_H/seed_12/best_model.zip --checkpoint hybrid=runs/full_H/seed_13/best_model.zip --checkpoint ablation=runs/full_A/seed_11/best_model.zip --checkpoint ablation=runs/full_A/seed_12/best_model.zip --checkpoint ablation=runs/full_A/seed_13/best_model.zip --output runs/benchmark_full
```

Không lặp lại cùng checkpoint hoặc cùng training seed để giả nhiều lần học độc lập; benchmark từ chối identity trùng. `summary.json` chỉ cho paired CI khi có đủ cấu trúc ghép tape/checkpoint; metric không có mẫu số giữ `null`, không tự biến thành 0. Đọc thêm `benchmark.png` (mean±SD; **không phải CI ghép cặp**) và raw CSV.

## 6. GUI, tạo sự cố và replay

```bash
python -m fleetrl gui --config configs/train.yaml --method fixed --seed 2000 --output runs/demo_B1
python -m fleetrl gui --config runs/H_seed11/config.yaml --method hybrid --checkpoint runs/H_seed11/best_model.zip --seed 2000 --output runs/demo_H
python -m fleetrl replay runs/eval_H_S4_2000/replay.jsonl.gz
```

| Thao tác | Tác dụng |
| --- | --- |
| Space hoặc RUN/PAUSE | Bắt đầu/tạm dừng; phiên mới bắt đầu ở trạng thái pause. |
| R hoặc RESET | Quay lại initial seed; phiên LIVE ghi vào `reset_001`, `reset_002`… riêng. |
| +/− hoặc FASTER/SLOWER | Đổi tốc độ phát 0,25×–128×; không sửa thời gian mô phỏng/KPI. |
| Click robot | Xem loại, pin, tải, task, port và booking. |
| T | Tiêm 5 đơn gấp và ghi vào event tape. |
| B | Chặn ô trống dưới con trỏ 300 giây mô phỏng; không xuyên robot/tường. |
| P | Tạm dừng robot đang chọn 60 giây mô phỏng. |
| Esc hoặc đóng cửa sổ | Kết thúc và ghi trạng thái complete/partial phù hợp. |

Reset LIVE trở về tape được sinh từ seed gốc, không giữ những sự kiện người dùng đã tiêm vào phiên trước. Muốn đối chiếu các sự kiện demo với baseline, dùng tape đã lưu của đúng `reset_###` trong lệnh `eval --tape ...` và cùng config/seed. REPLAY chỉ phát frame ghi trước, không chạy policy; nhãn REPLAY luôn hiện.

## 7. Đo tốc độ trước train dài

```bash
python -m fleetrl profile --config configs/train.yaml --robots 15 --decisions 1000 --output runs/profile_15.json
python -m fleetrl profile --config configs/train_20.yaml --robots 20 --decisions 10000 --output runs/profile_20.json
```

`profile` đo một environment fixed optimizer headless, có tốc độ end-to-end quyết định/s và online mean/p95. Nó không đo PPO update, inference neural policy hay tốc độ gộp nhiều worker. Dùng kết quả làm mốc máy thật; đo training wall-clock để dự toán cả validation/update/logging. So PPO/solver dùng latency benchmark đã gồm inference+commit; thời gian advance simulator được báo qua throughput bước, không gộp vào online decision latency.



## 8. Hiểu các file sau train

| File / thư mục trong output train | Dùng để làm gì |
| --- | --- |
| `config.yaml` | Cấu hình environment và train đã dùng; giữ cùng checkpoint. |
| `manifest.json` | Phiên bản runtime, budget và metadata run. |
| `best_model.zip` | Checkpoint tốt nhất trên validation theo safety → completed → total lateness → pending. |
| `final_model.zip` | Checkpoint cuối cùng; có thể kém `best_model.zip`. |
| `checkpoints/step_N.zip` | Snapshot định kỳ để tiếp tục/so sánh. |
| `validation.json`, `best_validation.json` | Các lần đo validation và lần tốt nhất. |
| `logs/progress.csv` | Chỉ số PPO: entropy, KL, clip fraction, value loss/explained variance, tốc độ; tên cột phụ thuộc logger SB3. |
| `logs/` TensorBoard | Đường học khi `train.tensorboard=true`. |
| `monitor/stage_*_worker_*.monitor.csv` | Episode reward/length theo worker và stage curriculum. |
| `interrupted_model.zip` | Checkpoint cứu khi run bị gián đoạn và tham số vẫn hữu hạn; chỉ có nếu nhánh cứu chạy được. |

`best_model.zip` không được chọn bằng reward đơn lẻ. Policy trước update cũng được đánh giá; nếu học làm tệ đi, best có thể vẫn là policy khởi tạo/resume. Cần kiểm `best_validation.json`, không tự hiểu chữ “best” là đã tốt hơn heuristic hoặc B1.

Mỗi run đánh giá lưu `tape.json`, `decisions.jsonl`, `replay.jsonl.gz`, `metrics.json`, `manifest.json`; file `runs.sqlite3` bên cạnh lập chỉ mục run. Benchmark tổng hợp `episodes.csv`, `episodes.jsonl` và `summary.json` với kết quả nhóm/CI; `benchmark.png` là biểu đồ mean±SD. Giữ nguyên các file gốc khi báo cáo kết quả, không chỉ giữ ảnh chart.

## 9. Protocol để có kết quả đáng tin

1. Chạy smoke và sửa lỗi engine trước khi tăng budget. Kiểm train log không NaN, model save/load được, evaluation kết thúc và có số liệu các đơn chưa xong.
2. Dùng train seeds 0–199 và map A. Validation dùng seeds từ 1.000 tách train; test cuối dùng 2.000–2.004, map B chỉ xuất hiện ở bài OOD. Không dùng test để chọn model hoặc chỉnh reward.
3. Tune B1 trên validation: so cả sáu fixed profiles cùng budget. Khóa profile tốt nhất; B0 là heuristic, không dùng model checkpoint giả.
4. Train H đầy đủ bằng **ít nhất 3 training seed độc lập**; train A bỏ sạc nâng cao bằng 3 seed với cùng budget. So `entities`/`mlp` là một ablation kiến trúc riêng, không đánh đồng với ablation sạc.
5. Đóng config/source/checkpoint trước test. Full proposal là 9 scenario × 5 tape × (3 H + 3 A + B0 + B1) = **360 ca, mỗi ca 1 giờ mô phỏng**. Phải cung cấp đủ checkpoint để chạy đúng số ca; một checkpoint không thay cho ba training seed.
6. Báo chênh lệch ghép cặp với B1, mean/std và CI, cùng throughput, pending, tardiness, energy, SoC đầu/cuối, charge wait, sự cố và online latency. Đơn chưa hoàn tất vẫn tính vào KPI; trễ/chờ của chúng là cận dưới quan sát tới cuối ca.

Mục tiêu +5% throughput, trễ không tăng quá 5%, p95≤250 ms và solver timeout<5% là **mục tiêu nghiên cứu** trong proposal. Nếu chưa đo hoặc chưa đạt, ghi đúng như vậy. Demo một phiên không thay 360 ca; CI từ ít seed còn bất định. Năng lượng/đơn thấp với pin cuối thấp hơn không tự chứng minh tiết kiệm điện.

## 10. Khi train xong nhưng kết quả xấu

| Dấu hiệu | Kiểm trước | Thay đổi có thể thử trên validation |
| --- | --- | --- |
| Reward tăng nhưng completed giảm/pending tăng | Thành phần reward, terminal penalty, từng nhóm tải/ưu tiên, số đơn blocked | Kiểm reward và candidate selection; không chỉnh chỉ để đường reward đẹp. |
| Action thay đổi nhưng kế hoạch gần như giống nhau | So sáu fixed profiles trên cùng tape; xem cost/candidate/reason trong decision log | Làm rõ sự khác biệt objective/candidate và áp lực sạc trước khi tăng network. |
| Robot chờ lâu, throughput thấp | Deadlocks, unresolved, route rejects, queue/booking, cổng bị chiếm | Sửa map/recovery/ETA hoặc giả định vận hành; đây có thể là lỗi engine, không phải PPO. |
| Solver UNKNOWN/fallback nhiều | Số ứng viên, time limit, tốc độ CPU, planner/commit latency | Giảm shortlist hoặc n_envs; đo lại. Giữ budget so sánh nhất quán. |
| Entropy sụp, KL/clip fraction cao | `logs/progress.csv`, learning rate, epochs, target KL | Giảm learning rate/epochs; điều chỉnh entropy có kiểm soát; giữ lại model trước thay đổi. |
| Value loss lớn, explained variance xấu | Quy mô reward, terminal penalty, sparse charging benefit, rollout length | So MLP/entity, xem credit assignment, thay một nhóm tham số mỗi trial. |
| Loss/observation có NaN | Log lỗi và checkpoint cuối hữu hạn; pin/timing/mask/normalization | Sửa phép tính hoặc config. Không tiếp tục từ checkpoint có tham số NaN. |
| Best kém B1 trên test | CI, tính ghép tape, đủ seed, cấu hình model/ablation | Báo kết quả âm. Dùng validation cho experiment mới; không tune trực tiếp test rồi tiếp tục gọi đó là holdout. |

Khi nhờ agent khác debug, gửi thư mục run gồm config/manifest/validation/progress, metrics/decision log của tape xấu và checkpoint tương ứng. Yêu cầu agent đọc `README.md` + `ABOUT.md`, tái lập một ca xấu trước, phân loại engine/optimizer/learning/evaluation rồi mới sửa. Không cần gửi venv.

## 11. Lỗi cài/chạy thường gặp

| Lỗi | Cách xử lý |
| --- | --- |
| `No module named fleetrl` | Đang dùng sai Python hoặc chưa `python -m pip install -r requirements.txt` ở đúng thư mục. Gọi `python -m pip ...` bằng chính interpreter sẽ train. |
| PowerShell không cho activate | Dùng `.\.venv\Scripts\python.exe` thay cho `python`; không cần đổi policy toàn máy. |
| `torch.cuda.is_available()` là false | Chạy CPU trước. Cài PyTorch/CUDA đúng máy nếu cần; không đặt device=cuda khi doctor chưa thấy GPU. |
| Multiprocessing trên Windows | Dùng CLI module entrypoint. Nếu tích hợp Python riêng, bảo vệ điểm khởi chạy bằng `if __name__ == '__main__':`; thử n_envs=1 để tách lỗi spawn. |
| Output directory đã có | Chọn thư mục run mới; resume cũng ghi sang thư mục mới. |
| Batch size lớn hơn rollout | Giữ `batch_size <= n_steps*n_envs`; **phải chia hết** rollout theo validation của train. |
| Config không nhận key | Key sai là lỗi chủ đích; xem các field trong `FleetConfig`/`TrainConfig`, không bỏ qua typo. |
| Không mở được Pygame | Cần desktop/display. Train/evaluate headless vẫn chạy; headless smoke GUI dùng SDL dummy chỉ kiểm render path. |
| Train chậm hơn ước lượng | CPU solver/planner thường chi phối; đo tốc độ máy thật. Không suy thời gian train từ GPU VRAM hoặc p95 đơn lẻ. |

## 12. Phạm vi kiểm chứng và giới hạn

Bộ tests kiểm số học năng lượng, scheduling/candidate/fallback, reservation/collision, seed/tape, observation/mask và PPO serialization; smoke command chạy đường tích hợp. `reports/VERIFICATION.md` cùng JSON/JUnit đi kèm là nguồn cho **số test và lệnh thật đã chạy**, cấu hình/horizon/seed, runtime và lỗi đã ghi nhận. Source có đầy đủ đường train nhưng không kèm tuyên bố đã đạt full benchmark hay policy chất lượng cao khi chưa chạy.

Planner ưu tiên/rolling horizon là bảo thủ và không đầy đủ: có tình huống tồn tại lời giải nhưng nó vẫn HOLD hoặc cần recovery. Mô hình năng lượng và deadline là giả định nghiên cứu, cần hiệu chỉnh trước vận hành thực. Replay là bản ghi với nhãn REPLAY. Resume giữ policy/optimizer nhưng không phục hồi chính xác từng RNG, rollout và tick của tiến trình cũ. Các khác biệt so proposal được ghi trong `docs/proposal_traceability.md`.
