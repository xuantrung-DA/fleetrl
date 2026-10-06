# Giải thích 14 phương pháp trong FleetRL

Tài liệu này giải thích 14 phương pháp được đăng ký trong `src/fleetrl/methods/registry.py`. Mục tiêu là làm rõ phương pháp đó là gì, ý tưởng chính và cách nó được dùng trong mô phỏng điều phối robot kho hàng.

## Bối cảnh chung

FleetRL có các robot cần nhận nhiệm vụ, di chuyển, sạc pin và tránh xung đột tài nguyên. Mỗi phương pháp quan sát trạng thái kho rồi quyết định cách điều phối. Các phương pháp cùng dùng mô phỏng và biểu diễn trạng thái chung, nhưng khác nhau ở cách chọn hành động, cách học và bộ lập lịch phía sau.

Trong tên phương pháp, `CPSAT` chỉ bộ lập lịch dùng CP-SAT; `MILP` chỉ bộ tối ưu quy hoạch tuyến tính nguyên hỗn hợp; `MPC` là điều khiển tối ưu hóa theo chân trời trượt. RL là học tăng cường: tác tử thử hành động, nhận thưởng/phạt và điều chỉnh chính sách để tăng tổng thưởng dài hạn. `CPSAT`/`MILP` xử lý tính khả thi và xung đột của lịch; không phải tên một thuật toán học.

Registry có 11 phương pháp học được và 3 bộ điều khiển không học: `heuristic`, `fixed_cpsat`, `forecast_mpc_milp`. Các số V01–V14 là mã nhận diện trong registry.

## Danh sách 14 phương pháp

### V01 — Heuristic (`heuristic`)

**Là gì:** Bộ điều khiển dựa trên luật ưu tiên, không cần huấn luyện. **Ý tưởng:** Dùng các quy tắc dễ hiểu để đưa ra quyết định nhanh. **Hoạt động:** Ưu tiên nhiệm vụ theo mức ưu tiên và thời gian chờ, xét robot gần nhất còn đủ điều kiện; sạc theo ngưỡng pin. Bộ này làm baseline đơn giản để so sánh với các cách tối ưu/học.

### V02 — Fixed CP-SAT (`fixed_cpsat`)

**Là gì:** Bộ điều khiển xác định dùng hồ sơ mục tiêu cố định cùng bộ lập lịch CP-SAT. **Ý tưởng:** Thay vì học trọng số từ dữ liệu, chọn cố định các tiêu chí và để solver tìm cách gán lịch khả thi, tốt theo các tiêu chí đó. **Hoạt động:** Tạo các lựa chọn nhiệm vụ/sạc hợp lệ, mô hình hóa giới hạn robot, nhiệm vụ, cổng sạc và thời gian; CP-SAT giải bài toán rồi trả về lịch. Đây là baseline tối ưu hóa có sạc nâng cao.

### V03 — Tabular Q-learning với CP-SAT (`qlearning_cpsat`)

**Là gì:** Q-learning dạng bảng, kết hợp với backend lập lịch CP-SAT. **Ý tưởng:** Học giá trị Q(s,a), tức mức hữu ích ước tính của hành động a trong trạng thái s. **Hoạt động:** Trạng thái được rời rạc hóa thành khóa hữu hạn; có sáu hành động tương ứng các hồ sơ mục tiêu. Mỗi lần chuyển trạng thái, cập nhật Q theo thưởng hiện tại cộng giá trị tốt nhất ước lượng ở trạng thái kế tiếp. Q-learning chọn hồ sơ; bộ lập lịch vẫn xử lý lịch cụ thể và tính khả thi.

### V04 — DQN với CP-SAT (`dqn_cpsat`)

**Là gì:** Deep Q-Network, dùng mạng nơ-ron để xấp xỉ giá trị Q thay vì lưu bảng. **Ý tưởng:** Mạng có thể khái quát giữa các trạng thái lớn hoặc ít gặp. **Hoạt động:** DQN học giá trị cho các hành động rời rạc, lấy mẫu trải nghiệm đã lưu và dùng mạng mục tiêu ổn định cập nhật. Khi chạy, hành động chọn hồ sơ điều phối; CP-SAT lo phần xếp lịch khả thi.

### V05 — A2C với CP-SAT (`a2c_cpsat`)

**Là gì:** Advantage Actor-Critic. **Ý tưởng:** Actor chọn hành động, Critic ước lượng giá trị trạng thái; phần “advantage” đánh giá hành động có tốt hơn mức kỳ vọng hay không. **Hoạt động:** Thu thập một đoạn trải nghiệm, tính lợi thế và cập nhật đồng thời chính sách cùng bộ ước lượng giá trị. Trong FleetRL nó chọn hồ sơ rời rạc, còn CP-SAT xử lý lịch và ràng buộc.

### V06 — PPO với CP-SAT (`ppo_cpsat`)

**Là gì:** Proximal Policy Optimization, thuật toán policy-gradient. **Ý tưởng:** Cải thiện chính sách từng bước có giới hạn để tránh một lần cập nhật làm hành vi thay đổi quá mạnh. **Hoạt động:** Thu thập trải nghiệm, tính lợi thế (GAE trong cấu hình hiện tại), rồi tối ưu mục tiêu PPO với clipping. PPO chọn hồ sơ hành động; CP-SAT tạo lịch triển khai thỏa ràng buộc. Đây là cấu hình PPO chuẩn để so sánh.

### V07 — QR-DQN với CP-SAT (`qrdqn_cpsat`)

**Là gì:** Quantile Regression DQN, biến thể DQN học phân phối lợi ích thay vì chỉ một giá trị trung bình Q. **Ý tưởng:** Biết độ biến động/rủi ro của kết quả giúp biểu diễn các tình huống có thưởng không chắc chắn tốt hơn. **Hoạt động:** Mạng dự đoán nhiều phân vị của tổng thưởng (cấu hình hiện tại dùng 50 quantile); chính sách lấy giá trị tổng hợp để chọn hành động rời rạc. CP-SAT vẫn lập lịch phía sau.

### V08 — Recurrent PPO với CP-SAT (`recurrent_ppo_cpsat`)

**Là gì:** PPO có bộ nhớ hồi quy LSTM. **Ý tưởng:** Khi một quan sát hiện tại chưa nói hết bối cảnh, chuỗi quan sát trước đó có thể giúp quyết định. **Hoạt động:** LSTM duy trì trạng thái ẩn qua thời gian; PPO cập nhật chính sách như thường lệ nhưng dựa trên biểu diễn có bộ nhớ. Hữu ích để khảo sát ảnh hưởng của lịch sử, với điều kiện chuỗi trạng thái và reset episode được xử lý đúng. Bộ lập lịch vẫn là CP-SAT.

### V09 — PPO với sạc theo ngưỡng (`ppo_threshold_cpsat`)

**Là gì:** Một biến thể PPO độc lập dùng cùng thuật toán PPO nhưng tắt cơ chế sạc nâng cao, chuyển sang điều khiển sạc theo ngưỡng. **Ý tưởng:** Làm thí nghiệm ablation để kiểm tra đóng góp của cơ chế sạc nâng cao. **Hoạt động:** PPO vẫn học cách chọn hồ sơ điều phối; phần sạc dùng chính sách ngưỡng đơn giản hơn. So sánh với V06 giúp quan sát khác biệt gắn với cấu hình sạc, nhưng không nên quy kết mọi chênh lệch chỉ do một thành phần nếu các cấu hình khác cũng khác nhau.

### V10 — SAC với hành động liên tục (`sac_cpsat`)

**Là gì:** Soft Actor-Critic, thuật toán actor-critic off-policy cho không gian hành động liên tục. **Ý tưởng:** Tối đa hóa thưởng đồng thời duy trì entropy để khuyến khích khám phá và chính sách đa dạng. **Hoạt động:** Lưu trải nghiệm vào replay buffer, cập nhật actor và critic từ mẫu cũ; entropy tự điều chỉnh theo cấu hình. Hành động liên tục được ánh xạ thành các trọng số/điều khiển của FleetRL rồi đưa vào bộ lập lịch CP-SAT.

### V11 — TD3 với hành động liên tục (`td3_cpsat`)

**Là gì:** Twin Delayed DDPG, thuật toán actor-critic off-policy cho hành động liên tục. **Ý tưởng:** Hai critic giảm xu hướng đánh giá quá cao giá trị; cập nhật actor chậm hơn để ổn định. **Hoạt động:** Học từ replay buffer, thêm nhiễu Gaussian vào hành động khi khám phá, và dùng cập nhật mục tiêu có làm mượt. Hành động được chuyển thành điều khiển liên tục của bộ tối ưu CP-SAT.

### V12 — MAPPO điều phối đa tác tử (`mappo_dispatch`)

**Là gì:** Multi-Agent PPO, trong đó mỗi robot là một tác tử chọn một lựa chọn điều phối. **Ý tưởng:** Học quyết định phân tán theo robot nhưng huấn luyện bằng thông tin toàn cục (centralized training, decentralized execution). **Hoạt động:** Mỗi robot chấm điểm các ứng viên như nhận nhiệm vụ, sạc, chờ hoặc tái định vị. Mặt nạ loại lựa chọn không hợp lệ; bộ phân xử lần lượt áp dụng hành động và từ chối xung đột nhiệm vụ, ô đỗ hoặc cổng sạc. Khác các biến thể chọn một hồ sơ toàn cục, MAPPO chọn trực tiếp quyết định theo robot.

### V13 — Forecast MPC với MILP (`forecast_mpc_milp`)

**Là gì:** Bộ điều khiển dự báo nhu cầu và tối ưu hóa chân trời trượt, không học. **Ý tưởng:** Dự đoán nhu cầu gần hạn, xem xét chuỗi quyết định kế tiếp, chỉ thực thi bước đầu rồi tính lại khi có quan sát mới. **Hoạt động:** Ước lượng tốc độ nhu cầu theo vùng bằng EWMA nhân quả; sinh các chuỗi tối đa hai thao tác khả thi cho mỗi robot; MILP chọn tổ hợp thỏa các giới hạn tài nguyên. Chỉ thao tác đầu tiên được cam kết. Dự báo không tạo nhiệm vụ tương lai giả và không đọc trước sự kiện chưa quan sát. Đây là MPC hữu hạn hai bước, không phải bộ tối ưu nhìn thấy toàn bộ tương lai.

### V14 — PPO với MILP (`ppo_milp`)

**Là gì:** PPO dùng cùng cấu trúc học như V06 nhưng thay backend CP-SAT bằng MILP/SCIP. **Ý tưởng:** Giữ chính sách PPO và các ứng viên/mục tiêu chung để khảo sát khác biệt do bộ tối ưu phía sau. **Hoạt động:** PPO học từ tương tác môi trường; lựa chọn được xử lý qua MILP để tìm lịch hợp lệ. Đây là phép so sánh backend; cần đối chiếu cấu hình, ngân sách thời gian solver và trạng thái tối ưu/timeout trước khi diễn giải kết quả.

## Cách đọc sự khác nhau

- **Khác thuật toán học:** V03–V08, V10–V12 là các họ thuật toán/chính sách khác nhau. V09 là PPO với cấu hình sạc ablation.
- **Khác bộ tối ưu:** V06 và V14 cùng là PPO nhưng dùng CP-SAT và MILP. V13 dùng MILP trong MPC dự báo.
- **Khác kiểu hành động:** Q-learning, DQN, A2C, PPO, QR-DQN, Recurrent PPO dùng lựa chọn rời rạc; SAC và TD3 dùng hành động liên tục; MAPPO đưa ra lựa chọn riêng cho từng robot.
- **Baseline không học:** V01 dựa trên luật; V02 tối ưu với hồ sơ cố định; V13 dự báo và tối ưu theo chân trời ngắn.

Không có thuật toán nào mặc nhiên tốt nhất. Kết quả phụ thuộc kịch bản, seed, ngân sách huấn luyện, cấu hình, giới hạn solver và metric. Nên so sánh trên cùng kịch bản và event tape, đồng thời báo cáo độ biến thiên giữa nhiều seed.

## Tài liệu mã nguồn đối chiếu

- Danh sách và cấu hình 14 phương pháp: `src/fleetrl/methods/registry.py`.
- Mỗi phương pháp học có module triển khai cùng tên trong `src/fleetrl/methods/`.
- Tạo ứng viên, chi phí và ràng buộc lịch dùng chung nằm trong optimizer và các backend tối ưu hóa.
