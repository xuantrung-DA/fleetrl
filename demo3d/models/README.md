# Model suy luận đã chọn

Chỉ MAPPO và DQN CP-SAT train seed 11. Lock tương đối với thư mục này. Dùng `--study demo3d/models`. ZIP, sidecar và lock cùng thuộc một gói; provenance ghi hash nguồn/hash xuất và xác nhận payload không đổi. MAPPO chỉ đổi đường dẫn lineage trong comment metadata. Không chứa replay buffer, không dùng gói này để resume train. Source hash huấn luyện là lịch sử.
