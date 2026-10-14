# Fallback cục bộ khi snapshot trễ

Người dùng cho phép nghiên cứu/sửa fallback. Source `6a65c6e57562b70b681c43032207d12e570a0b31fcffde541993723a6b34a44a`.

Giữ observation trễ của policy và giới hạn freshness10s. Khi quá hạn, bộ điều khiển cục bộ dùng telemetry vật lý hiện tại, chỉ xét các task ID đã được thấy trong snapshot trễ. Không đọc trước tape hoặc đưa task chưa quan sát vào dispatch. Dùng heuristic chung, không dùng action của policy trong các chu kỳ này. Fresh plan vẫn qua toàn bộ guard commit. Vì thế kết quả có fallback không được gọi là hiệu năng thuần của learner, và cần bộ điều khiển cục bộ có telemetry mới khi triển khai.

Cùng tape delay15/heuristic1005: trước8completed/56pending, sau62/2; fixed62/2, MAPPO61/3. Safety/lost=0, p95<18ms.357/360chu kỳ dùng fallback. CP-SAT timeout còn null nên gate đó CHƯA KIỂM. FIX_VERIFICATION.json chứa số chu kỳ và bằng chứng allowlist.

105test liên quan và208test đầy đủ đạt; Ruff/format và193checksum đạt. Simulator, optimizer, energy, observation, metrics/config giữ nguyên bytes; observation/action-space AST giữ nguyên. Không train/tune; MAPPO có nhãn checkpoint train trên source cũ. Ca âm/trace nguồn trước giữ trong readiness_pause_final_20261013.

Pilot198ca phải chạy mới trên source này: ../readiness_stale_final_20261013. Không trộn với Study14 hoặc pilot source cũ. Gặp lỗi mới dừng mục D; pha>6giờ phải hỏi trước.
