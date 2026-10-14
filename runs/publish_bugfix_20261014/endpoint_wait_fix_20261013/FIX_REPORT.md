# Sửa chu kỳ lease endpoint — kết quả kiểm chứng

Source trước6a65c6…a44a;source sau8bb96d90d4c7ac92595fdb4805c33ce5ffb958736237611a42f5b59948319c95. Người dùng cho phép tái hiện validation vàsửa khi xác nhận;không train/tune.

40S4 tự nhiên trên1000–1019 không tái hiện safetyfailure;không dùng điều đó khẳng định hếtbug. Fixture cókiểm soát validation1000 trênmaptrống,haiowner vừa rời endpoint vàchờlease nhau táihiện2route_wait_bound_exceeded ở300,5s,deadlock0. Nhánhendpointwait xóa blocked_since nên không vào recovery.

Bản sửa nhận diện chu kỳ owner/lease trongvùngapproach. Giữ timestampblocked đểvào deadlockrecovery theo deadlock_s cũ. Owner đangservice/pause khôngbị coi làcycle. Carrier chỉđược rút sangbay khi budget pin đủyêu cầu task banđầu cộng khứhồi vớitải, task/load/goal giữnguyên. Không nới routewait/reserve/chargecalendar hoặc thayđếm safety.

Sau sửa fixture hoànthành2/2task,safety/lost0,hai deadlock được ghi vàresolved. Biếnthể hai carrier đượctest;sửa đầu cònhold0completed đãgiữ XMLcaâm trước khi thêmguardedcarrierretreat.93targettest và213fulltest đạt;5test mới kiểm lại saubổsungannotation đạt. Ruff80file và193checksum đạt. SOURCE_REVIEW.json kiểmAST guard vàbyte cácmoduleprotected.

40S4 sau sửa safety/lost0. Heuristic giữnguyên;fixed giảmmean0,4completed,tăng0,4pending và186s lateness tronglượtđo;CI/t0trace giữ trongrun này,khôngtuyênbốperformance tốt hơn.198validation mởrộng sau đó bằng3controller khônghọc khôngcógate đãđo khôngđạt;timeout solver vẫnunknown.

Pha xácnhận mới3005–3009 đượcngười dùng cho phép sauđó dừng32/300 doS6/fixed3005 completion17/39<50%,safety0. Đây làfailurephụcvụ khác,cầnquyếtđịnh mụcD. Báo cáo ../readiness_nonlearning_fresh_20261013/REPORT.md vàS6_CALENDAR_WITNESS.json. Khôngsửa tiếp, khôngchạy33checkpoint. Tậpđãquan sát khônggọi làholdout hoàntoàn mớinếupháttriểnsource tiếp.


## Kết quả pha xác nhận sau phê duyệt tiếp tục

300/300 ca không học đã hoàn tất với source giữ nguyên. Safety/lost/crash=0; 6 deadlock đều được xử lý. Chỉ S6/fixed/3005 không đạt completion; 200 ca CP/MPC chưa xác minh timeout. Báo cáo cuối: ../readiness_nonlearning_fresh_20261013/FINAL_REPORT.md. Không train/tune hay chạy checkpoint học.
