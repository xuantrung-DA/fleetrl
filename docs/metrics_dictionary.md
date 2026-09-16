# Metrics dictionary (schema 2)

| ID | Name / keys | Unit | Formula | Denominator | Scope |
| --- | --- | --- | --- | --- | --- |
| M01 | throughput: throughput_per_hour | tasks/hour | completed / (sim_time_s/3600) | simulated hours | episode |
| M02 | completion_backlog: pending, completion_rate | tasks, fraction | completed/arrived; pending=arrived-completed | arrived tasks | episode |
| M03 | lateness: total_lateness_s | seconds | sum(max(0, completion_or_T-deadline)) | all arrived tasks | episode |
| M04 | deadline_violations: deadline_violation_rate | tasks, fraction | late_due / due | deadline<=T | episode |
| M05 | task_wait: mean_wait_s | seconds | sum(pickup_start_or_T-created)/arrived | arrived tasks | episode |
| M06 | energy: consumed_wh, energy_per_completed_wh | Wh, Wh/task | consumed/completed; initial+charged-consumed-final | completed tasks | episode |
| M07 | travel: empty_distance_ratio | metres, fraction | empty_distance/total_distance | traversed metres | episode |
| M08 | charge_wait: mean_charge_wait_s | seconds | mean(start_or_cancel_or_T-requested) | accepted charge requests | episode |
| M09 | charger_utilization: charger_utilization, port_occupied_utilization | seconds, fraction | active_seconds / (ports*T) | port-seconds | episode |
| M10 | safety: safety_incidents | incidents | unique state audit incidents + physical safety counters | all committed states | episode |
| M11 | deadlock_recovery: deadlock_recovery_rate, deadlock_recovery_s | incidents, seconds | confirmed_resolved / detected; resolved-detected | detected incidents | episode |
| M12 | online_latency: decision_p95_ms, latency_components_ms | milliseconds | percentiles of end-to-end measured decisions | dispatch decisions | episode |
| M13 | solver_fallback: solver_nonoptimal_rate, solver_timeout_rate, fallback_rate | calls, fraction | nonoptimal/calls; confirmed timeouts/calls; fallback/dispatch | solver calls or dispatch explicitly | episode |
| M14 | training_resources: resources | steps, updates, seconds, bytes | actual steps/updates, measured wall, max simultaneous process-tree RSS | training run | training manifest |
| M15 | statistical_stability: statistics, paired_vs_reference | mean, SD, CI95 | sample SD, crossed tape x training-seed bootstrap CI95 | complete expected run grid | study summary |
| M16 | generalization: generalization | KPI change, fraction | OOD-control; (OOD-control)/abs(control) | matched workload controls | study summary |

Null means unavailable or an undefined denominator. Pending tasks/requests are right-censored at T.
FEASIBLE/UNKNOWN do not establish a timeout. The solver wrapper leaves an unreported termination reason null.
Episode output includes M01–M13. M14 comes from training manifests; M15–M16 require the study grid.
