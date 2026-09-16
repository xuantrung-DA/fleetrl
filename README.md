# FleetRL v2 — 14 methods, 16 metric groups

Warehouse robot fleet task and charging coordination: 11 learning methods and 3 deterministic controllers share the physical simulator and planner. Each method has its own file in src/fleetrl/methods/; KPI formulas and statistical aggregation live in src/fleetrl/metrics.py.

- [Hướng dẫn train, study, resume và nghiệm thu](docs/READY_TO_TRAIN.md)
- [Bảng metrics và công thức](docs/metrics_dictionary.md)
- [Rà soát trước train: bằng chứng của phiên bản trước khi dọn source](reports/AUDIT_PRETRAIN.md)
- [Hướng dẫn kiểm tra code và chuẩn bị Git](docs/DEVELOPMENT.md)
- [Báo cáo kiểm chứng v2 trước đợt rà soát](reports/VERIFICATION_V2.md)
- [Plan đã duyệt](docs/IMPLEMENTATION_PLAN.md)
- [Proposal gốc](docs/PROPOSAL.md)

Use Python 3.12 and the isolated .venv. Run from this directory.

```powershell
.\.venv\Scripts\python.exe -m fleetrl doctor
.\.venv\Scripts\python.exe -m fleetrl study plan --config configs/study.yaml
.\.venv\Scripts\python.exe -m fleetrl train --config configs/methods/ppo_cpsat.yaml --output runs/ppo/seed11
```

Ready to train means verified execution, updates, checkpoint loading and resume. It does not mean convergence or a proven best algorithm. Historical v1 documentation is in [docs/README_V1.md](docs/README_V1.md); its dependency versions and PPO-only commands are superseded by the v2 guide.
