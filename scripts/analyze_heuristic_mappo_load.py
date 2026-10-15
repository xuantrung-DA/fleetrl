"""Summarize paired MAPPO versus heuristic episodes and plot the demand sweep."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "heuristic_mappo_load_20261015"
REPORTS = ROOT / "reports"
METRICS = ("completion_rate", "total_lateness_s")
FACTORS = (0.5, 1.0, 1.5, 2.0, 3.0)
TEST_SEEDS = (4100, 4101, 4102, 4103, 4104)
TRAINING_SEEDS = (11, 12, 13)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def paired_stats(rows: list[dict], metric: str, seeds: tuple[int, ...]) -> dict:
    cells = defaultdict(dict)
    for row in rows:
        key = int(row["seed"])
        if key not in seeds:
            raise ValueError(f"Unexpected seed: {key}")
        method = row["method"]
        if method == "heuristic":
            replica = "heuristic"
        elif method == "mappo_dispatch":
            replica = int(row["training_seed"])
        else:
            raise ValueError(f"Unexpected method: {method}")
        if replica in cells[key]:
            raise ValueError(f"Duplicate method/seed cell: {key}/{replica}")
        cells[key][replica] = float(row[metric])
    expected = {"heuristic", *TRAINING_SEEDS}
    if set(cells) != set(seeds) or any(set(cell) != expected for cell in cells.values()):
        raise ValueError("Incomplete paired tape/checkpoint grid")
    heuristic = np.array([cells[seed]["heuristic"] for seed in seeds], dtype=float)
    mappo = np.array(
        [[cells[seed][training_seed] for training_seed in TRAINING_SEEDS] for seed in seeds],
        dtype=float,
    )
    difference = mappo - heuristic[:, None]
    rng = np.random.default_rng(20261015)
    tape_draws = rng.integers(0, len(seeds), size=(20000, len(seeds)))
    policy_draws = rng.integers(0, len(TRAINING_SEEDS), size=(20000, len(TRAINING_SEEDS)))
    bootstrap = difference[tape_draws[:, :, None], policy_draws[:, None, :]].mean(axis=(1, 2))
    return {
        "heuristic_mean": float(heuristic.mean()),
        "mappo_mean": float(mappo.mean()),
        "difference": float(difference.mean()),
        "ci_low": float(np.quantile(bootstrap, 0.025)),
        "ci_high": float(np.quantile(bootstrap, 0.975)),
        "paired_tapes": len(seeds),
        "mappo_replicas": len(TRAINING_SEEDS),
    }


def historical() -> dict:
    main = read_csv(ROOT / "runs" / "study14_current_source_testonly" / "episodes.csv")
    if len(main) != 360:
        raise ValueError("Expected exactly 360 source-transfer episodes")
    scenarios = [f"S{i}" for i in range(1, 10)]
    output = {}
    for scenario in scenarios:
        rows = [
            row
            for row in main
            if row["scenario"] == scenario and row["method"] in {"heuristic", "mappo_dispatch"}
        ]
        output[scenario] = {
            metric: paired_stats(rows, metric, (3005, 3006, 3007, 3008, 3009)) for metric in METRICS
        }
    older = read_csv(ROOT / "runs" / "study14" / "episodes.csv")
    rows = [
        row
        for row in older
        if row["scenario"] == "OOD_A" and row["method"] in {"heuristic", "mappo_dispatch"}
    ]
    output["OOD_A"] = {
        metric: paired_stats(rows, metric, (2000, 2001, 2002, 2003, 2004)) for metric in METRICS
    }
    return output


def read_new(phase: str, factor: float | None = None) -> list[dict]:
    folder = RUN / phase
    if factor is not None:
        folder /= f"factor_{factor:.1f}"
    paths = sorted(folder.glob("*.json"))
    expected = 30 if phase == "tune" else 20
    if len(paths) != expected:
        raise ValueError(f"Incomplete {phase} grid: {len(paths)}/{expected} in {folder}")
    rows = []
    for path in paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        if row["status"] != "complete":
            raise ValueError(f"Incomplete episode: {path}")
        rows.append({**row, **row["metrics"]})
    return rows


def verify_tapes(rows: list[dict]) -> None:
    hashes = defaultdict(set)
    arrivals = defaultdict(set)
    for row in rows:
        hashes[row["seed"]].add(row["tape_hash"])
        arrivals[row["seed"]].add(row["arrived"])
    if any(len(values) != 1 for values in hashes.values()):
        raise ValueError("Methods did not share identical event tapes")
    if any(len(values) != 1 for values in arrivals.values()):
        raise ValueError("Methods did not receive the same number of tasks")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compact_episode(row: dict, phase: str) -> dict:
    return {
        "phase": phase,
        "scenario": row["scenario"],
        "demand_factor": row["demand_factor"],
        "method": row["method"],
        "charge_threshold": row["charge_threshold"],
        "seed": row["seed"],
        "training_seed": row["training_seed"],
        "tape_hash": row["tape_hash"],
        "source_hash": row["source_hash"],
        "checkpoint_sha256": row["checkpoint_sha256"],
        "arrived": row["arrived"],
        "completed": row["completed"],
        "completion_rate": row["completion_rate"],
        "total_lateness_s": row["total_lateness_s"],
        "safety_incidents": row["safety_incidents"],
        "lost_tasks": row["lost_tasks"],
        "wall_time_s": row["wall_time_s"],
    }


def fmt_ci(result: dict, scale: float = 1.0, digits: int = 1) -> str:
    return (
        f"{result['difference'] * scale:+.{digits}f} "
        f"[{result['ci_low'] * scale:+.{digits}f}, {result['ci_high'] * scale:+.{digits}f}]"
    )


def line_crossings(summary: dict, metric: str) -> list[float]:
    crossings = []
    for left, right in zip(FACTORS[:-1], FACTORS[1:]):
        first = summary[left][metric]["difference"]
        second = summary[right][metric]["difference"]
        if first * second < 0:
            crossings.append(left - first * (right - left) / (second - first))
    return crossings


def plot_sweep(summary: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for axis, metric, title, scale in (
        (axes[0], "completion_rate", "Completion rate", 100),
        (axes[1], "total_lateness_s", "Total lateness, s", 1),
    ):
        for field, label, color in (
            ("heuristic_mean", "Heuristic (tuned)", "#147d64"),
            ("mappo_mean", "MAPPO (3 checkpoints)", "#ba4a3c"),
        ):
            axis.plot(
                FACTORS,
                [summary[factor][metric][field] * scale for factor in FACTORS],
                marker="o",
                linewidth=2,
                label=label,
                color=color,
            )
        axis.set(xlabel="Demand multiplier", ylabel=title, xticks=FACTORS)
        axis.grid(alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)
    fig.savefig(REPORTS / "heuristic_mappo_load_sweep_20261015.png", dpi=180)
    plt.close(fig)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    old = historical()
    table = []
    for scenario in ["S1", "S2", "S3", "OOD_A", "S4", "S9", "S5", "S6", "S7", "S8"]:
        group = (
            "tải đều"
            if scenario in {"S1", "S2", "S3", "OOD_A"}
            else ("tải căng" if scenario in {"S4", "S9"} else "khác")
        )
        completion = old[scenario]["completion_rate"]
        lateness = old[scenario]["total_lateness_s"]
        table.append(
            {
                "scenario": scenario,
                "load_group": group,
                "source": "Study14 gốc" if scenario == "OOD_A" else "lượt 360",
                "heuristic_completion_pct": completion["heuristic_mean"] * 100,
                "mappo_completion_pct": completion["mappo_mean"] * 100,
                "mappo_minus_heuristic_completion_pp": completion["difference"] * 100,
                "completion_ci95_low_pp": completion["ci_low"] * 100,
                "completion_ci95_high_pp": completion["ci_high"] * 100,
                "heuristic_total_lateness_s": lateness["heuristic_mean"],
                "mappo_total_lateness_s": lateness["mappo_mean"],
                "mappo_minus_heuristic_lateness_s": lateness["difference"],
                "lateness_ci95_low_s": lateness["ci_low"],
                "lateness_ci95_high_s": lateness["ci_high"],
                "paired_tapes": completion["paired_tapes"],
                "mappo_replicas": completion["mappo_replicas"],
            }
        )
    write_csv(REPORTS / "heuristic_mappo_scenario_paired_20261015.csv", table)
    lines = [
        "# Heuristic so với MAPPO: so sánh ghép cặp, tune S4 và quét tải",
        "",
        "## Theo kịch bản",
        "",
        "Trong bảng, H là heuristic và M là MAPPO. **Hoàn thành** là tỷ lệ đơn xử lý xong "
        "(càng cao càng tốt). **Trễ** là tổng thời gian các đơn bị quá hạn "
        "(càng thấp càng tốt). Đơn chưa xong được tính trễ đến cuối lượt chạy.",
        "",
        "Cột **chênh lệch (Δ)** lấy kết quả MAPPO trừ heuristic. Vì vậy, Δ hoàn thành "
        "dương nghĩa là MAPPO hoàn thành nhiều hơn; Δ trễ âm nghĩa là MAPPO ít trễ hơn.",
        "",
        "Khoảng số trong ngoặc vuông là **CI 95%**: khoảng ước tính cho chênh lệch "
        "khi lấy mẫu lại 5 bộ đơn hàng và 3 mô hình MAPPO, tổng cộng 20.000 lần. "
        "Nếu khoảng này chứa số 0, kết quả chưa cho thấy bên nào hơn rõ ràng ở chỉ số đó.",
        "",
        "| Nhóm | Kịch bản | Nguồn | Hoàn thành H / M (%) | Δ hoàn thành (điểm %, CI 95%) | "
        "Trễ H / M (giây) | Δ trễ (giây, CI 95%) |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in table:
        scenario = row["scenario"]
        completion = old[scenario]["completion_rate"]
        lateness = old[scenario]["total_lateness_s"]
        lines.append(
            f"| {row['load_group']} | {scenario} | {row['source']} | "
            f"{completion['heuristic_mean'] * 100:.1f} / {completion['mappo_mean'] * 100:.1f} | "
            f"{fmt_ci(completion, 100, 1)} | "
            f"{lateness['heuristic_mean']:.0f} / {lateness['mappo_mean']:.0f} | "
            f"{fmt_ci(lateness, 1, 0)} |"
        )
    lines += [
        "",
        "Nguồn S1–S9: `runs/study14_current_source_testonly/episodes.csv` "
        "(360 episode, seed 3005–3009). OOD_A vắng mặt ở lượt 360; dòng OOD_A "
        "dùng `runs/study14/episodes.csv` gốc (seed 2000–2004), do đó không gộp "
        "với lượt 360. Checkpoint MAPPO đã train trên source cũ; lượt 360 là phép "
        "thử chuyển source và seed đã được quan sát trước đó. Heuristic gốc "
        "dùng ngưỡng sạc 20% theo `selected_baselines.json`.",
        "",
        "Source hash lượt 360: `"
        + json.loads(
            (ROOT / "runs" / "study14_current_source_testonly" / "identity.json").read_text(
                encoding="utf-8"
            )
        )["source_hash"]
        + "`; source hash thí nghiệm mới: `"
        + json.loads((RUN / "identity.json").read_text(encoding="utf-8"))["source_hash"]
        + "`.",
        "",
        "## Tune ngưỡng sạc heuristic trên S4",
        "",
    ]
    tune = read_new("tune")
    verify_tapes(tune)
    threshold_rows = []
    for threshold in (0.2, 0.3, 0.4):
        subset = [
            row
            for row in tune
            if row["method"] == "heuristic" and row["charge_threshold"] == threshold
        ]
        threshold_rows.append(
            {
                "threshold": threshold,
                "episodes": len(subset),
                "mean_completion_rate": mean(row["completion_rate"] for row in subset),
                "mean_total_lateness_s": mean(row["total_lateness_s"] for row in subset),
            }
        )
    write_csv(REPORTS / "heuristic_s4_threshold_tune_20261015.csv", threshold_rows)
    chosen = min(
        threshold_rows,
        key=lambda row: (
            -row["mean_completion_rate"],
            row["mean_total_lateness_s"],
            row["threshold"],
        ),
    )["threshold"]
    for row in threshold_rows:
        lines.append(
            f"- S4 tune ngưỡng {row['threshold']:.0%}: 5 episode, "
            f"hoàn thành {row['mean_completion_rate'] * 100:.2f}%, "
            f"trễ {row['mean_total_lateness_s']:.1f} giây."
        )
    tune_mappo = [row for row in tune if row["method"] == "mappo_dispatch"]
    lines.append(
        f"- MAPPO: 15 episode (3 checkpoint × 5 tape), "
        f"hoàn thành {mean(row['completion_rate'] for row in tune_mappo) * 100:.2f}%, "
        f"trễ {mean(row['total_lateness_s'] for row in tune_mappo):.1f} giây."
    )
    lines += [
        "",
        f"Ngưỡng chọn theo quy tắc định trước (hoàn thành cao nhất, rồi trễ thấp nhất): "
        f"**{chosen:.0%}**. MAPPO có 3 checkpoint × 5 tape = 15 episode trên "
        "cùng seed tune; heuristic có 3 ngưỡng × 5 tape = 15 episode. Đây là "
        "ngân sách đánh giá bằng nhau theo số episode/tape, không phải thời gian CPU "
        "hoặc ngân sách train bằng nhau. "
        "Kết quả tune không được dùng để tuyên bố thắng trên S4; xem tập test bên dưới.",
        "",
        "## Quét tải S4 trên seed test tách biệt",
        "",
        "Giữ cấu hình S4 và burst multiplier=2; nhân `demand_per_hour` 90 đơn/giờ "
        "với 0,5×, 1×, 1,5×, 2×, 3×. Mỗi mức dùng 5 tape seed 4100–4104, "
        "heuristic đã chọn ngưỡng và 3 checkpoint MAPPO trên cùng tape. "
        "8 đơn ban đầu giữ nguyên; do đó số đơn thực tế không nhân đúng tỷ lệ tuyệt đối.",
        "",
        "| Tải | Đơn đến TB | Hoàn thành H / M (%) | Δ hoàn thành (điểm %, CI 95%) | "
        "Trễ H / M (giây) | Δ trễ (giây, CI 95%) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    sweep = {}
    sweep_rows = []
    sweep_episodes = []
    for factor in FACTORS:
        records = read_new("sweep", factor)
        verify_tapes(records)
        sweep_episodes.extend(records)
        sweep[factor] = {metric: paired_stats(records, metric, TEST_SEEDS) for metric in METRICS}
        c = sweep[factor]["completion_rate"]
        lateness = sweep[factor]["total_lateness_s"]
        mean_arrived = mean(row["arrived"] for row in records if row["method"] == "heuristic")
        sweep_rows.append(
            {
                "factor": factor,
                "mean_arrived": mean_arrived,
                "heuristic_completion_pct": c["heuristic_mean"] * 100,
                "mappo_completion_pct": c["mappo_mean"] * 100,
                "completion_difference_pp": c["difference"] * 100,
                "completion_ci95_low_pp": c["ci_low"] * 100,
                "completion_ci95_high_pp": c["ci_high"] * 100,
                "heuristic_total_lateness_s": lateness["heuristic_mean"],
                "mappo_total_lateness_s": lateness["mappo_mean"],
                "lateness_difference_s": lateness["difference"],
                "lateness_ci95_low_s": lateness["ci_low"],
                "lateness_ci95_high_s": lateness["ci_high"],
            }
        )
        lines.append(
            f"| {factor:g}× | {mean_arrived:.1f} | "
            f"{c['heuristic_mean'] * 100:.1f} / {c['mappo_mean'] * 100:.1f} | "
            f"{fmt_ci(c, 100, 1)} | "
            f"{lateness['heuristic_mean']:.0f} / {lateness['mappo_mean']:.0f} | "
            f"{fmt_ci(lateness, 1, 0)} |"
        )
    write_csv(REPORTS / "heuristic_mappo_load_sweep_20261015.csv", sweep_rows)
    plot_sweep(sweep)
    lines += ["", "![Quét tải S4](heuristic_mappo_load_sweep_20261015.png)", ""]
    safety = sum(row["safety_incidents"] for row in sweep_episodes)
    lost = sum(row["lost_tasks"] for row in sweep_episodes)
    lines.append(
        f"Trên 100 episode quét tải: {safety:.0f} sự cố an toàn và {lost:.0f} tác vụ thất lạc."
    )
    lines.append("")
    baseline_completion = sweep[1.0]["completion_rate"]
    baseline_lateness = sweep[1.0]["total_lateness_s"]
    if baseline_completion["ci_low"] > 0 and baseline_lateness["ci_high"] < 0:
        lines.append(
            "Ở mức 1× trên seed test tách biệt, MAPPO hơn heuristic đã tune ở cả "
            "hoàn thành và trễ, với cả hai CI ghép cặp nằm về phía MAPPO. "
            "Đây là bằng chứng trong phạm vi S4 và checkpoint/source hiện tại."
        )
    else:
        lines.append(
            "Ở mức 1× trên seed test tách biệt, chưa có bằng chứng MAPPO hơn "
            "heuristic đã tune đồng thời ở cả hoàn thành và trễ theo CI ghép cặp. "
            "Không thể đưa ra kết luận MAPPO thắng S4 sau tune."
        )
    lines.append("")
    for metric, label in (("completion_rate", "hoàn thành"), ("total_lateness_s", "trễ")):
        crossing = line_crossings(sweep, metric)
        ties = [factor for factor in FACTORS if abs(sweep[factor][metric]["difference"]) < 1e-12]
        if crossing:
            points = ", ".join(f"{value:.2f}×" for value in crossing)
            lines.append(
                f"Đường trung bình {label} đổi thứ tự giữa hai mức tải đã đo; "
                f"nội suy tuyến tính cho điểm giao khoảng {points}. Đây là ước lượng "
                "từ lưới 5 mức, chưa phải điểm giao chính xác hoặc bằng chứng về "
                "cơ chế nhân quả."
            )
            brackets = [
                (left, right)
                for left, right in zip(FACTORS[:-1], FACTORS[1:])
                if sweep[left][metric]["difference"] * sweep[right][metric]["difference"] < 0
            ]
            supported = any(
                (sweep[left][metric]["ci_high"] < 0 and sweep[right][metric]["ci_low"] > 0)
                or (sweep[left][metric]["ci_low"] > 0 and sweep[right][metric]["ci_high"] < 0)
                for left, right in brackets
            )
            lines.append(
                "Hai mức kề ở hai phía điểm giao đều có CI ghép cặp tách khỏi 0."
                if supported
                else "CI ghép cặp ở hai phía chưa cùng tách khỏi 0; điểm giao mới là dấu hiệu mô tả."
            )
        elif ties:
            levels = ", ".join(f"{factor:g}×" for factor in ties)
            lines.append(
                f"Hai đường trung bình {label} bằng nhau ở {levels}; "
                "chưa quan sát được đổi thứ tự giữa hai mức tải kế cận."
            )
        else:
            lines.append(
                f"Không quan sát thấy đường trung bình {label} cắt nhau trong lưới tải đã đo."
            )
    if all(not line_crossings(sweep, metric) for metric in METRICS):
        lines.append(
            "Giả thuyết có điểm giao trong khoảng 0,5×–3× chưa được xác nhận bởi thí nghiệm này."
        )
    ood = read_new("ood")
    verify_tapes(ood)
    write_csv(
        REPORTS / "heuristic_mappo_new_episodes_20261015.csv",
        [compact_episode(row, "tune") for row in tune]
        + [compact_episode(row, "sweep") for row in sweep_episodes]
        + [compact_episode(row, "ood") for row in ood],
    )
    c = paired_stats(ood, "completion_rate", TEST_SEEDS)
    lateness = paired_stats(ood, "total_lateness_s", TEST_SEEDS)
    lines += [
        "",
        "## OOD_A trên source hiện tại",
        "",
        f"Với 5 seed test 4100–4104, hoàn thành heuristic/MAPPO: "
        f"{c['heuristic_mean'] * 100:.1f}% / {c['mappo_mean'] * 100:.1f}%; "
        f"Δ={fmt_ci(c, 100, 1)} điểm %. Trễ: "
        f"{lateness['heuristic_mean']:.0f} / {lateness['mappo_mean']:.0f} giây; "
        f"Δ={fmt_ci(lateness, 1, 0)} giây. Kết quả này thuộc thí nghiệm mới, "
        "không phải lượt 360.",
        "",
        "## Giới hạn",
        "",
        "- Mỗi mức chỉ có 5 tape và 3 checkpoint; CI bootstrap nhạy với mẫu ít. "
        "Không coi nhiều checkpoint trên cùng tape là 15 tape độc lập.",
        "- `total_lateness_s` gồm cả tác vụ còn chờ tại cuối 3600 giây; "
        "vì vậy nó phản ánh cả backlog bị kiểm duyệt ở chân trời mô phỏng.",
        "- MAPPO checkpoint train trên source cũ, nên phép đo này kiểm tra khả năng "
        "chuyển source của checkpoint; không chứng minh giới hạn tối đa của MAPPO.",
        "- Quét tải chỉ thay đổi cường độ đơn trên S4; không tự động suy rộng sang S9 hoặc "
        "các dạng OOD khác.",
        "",
        "## Tái lập",
        "",
        "Chạy từ gốc repository bằng môi trường Python đã cài FleetRL. Runner kiểm tra "
        "source hash và SHA-256 của ba checkpoint trước khi chạy; episode hoàn tất được "
        "lưu dưới `runs/heuristic_mappo_load_20261015/` và có thể tiếp tục bằng "
        "cùng lệnh. Bản CSV gọn của toàn bộ 150 episode mới là "
        "`reports/heuristic_mappo_new_episodes_20261015.csv`.",
        "",
        "```powershell",
        "python scripts/run_heuristic_mappo_load.py tune",
        "python scripts/run_heuristic_mappo_load.py sweep",
        "python scripts/run_heuristic_mappo_load.py ood",
        "python scripts/analyze_heuristic_mappo_load.py",
        "```",
        "",
    ]
    (REPORTS / "heuristic_mappo_followup_20261015.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"Wrote {len(table)} scenario rows, 3 thresholds and {len(sweep_rows)} loads")


if __name__ == "__main__":
    main()
