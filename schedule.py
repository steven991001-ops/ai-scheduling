# -*- coding: utf-8 -*-
"""
Job Shop Scheduling with OR-Tools CP-SAT
- 讀取同資料夾下的 orders.csv
- 目標：最小化總完工時間 (Makespan)
- 限制條件：
  1) 同一台機台同一時間只能做一道工序
  2) 同一張訂單要依 step 順序加工
- 輸出：
  1) 終端機顯示 Makespan 和每台機台工序開始/結束時間
  2) 排程結果存到 schedule_result.csv
  3) 甘特圖存到 gantt.png
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

# 讓程式在無 GUI 的環境中也能生成圖片
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ortools.sat.python import cp_model


def load_orders(csv_path: Path):
    """
    讀取訂單資料，格式範例：
    job,step,machine,duration
    J1,1,M1,3
    ...
    """
    rows = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "job": row["job"],
                "step": int(row["step"]),
                "machine": row["machine"],
                "duration": int(row["duration"]),
            })
    return rows


def build_schedule(rows):
    """
    建立 Job Shop 排程模型，使用 CP-SAT 求解。
    """
    # 1) 整理資料結構
    jobs = defaultdict(list)      # job -> [operation]
    machine_ops = defaultdict(list)  # machine -> [operation]
    op_info = {}  # (job, step) -> {job, step, machine, duration}

    all_ops = []

    for r in rows:
        job = r["job"]
        step = r["step"]
        machine = r["machine"]
        duration = r["duration"]
        op = (job, step)

        op_info[op] = {
            "job": job,
            "step": step,
            "machine": machine,
            "duration": duration,
        }
        jobs[job].append(op)
        machine_ops[machine].append(op)
        all_ops.append(op)

    # 對每個 job 內的工序依步驟排序
    for job in jobs:
        jobs[job].sort(key=lambda op: op_info[op]["step"])

    # 2) 建立 CP-SAT 模型
    model = cp_model.CpModel()

    start = {}
    end = {}
    interval = {}

    # 每個工序的開始時間、結束時間、區間變數
    for op in all_ops:
        info = op_info[op]
        duration = info["duration"]

        start[op] = model.NewIntVar(0, 100000, f"start_{op[0]}_{op[1]}")
        end[op] = model.NewIntVar(0, 100000, f"end_{op[0]}_{op[1]}")
        interval[op] = model.NewIntervalVar(
            start[op], duration, end[op], f"interval_{op[0]}_{op[1]}"
        )

    # 3) 同一張訂單要依 step 順序加工
    for job, job_ops in jobs.items():
        for prev_op, next_op in zip(job_ops, job_ops[1:]):
            model.Add(end[prev_op] <= start[next_op])

    # 4) 同一台機台同一時間只能做一道工序
    for machine, ops in machine_ops.items():
        model.AddNoOverlap([interval[op] for op in ops])

    # 5) 目標函數：最小化 Makespan
    # 估算一個合理的上界：所有工序持續時間總和
    horizon = sum(info["duration"] for info in op_info.values())
    makespan = model.NewIntVar(0, horizon, "makespan")
    for op in all_ops:
        model.Add(makespan >= end[op])
    model.Minimize(makespan)

    # 6) 求解
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30.0
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("找不到可行排程，請檢查資料或調整限制條件。")

    return {
        "solver": solver,
        "model": model,
        "start": start,
        "end": end,
        "interval": interval,
        "all_ops": all_ops,
        "jobs": jobs,
        "machine_ops": machine_ops,
        "op_info": op_info,
        "makespan": makespan,
    }


def print_schedule_result(result):
    """
    在終端機列印 Makespan 與每台機台的工序開始/結束時間。
    """
    solver = result["solver"]
    makespan = solver.Value(result["makespan"])
    machine_ops = result["machine_ops"]
    start = result["start"]
    end = result["end"]
    op_info = result["op_info"]

    print("=" * 80)
    print(f"Makespan = {makespan}")
    print("=" * 80)

    for machine in sorted(machine_ops):
        print(f"\nMachine {machine}")
        print("-" * 60)

        # 依開始時間排序，便於判讀排程
        ops = sorted(machine_ops[machine], key=lambda op: solver.Value(start[op]))
        for op in ops:
            info = op_info[op]
            job = info["job"]
            step = info["step"]
            s = solver.Value(start[op])
            e = solver.Value(end[op])
            dur = info["duration"]
            print(
                f"  Job {job}, Step {step}: "
                f"start={s}, end={e}, duration={dur}"
            )

    print("=" * 80)


def save_schedule_csv(result, output_path: Path):
    """
    將排程結果儲存成 schedule_result.csv，欄位：
    job,step,machine,start,end,duration
    """
    solver = result["solver"]
    start = result["start"]
    end = result["end"]
    op_info = result["op_info"]
    machine_ops = result["machine_ops"]

    rows = []
    for machine in sorted(machine_ops):
        ops = sorted(machine_ops[machine], key=lambda op: solver.Value(start[op]))
        for op in ops:
            info = op_info[op]
            rows.append({
                "job": info["job"],
                "step": info["step"],
                "machine": info["machine"],
                "start": solver.Value(start[op]),
                "end": solver.Value(end[op]),
                "duration": info["duration"],
            })

    # 為了方便查看，可按 machine、start 排序
    rows.sort(key=lambda x: (x["machine"], x["start"], x["job"], x["step"]))

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["job", "step", "machine", "start", "end", "duration"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"排程結果已儲存至: {output_path}")


def save_gantt_chart(result, output_path: Path):
    """
    畫出甘特圖，並保存為 gantt.png。
    """
    solver = result["solver"]
    machine_ops = result["machine_ops"]
    start = result["start"]
    end = result["end"]
    op_info = result["op_info"]

    # 橫軸時間，縱軸為機台
    machines = sorted(machine_ops.keys())
    machine_y = {machine: i for i, machine in enumerate(machines)}

    # 每個 job 都分配一個顏色，方便辨識
    jobs = sorted({op_info[op]["job"] for op in op_info})
    color_map = {}
    cmap = plt.get_cmap("tab10")
    for idx, job in enumerate(jobs):
        color_map[job] = cmap(idx % 10)

    fig, ax = plt.subplots(figsize=(12, max(5, 1.2 * len(machines) + 1)))

    # 先畫出每台機台的工序條
    for machine in machines:
        for op in sorted(machine_ops[machine], key=lambda op: solver.Value(start[op])):
            info = op_info[op]
            job = info["job"]
            step = info["step"]
            s = solver.Value(start[op])
            e = solver.Value(end[op])
            dur = e - s

            ax.barh(
                y=machine_y[machine],
                width=dur,
                left=s,
                height=0.6,
                color=color_map[job],
                edgecolor="black",
                linewidth=0.8,
                alpha=0.9,
            )

            # 在工序中央標示 job / step
            ax.text(
                s + dur / 2,
                machine_y[machine],
                f"{job}\nS{step}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )

    ax.set_yticks(range(len(machines)))
    ax.set_yticklabels(machines)
    ax.set_xlabel("Time")
    ax.set_title("Job Shop Schedule Gantt Chart")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.invert_yaxis()  # 讓第一台機台在上方

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"甘特圖已儲存至: {output_path}")


def main():
    """
    主函式：讀取 orders.csv，求解排程，輸出結果並儲存檔案。
    """
    base_dir = Path(__file__).resolve().parent
    orders_path = base_dir / "orders.csv"

    if not orders_path.exists():
        raise FileNotFoundError(f"找不到訂單檔：{orders_path}，請確認檔案位於 schedule.py 同一資料夾中。")

    rows = load_orders(orders_path)
    if not rows:
        raise ValueError("orders.csv 為空，請檢查檔案內容。")

    result = build_schedule(rows)

    print_schedule_result(result)

    result_path = base_dir / "schedule_result.csv"
    save_schedule_csv(result, result_path)

    gantt_path = base_dir / "gantt.png"
    save_gantt_chart(result, gantt_path)


if __name__ == "__main__":
    main()
