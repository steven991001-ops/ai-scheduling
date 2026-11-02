"""排程摘要（程式算事實＋AI 寫說明）
1. 程式從排程結果「計算」事實：Makespan、理論下限、各訂單完工時間、各機台忙碌時間／空檔／使用率、瓶頸機台
2. 把事實交給外部 AI API（Google Gemini），AI 只負責把事實寫成中文摘要與建議
3. 事實表＋AI 摘要一起存成 ai_summary.md
AI 服務忙碌時：每個模型重試 3 次，再換備用模型；全部失敗仍輸出事實表，流程不中斷。"""
import csv
import os
import re
import time
from collections import defaultdict
from pathlib import Path

from google import genai


# ---------- 1. 讀取排程結果（先讀 CSV，讀不到就解析終端機輸出） ----------
def load_ops():
    ops = []
    path = Path("schedule_result.csv")
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                r = {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
                pick = lambda *keys: next((r[k] for k in r if any(x in k for x in keys)), None)
                job, mach, st, en = pick("job"), pick("machine"), pick("start"), pick("end")
                if job and mach and st and en:
                    ops.append((job, mach, int(float(st)), int(float(en))))
    if ops:
        return ops
    machine = None
    for line in Path("schedule_output.txt").read_text(encoding="utf-8").splitlines():
        m = re.search(r"Machine\s+(\S+)", line)
        if m:
            machine = m.group(1)
        m = re.search(r"Job\s+(\S+?),.*?start=(\d+).*?end=(\d+)", line)
        if m and machine:
            ops.append((m.group(1), machine, int(m.group(2)), int(m.group(3))))
    return ops


ops = load_ops()
if not ops:
    raise SystemExit("讀不到排程結果，請確認 schedule.py 有輸出 schedule_result.csv")

# ---------- 2. 計算事實 ----------
makespan = max(e for _, _, _, e in ops)
by_machine, job_end = defaultdict(list), defaultdict(int)
for job, mach, s, e in ops:
    by_machine[mach].append((s, e, job))
    job_end[job] = max(job_end[job], e)

machines = {}
for mach, items in sorted(by_machine.items()):
    items.sort()
    busy = sum(e - s for s, e, _ in items)
    idle, t = [], 0
    for s, e, _ in items:
        if s > t:
            idle.append(f"{t}–{s}")
        t = max(t, e)
    if t < makespan:
        idle.append(f"{t}–{makespan}")
    machines[mach] = {"busy": busy, "idle": idle, "util": busy / makespan}

job_load = defaultdict(int)  # 每張訂單的總加工時間（另一個理論下限）
for job, _, s, e in ops:
    job_load[job] += e - s
lower_bound = max(max(m["busy"] for m in machines.values()), max(job_load.values()))
max_busy = max(m["busy"] for m in machines.values())
bottleneck = [k for k, m in machines.items() if m["busy"] == max_busy]
optimal = makespan == lower_bound

fact_lines = [
    f"- 總完工時間 Makespan：{makespan}",
    f"- 理論下限（最忙機台總加工時間與最長訂單總加工時間取大者）：{lower_bound}",
    f"- 是否已達最佳：{'是，Makespan 等於理論下限，無法再縮短' if optimal else '否，與理論下限相差 ' + str(makespan - lower_bound)}",
    f"- 瓶頸機台：{'、'.join(bottleneck)}（總加工時間 {max_busy}）",
    "- 各訂單完工時間：" + "、".join(f"{j}={job_end[j]}" for j in sorted(job_end)),
]
for k, m in machines.items():
    fact_lines.append(f"- {k}：忙碌 {m['busy']}、使用率 {m['util']:.0%}、空檔 {('、'.join(m['idle']) or '無')}")
facts = "\n".join(fact_lines)
print(facts)

# ---------- 3. 請 AI 把事實寫成摘要 ----------
prompt = f"""你是生產管理顧問。以下是「排程程式已計算好的事實」，請只根據這些事實，用繁體中文寫 200 字以內的摘要給生管人員。
規則：
- 不要自行重新計算，也不要加入事實以外的數字。
- 依序說明：總完工時間、是否已達最佳、瓶頸機台、各訂單完工時間、一項改善建議。
- 若已達最佳（Makespan 等於理論下限），改善建議不可以是「調整工序順序來縮短 Makespan」，
  應從瓶頸機台的產能著手（例如增加班次、外包或增加同類機台），或利用其他機台的空檔安排其他工作。

【程式計算的事實】
{facts}
"""

models = list(dict.fromkeys([os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite"),
                             "gemini-3.5-flash-lite", "gemini-3.8-flash"]))
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"], http_options={"timeout": 60_000})
summary, used, errors = None, None, []
for model in models:
    for attempt in range(1, 4):
        try:
            summary = client.models.generate_content(model=model, contents=prompt).text.strip()
            used = model
            break
        except Exception as e:  # 服務忙碌（503）、斷線、逾時等
            errors.append(f"{model} 第 {attempt} 次：{type(e).__name__}: {str(e)[:150]}")
            print(errors[-1])
            time.sleep(15 * attempt)
    if summary:
        break

# ---------- 4. 輸出 ----------
text = "# AI 排程摘要\n\n## 程式計算的事實\n\n" + facts + "\n\n"
if summary:
    text += f"## AI 說明（{used}）\n\n{summary}\n"
else:
    text += "## AI 說明\n\nAI 服務暫時無法使用，請稍後重新執行。\n\n" + "\n".join(f"- {e}" for e in errors) + "\n"
Path("ai_summary.md").write_text(text, encoding="utf-8")
print(text)
