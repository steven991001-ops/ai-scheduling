"""讀取排程結果，呼叫外部 AI API（Google Gemini）產生中文排程摘要，存成 ai_summary.md。
遇到 AI 服務忙碌或斷線時：每個模型最多重試 3 次，失敗再換備用模型；全部失敗則記錄原因，不讓整個流程中斷。"""
import os
import time
from pathlib import Path

from google import genai

# 1. 讀取 OR-Tools 排程結果（由 schedule.py 產生）
output = Path("schedule_output.txt").read_text(encoding="utf-8")
table = Path("schedule_result.csv").read_text(encoding="utf-8")

# 2. 撰寫給 AI 的指示（prompt）
prompt = f"""你是生產管理顧問。以下是 Job Shop 排程結果，請用繁體中文寫 200 字以內的摘要給生管人員，
內容依序包含：
1. 總完工時間（Makespan）
2. 瓶頸機台（全程最忙、沒有空檔的機台）
3. 各訂單的完工時間
4. 一項具體的改善建議

【排程程式輸出】
{output}

【排程表 schedule_result.csv】
{table}
"""

# 3. 要嘗試的模型：先用設定檔指定的模型，再依序換備用模型
models = [os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite"), "gemini-3.5-flash-lite", "gemini-3.8-flash"]
models = list(dict.fromkeys(models))  # 去除重複

# API 金鑰從環境變數 GEMINI_API_KEY 讀取（由 GitHub Secrets 提供）；單次呼叫最多等 60 秒
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"], http_options={"timeout": 60_000})

summary, used, errors = None, None, []
for model in models:
    for attempt in range(1, 4):
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            summary, used = response.text.strip(), model
            break
        except Exception as e:  # 服務忙碌（503）、斷線、逾時等
            errors.append(f"{model} 第 {attempt} 次：{type(e).__name__}: {str(e)[:150]}")
            print(errors[-1])
            time.sleep(15 * attempt)  # 等 15、30、45 秒再重試
    if summary:
        break

# 4. 輸出結果
if summary:
    text = f"# AI 排程摘要（{used}）\n\n{summary}\n"
else:
    text = "# AI 排程摘要\n\nAI 服務暫時無法使用，排程結果已正常產生，請稍後重新執行。\n\n" + "\n".join(f"- {e}" for e in errors) + "\n"
Path("ai_summary.md").write_text(text, encoding="utf-8")
print(text)
