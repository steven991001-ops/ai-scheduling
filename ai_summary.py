"""讀取排程結果，呼叫外部 AI API（Google Gemini）產生中文排程摘要，存成 ai_summary.md。"""
import os
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

# 3. 呼叫 Gemini：API 金鑰從環境變數 GEMINI_API_KEY 讀取（由 GitHub Secrets 提供）
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
model = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
response = client.models.generate_content(model=model, contents=prompt)
summary = response.text.strip()

# 4. 輸出結果
Path("ai_summary.md").write_text(f"# AI 排程摘要（{model}）\n\n{summary}\n", encoding="utf-8")
print(summary)
