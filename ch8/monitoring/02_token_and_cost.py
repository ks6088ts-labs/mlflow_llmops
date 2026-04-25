"""
第8章 - 8.2 トークン使用量とコストの可視化

自動トレーシングによるトークン使用量の追跡と、
コスト計算の基本的な流れを確認します。

実行方法:
  make cost
  または
  uv run python monitoring/02_token_and_cost.py
"""

from dotenv import load_dotenv

load_dotenv()

import os

import mlflow
from mlflow import MlflowClient
from openai import AzureOpenAI
from cost_calculator import calculate_cost

mlflow.set_experiment("ch8-monitoring-quickstart")
mlflow.openai.autolog()

LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
client = AzureOpenAI(
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
)
mlflow_client = MlflowClient()

# === 1. 異なるモデルでLLM呼び出し ===
# 注: Azure OpenAI では model にデプロイ名を指定する
models_and_prompts = [
    (LLM_MODEL, "Pythonのデコレータを簡潔に説明してください。"),
    (LLM_MODEL, "MLflowのトレーシング機能について、主な利点を3つ挙げてください。"),
    (
        LLM_MODEL,
        "RAGシステムの品質評価で重要な指標は何ですか?詳しく説明してください。",
    ),
]

print("=== トークン使用量とコスト ===\n")
total_cost = 0.0

for model, prompt in models_and_prompts:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )

    # トレースからトークン使用量を取得
    trace_id = mlflow.get_last_active_trace_id()
    trace = mlflow.get_trace(trace_id=trace_id)
    usage = trace.info.token_usage

    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)

    # コスト計算
    cost = calculate_cost(model, input_tokens, output_tokens)
    total_cost += cost

    # コストをトレースのタグに記録（完了済みトレースにはset_trace_tagを使用）
    mlflow_client.set_trace_tag(trace_id, "cost.total_usd", f"{cost:.6f}")
    mlflow_client.set_trace_tag(trace_id, "cost.model", model)

    print(f"プロンプト: {prompt[:40]}...")
    print(f"  モデル: {model}")
    print(f"  入力トークン: {input_tokens}")
    print(f"  出力トークン: {output_tokens}")
    print(f"  合計トークン: {total_tokens}")
    print(f"  コスト: ${cost:.6f}")
    print()

print(f"--- 合計コスト: ${total_cost:.6f} ---")

# === 2. トレースを検索してコスト集計 ===
print("\n=== コスト集計(トレース検索) ===")
traces = mlflow.search_traces(
    filter_string="tags.`cost.total_usd` != ''",
    max_results=100,
)
if len(traces) > 0:
    # tags列はdict型なので、そこからcost.total_usdを取り出す
    costs = traces["tags"].apply(lambda t: float(t.get("cost.total_usd", 0)))
    print(f"  トレース数: {len(costs)}")
    print(f"  合計コスト: ${costs.sum():.6f}")
    print(f"  平均コスト: ${costs.mean():.6f}")
    print(f"  最大コスト: ${costs.max():.6f}")
else:
    print("  コスト付きトレースが見つかりませんでした。")

print("\n✅ コスト分析完了。MLflow UIの各トレースで cost.* タグを確認できます。")
