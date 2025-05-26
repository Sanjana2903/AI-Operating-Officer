import time
import pandas as pd
import numpy as np
from main import ask_question

sample_queries = [
    {"query": "How can we migrate our legacy systems to Azure?", "role": "ceo"},
    {"query": "What are the key benefits of using copilots for engineers?", "role": "cto"},
    {"query": "How should we rethink Surface device design in the AI era?", "role": "product"},
    {"query": "How do we integrate GitHub Copilot into our development stack?", "role": "cto"},
    {"query": "What is Microsoft's position on sustainability in tech?", "role": "ceo"},
    {"query": "How can we reduce latency in AI-powered interfaces?", "role": "cto"},
    {"query": "What are the risks of hallucination in generative AI?", "role": "ceo"},
    {"query": "How to orchestrate Jira and GitHub workflows using AI?", "role": "product"},
    {"query": "What is Satya's view on responsible AI?", "role": "ceo"},
    {"query": "Explain the future of Microsoft Surface hardware.", "role": "product"},
    {"query": "How to reduce cost and carbon footprint in infra?", "role": "cto"},
    {"query": "How should teams onboard Copilot effectively?", "role": "product"},
    {"query": "How can we use Teams + AI to speed up decisions?", "role": "ceo"},
    {"query": "What tooling supports AI code generation in Azure?", "role": "cto"},
    {"query": "How should we prioritize accessibility in device design?", "role": "product"},
]


results = []
for i, item in enumerate(sample_queries, 1):
    print(f"{i}/{len(sample_queries)} | {item['role'].upper()} → {item['query']}")
    try:
        res = ask_question(item["query"], item["role"])
        results.append({
            "query": item["query"],
            "role": item["role"],
            "ragas_f1": res.get("score", 0.0),
            "hallucination": res.get("hallucination", 1.0),
            "deepeval_passed": res.get("deepeval_passed", False),
            "latency_ms": round(res.get("latency", -1), 2),
            "tools_used": res.get("reasoning", {}).get("tools", "None"),
            "actions": " | ".join(res.get("actions", [])),
            "citations": " | ".join(res.get("citations", [])),
            "answer": res.get("answer", "")
        })
    except Exception as e:
        print(f"❌ Failed: {e}")
        results.append({
            "query": item["query"],
            "role": item["role"],
            "ragas_f1": 0.0,
            "hallucination": 1.0,
            "deepeval_passed": False,
            "latency_ms": -1,
            "tools_used": "None",
            "actions": "",
            "citations": "",
            "answer": f"Error: {str(e)}"
        })

df = pd.DataFrame(results)

latencies = df[df["latency_ms"] > 0]["latency_ms"]
p95_latency = np.percentile(latencies, 95) if not latencies.empty else -1
avg_ragas = round(df["ragas_f1"].mean(), 3)
avg_hallucination = round(df["hallucination"].mean(), 3)
deepeval_pass_rate = f"{df['deepeval_passed'].mean()*100:.1f}%"

summary_row = pd.DataFrame([{
    "query": "✅ SUMMARY",
    "role": "",
    "ragas_f1": avg_ragas,
    "hallucination": avg_hallucination,
    "deepeval_passed": deepeval_pass_rate,
    "latency_ms": round(p95_latency, 1),
    "tools_used": "",
    "actions": "",
    "citations": "",
    "answer": ""
}])

df_final = pd.concat([df, summary_row], ignore_index=True)
df_final.to_csv("eval_report.csv", index=False)

print("\nEvaluation Summary:")
print(f"- Average RAGAS F1: {avg_ragas}")
print(f"- Average Hallucination Rate: {avg_hallucination}")
print(f"- DeepEval Pass Rate: {deepeval_pass_rate}")
print(f"- p95 Latency: {round(p95_latency, 1)} ms")
print("Results saved to eval_report.csv")
