# feedback.py
def get_user_feedback() -> str:
    print("\n🔁 Was the response helpful?")
    print("✅ = Good | 🔄 = Needs Improvement | ❌ = Incorrect")
    rating = input("Your feedback (✅ / 🔄 / ❌): ").strip()
    return rating

def auto_score_with_ragas(query, context, prediction) -> float:
    print("\n📊 Simulating auto-score (RAGAS fallback)...")
    score = 0.85 if "cloud" in query.lower() else 0.65
    return round(score, 2)

def auto_score_with_deepeval(query, context, prediction) -> float:
    print("📊 Simulating DeepEval scoring...")
    return 0.81 if "scalability" in prediction else 0.68
