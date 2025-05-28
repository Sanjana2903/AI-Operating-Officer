def get_user_feedback() -> str: # This function is fine for manual feedback collection
    print("\n🔁 Was the response helpful?")
    print("✅ = Good | 🔄 = Needs Improvement | ❌ = Incorrect")
    rating = input("Your feedback (✅ / 🔄 / ❌): ").strip()
    if rating not in ["✅", "🔄", "❌"]:
        print("Invalid input. Defaulting to 🔄.")
        return "🔄"
    return rating
