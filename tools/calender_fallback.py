def schedule_fallback_meeting(subject, start, end, attendees):
    return f"📅 Could not access Microsoft Graph. Log the meeting manually: {subject} on {start} with {', '.join(attendees)}"

def book_modern_core_clinic_with_fallback(prompt: str) -> str:
    try:
        from tools.calender import book_modern_core_clinic_from_prompt
        return book_modern_core_clinic_from_prompt(prompt)
    except Exception as e:
        print("🔁 Graph API failed. Using fallback logging...")
        return schedule_fallback_meeting("Modern Core Clinic", "2025-05-21T11:00:00", "2025-05-21T11:30:00", ["infra@example.com"])
