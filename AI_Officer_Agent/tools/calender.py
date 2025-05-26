import requests

def schedule_meeting(access_token, subject, start, end, attendees):
    url = "https://graph.microsoft.com/v1.0/me/events"
    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
    data = {
        "subject": subject,
        "start": {"dateTime": start, "timeZone": "UTC"},
        "end": {"dateTime": end, "timeZone": "UTC"},
        "attendees": [{"emailAddress": {"address": email}, "type": "required"} for email in attendees]
    }
    return requests.post(url, json=data, headers=headers).json()
