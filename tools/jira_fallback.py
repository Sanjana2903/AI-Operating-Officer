import os
import requests

def create_jira_ticket_fallback(base_url, auth, summary, description):
    url = f"{base_url}/rest/api/3/issue"
    payload = {
        "fields": {
            "project": {"key": "AAD"},  
            "summary": summary,
            "description": description,
            "issuetype": {"name": "Task"}
        }
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers, auth=auth)
    return response.json()

def create_core_jira_task_with_fallback(prompt: str) -> str:
    try:
        from tools.jira import create_core_jira_task_from_prompt
        return create_core_jira_task_from_prompt(prompt)
    except Exception as e:
        print("🔁 JIRA primary failed. Trying fallback...")

        base_url = os.getenv("JIRA_BASE_URL")
        email = os.getenv("JIRA_EMAIL")
        api_token = os.getenv("JIRA_API_TOKEN")

        if not all([base_url, email, api_token]):
            return "▪ Fallback JIRA failed – missing .env configuration"

        auth = (email, api_token)

        try:
            response = create_jira_ticket_fallback(
                base_url, auth,
                summary=f"Copilot Task: {prompt}",
                description="Auto-created fallback ticket via AI Operating Officer"
            )
            ticket_key = response.get("key")
            if ticket_key:
                ticket_url = f"{base_url}/browse/{ticket_key}"
                return f"▪ View fallback JIRA ticket: {ticket_url}"
            else:
                return f"▪ Fallback JIRA failed – ticket key missing from response: {response}"
        except Exception as e2:
            return f"▪ Fallback JIRA failed: {str(e2)}"
