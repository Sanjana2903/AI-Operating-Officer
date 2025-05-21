import requests
import os
def create_jira_ticket(base_url, auth, project_key, summary, description):
    url = f"{base_url}/rest/api/3/issue"
    payload = {
        "fields": {
            "project": {"key": project_key},
            "summary": summary,
            "description": description,
            "issuetype": {"name": "Task"}
        }
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers, auth=auth)
    return response.json()
def create_core_jira_task_from_prompt(prompt: str) -> str:
    base_url = os.getenv("JIRA_BASE_URL")
    email = os.getenv("JIRA_EMAIL")
    api_token = os.getenv("JIRA_API_TOKEN")
    project_key = os.getenv("JIRA_PROJECT_KEY", "AAD")

    if not all([base_url, email, api_token]):
        raise ValueError("Missing JIRA configuration in .env")

    auth = (email, api_token)

    response = create_jira_ticket(
        base_url, auth, project_key,
        summary=f"Copilot Task: {prompt}",
        description="Auto-created ticket via AI Operating Officer"
    )

    ticket_key = response.get("key")
    if ticket_key:
        ticket_url = f"{base_url}/browse/{ticket_key}"
        return f"▪ View JIRA ticket: {ticket_url}"
    else:
        raise RuntimeError(f"JIRA ticket creation failed: {response}")
