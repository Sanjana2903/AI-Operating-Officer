import os
import requests
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename="automation.log", level=logging.INFO)

def create_jira_ticket_fallback(base_url, auth, project_key, summary, description):
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

def create_core_jira_task_with_fallback(prompt: str) -> str:
    try:
        from tools.jira import create_core_jira_task_from_prompt
        return create_core_jira_task_from_prompt(prompt)
    except Exception as e:
        print("🔁 JIRA primary failed. Trying fallback...")
        logger.warning(f"Primary JIRA failed: {e}")

        base_url = os.getenv("JIRA_BASE_URL")
        email = os.getenv("JIRA_EMAIL")
        api_token = os.getenv("JIRA_API_TOKEN")
        project_key = os.getenv("JIRA_PROJECT_KEY", "AAD")

        if not all([base_url, email, api_token]):
            return "▪ Fallback JIRA failed – missing .env configuration"

        auth = (email, api_token)

        try:
            response = create_jira_ticket_fallback(
                base_url, auth, project_key,
                summary=f"Copilot Task: {prompt}",
                description="Auto-created fallback ticket via AI Operating Officer"
            )
            if response.get("key"):
                ticket_url = f"{base_url}/browse/{response['key']}"
                return f"▪ View fallback JIRA ticket: {ticket_url}"
            elif response.get("errorMessages"):
                errors = "; ".join(response["errorMessages"])
                return f"▪ Fallback JIRA failed – {errors}. Check project permissions and API token access."

            else:
                return f"▪ Fallback JIRA failed – unexpected response: {response}"
        except Exception as e2:
            logger.error(f"Fallback JIRA exception: {e2}")
            return f"▪ Fallback JIRA failed: {str(e2)}"
