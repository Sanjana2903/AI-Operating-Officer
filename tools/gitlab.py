import requests
import os
from .github import create_poc_repo_from_prompt
def create_gitlab_repo(token, name):
    url = "https://gitlab.com/api/v4/projects"
    headers = {"PRIVATE-TOKEN": token}
    data = {"name": name, "visibility": "private"}
    response = requests.post(url, headers=headers, data=data)
    return response.json()

def create_poc_repo_gitlab_fallback(prompt: str) -> str:
    token = os.getenv("GITLAB_TOKEN")
    #namespace_id = os.getenv("GITLAB_NAMESPACE_ID")
    if not all([token]):
        return "❌ GitLab fallback misconfigured: missing token "

    repo_name = f"poc-{prompt.replace(' ', '-').lower()}"
    response = create_gitlab_repo(token, repo_name)
    web_url = response.get('web_url')
    if web_url:
        return f"▪ View GitLab Repo: {web_url}"
    else:
        return f"❌ GitLab repo creation failed: {response}"
    
def create_poc_repo_with_fallback(prompt: str) -> str:
    try:
        result = create_poc_repo_from_prompt(prompt)
        if "http" in result:
            return result
        raise Exception("GitHub result invalid")
    except Exception as e:
        print(f"GitHub failed: {e}. Trying GitLab fallback...")
        try:
            result = create_poc_repo_gitlab_fallback(prompt)
            if "http" in result:
                return result
            raise Exception("GitLab result invalid")
        except Exception as e2:
            return "❌ All repo creation attempts failed."


