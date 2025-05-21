import requests

def create_gitlab_repo(token, name):
    url = "https://gitlab.com/api/v4/projects"
    headers = {"PRIVATE-TOKEN": token}
    data = {"name": name, "visibility": "private"}
    response = requests.post(url, headers=headers, data=data)
    return response.json()

def create_poc_repo_gitlab_fallback(prompt: str) -> str:
    
    token = "gitlab-token"
    
    repo_name = "poc-core-migration"
    response = create_gitlab_repo(token, repo_name)
    web_url = response.get('web_url')
    if web_url:
        return f"▪ View GitLab Repo: {web_url}"
    else:
        return "❌ GitLab repo creation failed: URL not available"
        
