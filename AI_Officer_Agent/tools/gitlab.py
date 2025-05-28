import requests
import os 

def create_gitlab_repo(token, name):
    url = "https://gitlab.com/api/v4/projects"
    headers = {"PRIVATE-TOKEN": token}
    data = {"name": name, "visibility": "private"}
    response = requests.post(url, headers=headers, data=data)
    response.raise_for_status() # Add error handling for the request
    return response.json()

def create_poc_repo_gitlab_fallback(prompt: str) -> str:
    token = os.getenv("GITLAB_TOKEN") # Use environment variable
    if not token:
        return "❌ GitLab fallback failed: GITLAB_TOKEN not set in environment."

   
    repo_name = f"poc-gitlab-{prompt.replace(' ', '-').lower()}"
    try:
        response = create_gitlab_repo(token, repo_name)
        web_url = response.get('web_url')
        if web_url:
            return f"▪ View GitLab Repo: {web_url}"
        else:
            # More specific error based on potential GitLab response structure
            error_message = response.get('message', 'URL not available and no specific error message.')
            return f"❌ GitLab repo creation failed: {error_message}"
    except requests.exceptions.RequestException as e:
        return f"❌ GitLab repo creation failed due to a network or request error: {str(e)}"
    except Exception as e: # Catch other potential errors
        return f"❌ GitLab repo creation failed: {str(e)}"
