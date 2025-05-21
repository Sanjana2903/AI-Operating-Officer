import os
import requests

def create_github_repo(token, name):
    url = "https://api.github.com/user/repos"
    headers = {
        "Authorization": f"Bearer {token}"
    }
    data = {
        "name": name,
        "private": True
    }
    res = requests.post(url, json=data, headers=headers)
    if res.status_code != 201:
        raise Exception(f"GitHub repo creation failed: {res.status_code}, {res.text}")
    return res.json()["html_url"]

def create_poc_repo_from_prompt(prompt: str) -> str:
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise ValueError("Missing GITHUB_TOKEN in environment")

    repo_name = f"poc-{prompt.replace(' ', '-').lower()}"
    url = create_github_repo(token, repo_name)
    return f"▪ View GitHub Repo: {url}"
