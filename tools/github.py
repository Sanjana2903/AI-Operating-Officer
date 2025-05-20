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
    return res.json()
