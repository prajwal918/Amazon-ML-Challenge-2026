import json
import os
from importlib import resources
from google_auth_oauthlib.flow import InstalledAppFlow

PUBLIC_SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/colaboratory",
    "https://www.googleapis.com/auth/drive.file",
]
REMOTE_REDIRECT_URI = "https://sdk.cloud.google.com/applicationdefaultauthcode.html"

config_resource = resources.files("colab_cli").joinpath("oauth_config.json")
client_config = json.loads(config_resource.read_text())

flow = InstalledAppFlow.from_client_config(client_config, PUBLIC_SCOPES)
flow.redirect_uri = REMOTE_REDIRECT_URI
auth_url, _ = flow.authorization_url(prompt="consent", token_usage="remote")

print("AUTH_URL_START")
print(auth_url)
print("AUTH_URL_END")
code = input("Enter authorization code: ").strip()

flow.fetch_token(code=code)
creds = flow.credentials

os.makedirs("/root/.config/colab-cli", exist_ok=True)
with open("/root/.config/colab-cli/token_account2.json", "w") as f:
    f.write(creds.to_json())

print("SUCCESS_ACCOUNT2_SAVED")
