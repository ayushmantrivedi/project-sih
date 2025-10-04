import requests

API_URL = "https://api.infermedica.com/v3/"
APP_ID = "YOUR_APP_ID"
APP_KEY = "YOUR_APP_KEY"

def get_symptom_info(symptom):
    headers = {
        "App-Id": APP_ID,
        "App-Key": APP_KEY,
        "Content-Type": "application/json"
    }
    response = requests.get(f"{API_URL}symptoms", headers=headers, params={"name": symptom})
    return response.json()