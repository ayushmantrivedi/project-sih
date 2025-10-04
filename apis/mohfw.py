import requests

def get_mohfw_data():
    url = "https://data.covid19india.org/data.json"
    r = requests.get(url)
    if r.ok:
        return r.json().get("statewise", [])
    return []