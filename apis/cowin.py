import requests

def get_cowin_stats():
    url = "https://data.cowin.gov.in/api/v1/reports/vaccine/vaccination-by-age"
    r = requests.get(url)
    if r.ok:
        return r.json()
    return {}