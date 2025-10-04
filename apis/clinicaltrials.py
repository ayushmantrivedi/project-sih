import requests

def get_clinical_trials(query):
    url = f"https://clinicaltrials.gov/api/query/full_studies?expr={query}&min_rnk=1&max_rnk=3&fmt=json"
    r = requests.get(url)
    if r.ok:
        studies = r.json().get("FullStudiesResponse", {}).get("FullStudies", [])
        results = []
        for s in studies:
            protocol = s.get("Study", {}).get("ProtocolSection", {})
            title = protocol.get("IdentificationModule", {}).get("BriefTitle", "")
            condition = protocol.get("ConditionsModule", {}).get("ConditionList", {}).get("Condition", [])
            results.append({"title": title, "condition": condition})
        return results
    return []