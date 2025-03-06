import requests
import json
import time
import logging
from app import crew

# Configure logging
logging.basicConfig(
    filename="career_processing.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# Load career slugs from JSON file
with open("Career_Names_endpoints.json", "r") as f:
    careers_data = json.load(f)

API_URL = "https://devie4nodeapis.azurewebsites.net/api/careerLibrary/career"
HEADERS = {
    "accept": "*/*",
    "Authorization": "=="  
}

final_results = []
failed_cases = []

# Iterate through each career QueryParam
for career in careers_data["Careers"]:
    slug = career["QueryParam"]
    career_name = career["CareerName"]
    logging.info(f"Processing career: {career_name} ({slug})")
    
    try:
        response = requests.get(f"{API_URL}?slug={slug}&onlyActive=true", headers=HEADERS)
        
        if response.status_code == 200:
            data = response.json().get("data", {})
            
            # Extract relevant fields
            inputs = {
                "career_id": data.get("careerDetails", [{}])[0].get("CareerID", None),
                "career_name": data.get("careerDetails", [{}])[0].get("CareerName", None),
                "description": data.get("description", None),
                "top_colleges": ", ".join([inst["InstituteName"] for inst in data.get("careerInstitute", [])]) or None,
                "famous_personalities_name": ", ".join([p["FirstName"] for p in data.get("careerPersonality", [])]) or None,
                "top_companies": ", ".join([c["CompanyName"] for c in data.get("careerCompany", [])]) or None,
            }
            
            logging.info(f"Calling crew.kickoff() for career: {career_name}")
            result = crew.kickoff(inputs=inputs)  # This takes time
            
            # Store final result
            final_result = {
                "id": inputs["career_id"],
                "careerName":inputs['career_name'],
                "status": result.raw
            }
            final_results.append(final_result)
            
            # Save results incrementally
            with open("final_results.json", "w") as f:
                json.dump(final_results, f, indent=4)
            
            logging.info(f"Successfully processed {career_name} ({slug})")
        else:
            logging.error(f"Failed to fetch data for {career_name} ({slug}) - Status Code: {response.status_code}")
            failed_cases.append({"career": career_name, "slug": slug, "reason": "API request failed", "status_code": response.status_code})
    
    except Exception as e:
        logging.error(f"Exception occurred while processing {career_name} ({slug}): {str(e)}")
        failed_cases.append({"career": career_name, "slug": slug, "reason": str(e)})
    
    # Save failed cases incrementally
    with open("failed_cases.json", "w") as f:
        json.dump(failed_cases, f, indent=4)
    
print("All careers processed! Results saved in final_results.json and failed_cases.json")
