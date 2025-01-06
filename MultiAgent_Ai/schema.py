def get_use_cases(self):
    use_cases = []
    current_use_case = {}
    
    # Assuming the response is stored in self.response or a similar attribute
    lines = self.response.split('\n')
    
    for line in lines:
        if line.startswith("Use Case"):
            if current_use_case:
                use_cases.append(current_use_case)
            current_use_case = {"title": line.strip()}
        elif "Objective:" in line:
            current_use_case["objective"] = line.split("Objective:")[1].strip()
        elif "AI Application:" in line:
            current_use_case["ai_application"] = line.split("AI Application:")[1].strip()
        elif "Cross-Functional Benefits:" in line:
            current_use_case["benefits"] = []
        elif line.strip().startswith("•"):
            if "benefits" in current_use_case:
                current_use_case["benefits"].append(line.strip()[2:])
        elif "Department Name:" in line:
            current_use_case["department"] = line.split("Department Name:")[1].strip()
        elif "Field Name:" in line:
            current_use_case["field"] = line.split("Field Name:")[1].strip()
        elif "Sector Name:" in line:
            current_use_case["sector"] = line.split("Sector Name:")[1].strip()
    
    if current_use_case:
        use_cases.append(current_use_case)
    
    return use_cases

