from config import issue_department_mapping

def route_issue(issue_type: str) -> str:
    return issue_department_mapping.get(issue_type, "Municipal Corporation")
