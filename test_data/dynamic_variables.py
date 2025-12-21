# Dynamic Python variable file with get_variables function
def get_variables(environment="test", api_key="default"):
    """Return variables based on arguments"""
    variables = {
        "ENVIRONMENT": environment,
        "API_KEY": api_key,
        "BASE_URL": f"https://{environment}.example.com",
        "LIST__ENDPOINTS": [f"/{environment}/api", f"/{environment}/health"],
        "DICT__CONFIG": {
            "env": environment,
            "debug": environment == "dev",
            "timeout": 30 if environment == "prod" else 10
        }
    }
    
    if environment == "prod":
        variables["PROD_ONLY_VAR"] = "production setting"
    
    return variables