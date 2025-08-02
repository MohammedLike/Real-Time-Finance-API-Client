import os
from dotenv import load_dotenv

# Load environment variables from apikey.env file
load_dotenv('apikey.env')

# Get API key from environment
ALPHAVANTAGE_API_KEY = os.getenv('ALPHAVANTAGE_API_KEY')

# Base URL for Alpha Vantage API
ALPHAVANTAGE_BASE_URL = "https://www.alphavantage.co/query"

def get_api_key():
    """Get the Alpha Vantage API key from environment variables."""
    if not ALPHAVANTAGE_API_KEY:
        raise ValueError("ALPHAVANTAGE_API_KEY not found in environment variables. Please check your apikey.env file.")
    return ALPHAVANTAGE_API_KEY

def get_base_url():
    """Get the Alpha Vantage base URL."""
    return ALPHAVANTAGE_BASE_URL 