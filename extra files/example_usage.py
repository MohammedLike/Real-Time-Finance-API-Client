import requests
import json
import pandas as pd
from datetime import datetime
from config import get_api_key, get_base_url

def fetch_stock_data(symbol="AAPL", function="TIME_SERIES_DAILY_ADJUSTED"):
    """
    Fetch stock data from Alpha Vantage API
    
    Args:
        symbol (str): Stock symbol (e.g., 'AAPL', 'MSFT', 'GOOGL')
        function (str): API function to call
    
    Returns:
        dict: JSON response from API
    """
    try:
        # Get API key from config
        api_key = get_api_key()
        base_url = get_base_url()
        
        # Set up parameters
        params = {
            "function": function,
            "symbol": symbol,
            "outputsize": "compact",  # Use 'full' for complete data
            "datatype": "json",
            "apikey": api_key
        }
        
        # Make API request
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise exception for bad status codes
        
        # Parse JSON response
        data = response.json()
        
        # Check for API errors
        if "Error Message" in data:
            raise ValueError(f"API Error: {data['Error Message']}")
        
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return None
    except ValueError as e:
        print(f"Value error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def fetch_crypto_data(symbol="BTC", market="USD"):
    """
    Fetch cryptocurrency data from Alpha Vantage API
    
    Args:
        symbol (str): Cryptocurrency symbol (e.g., 'BTC', 'ETH')
        market (str): Market currency (e.g., 'USD', 'EUR')
    
    Returns:
        dict: JSON response from API
    """
    try:
        # Get API key from config
        api_key = get_api_key()
        base_url = get_base_url()
        
        # Set up parameters for crypto
        params = {
            "function": "DIGITAL_CURRENCY_DAILY",
            "symbol": symbol,
            "market": market,
            "apikey": api_key
        }
        
        # Make API request
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        
        # Parse JSON response
        data = response.json()
        
        # Check for API errors
        if "Error Message" in data:
            raise ValueError(f"API Error: {data['Error Message']}")
        
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return None
    except ValueError as e:
        print(f"Value error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def main():
    """Main function to demonstrate API usage"""
    print("Testing Alpha Vantage API connection...")
    
    # Test API key loading
    try:
        api_key = get_api_key()
        print(f"✅ API key loaded successfully: {api_key[:8]}...")
    except ValueError as e:
        print(f"❌ Error loading API key: {e}")
        return
    
    # Test stock data fetch
    print("\n📈 Fetching Apple stock data...")
    stock_data = fetch_stock_data("AAPL")
    if stock_data:
        print("✅ Stock data fetched successfully!")
        # Print first few keys to show data structure
        if "Time Series (Daily)" in stock_data:
            dates = list(stock_data["Time Series (Daily)"].keys())[:3]
            print(f"   Available dates: {dates}")
    
    # Test crypto data fetch
    print("\n🪙 Fetching Bitcoin data...")
    crypto_data = fetch_crypto_data("BTC")
    if crypto_data:
        print("✅ Crypto data fetched successfully!")
        # Print first few keys to show data structure
        if "Time Series (Digital Currency Daily)" in crypto_data:
            dates = list(crypto_data["Time Series (Digital Currency Daily)"].keys())[:3]
            print(f"   Available dates: {dates}")

if __name__ == "__main__":
    main() 