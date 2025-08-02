import requests
import json
import pandas as pd
from datetime import datetime
import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_api_key, get_base_url

def fetch_crypto_prices():
    """Fetch cryptocurrency prices from Alpha Vantage API"""
    try:
        # Get API key and base URL
        api_key = get_api_key()
        base_url = get_base_url()
        
        # List of cryptocurrencies to fetch
        crypto_symbols = ['BTC', 'ETH', 'DOGE', 'SOL', 'USDT']
        crypto_data = []
        
        for symbol in crypto_symbols:
            # Set up parameters for crypto data
            params = {
                "function": "DIGITAL_CURRENCY_DAILY",
                "symbol": symbol,
                "market": "USD",
                "apikey": api_key
            }
            
            # Make API request
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            
            # Parse JSON response
            data = response.json()
            
            # Check for API errors
            if "Error Message" in data:
                print(f"API Error for {symbol}: {data['Error Message']}")
                continue
            
            # Extract latest price data
            if "Time Series (Digital Currency Daily)" in data:
                time_series = data["Time Series (Digital Currency Daily)"]
                # Get the most recent date
                latest_date = max(time_series.keys())
                latest_data = time_series[latest_date]
                
                crypto_data.append({
                    'cryptocurrency': symbol,
                    'price_usd': float(latest_data['4a. close (USD)']),
                    'volume': float(latest_data['5. volume']),
                    'market_cap': float(latest_data['6. market cap (USD)']),
                    'date': latest_date
                })
            
            # Add delay to respect API rate limits
            import time
            time.sleep(0.1)
        
        return crypto_data
        
    except Exception as e:
        print(f"Error fetching crypto data: {e}")
        return []

def main():
    """Main function to fetch and save crypto data"""
    print("Fetching cryptocurrency price data from Alpha Vantage...")
    
    # Fetch data
    crypto_data = fetch_crypto_prices()
    
    if not crypto_data:
        print("No data retrieved. Please check your API key and internet connection.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(crypto_data)
    
    # Add timestamp
    df['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print("DataFrame:")
    print(df)
    
    # Save to CSV file
    csv_filename = 'cryptocurrency_prices_alphavantage.csv'
    df.to_csv(csv_filename, index=False)
    print(f"\nData saved to {csv_filename}")
    
    # Display the first few rows of the saved data
    print(f"\nFirst few rows of {csv_filename}:")
    print(df.head())

if __name__ == "__main__":
    main() 