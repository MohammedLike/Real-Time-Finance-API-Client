"""
Main script to demonstrate real-time financial data fetching using Alpha Vantage API
"""

from financial_data_fetch import StockDataFetcher
from authentication import AuthenticationManager

def main():
    """Main function to fetch and save stock data"""

    # Step 1: Get API Key
    auth_manager = AuthenticationManager()
    api_key = auth_manager.get_api_key()

    # Step 2: Initialize Fetcher
    fetcher = StockDataFetcher(api_key)

    # Step 3: Define stock symbols to fetch
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

    # Step 4: Fetch data
    all_data = fetcher.fetch_multiple_stocks(symbols)

    # Optional: Print a preview
    for symbol, df in all_data.items():
        print(f"\n📊 {symbol} - Last 5 Entries:\n{df.tail()}")

if __name__ == "__main__":
    main()
