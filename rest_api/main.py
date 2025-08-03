"""
Main script to demonstrate real-time financial data fetching using Alpha Vantage API
"""

from financial_data_fetch import StockDataFetcher
from authentication import AuthenticationManager

def main():
    """Main function to fetch and save stock data"""

    auth_manager = AuthenticationManager()
    api_key = auth_manager.get_api_key()

    fetcher = StockDataFetcher(api_key)

    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

    all_data = fetcher.fetch_multiple_stocks(symbols)

    for symbol, df in all_data.items():
        print(f"\n📊 {symbol} - Last 5 Entries:\n{df.tail()}")

if __name__ == "__main__":
    main()
