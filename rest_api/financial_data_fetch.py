import requests
import pandas as pd
import os
from authentication import AuthenticationManager 

class StockDataFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"

    def fetch_daily_adjusted(self, symbol):
        print(f"\n📈 Fetching data for {symbol}...")
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": "full",
            "apikey": self.api_key
        }

        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()

            if "Time Series (Daily)" in data:
                df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                df = df.astype(float)

                os.makedirs("data", exist_ok=True)
                file_path = f"data/{symbol}_daily.csv"
                df.to_csv(file_path)
                print(f"✅ Data for {symbol} saved to {file_path}")
                return df

            else:
                error_msg = data.get("Note") or data.get("Error Message") or "Unknown error"
                print(f"❌ Error: Unexpected response for {symbol}: {error_msg}")
                return None

        except requests.RequestException as e:
            print(f"❌ Network error fetching data for {symbol}: {e}")
            return None

    def fetch_multiple_stocks(self, symbols):
        all_data = {}
        for symbol in symbols:
            df = self.fetch_daily_adjusted(symbol)
            if df is not None:
                all_data[symbol] = df
        return all_data


if __name__ == "__main__":
    auth = AuthenticationManager()
    api_key = auth.get_api_key()
    fetcher = StockDataFetcher(api_key)
    fetcher.fetch_multiple_stocks(["AAPL", "MSFT", "GOOGL"])
