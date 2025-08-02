#!/usr/bin/env python3
"""
Main entry point for the Real Time Finance API Client
Demonstrates how to use the financial data fetcher
"""

import sys
import os
import pandas as pd
from datetime import datetime
import time

# Set UTF-8 encoding for Windows console
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add the current directory to Python path to allow imports
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from rest_api import StockDataFetcher, AuthenticationManager
    print("SUCCESS: Modules imported successfully!")
except ImportError as e:
    print(f"ERROR: Failed to import modules: {e}")
    sys.exit(1)

class StockAnalyzer:
    """Enhanced wrapper for StockDataFetcher with additional functionality"""
    
    def __init__(self, env_file='apikey.env'):
        """Initialize the stock analyzer"""
        self.auth = AuthenticationManager(env_file)
        self.api_key = self.auth.get_api_key()
        self.fetcher = StockDataFetcher(self.api_key)
        self.calls_made = 0
        self.data_cache = {}
        
    def is_ready(self):
        """Check if the analyzer is ready to make API calls"""
        return self.api_key is not None
    
    def fetch_single_stock(self, symbol, output_size='compact'):
        """Fetch data for a single stock and convert to DataFrame"""
        try:
            print(f"  Fetching {symbol}...")
            raw_data = self.fetcher.fetch_daily_adjusted(symbol, output_size)
            self.calls_made += 1
            
            if "Time Series (Daily)" not in raw_data:
                print(f"    WARNING: No time series data for {symbol}")
                return None
            
            time_series = raw_data["Time Series (Daily)"]
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient="index", dtype='float')
            df.index = pd.to_datetime(df.index)
            df.sort_index(ascending=False, inplace=True)  # Latest first
            
            # Rename columns to be more readable
            df.columns = [
                "Open", "High", "Low", "Close", "Adjusted_Close",
                "Volume", "Dividend_Amount", "Split_Coefficient"
            ]
            
            return df
            
        except Exception as e:
            print(f"    ERROR: Failed to fetch {symbol}: {e}")
            return None
    
    def fetch_multiple_stocks(self, symbols, output_size='compact', delay=12):
        """Fetch data for multiple stocks with rate limiting"""
        stock_data = {}
        
        for i, symbol in enumerate(symbols):
            df = self.fetch_single_stock(symbol, output_size)
            if df is not None:
                stock_data[symbol] = df
                self.data_cache[symbol] = df
            
            # Rate limiting: Alpha Vantage free tier allows 5 calls per minute
            if i < len(symbols) - 1:  # Don't wait after the last symbol
                print(f"    Waiting {delay} seconds to respect rate limits...")
                time.sleep(delay)
        
        return stock_data
    
    def save_data_to_csv(self, stock_data, output_dir='Data'):
        """Save stock data to CSV files"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        saved_files = []
        for symbol, df in stock_data.items():
            if df is not None and not df.empty:
                filename = f"{symbol}_daily_data_{datetime.now().strftime('%Y%m%d')}.csv"
                filepath = os.path.join(output_dir, filename)
                df.to_csv(filepath)
                saved_files.append(filepath)
                print(f"    Saved {symbol} data to {filepath}")
        
        return saved_files
    
    def get_stats(self):
        """Get API usage statistics"""
        # Alpha Vantage free tier: 25 calls per day, 5 calls per minute
        return {
            'calls_made': self.calls_made,
            'calls_remaining_today': max(0, 25 - self.calls_made),
            'calls_remaining_minute': 'Rate limited (5/min)',
            'data_cached': len(self.data_cache)
        }
    
    def close(self):
        """Clean up resources"""
        if hasattr(self.fetcher, 'session'):
            self.fetcher.session.close()

def main():
    """Main function to demonstrate the API client usage"""
    print("Real Time Finance API Client")
    print("=" * 40)
    
    # Initialize the stock analyzer
    try:
        analyzer = StockAnalyzer(env_file='apikey.env')
        print("SUCCESS: Stock analyzer initialized successfully!")
    except Exception as e:
        print(f"ERROR: Failed to initialize analyzer: {e}")
        return
    
    # Check if analyzer is ready
    if not analyzer.is_ready():
        print("ERROR: Analyzer is not ready. Please check your API key configuration.")
        return
    
    # List of popular stocks to fetch
    stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    print(f"\nFetching daily data for {len(stocks)} stocks...")
    print("NOTE: This will take about 1 minute due to API rate limits (12s between calls)")
    
    # Fetch data for multiple stocks
    stock_data = analyzer.fetch_multiple_stocks(stocks, output_size='compact')
    
    if stock_data:
        print(f"\nSUCCESS: Successfully fetched data for {len(stock_data)} stocks!")
        
        # Display summary for each stock
        print("\nStock Summary:")
        print("-" * 50)
        for symbol, data in stock_data.items():
            if data is not None and not data.empty:
                latest_date = data.index[0].strftime('%Y-%m-%d')
                latest_close = data['Close'].iloc[0]
                volume = data['Volume'].iloc[0]
                print(f"  {symbol:6}: ${latest_close:8.2f} | {latest_date} | Vol: {volume:,.0f}")
            else:
                print(f"  {symbol:6}: No data available")
        
        # Save data to CSV
        print(f"\nSaving data to CSV files...")
        saved_files = analyzer.save_data_to_csv(stock_data, output_dir='Data')
        print(f"SUCCESS: Saved {len(saved_files)} files!")
        
        # Display API usage statistics
        stats = analyzer.get_stats()
        print(f"\nAPI Usage Statistics:")
        print(f"  Calls made: {stats.get('calls_made', 0)}")
        print(f"  Calls remaining today: {stats.get('calls_remaining_today', 'Unknown')}")
        print(f"  Rate limit: {stats.get('calls_remaining_minute', 'Unknown')}")
        print(f"  Data cached: {stats.get('data_cached', 0)} stocks")
        
        # Optional: Display some basic analysis
        print(f"\nQuick Analysis:")
        print("-" * 30)
        for symbol, data in stock_data.items():
            if data is not None and len(data) >= 5:
                recent_avg = data['Close'].head(5).mean()
                latest_price = data['Close'].iloc[0]
                change_pct = ((latest_price - recent_avg) / recent_avg) * 100
                trend = "UP" if change_pct > 0 else "DOWN"
                print(f"  {symbol}: {trend} {change_pct:+.1f}% vs 5-day avg")
    
    else:
        print("ERROR: Failed to fetch stock data")
        print("\nPossible reasons:")
        print("  - API key has reached daily limit (25 requests/day for free tier)")
        print("  - Invalid API key")
        print("  - Network connectivity issues")
        print("  - API service temporarily unavailable")
        print("\nSolutions:")
        print("  - Wait until tomorrow for daily limit reset")
        print("  - Upgrade to premium plan for unlimited requests")
        print("  - Check your internet connection")
        print("  - Verify your API key is correct")
    
    # Clean up
    analyzer.close()
    print("\nAPI client session closed.")

if __name__ == "__main__":
    main()