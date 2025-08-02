"""
Financial data fetching module
Combines all components to fetch and process stock data from Alpha Vantage API
"""

import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime

from .authentication import AuthenticationManager
from .param_header import QueryParamsBuilder
from .get_request import HTTPRequestHandler
from .rate_limiting_error import APICallManager


class StockDataFetcher:
    """Main class for fetching stock data from Alpha Vantage API"""
    
    def __init__(self, env_file='apikey.env', calls_per_minute=5, calls_per_day=500):
        """
        Initialize stock data fetcher
        
        Args:
            env_file (str): Path to environment file with API key
            calls_per_minute (int): Rate limit for calls per minute
            calls_per_day (int): Rate limit for calls per day
        """
        # Initialize components
        self.auth_manager = AuthenticationManager(env_file)
        self.api_call_manager = APICallManager(calls_per_minute, calls_per_day)
        self.http_handler = HTTPRequestHandler()
        
        # Initialize query builder if authentication is successful
        if self.auth_manager.is_authenticated():
            self.query_builder = QueryParamsBuilder(self.auth_manager.get_api_key())
        else:
            self.query_builder = None
            print("❌ Cannot initialize: Authentication failed")
    
    def is_ready(self) -> bool:
        """
        Check if fetcher is ready to make API calls
        
        Returns:
            bool: True if all components are properly initialized
        """
        return (self.auth_manager.is_authenticated() and 
                self.query_builder is not None)
    
    def fetch_daily_stock_data(self, symbol: str, output_size='compact') -> Optional[pd.DataFrame]:
        """
        Fetch daily stock data for a single symbol
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL')
            output_size (str): 'compact' for last 100 days, 'full' for all data
            
        Returns:
            pandas.DataFrame: Stock data or None if failed
        """
        if not self.is_ready():
            print("❌ Fetcher not ready. Check authentication.")
            return None
        
        # Validate symbol
        if not self.query_builder.validate_symbol(symbol):
            print(f"❌ Invalid symbol format: {symbol}")
            return None
        
        # Prepare for API call (rate limiting)
        if not self.api_call_manager.prepare_call():
            return None
        
        # Build request parameters
        params = self.query_builder.build_daily_params(symbol, output_size)
        headers = self.query_builder.get_headers()
        
        # Make API request
        print(f"🔄 Fetching daily data for {symbol}...")
        data = self.http_handler.make_get_request(
            url=self.query_builder.BASE_URL,
            params=params,
            headers=headers
        )
        
        # Handle response
        error_analysis = self.api_call_manager.record_call_result(data, symbol)
        
        if error_analysis['has_error']:
            return None
        
        # Convert to DataFrame
        return self._convert_to_dataframe(data, symbol)
    
    def fetch_multiple_stocks(self, symbols: List[str], output_size='compact') -> Dict[str, pd.DataFrame]:
        """
        Fetch daily data for multiple stocks
        
        Args:
            symbols (list): List of stock symbols
            output_size (str): 'compact' or 'full'
            
        Returns:
            dict: Dictionary mapping symbols to DataFrames
        """
        if not self.is_ready():
            print("❌ Fetcher not ready. Check authentication.")
            return {}
        
        stock_data = {}
        valid_symbols = [s for s in symbols if self.query_builder.validate_symbol(s)]
        
        if len(valid_symbols) != len(symbols):
            invalid = set(symbols) - set(valid_symbols)
            print(f"⚠️ Skipping invalid symbols: {invalid}")
        
        print(f"📈 Fetching data for {len(valid_symbols)} symbols...")
        
        for i, symbol in enumerate(valid_symbols):
            print(f"[{i+1}/{len(valid_symbols)}] Processing: {symbol}")
            
            df = self.fetch_daily_stock_data(symbol, output_size)
            if df is not None:
                stock_data[symbol] = df
                print(f"✓ {symbol}: Successfully fetched {len(df)} rows")
            else:
                print(f"❌ {symbol}: Failed to fetch data")
        
        # Print final statistics
        self.api_call_manager.print_stats()
        
        return stock_data
    
    def fetch_intraday_data(self, symbol: str, interval='5min', output_size='compact') -> Optional[pd.DataFrame]:
        """
        Fetch intraday stock data
        
        Args:
            symbol (str): Stock symbol
            interval (str): Time interval ('1min', '5min', '15min', '30min', '60min')
            output_size (str): 'compact' or 'full'
            
        Returns:
            pandas.DataFrame: Intraday data or None if failed
        """
        if not self.is_ready():
            return None
        
        if not self.query_builder.validate_symbol(symbol):
            print(f"❌ Invalid symbol format: {symbol}")
            return None
        
        if not self.api_call_manager.prepare_call():
            return None
        
        try:
            params = self.query_builder.build_intraday_params(symbol, interval, output_size)
            headers = self.query_builder.get_headers()
            
            print(f"🔄 Fetching {interval} intraday data for {symbol}...")
            data = self.http_handler.make_get_request(
                url=self.query_builder.BASE_URL,
                params=params,
                headers=headers
            )
            
            error_analysis = self.api_call_manager.record_call_result(data, symbol)
            
            if error_analysis['has_error']:
                return None
            
            return self._convert_to_dataframe(data, symbol, is_intraday=True, interval=interval)
            
        except ValueError as e:
            print(f"❌ Parameter error: {e}")
            return None
    
    def _convert_to_dataframe(self, data: Dict[str, Any], symbol: str, 
                            is_intraday: bool = False, interval: str = None) -> Optional[pd.DataFrame]:
        """
        Convert API response to pandas DataFrame
        
        Args:
            data (dict): API response data
            symbol (str): Stock symbol
            is_intraday (bool): Whether this is intraday data
            interval (str): Time interval for intraday data
            
        Returns:
            pandas.DataFrame: Processed stock data
        """
        try:
            # Find the time series key in the response
            time_series_key = None
            if is_intraday:
                time_series_key = f"Time Series ({interval})"
            else:
                for key in data.keys():
                    if key.startswith("Time Series"):
                        time_series_key = key
                        break
            
            if not time_series_key or time_series_key not in data:
                print(f"❌ No time series data found for {symbol}")
                return None
            
            time_series = data[time_series_key]
            
            if not time_series:
                print(f"❌ Empty time series data for {symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            
            # Standardize column names
            column_mapping = {
                '1. open': 'Open',
                '2. high': 'High', 
                '3. low': 'Low',
                '4. close': 'Close',
                '5. volume': 'Volume'
            }
            
            df.columns = [column_mapping.get(col, col) for col in df.columns]
            
            # Convert to numeric types
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert index to datetime
            df.index = pd.to_datetime(df.index)
            df.index.name = 'Date'
            
            # Sort by date (oldest first)
            df = df.sort_index()
            
            # Add metadata
            df.attrs['symbol'] = symbol
            df.attrs['fetch_time'] = datetime.now()
            df.attrs['data_type'] = 'intraday' if is_intraday else 'daily'
            if is_intraday:
                df.attrs['interval'] = interval
            
            return df
            
        except Exception as e:
            print(f"❌ Error converting data for {symbol}: {e}")
            return None
    
    def get_available_data_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get information about available data without fetching full dataset
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            dict: Information about available data
        """
        if not self.is_ready():
            return None
        
        # Fetch compact data to get info
        df = self.fetch_daily_stock_data(symbol, output_size='compact')
        if df is None:
            return None
        
        return {
            'symbol': symbol,
            'total_days': len(df),
            'date_range': {
                'start': df.index.min().strftime('%Y-%m-%d'),
                'end': df.index.max().strftime('%Y-%m-%d')
            },
            'latest_price': float(df['Close'].iloc[-1]),
            'latest_volume': int(df['Volume'].iloc[-1]),
            'price_range': {
                'min': float(df['Close'].min()),
                'max': float(df['Close'].max())
            }
        }
    
    def save_data_to_csv(self, stock_data: Dict[str, pd.DataFrame], 
                        output_dir: str = 'stock_data') -> None:
        """
        Save stock data to CSV files
        
        Args:
            stock_data (dict): Dictionary of symbol -> DataFrame
            output_dir (str): Directory to save CSV files
        """
        import os
        
        if not stock_data:
            print("❌ No data to save")
            return
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        for symbol, df in stock_data.items():
            filename = f"{symbol}_daily_stock_data.csv"
            filepath = os.path.join(output_dir, filename)
            
            try:
                df.to_csv(filepath)
                print(f"💾 Saved {symbol} data to {filepath}")
            except Exception as e:
                print(f"❌ Failed to save {symbol}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about API usage
        
        Returns:
            dict: API usage statistics
        """
        return self.api_call_manager.get_stats()
    
    def close(self):
        """Close HTTP session and cleanup resources"""
        self.http_handler.close_session()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


# Example usage and main function
def main():
    """Example usage of the StockDataFetcher"""
    
    # List of popular stock symbols
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META"]
    
    # Initialize fetcher with context manager for proper cleanup
    with StockDataFetcher() as fetcher:
        if not fetcher.is_ready():
            print("❌ Failed to initialize fetcher. Check your API key.")
            return
        
        print("🚀 Starting stock data fetch...")
        
        # Fetch data for multiple stocks
        stock_data = fetcher.fetch_multiple_stocks(symbols, output_size='compact')
        
        if stock_data:
            print(f"\n✅ Successfully fetched data for {len(stock_data)} stocks")
            
            # Save to CSV files
            fetcher.save_data_to_csv(stock_data)
            
            # Display sample data for each stock
            for symbol, df in stock_data.items():
                print(f"\n📊 {symbol} - Latest 5 days:")
                print(df.tail().round(2))
            
            # Show API usage stats
            print(f"\n📈 Final Statistics:")
            stats = fetcher.get_stats()
            for key, value in stats.items():
                print(f"   {key.replace('_', ' ').title()}: {value}")
        
        else:
            print("❌ No data was successfully fetched")


if __name__ == "__main__":
    main()