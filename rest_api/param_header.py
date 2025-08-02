"""
Query parameters and headers configuration for Alpha Vantage API
Handles request configuration and parameter validation
"""

class QueryParamsBuilder:
    """Builds and validates query parameters for Alpha Vantage API requests"""
    
    BASE_URL = "https://www.alphavantage.co/query"
    
    SUPPORTED_FUNCTIONS = {
        'TIME_SERIES_DAILY': 'Daily time series data',
        'TIME_SERIES_WEEKLY': 'Weekly time series data',
        'TIME_SERIES_MONTHLY': 'Monthly time series data',
        'TIME_SERIES_INTRADAY': 'Intraday time series data'
    }
    
    def __init__(self, api_key):
        """
        Initialize query params builder
        
        Args:
            api_key (str): Alpha Vantage API key
        """
        self.api_key = api_key
    
    def build_daily_params(self, symbol, output_size='compact'):
        """
        Build parameters for daily time series request
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL')
            output_size (str): 'compact' for last 100 days, 'full' for all data
            
        Returns:
            dict: Query parameters for the request
        """
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol.upper(),
            "apikey": self.api_key,
            "outputsize": output_size
        }
        return params
    
    def build_intraday_params(self, symbol, interval='5min', output_size='compact'):
        """
        Build parameters for intraday time series request
        
        Args:
            symbol (str): Stock symbol
            interval (str): Time interval ('1min', '5min', '15min', '30min', '60min')
            output_size (str): 'compact' or 'full'
            
        Returns:
            dict: Query parameters for the request
        """
        valid_intervals = ['1min', '5min', '15min', '30min', '60min']
        if interval not in valid_intervals:
            raise ValueError(f"Invalid interval. Must be one of: {valid_intervals}")
        
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol.upper(),
            "interval": interval,
            "apikey": self.api_key,
            "outputsize": output_size
        }
        return params
    
    def build_weekly_params(self, symbol):
        """
        Build parameters for weekly time series request
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            dict: Query parameters for the request
        """
        params = {
            "function": "TIME_SERIES_WEEKLY",
            "symbol": symbol.upper(),
            "apikey": self.api_key
        }
        return params
    
    def build_monthly_params(self, symbol):
        """
        Build parameters for monthly time series request
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            dict: Query parameters for the request
        """
        params = {
            "function": "TIME_SERIES_MONTHLY",
            "symbol": symbol.upper(),
            "apikey": self.api_key
        }
        return params
    
    @staticmethod
    def get_headers():
        """
        Get standard headers for Alpha Vantage requests
        
        Returns:
            dict: HTTP headers for the request
        """
        return {
            'User-Agent': 'Python-Stock-Fetcher/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
    
    def validate_symbol(self, symbol):
        """
        Basic validation for stock symbol
        
        Args:
            symbol (str): Stock symbol to validate
            
        Returns:
            bool: True if symbol appears valid, False otherwise
        """
        if not symbol or not isinstance(symbol, str):
            return False
        
        # Basic validation: 1-5 characters, alphanumeric
        symbol = symbol.strip().upper()
        return len(symbol) >= 1 and len(symbol) <= 5 and symbol.isalnum()