"""
Rate limiting and error handling for Alpha Vantage API
Manages API call frequency and handles various error conditions
"""

import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

class RateLimiter:
    """Manages rate limiting for API calls"""
    
    def __init__(self, calls_per_minute=5, calls_per_day=500):
        """
        Initialize rate limiter
        
        Args:
            calls_per_minute (int): Maximum calls per minute (Alpha Vantage free tier: 5)
            calls_per_day (int): Maximum calls per day (Alpha Vantage free tier: 500)
        """
        self.calls_per_minute = calls_per_minute
        self.calls_per_day = calls_per_day
        self.call_history = []
        self.daily_calls = 0
        self.last_reset_date = datetime.now().date()
    
    def can_make_call(self) -> bool:
        """
        Check if a call can be made within rate limits
        
        Returns:
            bool: True if call is allowed, False otherwise
        """
        now = datetime.now()
        
        if now.date() > self.last_reset_date:
            self.daily_calls = 0
            self.last_reset_date = now.date()
        
        if self.daily_calls >= self.calls_per_day:
            return False
        
        one_minute_ago = now - timedelta(minutes=1)
        self.call_history = [call_time for call_time in self.call_history 
                           if call_time > one_minute_ago]
        
        return len(self.call_history) < self.calls_per_minute
    
    def record_call(self):
        """Record that a call was made"""
        now = datetime.now()
        self.call_history.append(now)
        self.daily_calls += 1
    
    def get_wait_time(self) -> float:
        """
        Get time to wait before next call is allowed
        
        Returns:
            float: Seconds to wait, 0 if call can be made immediately
        """
        if not self.call_history:
            return 0
        
        now = datetime.now()
        one_minute_ago = now - timedelta(minutes=1)
        recent_calls = [call_time for call_time in self.call_history 
                       if call_time > one_minute_ago]
        
        if len(recent_calls) < self.calls_per_minute:
            return 0
        
        oldest_call = min(recent_calls)
        wait_until = oldest_call + timedelta(minutes=1)
        wait_seconds = (wait_until - now).total_seconds()
        
        return max(0, wait_seconds)
    
    def wait_if_needed(self):
        """Wait if necessary to respect rate limits"""
        wait_time = self.get_wait_time()
        if wait_time > 0:
            print(f"⏳ Rate limit reached. Waiting {wait_time:.1f} seconds...")
            time.sleep(wait_time)


class ErrorHandler:
    """Handles and categorizes API errors"""
    
    KNOWN_ERRORS = {
        "Invalid API call": "API call format is incorrect",
        "the parameter apikey is invalid": "API key is invalid or missing",
        "Thank you for using Alpha Vantage": "Rate limit exceeded",
        "Note": "API limit reached or temporary issue"
    }
    
    @staticmethod
    def analyze_response(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze API response for errors
        
        Args:
            data (dict): JSON response from API
            
        Returns:
            dict: Error analysis results
        """
        result = {
            'has_error': False,
            'error_type': None,
            'error_message': None,
            'is_rate_limit': False,
            'is_auth_error': False,
            'is_recoverable': True
        }
        
        if not data:
            result.update({
                'has_error': True,
                'error_type': 'no_data',
                'error_message': 'No data received from API',
                'is_recoverable': False
            })
            return result
        
        if "Error Message" in data:
            error_msg = data["Error Message"]
            result.update({
                'has_error': True,
                'error_type': 'api_error',
                'error_message': error_msg,
                'is_auth_error': 'apikey' in error_msg.lower(),
                'is_recoverable': 'apikey' not in error_msg.lower()
            })
        
        elif "Note" in data:
            note_msg = data["Note"]
            result.update({
                'has_error': True,
                'error_type': 'rate_limit',
                'error_message': note_msg,
                'is_rate_limit': True,
                'is_recoverable': True
            })
        
        elif not any(key.startswith("Time Series") for key in data.keys()):
            if not any(key in ["Meta Data", "Realtime Currency Exchange Rate"] for key in data.keys()):
                result.update({
                    'has_error': True,
                    'error_type': 'no_time_series',
                    'error_message': 'No time series data found in response',
                    'is_recoverable': True
                })
        
        return result
    
    @staticmethod
    def handle_error(error_analysis: Dict[str, Any], symbol: str = None) -> None:
        """
        Handle errors based on analysis
        
        Args:
            error_analysis (dict): Results from analyze_response
            symbol (str, optional): Stock symbol being processed
        """
        if not error_analysis['has_error']:
            return
        
        symbol_info = f" for {symbol}" if symbol else ""
        error_msg = error_analysis['error_message']
        
        if error_analysis['is_auth_error']:
            print(f"🔑 Authentication Error{symbol_info}: {error_msg}")
            print("   Please check your API key in the .env file")
        
        elif error_analysis['is_rate_limit']:
            print(f"⏱️ Rate Limit{symbol_info}: {error_msg}")
            print("   Consider upgrading your API plan or increasing delays")
        
        elif error_analysis['error_type'] == 'no_data':
            print(f"📭 No Data{symbol_info}: {error_msg}")
        
        elif error_analysis['error_type'] == 'no_time_series':
            print(f"📊 Data Issue{symbol_info}: {error_msg}")
            print("   The symbol might be invalid or delisted")
        
        else:
            print(f"⚠️ API Error{symbol_info}: {error_msg}")
    
    @staticmethod
    def suggest_recovery(error_analysis: Dict[str, Any]) -> Optional[str]:
        """
        Suggest recovery actions for errors
        
        Args:
            error_analysis (dict): Results from analyze_response
            
        Returns:
            str: Recovery suggestion or None
        """
        if not error_analysis['has_error'] or not error_analysis['is_recoverable']:
            return None
        
        if error_analysis['is_rate_limit']:
            return "Wait and retry with longer delays between requests"
        elif error_analysis['error_type'] == 'no_time_series':
            return "Verify the stock symbol is correct and currently traded"
        elif error_analysis['error_type'] == 'api_error':
            return "Check API parameters and symbol format"
        else:
            return "Retry the request after a short delay"


class APICallManager:
    """Combines rate limiting and error handling for API calls"""
    
    def __init__(self, calls_per_minute=5, calls_per_day=500):
        """
        Initialize API call manager
        
        Args:
            calls_per_minute (int): Maximum calls per minute
            calls_per_day (int): Maximum calls per day
        """
        self.rate_limiter = RateLimiter(calls_per_minute, calls_per_day)
        self.error_handler = ErrorHandler()
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
    
    def prepare_call(self) -> bool:
        """
        Prepare for an API call (handle rate limiting)
        
        Returns:
            bool: True if call can proceed, False if daily limit reached
        """
        if not self.rate_limiter.can_make_call():
            if self.rate_limiter.daily_calls >= self.rate_limiter.calls_per_day:
                print("🚫 Daily API limit reached. Cannot make more calls today.")
                return False
            
            self.rate_limiter.wait_if_needed()
        
        return True
    
    def record_call_result(self, data: Dict[str, Any], symbol: str = None) -> Dict[str, Any]:
        """
        Record the result of an API call
        
        Args:
            data (dict): API response data
            symbol (str, optional): Stock symbol
            
        Returns:
            dict: Error analysis results
        """
        self.rate_limiter.record_call()
        self.total_calls += 1
        
        error_analysis = self.error_handler.analyze_response(data)
        
        if error_analysis['has_error']:
            self.failed_calls += 1
            self.error_handler.handle_error(error_analysis, symbol)
        else:
            self.successful_calls += 1
        
        return error_analysis
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about API calls
        
        Returns:
            dict: Call statistics
        """
        return {
            'total_calls': self.total_calls,
            'successful_calls': self.successful_calls,
            'failed_calls': self.failed_calls,
            'success_rate': (self.successful_calls / self.total_calls * 100) if self.total_calls > 0 else 0,
            'daily_calls_used': self.rate_limiter.daily_calls,
            'daily_calls_remaining': self.rate_limiter.calls_per_day - self.rate_limiter.daily_calls
        }
    
    def print_stats(self):
        """Print call statistics"""
        stats = self.get_stats()
        print(f"\n📊 API Call Statistics:")
        print(f"   Total calls: {stats['total_calls']}")
        print(f"   Successful: {stats['successful_calls']}")
        print(f"   Failed: {stats['failed_calls']}")
        print(f"   Success rate: {stats['success_rate']:.1f}%")
        print(f"   Daily calls used: {stats['daily_calls_used']}/{self.rate_limiter.calls_per_day}")