"""
HTTP GET request handler for Alpha Vantage API
Handles low-level HTTP communication with error handling
"""

import requests
import json
from typing import Dict, Optional, Any

class HTTPRequestHandler:
    """Handles HTTP GET requests with proper error handling and response parsing"""
    
    def __init__(self, timeout=30):
        """
        Initialize HTTP request handler
        
        Args:
            timeout (int): Request timeout in seconds
        """
        self.timeout = timeout
        self.session = requests.Session()
    
    def make_get_request(self, url: str, params: Dict[str, Any], 
                        headers: Optional[Dict[str, str]] = None) -> Optional[Dict]:
        """
        Make a GET request to the specified URL with parameters
        
        Args:
            url (str): The URL to make the request to
            params (dict): Query parameters for the request
            headers (dict, optional): HTTP headers for the request
            
        Returns:
            dict: Parsed JSON response if successful, None otherwise
        """
        try:
            response = self.session.get(
                url=url,
                params=params,
                headers=headers or {},
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                print(f"❌ HTTP Error {response.status_code}: {response.reason}")
                return None
            
            try:
                data = response.json()
                return data
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing error: {e}")
                print(f"Response content: {response.text[:200]}...")
                return None
                
        except requests.exceptions.Timeout:
            print(f"❌ Request timeout after {self.timeout} seconds")
            return None
            
        except requests.exceptions.ConnectionError:
            print("❌ Connection error - check your internet connection")
            return None
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Request exception: {e}")
            return None
    
    def test_connection(self, url: str) -> bool:
        """
        Test connection to the API endpoint
        
        Args:
            url (str): URL to test
            
        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            response = self.session.head(url, timeout=10)
            return response.status_code in [200, 405]  
        except Exception:
            return False
    
    def get_response_info(self, url: str, params: Dict[str, Any], 
                         headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Get detailed information about the response without parsing JSON
        
        Args:
            url (str): The URL to make the request to
            params (dict): Query parameters
            headers (dict, optional): HTTP headers
            
        Returns:
            dict: Response information including status, headers, etc.
        """
        try:
            response = self.session.get(
                url=url,
                params=params,
                headers=headers or {},
                timeout=self.timeout
            )
            
            return {
                'status_code': response.status_code,
                'headers': dict(response.headers),
                'content_length': len(response.content),
                'url': response.url,
                'encoding': response.encoding,
                'elapsed_seconds': response.elapsed.total_seconds()
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'status_code': None
            }
    
    def close_session(self):
        """Close the HTTP session"""
        self.session.close()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - closes session"""
        self.close_session()