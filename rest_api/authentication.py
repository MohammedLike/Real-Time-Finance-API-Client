import os
from dotenv import load_dotenv

class AuthenticationManager:
    """Manages API authentication for Alpha Vantage"""
    
    def __init__(self, env_file='apikey.env'):
        """
        Initialize authentication manager
        
        Args:
            env_file (str): Path to environment file containing API key
        """
        self.env_file = env_file
        self.api_key = None
        self._load_api_key()
    
    def _load_api_key(self):
        """Load API key from environment file"""
        try:
            load_dotenv(dotenv_path=self.env_file)
            self.api_key = os.getenv("ALPHAVANTAGE_API_KEY")
            
            if not self.api_key:
                raise ValueError("ALPHAVANTAGE_API_KEY not found in environment variables")
                
        except Exception as e:
            print(f"❌ Error loading API key: {e}")
            self.api_key = None
    
    def get_api_key(self):
        """
        Get the loaded API key
        
        Returns:
            str: API key if loaded successfully, None otherwise
        """
        return self.api_key
    
    def is_authenticated(self):
        """
        Check if API key is loaded and valid
        
        Returns:
            bool: True if API key is available, False otherwise
        """
        return self.api_key is not None
    
    def reload_api_key(self):
        """Reload API key from environment file"""
        self._load_api_key()
        return self.is_authenticated()
