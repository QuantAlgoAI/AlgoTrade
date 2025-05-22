import os
import secrets
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

class Config:
    """Main configuration class."""
    
    # Application Settings
    LOG_DIR = os.getenv('LOG_DIR', 'logs')
    LOG_RETENTION_DAYS = int(os.getenv('LOG_RETENTION_DAYS', '30'))
    
    # Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or secrets.token_hex(16)
    
    @staticmethod
    def validate():
        """Validate that all required environment variables are set."""
        required_vars = [
            'LOG_DIR',
            'LOG_RETENTION_DAYS'
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            ) 