from cryptography.fernet import Fernet
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import logging
import base64
from typing import Optional

# Load environment variables
load_dotenv()

# Security configuration
ENCRYPTION_KEY = os.getenv('ENCRYPTION_KEY')
JWT_SECRET = os.getenv('JWT_SECRET')
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Initialize encryption
if not ENCRYPTION_KEY:
    ENCRYPTION_KEY = Fernet.generate_key()
    logging.warning("No encryption key found. Generated new key.")

fernet = Fernet(ENCRYPTION_KEY)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def encrypt_data(data: str) -> str:
    """Encrypt sensitive data"""
    try:
        return fernet.encrypt(data.encode()).decode()
    except Exception as e:
        logging.error(f"Error encrypting data: {str(e)}")
        raise

def decrypt_data(encrypted_data: str) -> str:
    """Decrypt sensitive data"""
    try:
        return fernet.decrypt(encrypted_data.encode()).decode()
    except Exception as e:
        logging.error(f"Error decrypting data: {str(e)}")
        raise

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Generate password hash"""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Optional[dict]:
    """Verify JWT token"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        return payload
    except JWTError as e:
        logging.error(f"Error verifying token: {str(e)}")
        return None

class SecurityManager:
    def __init__(self):
        self.encryption_key = ENCRYPTION_KEY
        self.jwt_secret = JWT_SECRET
        self.fernet = fernet
        self.pwd_context = pwd_context

    def encrypt_api_credentials(self, api_key: str, api_secret: str) -> dict:
        """Encrypt API credentials"""
        return {
            'api_key': encrypt_data(api_key),
            'api_secret': encrypt_data(api_secret)
        }

    def decrypt_api_credentials(self, encrypted_credentials: dict) -> dict:
        """Decrypt API credentials"""
        return {
            'api_key': decrypt_data(encrypted_credentials['api_key']),
            'api_secret': decrypt_data(encrypted_credentials['api_secret'])
        }

    def create_user_token(self, user_id: str) -> str:
        """Create user authentication token"""
        return create_access_token(
            data={"sub": user_id},
            expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        )

    def verify_user_token(self, token: str) -> Optional[str]:
        """Verify user authentication token"""
        payload = verify_token(token)
        if payload:
            return payload.get("sub")
        return None

    def hash_sensitive_data(self, data: str) -> str:
        """Hash sensitive data for storage"""
        return pwd_context.hash(data)

    def verify_hashed_data(self, plain_data: str, hashed_data: str) -> bool:
        """Verify hashed data"""
        return pwd_context.verify(plain_data, hashed_data)

# Initialize security manager
security_manager = SecurityManager() 