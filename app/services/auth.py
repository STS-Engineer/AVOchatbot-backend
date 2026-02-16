"""
Authentication service - JWT token generation and password hashing.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from loguru import logger
from app.core.config import settings
import hashlib
import base64
import bcrypt  # Use bcrypt directly instead of through passlib


class AuthService:
    """Handles authentication operations."""
    
    def __init__(self):
        """Initialize authentication service."""
        self.secret_key = settings.JWT_SECRET_KEY
        self.algorithm = settings.JWT_ALGORITHM
        self.access_token_expire = timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
        self.refresh_token_expire = timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS)
    
    def _normalize_password(self, password: str) -> str:
        """
        Normalize password to work with bcrypt's 72-byte limit.
        Uses SHA256 pre-hashing for long passwords.
        """
        password_bytes = password.encode('utf-8')
        actual_length = len(password_bytes)
        
        logger.debug(f"Password byte length: {actual_length}")
        
        # Bcrypt has a strict 72-byte limit
        # If password is longer, use SHA256 pre-hash
        if actual_length > 72:
            logger.warning(f"Password is {actual_length} bytes, using SHA256 pre-hash")
            # Create SHA256 hash and encode as base64
            sha_hash = hashlib.sha256(password_bytes).digest()
            # Base64 encode to make it a valid string (44 chars, well under 72 bytes)
            normalized = base64.b64encode(sha_hash).decode('ascii')
            logger.debug(f"Normalized to {len(normalized)} chars, {len(normalized.encode('utf-8'))} bytes")
            return normalized
        
        logger.debug("Password is within 72-byte limit, no pre-hashing needed")
        return password
    
    def hash_password(self, password: str) -> str:
        """
        Hash a password using bcrypt directly (bypasses passlib to avoid version conflicts).
        
        Bcrypt has a 72-byte limit. For longer passwords, we use SHA256 pre-hashing.
        """
        logger.info(f"🔐 Hashing password (length: {len(password)} chars)")
        
        try:
            # Normalize password (apply SHA256 pre-hash if needed)
            normalized = self._normalize_password(password)
            logger.info(f"🔐 Normalized password ready for bcrypt")
            
            # Ensure the normalized password is truly under 72 bytes
            normalized_bytes = normalized.encode('utf-8')
            if len(normalized_bytes) > 72:
                # This should never happen, but add safety check
                logger.error(f"Normalized password still too long: {len(normalized_bytes)} bytes")
                raise ValueError(f"Password normalization failed: {len(normalized_bytes)} bytes")
            
            # Hash with bcrypt directly
            logger.info("🔐 Calling bcrypt.hashpw()...")
            salt = bcrypt.gensalt()
            hashed = bcrypt.hashpw(normalized_bytes, salt)
            logger.info("✅ Password hashing completed successfully")
            
            # Return as string (decode from bytes)
            return hashed.decode('utf-8')
        except Exception as e:
            logger.error(f"❌ Bcrypt hashing error: {e}", exc_info=True)
            raise
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        Verify a password against its hash using bcrypt directly.
        
        Handles the same normalization as hash_password for consistency.
        """
        try:
            # Apply same normalization as during hashing
            normalized = self._normalize_password(plain_password)
            normalized_bytes = normalized.encode('utf-8')
            hashed_bytes = hashed_password.encode('utf-8')
            
            # Verify with bcrypt directly
            return bcrypt.checkpw(normalized_bytes, hashed_bytes)
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False
    
    def create_access_token(self, data: Dict[str, Any]) -> str:
        """
        Create a JWT access token.
        
        Args:
            data: Payload to encode (should include 'sub' for user_id)
        
        Returns:
            Encoded JWT token
        """
        to_encode = data.copy()
        expire = datetime.utcnow() + self.access_token_expire
        to_encode.update({
            "exp": expire,
            "type": "access"
        })
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """
        Create a JWT refresh token.
        
        Args:
            data: Payload to encode (should include 'sub' for user_id)
        
        Returns:
            Encoded JWT token
        """
        to_encode = data.copy()
        expire = datetime.utcnow() + self.refresh_token_expire
        to_encode.update({
            "exp": expire,
            "type": "refresh"
        })
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str, token_type: str = "access") -> Optional[Dict[str, Any]]:
        """
        Verify and decode a JWT token.
        
        Args:
            token: JWT token to verify
            token_type: Expected token type ('access' or 'refresh')
        
        Returns:
            Decoded payload if valid, None otherwise
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check token type
            if payload.get("type") != token_type:
                logger.warning(f"Invalid token type. Expected {token_type}, got {payload.get('type')}")
                return None
            
            # Check expiration
            exp = payload.get("exp")
            if exp is None:
                logger.warning("Token missing expiration")
                return None
            
            if datetime.utcnow() > datetime.fromtimestamp(exp):
                logger.warning("Token has expired")
                return None
            
            return payload
        
        except JWTError as e:
            logger.error(f"JWT verification error: {e}")
            return None
    
    def decode_token_user_id(self, token: str) -> Optional[str]:
        """
        Extract user_id from a valid access token.
        
        Args:
            token: JWT access token
        
        Returns:
            user_id if valid, None otherwise
        """
        payload = self.verify_token(token, token_type="access")
        if payload:
            return payload.get("sub")
        return None


# Singleton instance
_auth_service = None


def get_auth_service() -> AuthService:
    """Get or create authentication service instance."""
    global _auth_service
    if _auth_service is None:
        _auth_service = AuthService()
        logger.info("Authentication service initialized")
    return _auth_service
