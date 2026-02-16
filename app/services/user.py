"""
User service - User management and database operations.
"""

from typing import Optional, Dict, Any
from datetime import datetime
from sqlalchemy import text
from loguru import logger
from app.services.database import get_database
from app.services.auth import get_auth_service


class UserService:
    """Handles user-related database operations."""
    
    def __init__(self):
        """Initialize user service."""
        self.db = get_database()
        self.auth = get_auth_service()
        logger.info("User service initialized")
    
    def create_user(
        self,
        email: str,
        username: str,
        password: str,
        full_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Create a new user.
        
        Args:
            email: User's email address
            username: Unique username
            password: Plain text password (will be hashed)
            full_name: Optional full name
        
        Returns:
            User data if created successfully, None otherwise
        """
        try:
            with self.db.get_session() as session:
                # Check if user already exists
                existing = session.execute(
                    text("""
                    SELECT id FROM users 
                    WHERE email = :email OR username = :username
                    """),
                    {"email": email, "username": username}
                ).fetchone()
                
                if existing:
                    logger.warning(f"User already exists with email or username: {email}/{username}")
                    return None
                
                # Hash password
                try:
                    logger.info(f"Hashing password for user: {username}")
                    password_hash = self.auth.hash_password(password)
                    logger.info("Password hashed successfully")
                except Exception as hash_error:
                    logger.error(f"Password hashing failed: {hash_error}")
                    raise
                
                # Insert user
                result = session.execute(
                    text("""
                    INSERT INTO users (email, username, password_hash, full_name, created_at, is_active, is_verified)
                    VALUES (:email, :username, :password_hash, :full_name, :created_at, TRUE, FALSE)
                    RETURNING id, email, username, full_name, created_at, is_active, is_verified
                    """),
                    {
                        "email": email,
                        "username": username,
                        "password_hash": password_hash,
                        "full_name": full_name,
                        "created_at": datetime.utcnow()
                    }
                )
                session.commit()
                
                user = result.fetchone()
                if user:
                    logger.info(f"User created successfully: {username}")
                    return dict(user._mapping)
                
                return None
        
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return None
    
    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get user by email address."""
        try:
            with self.db.get_session() as session:
                result = session.execute(
                    text("""
                    SELECT id, email, username, password_hash, full_name, 
                           created_at, last_login, is_active, is_verified
                    FROM users 
                    WHERE email = :email
                    """),
                    {"email": email}
                )
                user = result.fetchone()
                return dict(user._mapping) if user else None
        except Exception as e:
            logger.error(f"Error getting user by email: {e}")
            return None
    
    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user by username."""
        try:
            with self.db.get_session() as session:
                result = session.execute(
                    text("""
                    SELECT id, email, username, password_hash, full_name, 
                           created_at, last_login, is_active, is_verified
                    FROM users 
                    WHERE username = :username
                    """),
                    {"username": username}
                )
                user = result.fetchone()
                return dict(user._mapping) if user else None
        except Exception as e:
            logger.error(f"Error getting user by username: {e}")
            return None
    
    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID."""
        try:
            with self.db.get_session() as session:
                result = session.execute(
                    text("""
                    SELECT id, email, username, full_name, 
                           created_at, last_login, is_active, is_verified
                    FROM users 
                    WHERE id = :user_id
                    """),
                    {"user_id": user_id}
                )
                user = result.fetchone()
                return dict(user._mapping) if user else None
        except Exception as e:
            logger.error(f"Error getting user by ID: {e}")
            return None
    
    def authenticate_user(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        """
        Authenticate a user with email and password.
        
        Args:
            email: User's email
            password: Plain text password
        
        Returns:
            User data (without password_hash) if authenticated, None otherwise
        """
        try:
            user = self.get_user_by_email(email)
            
            if not user:
                logger.warning(f"Authentication failed: User not found - {email}")
                return None
            
            if not user.get("is_active"):
                logger.warning(f"Authentication failed: User inactive - {email}")
                return None
            
            # Verify password
            if not self.auth.verify_password(password, user["password_hash"]):
                logger.warning(f"Authentication failed: Invalid password - {email}")
                return None
            
            # Update last login
            self.update_last_login(str(user["id"]))
            
            # Remove password_hash from returned data
            user.pop("password_hash", None)
            
            logger.info(f"User authenticated successfully: {email}")
            return user
        
        except Exception as e:
            logger.error(f"Error authenticating user: {e}")
            return None
    
    def update_last_login(self, user_id: str):
        """Update user's last login timestamp."""
        try:
            with self.db.get_session() as session:
                session.execute(
                    text("""
                    UPDATE users 
                    SET last_login = :last_login
                    WHERE id = :user_id
                    """),
                    {"user_id": user_id, "last_login": datetime.utcnow()}
                )
                session.commit()
        except Exception as e:
            logger.error(f"Error updating last login: {e}")
    
    def store_refresh_token(self, user_id: str, token_hash: str, expires_at: datetime) -> bool:
        """Store a refresh token in the database."""
        try:
            with self.db.get_session() as session:
                session.execute(
                    text("""
                    INSERT INTO refresh_tokens (user_id, token_hash, expires_at, created_at)
                    VALUES (:user_id, :token_hash, :expires_at, :created_at)
                    """),
                    {
                        "user_id": user_id,
                        "token_hash": token_hash,
                        "expires_at": expires_at,
                        "created_at": datetime.utcnow()
                    }
                )
                session.commit()
                return True
        except Exception as e:
            logger.error(f"Error storing refresh token: {e}")
            return False
    
    def verify_refresh_token(self, user_id: str, token_hash: str) -> bool:
        """Verify a refresh token exists and is not revoked."""
        try:
            with self.db.get_session() as session:
                result = session.execute(
                    text("""
                    SELECT id FROM refresh_tokens
                    WHERE user_id = :user_id 
                    AND token_hash = :token_hash
                    AND expires_at > :now
                    AND revoked = FALSE
                    """),
                    {
                        "user_id": user_id,
                        "token_hash": token_hash,
                        "now": datetime.utcnow()
                    }
                )
                return result.fetchone() is not None
        except Exception as e:
            logger.error(f"Error verifying refresh token: {e}")
            return False
    
    def revoke_refresh_token(self, user_id: str, token_hash: str) -> bool:
        """Revoke a refresh token."""
        try:
            with self.db.get_session() as session:
                session.execute(
                    text("""
                    UPDATE refresh_tokens
                    SET revoked = TRUE
                    WHERE user_id = :user_id AND token_hash = :token_hash
                    """),
                    {"user_id": user_id, "token_hash": token_hash}
                )
                session.commit()
                return True
        except Exception as e:
            logger.error(f"Error revoking refresh token: {e}")
            return False


# Singleton instance
_user_service = None


def get_user_service() -> UserService:
    """Get or create user service instance."""
    global _user_service
    if _user_service is None:
        _user_service = UserService()
    return _user_service
