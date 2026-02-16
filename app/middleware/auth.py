"""
Authentication middleware - JWT verification and user extraction.
"""

from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from loguru import logger
from app.services.auth import get_auth_service
from app.services.user import get_user_service


# HTTP Bearer token scheme
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    """
    Dependency to get the current authenticated user.
    
    Validates the JWT token and returns the user data.
    Raises 401 if token is invalid or user not found.
    
    Usage in routes:
        @router.get("/protected")
        async def protected_route(current_user: dict = Depends(get_current_user)):
            user_id = current_user["id"]
            ...
    """
    auth_service = get_auth_service()
    user_service = get_user_service()
    
    # Extract token from Authorization header
    token = credentials.credentials
    
    # Verify token and extract user_id
    user_id = auth_service.decode_token_user_id(token)
    
    if not user_id:
        logger.warning("Invalid or expired token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get user from database
    user = user_service.get_user_by_id(user_id)
    
    if not user:
        logger.warning(f"User not found for token: {user_id}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.get("is_active"):
        logger.warning(f"Inactive user attempted access: {user_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive",
        )
    
    return user


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))
) -> Optional[dict]:
    """
    Optional authentication dependency.
    Returns user if token is valid, None otherwise.
    Does not raise exceptions for missing/invalid tokens.
    
    Usage in routes that work with or without authentication:
        @router.get("/content")
        async def content(user: Optional[dict] = Depends(get_current_user_optional)):
            if user:
                # Authenticated user
                ...
            else:
                # Anonymous user
                ...
    """
    if not credentials:
        return None
    
    try:
        auth_service = get_auth_service()
        user_service = get_user_service()
        
        token = credentials.credentials
        user_id = auth_service.decode_token_user_id(token)
        
        if user_id:
            user = user_service.get_user_by_id(user_id)
            if user and user.get("is_active"):
                return user
        
        return None
    
    except Exception as e:
        logger.debug(f"Optional auth failed (expected): {e}")
        return None
