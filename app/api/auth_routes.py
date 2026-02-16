"""
Authentication routes - Register, login, and token management.
"""

from fastapi import APIRouter, HTTPException, status, Depends
from loguru import logger
from datetime import datetime, timedelta
from hashlib import sha256
from app.models.schemas import (
    UserRegister,
    UserLogin,
    TokenResponse,
    RefreshTokenRequest,
    UserResponse,
    ConversationResponse
)
from app.services.auth import get_auth_service
from app.services.user import get_user_service
from app.services.conversation import get_conversation_service
from app.middleware.auth import get_current_user
from app.core.config import settings


router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserRegister) -> TokenResponse:
    """
    Register a new user.
    
    **Request Body:**
    - `email`: User's email address (required, unique)
    - `username`: Unique username (required, 3-100 characters)
    - `password`: Password (required, minimum 6 characters)
    - `full_name`: Optional full name
    
    **Response:**
    - `access_token`: JWT access token (valid for 30 minutes)
    - `refresh_token`: JWT refresh token (valid for 7 days)
    - `token_type`: Always "bearer"
    - `expires_in`: Token expiration time in seconds
    - `user`: User profile data
    """
    logger.info(f"🚀 REGISTER endpoint called - Email: {user_data.email}, Username: {user_data.username}")
    logger.info(f"🔍 Password length: {len(user_data.password)} chars, {len(user_data.password.encode('utf-8'))} bytes")
    
    try:
        user_service = get_user_service()
        auth_service = get_auth_service()
        
        # Create user
        logger.info(f"Creating user: {user_data.username}")
        user = user_service.create_user(
            email=user_data.email,
            username=user_data.username,
            password=user_data.password,
            full_name=user_data.full_name
        )
        
        if not user:
            logger.warning(f"User creation failed - email or username exists: {user_data.email}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email or username already exists"
            )
        
        # Generate tokens
        user_id = str(user["id"])
        logger.info(f"Generating tokens for user ID: {user_id}")
        access_token = auth_service.create_access_token({"sub": user_id})
        refresh_token = auth_service.create_refresh_token({"sub": user_id})
        
        # Store refresh token hash
        token_hash = sha256(refresh_token.encode()).hexdigest()
        expires_at = datetime.utcnow() + timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS)
        user_service.store_refresh_token(user_id, token_hash, expires_at)
        
        logger.info(f"✅ User registered successfully: {user_data.username}")
        
        # Convert user dict for response (ensure id is string)
        user_response_data = {
            "id": str(user["id"]),
            "email": user["email"],
            "username": user["username"],
            "full_name": user.get("full_name"),
            "created_at": user["created_at"],
            "last_login": user.get("last_login"),
            "is_active": user["is_active"],
            "is_verified": user["is_verified"]
        }
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user=UserResponse(**user_response_data)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Registration error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )


@router.post("/login", response_model=TokenResponse)
async def login(credentials: UserLogin) -> TokenResponse:
    """
    Login with email and password.
    
    **Request Body:**
    - `email`: User's email address
    - `password`: User's password
    
    **Response:**
    - `access_token`: JWT access token (valid for 30 minutes)
    - `refresh_token`: JWT refresh token (valid for 7 days)
    - `token_type`: Always "bearer"
    - `expires_in`: Token expiration time in seconds
    - `user`: User profile data
    """
    logger.info(f"🚀 LOGIN endpoint called - Email: {credentials.email}")
    
    try:
        user_service = get_user_service()
        auth_service = get_auth_service()
        
        # Authenticate user
        logger.info(f"Authenticating user: {credentials.email}")
        user = user_service.authenticate_user(credentials.email, credentials.password)
        
        if not user:
            logger.warning(f"Authentication failed for: {credentials.email}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Generate tokens
        user_id = str(user["id"])
        logger.info(f"Generating tokens for user ID: {user_id}")
        access_token = auth_service.create_access_token({"sub": user_id})
        refresh_token = auth_service.create_refresh_token({"sub": user_id})
        
        # Store refresh token hash
        token_hash = sha256(refresh_token.encode()).hexdigest()
        expires_at = datetime.utcnow() + timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS)
        user_service.store_refresh_token(user_id, token_hash, expires_at)
        
        logger.info(f"✅ User logged in successfully: {credentials.email}")
        
        # Convert user dict for response (ensure id is string)
        user_response_data = {
            "id": str(user["id"]),
            "email": user["email"],
            "username": user["username"],
            "full_name": user.get("full_name"),
            "created_at": user["created_at"],
            "last_login": user.get("last_login"),
            "is_active": user["is_active"],
            "is_verified": user["is_verified"]
        }
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user=UserResponse(**user_response_data)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Login error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}"
        )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_access_token(token_request: RefreshTokenRequest) -> TokenResponse:
    """
    Refresh access token using a valid refresh token.
    
    **Request Body:**
    - `refresh_token`: Valid JWT refresh token
    
    **Response:**
    - New access token and refresh token
    - User profile data
    """
    auth_service = get_auth_service()
    user_service = get_user_service()
    
    # Verify refresh token
    payload = auth_service.verify_token(token_request.refresh_token, token_type="refresh")
    
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
        )
    
    # Verify token exists in database and is not revoked
    token_hash = sha256(token_request.refresh_token.encode()).hexdigest()
    if not user_service.verify_refresh_token(user_id, token_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token has been revoked or does not exist",
        )
    
    # Get user
    user = user_service.get_user_by_id(user_id)
    if not user or not user.get("is_active"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )
    
    # Revoke old refresh token
    user_service.revoke_refresh_token(user_id, token_hash)
    
    # Generate new tokens
    access_token = auth_service.create_access_token({"sub": user_id})
    new_refresh_token = auth_service.create_refresh_token({"sub": user_id})
    
    # Store new refresh token
    new_token_hash = sha256(new_refresh_token.encode()).hexdigest()
    expires_at = datetime.utcnow() + timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS)
    user_service.store_refresh_token(user_id, new_token_hash, expires_at)
    
    logger.info(f"Token refreshed for user: {user_id}")
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=new_refresh_token,
        token_type="bearer",
        expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=UserResponse(**user)
    )


@router.post("/logout")
async def logout(
    token_request: RefreshTokenRequest,
    current_user: dict = Depends(get_current_user)
) -> dict:
    """
    Logout user by revoking refresh token.
    
    **Request Body:**
    - `refresh_token`: Refresh token to revoke
    
    **Headers:**
    - `Authorization`: Bearer {access_token}
    """
    user_service = get_user_service()
    
    user_id = str(current_user["id"])
    token_hash = sha256(token_request.refresh_token.encode()).hexdigest()
    
    user_service.revoke_refresh_token(user_id, token_hash)
    
    logger.info(f"User logged out: {current_user['email']}")
    
    return {
        "success": True,
        "message": "Logged out successfully"
    }


@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(current_user: dict = Depends(get_current_user)) -> UserResponse:
    """
    Get current user profile.
    
    **Headers:**
    - `Authorization`: Bearer {access_token}
    
    **Response:**
    - User profile data
    """
    # Convert user dict for response (ensure id is string)
    user_response_data = {
        "id": str(current_user["id"]),
        "email": current_user["email"],
        "username": current_user["username"],
        "full_name": current_user.get("full_name"),
        "created_at": current_user["created_at"],
        "last_login": current_user.get("last_login"),
        "is_active": current_user["is_active"],
        "is_verified": current_user["is_verified"]
    }
    return UserResponse(**user_response_data)


@router.get("/conversations", response_model=list[ConversationResponse])
async def list_user_conversations(current_user: dict = Depends(get_current_user)) -> list[ConversationResponse]:
    """
    List all conversations for the authenticated user.
    
    **Headers:**
    - `Authorization`: Bearer {access_token}
    
    **Response:**
    - List of conversations with metadata
    """
    conversation_service = get_conversation_service()
    user_id = str(current_user["id"])
    
    conversations = conversation_service.list_conversations(user_id, include_archived=False)
    
    # Convert UUID to string for Pydantic validation
    return [
        ConversationResponse(
            id=str(conv["id"]),
            title=conv["title"],
            created_at=conv["created_at"],
            updated_at=conv["updated_at"],
            is_archived=conv["is_archived"],
            message_count=conv.get("message_count")
        ) for conv in conversations
    ]
