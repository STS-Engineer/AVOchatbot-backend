
from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from app.models.otp_schemas import OTPRequest, OTPVerifyRequest
from app.models.schemas import TokenResponse, UserResponse
from app.services.otp import OTPService
from app.services.user import UserService
from app.services.auth import get_auth_service
from app.utils.mailer import send_email

router = APIRouter()
otp_service = OTPService()

user_service = UserService()
auth_service = get_auth_service()

@router.post("/send-otp")
async def send_otp(request: OTPRequest, background_tasks: BackgroundTasks):
    otp = otp_service.generate_otp()
    otp_service.store_otp(request.email, otp)
    # Send OTP via email (async)
    background_tasks.add_task(
        send_email,
        request.email,
        "Your Login OTP",
        f"Your OTP code is: {otp}"
    )
    return {"success": True, "message": "OTP sent to email"}

@router.post("/verify-otp")
async def verify_otp(request: OTPVerifyRequest):
    if otp_service.verify_otp(request.email, request.otp):
        return {"success": True, "message": "OTP verified"}
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired OTP")


# OTP login endpoint: verifies OTP and returns JWT tokens
@router.post("/login-otp", response_model=TokenResponse)
async def login_otp(request: OTPVerifyRequest):
    if not otp_service.verify_otp(request.email, request.otp):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired OTP")
    user = user_service.get_user_by_email(request.email)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    user_id = str(user["id"])
    access_token = auth_service.create_access_token({"sub": user_id})
    refresh_token = auth_service.create_refresh_token({"sub": user_id})
    # Optionally update last_login here
    user_response_data = {
        "id": user_id,
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
        expires_in=30*60,  # 30 minutes
        user=UserResponse(**user_response_data)
    )
