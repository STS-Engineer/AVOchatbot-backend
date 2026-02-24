from pydantic import BaseModel, Field

class OTPRequest(BaseModel):
    email: str = Field(..., description="Email address to send OTP to")

class OTPVerifyRequest(BaseModel):
    email: str = Field(..., description="Email address")
    otp: str = Field(..., description="OTP code received by user")
