import random
import string
from datetime import datetime, timedelta, timezone
from sqlalchemy import text
from app.services.database import get_users_database
from loguru import logger

class OTPService:
    def __init__(self):
        self.db = get_users_database()

    def generate_otp(self, length=6):
        return ''.join(random.choices(string.digits, k=length))

    def store_otp(self, email: str, otp: str, expires_in=300):
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
        with self.db.get_session() as session:
            session.execute(
                text("""
                INSERT INTO otps (email, otp, expires_at)
                VALUES (:email, :otp, :expires_at)
                ON CONFLICT (email) DO UPDATE SET otp = :otp, expires_at = :expires_at
                """),
                {"email": email, "otp": otp, "expires_at": expires_at}
            )
            session.commit()
        logger.info(f"OTP stored for {email}, expires at {expires_at}")

    def verify_otp(self, email: str, otp: str):
        with self.db.get_session() as session:
            result = session.execute(
                text("""
                SELECT otp, expires_at FROM otps WHERE email = :email
                """),
                {"email": email}
            ).fetchone()
            if not result:
                logger.warning(f"No OTP found for {email}")
                return False
            db_otp, expires_at = result
            now = datetime.now(timezone.utc)
            # Ensure expires_at is timezone-aware (UTC)
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            logger.info(f"OTP debug: input={otp!r}, db_otp={db_otp!r}, expires_at={expires_at!r}, now={now!r}")
            if db_otp == otp and expires_at > now:
                session.execute(text("DELETE FROM otps WHERE email = :email"), {"email": email})
                session.commit()
                return True
            logger.warning(f"OTP check failed: match={db_otp == otp}, not expired={expires_at > now}")
            return False
