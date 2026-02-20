import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

def send_email(to_email: str, subject: str, html_body: str, cc_emails: list = None):
    """
    Send email using SMTP (Outlook or standard SMTP server)
    Supports both authenticated and unauthenticated SMTP
    """
    try:
        logger.info(f"Attempting to send email to {to_email}")
        logger.debug(f"SMTP Config - Host: {settings.SMTP_HOST}, Port: {settings.SMTP_PORT}, From: {settings.SMTP_FROM}")
        
        msg = MIMEMultipart("alternative")
        msg["From"] = settings.SMTP_FROM
        msg["To"] = to_email
        msg["Subject"] = subject
        
        if cc_emails:
            msg["Cc"] = ", ".join(cc_emails)
        
        msg.attach(MIMEText(html_body, "html"))

        recipients = [to_email]
        if cc_emails:
            recipients.extend(cc_emails)

        # Try to use SMTP with TLS (for Outlook/Office 365)
        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=30) as server:
            logger.debug(f"Connected to SMTP server {settings.SMTP_HOST}:{settings.SMTP_PORT}")
            
            # Try STARTTLS for encrypted connection if on port 587
            try:
                if settings.SMTP_PORT == 587:
                    server.starttls()
                    logger.debug("STARTTLS connection established")
            except Exception as e:
                logger.warning(f"STARTTLS not available: {str(e)}")
            
            # Try to authenticate if credentials are provided
            try:
                if settings.SMTP_USER and settings.SMTP_PASSWORD:
                    logger.debug(f"Attempting authentication as {settings.SMTP_USER}")
                    server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
                    logger.info(f"Successfully authenticated as {settings.SMTP_USER}")
                else:
                    logger.warning("No SMTP credentials provided, attempting unauthenticated send")
            except smtplib.SMTPAuthenticationError as e:
                logger.error(f"SMTP Authentication failed: {str(e)}")
                raise
            except smtplib.SMTPNotSupportedError:
                logger.debug("SMTP AUTH extension not supported by server, continuing without authentication")
            
            # Send email
            server.sendmail(
                settings.SMTP_FROM,
                recipients,
                msg.as_string(),
            )
            logger.info(f"Email sent successfully to {to_email}")
        
    except Exception as e:
        logger.error(f"Failed to send email to {to_email}: {type(e).__name__} - {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Log but don't re-raise - we want the request to succeed even if email fails
        return False
    
    return True
