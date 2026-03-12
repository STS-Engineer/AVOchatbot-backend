"""
Configuration settings for the Backend API.
"""

from pydantic_settings import BaseSettings
from typing import List
import os
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file in project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
else:
    # Try parent directory
    PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
    ENV_FILE = PROJECT_ROOT / ".env"
    if ENV_FILE.exists():
        load_dotenv(ENV_FILE)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Server
    SERVER_HOST: str = "0.0.0.0"
    SERVER_PORT: int = 8000
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    
    # Database (required from environment)
    DB_HOST: str
    DB_PORT: int
    DB_NAME: str
    DB_USER: str
    DB_PASSWORD: str
    DB_SSLMODE: str

    # Central users database (required from environment)
    USERS_DB_HOST: str
    USERS_DB_PORT: int
    USERS_DB_NAME: str
    USERS_DB_USER: str
    USERS_DB_PASSWORD: str
    USERS_DB_SSLMODE: str
    
    # Groq LLM (required from environment)
    LLM_PROVIDER: str = "groq"  # groq | openai
    GROQ_API_KEY: str
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 2048
    LLM_MODEL: str = "openai/gpt-oss-120b"

    # OpenAI chat model (used when LLM_PROVIDER=openai)
    OPENAI_LLM_MODEL: str = "gpt-4o-mini"
    
    # OpenAI (Embeddings) (required from environment)
    OPENAI_API_KEY: str
    EMBEDDING_MODEL: str = "text-embedding-3-large"
    EMBEDDING_DIMENSION: int = 1536
    
    # RAG
    TOP_K_RESULTS: int = 8
    SIMILARITY_THRESHOLD: float = 0.4  # Raised from 0.2 to reduce false positives
    
    # Authentication / JWT
    JWT_SECRET_KEY: str = "your-secret-key-change-in-production"  # Should be in .env
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30  # 30 minutes
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7  # 7 days
    
    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "https://knowledge-chat.azurewebsites.net"
    ]
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/backend.log"

    # Uploads
    UPLOADS_DIR: str = ""

    # SMTP (for assistant/complaint email feature)
    SMTP_HOST: str = "smtp.avocarbon.com"  # Default, can be overridden in .env
    SMTP_PORT: int = 25
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""
    SMTP_FROM: str = ""
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
    

    @property
    def database_url(self) -> str:
        """Build knowledge base database connection URL."""
        from urllib.parse import quote
        password = quote(self.DB_PASSWORD, safe='')
        return f"postgresql://{self.DB_USER}:{password}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    @property
    def users_database_url(self) -> str:
        """Build central users database connection URL."""
        from urllib.parse import quote
        password = quote(self.USERS_DB_PASSWORD, safe='')
        return f"postgresql://{self.USERS_DB_USER}:{password}@{self.USERS_DB_HOST}:{self.USERS_DB_PORT}/{self.USERS_DB_NAME}"

    @property
    def uploads_dir_path(self) -> Path:
        """Resolve the uploads directory, preferring persistent Azure App Service storage."""
        if self.UPLOADS_DIR:
            uploads_dir = Path(self.UPLOADS_DIR)
            source = "UPLOADS_DIR env var"
        else:
            azure_home = os.getenv("HOME")
            is_azure_app_service = bool(os.getenv("WEBSITE_SITE_NAME") or os.getenv("WEBSITE_INSTANCE_ID"))

            if azure_home and is_azure_app_service:
                uploads_dir = Path(azure_home) / "data" / "uploads"
                source = f"Azure persistent storage: HOME={azure_home}"
            else:
                uploads_dir = PROJECT_ROOT / "uploads"
                source = f"Local project folder (not Azure or HOME not set): HOME={azure_home}"

        uploads_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Uploads directory resolved: {uploads_dir} (source: {source})")
        return uploads_dir


# Create settings instance
settings = Settings()

# Ensure logs directory exists
os.makedirs(os.path.dirname(settings.LOG_FILE), exist_ok=True)
