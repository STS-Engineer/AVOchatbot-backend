"""
Configuration settings for the Backend API.
"""

from pydantic_settings import BaseSettings
from typing import List
import os
from pathlib import Path
from dotenv import load_dotenv

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
    
    # Groq LLM (required from environment)
    GROQ_API_KEY: str
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 2048
    LLM_MODEL: str = "openai/gpt-oss-120b"
    
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
        """Build database connection URL."""
        from urllib.parse import quote
        password = quote(self.DB_PASSWORD, safe='')
        return f"postgresql://{self.DB_USER}:{password}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"


# Create settings instance
settings = Settings()

# Ensure logs directory exists
os.makedirs(os.path.dirname(settings.LOG_FILE), exist_ok=True)
