"""
Main FastAPI application.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from loguru import logger
from pathlib import Path
import sys

from app.core.config import settings
from app.core.logging import setup_logging
from app.api.routes import router as chat_router, init_services, start_periodic_sync
from app.api.health import router as health_router
from app.api.auth_routes import router as auth_router
from app.api.otp_routes import router as otp_router

# Setup logging
logger.remove()
logger_ = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application startup and shutdown.
    """
    # Startup
    logger.info("=" * 50)
    logger.info("Starting Chatbot Backend API")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Debug: {settings.DEBUG}")
    logger.info(f"Database: {settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}")
    logger.info("=" * 50)
    
    init_services()
    
    yield
    
    # Shutdown
    logger.info("=" * 50)
    logger.info("Shutting down Chatbot Backend API")
    logger.info("=" * 50)


# Create FastAPI app
app = FastAPI(
    title="Chatbot Backend API",
    description="RAG-based chatbot API with knowledge base integration",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan
)

start_periodic_sync(app)
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router)
app.include_router(otp_router)
app.include_router(chat_router)
app.include_router(health_router)

# Mount uploads directory for static file serving
# Check for uploads directory in project root
uploads_path = Path(__file__).parent.parent / "uploads"
if not uploads_path.exists():
    uploads_path = Path(__file__).parent.parent.parent / "rag_project" / "uploads"

if uploads_path.exists():
    app.mount("/uploads", StaticFiles(directory=str(uploads_path)), name="uploads")
    logger.info(f"Mounted uploads directory: {uploads_path}")
else:
    logger.warning(f"Uploads directory not found: {uploads_path}")
@app.get("/", tags=["root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to Chatbot Backend API",
        "docs": "/api/docs",
        "version": "1.0.0"
    }


# Startup event
@app.on_event("startup")
async def startup_event():
    """Handle startup events."""
    logger.info("API startup event triggered")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Handle shutdown events."""
    logger.info("API shutdown event triggered")


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting server on {settings.SERVER_HOST}:{settings.SERVER_PORT}")
    
    uvicorn.run(
        "app.main:app",
        host=settings.SERVER_HOST,
        port=settings.SERVER_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
