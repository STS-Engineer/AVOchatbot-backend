#!/usr/bin/env python
"""
Backend API development server runner.
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

if __name__ == "__main__":
    import uvicorn
    from pathlib import Path
    
    # Get the path for uvicorn to import the app
    PROJECT_ROOT = Path(__file__).parent
    
    print("=" * 60)
    print("Starting Chatbot Backend API")
    print("=" * 60)
    
    # Import settings to show info
    sys.path.insert(0, str(PROJECT_ROOT))
    from app.core.config import settings
    
    print(f"Server:     {settings.SERVER_HOST}:{settings.SERVER_PORT}")
    print(f"Environment: {settings.ENVIRONMENT}")
    print(f"Database:   {settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}")
    print(f"Docs URL:   http://{settings.SERVER_HOST}:{settings.SERVER_PORT}/api/docs")
    print("=" * 60)
    
    # Run with proper import string for reload to work
    uvicorn.run(
        "app.main:app",
        host=settings.SERVER_HOST,
        port=settings.SERVER_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
