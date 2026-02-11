"""
Main API routes for the chatbot backend.
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from loguru import logger
from pathlib import Path
from typing import Optional
from app.models.schemas import (
    ChatRequest, ChatResponse, HistoryRequest, HistoryResponse,
    SearchRequest, SearchResponse, HistoryMessage, EditMessageRequest
)
from app.services.chat import get_chat_service
from app.services.rag import get_rag_service
from datetime import datetime

router = APIRouter(prefix="/api", tags=["chat"])

# Get service instances
chat_service = None
rag_service = None


def init_services():
    """Initialize services (called on startup)."""
    global chat_service, rag_service
    chat_service = get_chat_service()
    rag_service = get_rag_service()


@router.post("/chat", response_model=ChatResponse, summary="Send a chat message")
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Send a message to the chatbot and receive an AI-generated response with knowledge base context.
    
    **Request Body:**
    - `message`: The user's question or query (required, 1-5000 characters)
    - `include_context`: Whether to include retrieved context in response (default: true)
    - `top_k`: Number of context items to retrieve (default: 8, min: 1, max: 20)
    
    **Response:**
    - `success`: Whether the request was successful
    - `message`: AI-generated response
    - `context`: Formatted knowledge base context
    - `context_items`: Detailed context items retrieved
    - `context_count`: Number of context items
    - `timestamp`: Response timestamp
    """
    try:
        if not chat_service:
            init_services()
        
        logger.info(f"Chat request: {request.message[:100]}")
        
        result = chat_service.process_message(
            message=request.message,
            top_k=request.top_k,
            include_context=request.include_context,
            conversation_id=request.conversation_id,
        )
        
        if result["success"]:
            return ChatResponse(**result)
        else:
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to process message"))
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/edit-message", response_model=ChatResponse, summary="Edit a message and regenerate response")
async def edit_message(request: EditMessageRequest) -> ChatResponse:
    """
    Edit a prior user message and regenerate the assistant response.

    **Request Body:**
    - `message`: Updated user message
    - `message_index`: Zero-based index of the user message in history
    - `include_context`: Whether to include retrieved context in response
    - `top_k`: Number of context items to retrieve
    - `conversation_id`: Conversation identifier
    """
    try:
        if not chat_service:
            init_services()

        result = chat_service.edit_message_and_respond(
            message_index=request.message_index,
            new_message=request.message,
            top_k=request.top_k,
            include_context=request.include_context,
            conversation_id=request.conversation_id,
        )

        if result["success"]:
            return ChatResponse(**result)
        else:
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to edit message"))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in edit-message endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history", response_model=HistoryResponse, summary="Get conversation history")
async def get_history(
    limit: int = Query(50, ge=1, le=200, description="Number of messages to retrieve"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    conversation_id: Optional[str] = Query(None, description="Conversation identifier")
) -> HistoryResponse:
    """
    Retrieve the conversation history with pagination support.
    
    **Query Parameters:**
    - `limit`: Maximum number of messages (default: 50, min: 1, max: 200)
    - `offset`: Number of messages to skip (default: 0)
    
    **Response:**
    - `success`: Whether the request was successful
    - `messages`: List of messages with role, content, and timestamp
    - `total`: Total number of messages in history
    - `timestamp`: Response timestamp
    """
    try:
        if not chat_service:
            init_services()
        
        messages = chat_service.get_history(limit=limit, offset=offset, conversation_id=conversation_id)
        
        # Convert to HistoryMessage objects
        history_messages = [
            HistoryMessage(
                role=msg["role"],
                content=msg["content"],
                timestamp=msg["timestamp"] if isinstance(msg["timestamp"], datetime) else datetime.fromisoformat(msg["timestamp"]),
                context_count=msg.get("context_count")
            )
            for msg in messages
        ]
        
        return HistoryResponse(
            success=True,
            messages=history_messages,
            total=chat_service.get_history_count(conversation_id=conversation_id),
            timestamp=datetime.now()
        )
    
    except Exception as e:
        logger.error(f"Error in history endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear-history", summary="Clear conversation history")
async def clear_history(
    conversation_id: Optional[str] = Query(None, description="Conversation identifier")
):
    """Clear conversation history."""
    try:
        if not chat_service:
            init_services()
        
        success = chat_service.clear_history(conversation_id=conversation_id)
        
        return {
            "success": success,
            "message": "Conversation history cleared" if success else "Failed to clear history",
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error clearing history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", response_model=SearchResponse, summary="Search knowledge base")
async def search(request: SearchRequest) -> SearchResponse:
    """
    Search the knowledge base directly without generating an AI response.
    
    **Request Body:**
    - `query`: Search query (required, 1-5000 characters)
    - `top_k`: Number of results to return (default: 5, min: 1, max: 20)
    
    **Response:**
    - `success`: Whether the request was successful
    - `results`: List of matching knowledge base items
    - `count`: Number of results
    - `timestamp`: Response timestamp
    """
    try:
        if not rag_service:
            init_services()
        
        logger.info(f"Search request: {request.query}")
        
        _, context_items = rag_service.retrieve_context(request.query, k=request.top_k)
        
        return SearchResponse(
            success=True,
            results=context_items,
            count=len(context_items),
            timestamp=datetime.now()
        )
    
    except Exception as e:
        logger.error(f"Error in search endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download/{file_path:path}", summary="Download attachment file")
async def download_file(file_path: str):
    """
    Download an attachment file from the uploads directory.
    
    **Parameters:**
    - `file_path`: The filename or path within uploads directory (e.g., 'filename.pdf')
    
    **Response:**
    - File download with proper MIME type and Content-Disposition headers
    """
    try:
        # Security: prevent directory traversal attacks
        if ".." in file_path or file_path.startswith("/"):
            raise HTTPException(status_code=400, detail="Invalid file path")
        
        # Build the full file path
        uploads_dir = Path(__file__).parent.parent.parent / "uploads"
        full_path = uploads_dir / file_path
        
        # Verify the file exists and is within uploads directory
        full_path = full_path.resolve()
        if not full_path.exists() or not str(full_path).startswith(str(uploads_dir.resolve())):
            logger.warning(f"File not found or access denied: {file_path}")
            raise HTTPException(status_code=404, detail="File not found")
        
        logger.info(f"Downloading file: {file_path}")
        
        # Extract filename for Content-Disposition header
        filename = full_path.name
        
        # Determine MIME type based on file extension
        mime_type = "application/octet-stream"  # default
        suffix = full_path.suffix.lower()
        
        mime_types = {
            ".pdf": "application/pdf",
            ".doc": "application/msword",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".xls": "application/vnd.ms-excel",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".txt": "text/plain",
            ".csv": "text/csv",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".zip": "application/zip",
            ".rar": "application/x-rar-compressed",
        }
        
        mime_type = mime_types.get(suffix, "application/octet-stream")
        
        # Return file with proper headers for download
        return FileResponse(
            path=full_path,
            media_type=mime_type,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

