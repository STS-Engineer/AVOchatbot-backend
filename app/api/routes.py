
"""
Main API routes for the chatbot backend.
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import FileResponse
from loguru import logger
from pathlib import Path
from typing import Optional
from app.models.schemas import (
    ChatRequest, ChatResponse, HistoryRequest, HistoryResponse,
    SearchRequest, SearchResponse, HistoryMessage, EditMessageRequest,
    AssistantHelpRequest, AssistantHelpResponse
)
from app.services.chat import get_chat_service
from app.services.rag import get_rag_service
from app.middleware.auth import get_current_user
from datetime import datetime

router = APIRouter(prefix="/api", tags=["chat"])

# --- Assistant Help Q/A endpoint with escalation ---
@router.post("/assistant-help", response_model=AssistantHelpResponse, summary="Assistant help Q/A with escalation")
async def assistant_help(
    request: AssistantHelpRequest,
    current_user: dict = Depends(get_current_user)
) -> AssistantHelpResponse:
    """
    Assistant answers user help/complaint. If escalation is needed, an email is sent to the manager.
    """
    # Use LLM to answer the question and decide escalation
    from app.services.llm import get_llm_service
    llm = get_llm_service()
    escalation_prompt = f"""
You are a helpful assistant. Answer the user's message below. If the issue requires manager/admin intervention (e.g., cannot be solved by assistant, is a serious complaint, or needs human action), respond with:

ESCALATE: <reason for escalation>
<Your answer to the user>

Otherwise, just answer the user directly.

User message: {request.message}
"""
    llm_response = llm.generate_response(escalation_prompt)
    # Check if user explicitly requests manager/admin help
    user_message_lower = request.message.lower()
    force_escalate_keywords = [
        "tell the manager", "contact the manager", "admin's help", "need admin", "need manager", "escalate", "manager help", "admin help"
    ]
    force_escalate = any(keyword in user_message_lower for keyword in force_escalate_keywords)

    if llm_response.strip().upper().startswith("ESCALATE:") or force_escalate:
        # Extract escalation reason and answer
        if llm_response.strip().upper().startswith("ESCALATE:"):
            lines = llm_response.split("\n", 2)
            escalation_reason = lines[0][9:].strip() if len(lines) > 0 else "Escalation required"
            answer = lines[1].strip() if len(lines) > 1 else ""
        else:
            escalation_reason = "User explicitly requested manager/admin intervention."
            answer = "Your request has been escalated to the manager as you requested."
        # Compose a professional recap message for the manager
        recap_message = f"""
        <html><body>
        <h3>Escalation Recap</h3>
        <p><b>User:</b> {current_user.get('email','user')}</p>
        <p><b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><b>Situation:</b> The user requested assistance and the issue requires manager/admin intervention.</p>
        <p><b>User Message:</b> {request.message}</p>
        <p><b>Reason for Escalation:</b> {escalation_reason}</p>
        </body></html>
        """
        manager_email = "rihem.arfaoui@avocarbon.com"
        subject = f"[Assistant Escalation] {escalation_reason}"
        import mailer
        mailer.send_email(to_email=manager_email, subject=subject, html_body=recap_message)
        return AssistantHelpResponse(success=True, answer=answer, escalated=True, escalation_message="Your request was escalated to the manager.")
    else:
        return AssistantHelpResponse(success=True, answer=llm_response, escalated=False, escalation_message="")

# Get service instances
chat_service = None
rag_service = None


def init_services():
    """Initialize services (called on startup)."""
    global chat_service, rag_service
    chat_service = get_chat_service()
    rag_service = get_rag_service()



# Complaint/Assistance endpoint
from app.models.schemas import ComplaintRequest, ComplaintResponse
from app.core.config import settings
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
import mailer

@router.post("/complaint", response_model=ComplaintResponse, summary="Submit a complaint or assistance request")
async def submit_complaint(
    request: ComplaintRequest,
    current_user: dict = Depends(get_current_user)
) -> ComplaintResponse:
    """
    Submit a complaint or assistance request. If escalation is needed, an email is sent to the manager.
    """
    # Compose email to admin/manager
    manager_email = "rihem.arfaoui@avocarbon.com"  # Updated to real admin/manager email
    subject = f"[Chatbot Assistance/Complaint] {request.subject}"
    html_body = f"""
    <html><body>
    <h3>New Complaint/Assistance Request from {current_user.get('email','user')}</h3>
    <p><b>Subject:</b> {request.subject}</p>
    <p><b>Message:</b><br>{request.message}</p>
    </body></html>
    """
    # Send email using mailer.py
    email_sent = mailer.send_email(
        to_email=manager_email,
        subject=subject,
        html_body=html_body
    )
    if email_sent:
        return ComplaintResponse(success=True, message="Your complaint/request was received. The manager has been notified.", escalated=True)
    else:
        return ComplaintResponse(success=False, message="Failed to notify manager by email, but your complaint was received.", escalated=False)


@router.post("/chat", response_model=ChatResponse, summary="Send a chat message")
async def chat(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user)
) -> ChatResponse:
    """
    Send a message to the chatbot and receive an AI-generated response with knowledge base context.
    
    **Authentication Required:** Bearer token in Authorization header
    
    **Request Body:**
    - `message`: The user's question or query (required, 1-5000 characters)
    - `include_context`: Whether to include retrieved context in response (default: true)
    - `top_k`: Number of context items to retrieve (default: 8, min: 1, max: 20)
    - `conversation_id`: Optional conversation ID (creates new conversation if not provided)
    
    **Response:**
    - `success`: Whether the request was successful
    - `message`: AI-generated response
    - `context`: Formatted knowledge base context
    - `context_items`: Detailed context items retrieved
    - `context_count`: Number of context items
    - `conversation_id`: Current conversation ID
    - `timestamp`: Response timestamp
    """
    try:
        if not chat_service:
            init_services()
        
        user_id = str(current_user["id"])
        logger.info(f"Chat request from user {user_id}: {request.message[:100]}")
        
        result = chat_service.process_message(
            message=request.message,
            user_id=user_id,
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
async def edit_message(
    request: EditMessageRequest,
    current_user: dict = Depends(get_current_user)
) -> ChatResponse:
    """
    Edit a prior user message and regenerate the assistant response.

    **Authentication Required:** Bearer token in Authorization header

    **Request Body:**
    - `message`: Updated user message
    - `message_index`: Zero-based index of the user message in history
    - `include_context`: Whether to include retrieved context in response
    - `top_k`: Number of context items to retrieve
    - `conversation_id`: Conversation identifier (required)
    """
    try:
        if not chat_service:
            init_services()

        user_id = str(current_user["id"])
        
        if not request.conversation_id:
            raise HTTPException(status_code=400, detail="conversation_id is required for editing messages")

        result = chat_service.edit_message_and_respond(
            message_index=request.message_index,
            new_message=request.message,
            user_id=user_id,
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
    conversation_id: str = Query(..., description="Conversation identifier"),
    limit: int = Query(50, ge=1, le=200, description="Number of messages to retrieve"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    current_user: dict = Depends(get_current_user)
) -> HistoryResponse:
    """
    Retrieve the conversation history with pagination support.
    
    **Authentication Required:** Bearer token in Authorization header
    
    **Query Parameters:**
    - `conversation_id`: Conversation ID (required)
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
        
        user_id = str(current_user["id"])
        
        messages = chat_service.get_history(
            user_id=user_id,
            conversation_id=conversation_id,
            limit=limit,
            offset=offset
        )
        
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
            total=len(history_messages),
            timestamp=datetime.now()
        )
    
    except Exception as e:
        logger.error(f"Error retrieving history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/conversation/{conversation_id}", summary="Delete conversation")
async def delete_conversation(
    conversation_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Delete a conversation and all its messages.
    
    **Authentication Required:** Bearer token in Authorization header
    
    **Path Parameters:**
    - `conversation_id`: Conversation ID to delete
    """
    try:
        if not chat_service:
            init_services()
        
        user_id = str(current_user["id"])
        success = chat_service.clear_history(user_id=user_id, conversation_id=conversation_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Conversation not found or access denied")
        
        return {
            "success": success,
            "message": "Conversation deleted successfully",
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation: {str(e)}")
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

