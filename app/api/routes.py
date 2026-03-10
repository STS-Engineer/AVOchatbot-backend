
"""
Main API routes for the chatbot backend.
"""

from fastapi import APIRouter, HTTPException, Query, Depends, File, UploadFile, BackgroundTasks
from fastapi import FastAPI
from fastapi.responses import FileResponse
from loguru import logger
from pathlib import Path
from typing import Optional
from html import escape
import json
import re
from app.models.schemas import (
    ChatRequest, ChatResponse, HistoryRequest, HistoryResponse,
    SearchRequest, SearchResponse, HistoryMessage, EditMessageRequest,
    AssistantHelpRequest, AssistantHelpResponse
)
from app.services.chat import get_chat_service
from app.services.rag import get_rag_service
from app.services.file_analysis import get_file_analysis_service
from app.middleware.auth import get_current_user, get_current_user_optional
from datetime import datetime

router = APIRouter(prefix="/api", tags=["chat"])
# --- Periodic User Sync Background Task ---
import threading
import time
from app.services.user import get_user_service

# --- Manual User Sync Endpoint ---
@router.post("/sync-users", summary="Manually sync users from central DB to local DB")
async def manual_sync_users(current_user: dict = Depends(get_current_user)):
    """
    Manually trigger user sync from central users DB to local knowledge_base DB.
    Requires authentication.
    """
    user_service = get_user_service()
    try:
        user_service.sync_users_to_local_db()
        logger.info("Manual user sync triggered by: %s", current_user.get("email", "unknown"))
        return {"success": True, "message": "Users synced successfully."}
    except Exception as e:
        logger.error(f"Manual user sync failed: {e}")
        return {"success": False, "message": str(e)}

def periodic_user_sync(interval_seconds=2):
    def sync_loop():
        user_service = get_user_service()
        while True:
            try:
                user_service.sync_users_to_local_db()
            except Exception as e:
                logger.error(f"Periodic user sync failed: {e}")
            time.sleep(interval_seconds)
    t = threading.Thread(target=sync_loop, daemon=True)
    t.start()

# Start the background sync when the app starts
def start_periodic_sync(app: FastAPI):
    @app.on_event("startup")
    async def _start_sync():
        periodic_user_sync()

# --- Image/File Upload Endpoint ---
@router.post("/upload", summary="Upload an image or file")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload an image or document, then attempt text extraction and LLM analysis.
    """
    from pathlib import Path
    import shutil
    import uuid

    uploads_dir = Path(__file__).parent.parent.parent / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(file.filename or "upload.bin").name
    stored_name = f"{uuid.uuid4().hex[:8]}_{safe_name}"
    file_path = uploads_dir / stored_name

    # Save file
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    analyzer = get_file_analysis_service()
    analysis_result = analyzer.analyze_file(file_path)

    return {
        "success": True,
        "file_path": stored_name,
        "original_name": safe_name,
        "url": f"/uploads/{stored_name}",
        "readable": analysis_result.get("readable", False),
        "analysis": analysis_result.get("analysis"),
        "analysis_error": analysis_result.get("error"),
        "extracted_chars": analysis_result.get("extracted_chars", 0),
        "preview": analysis_result.get("preview"),
    }


def _truncate_for_subject(text: str, max_length: int = 70) -> str:
    """Keep subject lines concise and readable."""
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return "Escalation Required"
    return cleaned if len(cleaned) <= max_length else f"{cleaned[:max_length - 1].rstrip()}…"


def _format_professional_issue(text: str) -> str:
    """Format user issue text into a professional single sentence."""
    cleaned = " ".join((text or "").split()).strip()
    if not cleaned:
        return "No additional details were provided by the user."
    cleaned = cleaned[0].upper() + cleaned[1:] if len(cleaned) > 1 else cleaned.upper()
    if cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned

# --- Assistant Help Q/A endpoint with escalation ---
@router.post("/assistant-help", summary="Assistant help Q/A with escalation", response_model=AssistantHelpResponse)
async def assistant_help(
    request: AssistantHelpRequest,
    background_tasks: BackgroundTasks,
    current_user: Optional[dict] = Depends(get_current_user_optional)
) -> AssistantHelpResponse:
    """
    Assistant answers user help/complaint. If escalation is needed, an email is sent to the manager.
    """
    reporter_email = (current_user or {}).get("email", "anonymous-user")
    logger.info(f"[assistant_help] Endpoint called. Payload: {{'message': request.message, 'user': {reporter_email}}}")
    # Use LLM to answer the question and decide escalation
    from app.services.llm import get_llm_service
    llm = get_llm_service()
    escalation_prompt = f"""
You are a professional support assistant.
Decide whether the user's request must be escalated to the technical/support team.

Rules:
1) If the user reports a technical issue, bug, service disruption, malfunction, or explicitly asks to contact/report to a human team, set "escalate" to true.
2) If escalation is not needed, set "escalate" to false.
3) Write a very short, clear end-user reply in the same language as the user message.
4) Return ONLY valid JSON (no markdown, no code fences, no extra text) with exactly these keys:
   {{"answer": "<short reply>", "escalate": true|false}}

User message: {request.message}
"""
    llm_response_raw = llm.generate_response(escalation_prompt)
    logger.info(f"[assistant_help] LLM raw response: {llm_response_raw}")

    parsed_response = None
    try:
        parsed_response = json.loads(llm_response_raw)
    except Exception:
        json_match = re.search(r"\{[\s\S]*\}", llm_response_raw or "")
        if json_match:
            try:
                parsed_response = json.loads(json_match.group(0))
            except Exception:
                parsed_response = None

    escalate = False
    assistant_answer = ""

    if isinstance(parsed_response, dict):
        escalate = bool(parsed_response.get("escalate", False))
        assistant_answer = str(parsed_response.get("answer", "")).strip()

    if not assistant_answer:
        assistant_answer = (llm_response_raw or "").strip()

    if not isinstance(parsed_response, dict):
        logger.warning("[assistant_help] LLM response was not valid JSON. Falling back to keyword-based escalation detection.")
        llm_response_lower = assistant_answer.lower()
        if (
            "escalate_now" in llm_response_lower
            or "has been escalated" in llm_response_lower
            or "notified" in llm_response_lower
            or "escalated to the technical team" in llm_response_lower
            or "escalated" in llm_response_lower
            or "escalating" in llm_response_lower
            or "issue reported" in llm_response_lower
            or "forwarded" in llm_response_lower
            or "team member" in llm_response_lower
            or "arrange a call" in llm_response_lower
        ):
            escalate = True

    if not assistant_answer:
        assistant_answer = "Your request has been escalated." if escalate else "How can I help you?"
    logger.info(f"[assistant_help] Escalation decision: escalate={escalate} for user: {reporter_email}")
    if escalate:
        logger.info(f"[assistant_help] Entered escalation block for user: {reporter_email}")
        escaped_user = escape(reporter_email)
        # Fetch recent conversation history (stub: empty for now)
        conversation_history = []
        # Example: conversation_history = chat_service.get_history(user_id=current_user["id"], limit=5)
        llm_analysis_prompt = f"""
    Summarize the user's technical issue for the escalation report. Write a short, clear paragraph using simple, explainable phrases. Avoid tables, bullet points, or lists. Focus on the main problem, user observation, and impact. Do not include troubleshooting steps. Make it easy for anyone to understand.
    User message:
    {request.message}
    """
        llm_analysis = llm.generate_response(llm_analysis_prompt)
        manager_email = "rihem.arfaoui@avocarbon.com"
        recap_message = f"""
        <html><body style='font-family:Segoe UI,Arial,sans-serif;'>
        <h2 style='color:#1a73e8;margin-bottom:16px;'>🚨 Knowledge-Base Chatbot Incident Report</h2>
        <div style='background:#fff;border-radius:8px;box-shadow:0 2px 8px #eee;padding:24px;'>
            <div style='font-size:16px;background:#fafafa;padding:18px;border-radius:8px;margin-bottom:24px;'>
                <strong>Reported By:</strong> {escaped_user}<br>
                <strong>Reported At:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
                <strong>Summary:</strong>
                <div style='background:#ffe0e0;border-radius:6px;padding:12px;font-size:17px;color:#b71c1c;font-weight:bold;margin-top:8px;'>{escape(llm_analysis)}</div>
                <strong>User Problem Details:</strong> {escape(request.message)}
            </div>
            <div style='margin-top:18px;'>
                <a href='mailto:{manager_email}?subject=Incident%20Acknowledged' style='display:inline-block;background:#1a73e8;color:#fff;padding:10px 22px;border-radius:6px;text-decoration:none;font-weight:bold;margin-right:12px;'>Acknowledge</a>
                <a href='mailto:{escaped_user}?subject=Follow-up%20on%20Chatbot%20Issue' style='display:inline-block;background:#43a047;color:#fff;padding:10px 22px;border-radius:6px;text-decoration:none;font-weight:bold;'>Reply to User</a>
            </div>
        </div>
        <p style='color:#888;font-size:14px;margin-top:18px;'>This is a knowledge-base Chatbot issue. Please investigate promptly.</p>
        </body></html>
        """
        subject = f"Technical Issue Reported by {escaped_user} (Knowledge-Base Chatbot)"
        escalation_cc_emails = ["taha.khiari@avocarbon.com"]
        import logging
        logging.info(f"[assistant_help] Scheduling escalation email to {manager_email} with subject '{subject}'")
        def log_and_send_email(*args, **kwargs):
            result = mailer.send_email(*args, **kwargs)
            logging.info(f"[assistant_help] mailer.send_email returned: {result}")
            return result
        background_tasks.add_task(
            log_and_send_email,
            to_email=manager_email,
            subject=subject,
            html_body=recap_message,
            cc_emails=escalation_cc_emails,
        )
        return AssistantHelpResponse(success=True, answer=assistant_answer, escalated=True, escalation_message="Your request was escalated to the technical team.")
    return AssistantHelpResponse(success=True, answer=assistant_answer, escalated=False, escalation_message="")

# Get service instances
chat_service = None
rag_service = None


def init_services():
    """Initialize services (called on startup)."""
    global chat_service, rag_service
    chat_service = get_chat_service()
    rag_service = get_rag_service()



from app.models.schemas import ComplaintRequest, ComplaintResponse
from app.core.config import settings
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from app.utils import mailer

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
            uploaded_files=request.uploaded_files,
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
