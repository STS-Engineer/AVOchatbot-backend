"""
Pydantic models for API requests and responses.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


# Request Models
class ChatRequest(BaseModel):
    """Chat message request model."""
    message: str = Field(..., min_length=1, max_length=5000, description="User's query")
    include_context: bool = Field(default=True, description="Include retrieved context in response")
    top_k: Optional[int] = Field(default=8, ge=1, le=20, description="Number of context items to retrieve")
    conversation_id: Optional[str] = Field(default=None, description="Conversation identifier")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "What is the process for handling late payment requests?",
                "include_context": True,
                "top_k": 8
            }
        }


class HistoryRequest(BaseModel):
    """Request to get conversation history."""
    limit: Optional[int] = Field(default=50, ge=1, le=200, description="Number of messages to retrieve")
    offset: Optional[int] = Field(default=0, ge=0, description="Offset for pagination")
    conversation_id: Optional[str] = Field(default=None, description="Conversation identifier")


class SearchRequest(BaseModel):
    """Knowledge base search request."""
    query: str = Field(..., min_length=1, max_length=5000, description="Search query")
    top_k: Optional[int] = Field(default=5, ge=1, le=20, description="Number of results")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "payment management",
                "top_k": 5
            }
        }


    class EditMessageRequest(BaseModel):
        """Edit a user message and regenerate the assistant response."""
        message: str = Field(..., min_length=1, max_length=5000, description="Updated user query")
        message_index: int = Field(..., ge=0, description="Zero-based index of the user message in history")
        include_context: bool = Field(default=True, description="Include retrieved context in response")
        top_k: Optional[int] = Field(default=8, ge=1, le=20, description="Number of context items to retrieve")
        conversation_id: Optional[str] = Field(default=None, description="Conversation identifier")


# Response Models
class Attachment(BaseModel):
    """File attachment in knowledge base."""
    id: str
    file_name: str
    file_type: str
    file_path: str
    uploaded_at: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "1e006fc9-2860-4790-92bd-798c754c87f8",
                "file_name": "trust.png",
                "file_type": "image/png",
                "file_path": "uploads/6e6eb4c8_1770113882_trust.png"
            }
        }


class ContextItem(BaseModel):
    """A single context item from knowledge base."""
    id: str
    title: str
    node_type: str
    content: Optional[str] = None
    similarity: Optional[float] = None
    parent_id: Optional[str] = None
    attachments: Optional[List[Attachment]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "04d8ee21-775e-4a78-a165-00cc8da0caf7",
                "title": "Communication client en situation de retard de paiement",
                "node_type": "instruction",
                "similarity": 0.89,
                "parent_id": "078e76c7-7738-4d81-aa20-c037b24c2011",
                "attachments": []
            }
        }


class ChatResponse(BaseModel):
    """Chat response model."""
    success: bool = Field(..., description="Whether the request was successful")
    message: Optional[str] = Field(None, description="AI-generated response")
    context: Optional[str] = Field(None, description="Formatted knowledge base context")
    context_items: Optional[List[ContextItem]] = Field(None, description="Retrieved context items")
    context_count: Optional[int] = Field(None, description="Number of context items retrieved")
    error: Optional[str] = Field(None, description="Error message if unsuccessful")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Based on the knowledge base...",
                "context": "Formatted context here",
                "context_items": [],
                "context_count": 3,
                "timestamp": "2026-02-05T10:30:00"
            }
        }


class SearchResponse(BaseModel):
    """Search response model."""
    success: bool
    results: List[ContextItem]
    count: int
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "results": [],
                "count": 0,
                "timestamp": "2026-02-05T10:30:00"
            }
        }


class HistoryMessage(BaseModel):
    """A single message in conversation history."""
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(..., description="Message timestamp")
    context_count: Optional[int] = Field(None, description="Number of context items used")


class HistoryResponse(BaseModel):
    """Conversation history response."""
    success: bool
    messages: List[HistoryMessage]
    total: int
    timestamp: datetime = Field(default_factory=datetime.now)


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    database_connected: bool = Field(..., description="Database connection status")
    llm_configured: bool = Field(..., description="LLM configuration status")
    timestamp: datetime = Field(default_factory=datetime.now)
