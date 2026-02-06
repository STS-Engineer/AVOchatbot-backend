# Backend & Frontend Integration Guide

## Overview

The backend provides a REST API for a RAG (Retrieval Augmented Generation) based chatbot. The frontend communicates with this API to send messages, retrieve conversation history, and search the knowledge base.

## API Specification

### Base URL
```
http://localhost:8000
```

### Health & Configuration

#### GET `/health`
Check API health and dependency status.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "database_connected": true,
  "llm_configured": true,
  "timestamp": "2026-02-05T12:34:56.789Z"
}
```

#### GET `/config`
Get API configuration details.

---

### Chat & Conversation

#### POST `/api/chat`
Send a message to the chatbot and receive an AI-generated response with knowledge base context.

**Request:**
```json
{
  "message": "What is the process for handling late payment requests?",
  "include_context": true,
  "top_k": 8
}
```

**Parameters:**
- `message` (string, required): User's question/query (1-5000 characters)
- `include_context` (boolean, optional): Include retrieved context in response (default: true)
- `top_k` (integer, optional): Number of context items to retrieve (1-20, default: 8)

**Response:**
```json
{
  "success": true,
  "message": "AI-generated response with detailed information...",
  "context": "Formatted context from knowledge base...",
  "context_items": [
    {
      "id": "04d8ee21-775e-4a78-a165-00cc8da0caf7",
      "title": "Communication client en situation de retard de paiement",
      "node_type": "instruction",
      "content": "Instruction content...",
      "similarity": 0.89,
      "attachments": []
    }
  ],
  "context_count": 3,
  "timestamp": "2026-02-05T12:34:56.789Z"
}
```

#### GET `/api/history`
Retrieve conversation history with pagination support.

**Query Parameters:**
- `limit` (integer, optional): Maximum messages to retrieve (1-200, default: 50)
- `offset` (integer, optional): Number of messages to skip for pagination (default: 0)

**Response:**
```json
{
  "success": true,
  "messages": [
    {
      "role": "user",
      "content": "User's message",
      "timestamp": "2026-02-05T12:34:56.789Z"
    },
    {
      "role": "assistant",
      "content": "Assistant's response",
      "timestamp": "2026-02-05T12:34:57.890Z",
      "context_count": 3
    }
  ],
  "total": 42,
  "timestamp": "2026-02-05T12:34:58.901Z"
}
```

#### POST `/api/clear-history`
Clear all conversation history.

**Response:**
```json
{
  "success": true,
  "message": "Conversation history cleared",
  "timestamp": "2026-02-05T12:34:56.789Z"
}
```

---

### Knowledge Base Search

#### POST `/api/search`
Search the knowledge base directly without generating an AI response.

**Request:**
```json
{
  "query": "payment management",
  "top_k": 5
}
```

**Parameters:**
- `query` (string, required): Search query (1-5000 characters)
- `top_k` (integer, optional): Number of results to return (1-20, default: 5)

**Response:**
```json
{
  "success": true,
  "results": [
    {
      "id": "node-id",
      "title": "Node Title",
      "node_type": "instruction|procedure|guideline",
      "similarity": 0.85,
      "attachments": []
    }
  ],
  "count": 5,
  "timestamp": "2026-02-05T12:34:56.789Z"
}
```

---

## Data Types

### ContextItem
```typescript
{
  id: string;
  title: string;
  node_type: string;  // "instruction", "procedure", "guideline", etc.
  content?: string;
  similarity?: number; // 0-1 score
  parent_id?: string;
  attachments?: Attachment[];
}
```

### Attachment
```typescript
{
  id: string;
  file_name: string;
  file_type: string;  // MIME type
  file_path: string;  // URL to file
  uploaded_at?: string;
}
```

### HistoryMessage
```typescript
{
  role: "user" | "assistant";
  content: string;
  timestamp: string; // ISO 8601
  context_count?: number; // For assistant messages
}
```

---

## Setup & Requirements

### Environment Variables
```
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=knowledge_DB
DB_USER=postgres
DB_PASSWORD=your_password

# Groq LLM
GROQ_API_KEY=your_groq_key
LLM_MODEL=llama-3.3-70b-versatile
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=1024

# OpenAI Embeddings
OPENAI_API_KEY=your_openai_key
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIMENSION=1536

# Server
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
ENVIRONMENT=development
DEBUG=true
```

### Installation
```bash
cd backend
pip install -r requirements.txt
python run.py
```

---

## CORS Configuration

The backend is configured to accept requests from the frontend. Ensure the frontend origin is in `ALLOWED_ORIGINS` in `config.py`.

```python
ALLOWED_ORIGINS = [
    "http://localhost:5173",  # Vite dev server
    "http://localhost:3000",  # Alternative port
    "http://localhost:8080",  # Alternative port
]
```

---

## Error Handling

All endpoints return appropriate HTTP status codes:

- **200 OK**: Successful request
- **400 Bad Request**: Invalid parameters
- **500 Internal Server Error**: Server-side error

Error responses include:
```json
{
  "success": false,
  "error": "Error description",
  "timestamp": "2026-02-05T12:34:56.789Z"
}
```

---

## Performance Considerations

- Message processing may take 2-5 seconds depending on:
  - Model response generation
  - Knowledge base retrieval
  - Number of context items requested

- History retrieval is fast and cached

- Search results are ordered by relevance (similarity score)

---

## Logging

Backend logs important events:
- Message processing
- API requests/responses
- Errors and exceptions
- Service initialization/shutdown

Check `logs/` directory for application logs.

---

## Frontend Integration

The frontend (`../frontend`) connects to this API through:
- API client in `src/api/client.ts`
- Type definitions in `src/api/types.ts`
- Configuration in `src/api/config.ts`

See `../frontend/INTEGRATION.md` for frontend setup details.

---

## Development

### Running Tests
```bash
pytest tests/
```

### API Documentation
- Swagger UI: http://localhost:8000/api/docs
- ReDoc: http://localhost:8000/api/redoc
- OpenAPI JSON: http://localhost:8000/api/openapi.json

---

## Troubleshooting

### Messages Not Processing
- Check `GROQ_API_KEY` and `OPENAI_API_KEY` are set
- Verify database connection with `/health` endpoint
- Check logs for detailed error messages

### Knowledge Base Empty
- Ensure CSV files are loaded into database
- Check database connection
- Verify file paths in `knowledge_node.csv` and `knowledge_attachment.csv`

### CORS Errors
- Add frontend origin to `ALLOWED_ORIGINS` in config
- Ensure backend is accessible from frontend

### Slow Responses
- Reduce `top_k` parameter for faster retrieval
- Check database query performance
- Monitor API logs for bottlenecks
