# Backend API for Chatbot

FastAPI-based backend for the RAG chatbot system. Integrates with the React frontend to provide intelligent question-answering based on a knowledge database.

## Features

- **RAG System**: Retrieval Augmented Generation for accurate, context-aware responses
- **Multi-level Search**: Title matching → Semantic similarity → Fallback search
- **LLM Integration**: Groq API for high-quality response generation
- **Knowledge Base**: PostgreSQL with pgvector for semantic search
- **Conversation History**: Track and retrieve conversation context
- **API Documentation**: Auto-generated Swagger/OpenAPI docs

## Project Structure

```
backend/
├── app/
│   ├── api/
│   │   ├── routes.py        # Main chat endpoints
│   │   └── health.py        # Health check endpoints
│   ├── services/
│   │   ├── chat.py          # Chat orchestration service
│   │   ├── rag.py           # RAG retrieval service
│   │   ├── llm.py           # LLM (Groq) integration
│   │   ├── embedding.py     # Embedding service (OpenAI)
│   │   └── database.py      # Database operations
│   ├── models/
│   │   └── schemas.py       # Pydantic request/response models
│   ├── core/
│   │   ├── config.py        # Settings from environment
│   │   └── logging.py       # Logging configuration
│   └── main.py              # FastAPI application
├── logs/                     # Application logs
├── requirements.txt          # Python dependencies
├── .env.example             # Example environment variables
├── run.py                   # Development server runner
└── README.md                # This file
```

## Installation

### Prerequisites
- Python 3.9+
- PostgreSQL with pgvector extension
- GROQ API key
- OpenAI API key (for embeddings)

### Setup Steps

1. **Navigate to backend folder**
```bash
cd d:\test\backend
```

2. **Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment**
```bash
# Copy example to actual .env
copy .env.example .env

# Edit .env with your actual values
# Required:
# - DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
# - GROQ_API_KEY
# - OPENAI_API_KEY
```

5. **Run development server**
```bash
python run.py
```

Server will start at `http://localhost:8000`

## API Endpoints

### Chat Endpoint
**POST** `/api/chat`
- Send a message to the chatbot
- Returns AI response with knowledge base context
- Query parameters: `top_k` (default: 8), `include_context` (default: true)

Example request:
```json
{
  "message": "What is the process for handling late payments?",
  "include_context": true,
  "top_k": 8
}
```

### History Endpoint
**GET** `/api/history`
- Retrieve conversation history
- Query parameters: `limit` (default: 50), `offset` (default: 0)

### Clear History
**POST** `/api/clear-history`
- Clear all conversation history

### Search Endpoint
**POST** `/api/search`
- Search knowledge base directly
- Returns matching items without AI generation

### Health Check
**GET** `/health`
- Check service health status
- Returns database and LLM configuration status

### Configuration
**GET** `/config`
- Get current API configuration

## Documentation

Once the server is running, access interactive API documentation:
- **Swagger UI**: http://localhost:8000/api/docs
- **ReDoc**: http://localhost:8000/api/redoc

## Environment Variables

See `.env.example` for all available options:

```bash
# Server
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
ENVIRONMENT=development
DEBUG=True

# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=knowledge_DB
DB_USER=postgres
DB_PASSWORD=your_password
DB_SSLMODE=disable

# APIs
GROQ_API_KEY=your_groq_key
OPENAI_API_KEY=your_openai_key

# RAG Settings
TOP_K_RESULTS=8
SIMILARITY_THRESHOLD=0.3

# LLM Settings
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=1024

# CORS
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:3000
```

## Service Architecture

### ChatService
Main orchestration service that:
- Processes user messages
- Calls RAG system for context retrieval
- Calls LLM for response generation
- Manages conversation history

### RAGService
Retrieval Augmented Generation:
- Multi-level search strategy
- Title matching (exact/similarity)
- Semantic similarity search (vector-based)
- Keyword fallback
- Returns formatted context and items

### LLMService
Groq API integration:
- Generates responses based on query + context
- Configurable temperature and max tokens
- System prompt for role definition

### EmbeddingService
OpenAI embeddings:
- Text to vector conversion (1536 dimensions)
- Batch processing support
- Dimension handling

### DatabaseService
PostgreSQL operations:
- Vector similarity search
- Title-based search
- Knowledge node retrieval

## Development

### Adding New Endpoints
1. Create handler in `app/api/routes.py` or new file in `app/api/`
2. Add Pydantic models in `app/models/schemas.py`
3. Add business logic in `app/services/`
4. Include router in `app/main.py`

### Debugging
Enable debug mode in `.env`:
```bash
DEBUG=True
LOG_LEVEL=DEBUG
```

Check logs in `logs/backend.log`

## Integration with Frontend

The frontend (React + Vite at `d:\test\frontend`) connects to this backend:

1. Frontend makes requests to `http://localhost:8000/api/chat`
2. Backend returns structured responses with AI message and context
3. Frontend displays message in chat UI
4. Supports streaming and real-time updates

### CORS Configuration
Make sure `ALLOWED_ORIGINS` includes your frontend URL:
```bash
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:3000
```

## Troubleshooting

### Database Connection Issues
- Verify PostgreSQL is running
- Check credentials in `.env`
- Ensure pgvector extension is installed: `CREATE EXTENSION IF NOT EXISTS vector;`

### API Key Issues
- Verify GROQ_API_KEY is set correctly
- Verify OPENAI_API_KEY is set correctly
- Check API key format (should start with specific prefixes)

### No Context Retrieved
- Check database has knowledge_node entries
- Verify embeddings exist in database
- Lower SIMILARITY_THRESHOLD in .env

## Performance Tips

- Adjust `TOP_K_RESULTS` for context retrieval (higher = more context, slower)
- Enable GZIP compression (enabled by default)
- Use connection pooling (configured in database service)
- Monitor logs for bottlenecks

## Future Enhancements

- [ ] Streaming responses
- [ ] File upload support
- [ ] Knowledge base management API
- [ ] Analytics and usage tracking
- [ ] Custom embedding models
- [ ] Fine-tuned LLM models
- [ ] Caching layer (Redis)
- [ ] Rate limiting

## License

Proprietary
