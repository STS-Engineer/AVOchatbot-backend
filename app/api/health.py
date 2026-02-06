"""
Health check and utility routes.
"""

from fastapi import APIRouter, HTTPException
from loguru import logger
from datetime import datetime
from app.core.config import settings
from app.services.database import get_database
from app.services.llm import get_llm_service
from app.models.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Health check endpoint")
async def health_check() -> HealthResponse:
    """
    Check the health status of the backend service and its dependencies.
    
    **Response:**
    - `status`: Service status ("healthy" or "degraded")
    - `version`: API version
    - `database_connected`: Database connection status
    - `llm_configured`: LLM API key configuration status
    - `timestamp`: Response timestamp
    """
    try:
        db = get_database()
        db_connected = db.test_connection()
        
        llm = get_llm_service()
        llm_configured = llm.client is not None
        
        status = "healthy" if db_connected and llm_configured else "degraded"
        
        return HealthResponse(
            status=status,
            version="1.0.0",
            database_connected=db_connected,
            llm_configured=llm_configured,
            timestamp=datetime.now()
        )
    
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config", summary="Get API configuration")
async def get_config():
    """Get current API configuration (safe values only)."""
    try:
        return {
            "environment": settings.ENVIRONMENT,
            "debug": settings.DEBUG,
            "server": {
                "host": settings.SERVER_HOST,
                "port": settings.SERVER_PORT
            },
            "rag": {
                "top_k_results": settings.TOP_K_RESULTS,
                "similarity_threshold": settings.SIMILARITY_THRESHOLD
            },
            "llm": {
                "model": settings.LLM_MODEL,
                "temperature": settings.LLM_TEMPERATURE,
                "max_tokens": settings.LLM_MAX_TOKENS
            },
            "embedding": {
                "model": settings.EMBEDDING_MODEL,
                "dimension": settings.EMBEDDING_DIMENSION
            }
        }
    except Exception as e:
        logger.error(f"Error getting config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/debug/attachments", summary="Debug: List all attachments")
async def debug_attachments():
    """Debug endpoint to inspect all attachments in the knowledge base."""
    try:
        db = get_database()
        all_attachments = []
        
        # Get a sample of nodes and their attachments
        nodes = db.search_by_keyword("", limit=20)
        for node in nodes:
            node_id = node.get('id')
            attachments = db.get_attachments_for_node(node_id)
            for att in attachments:
                all_attachments.append({
                    "node_title": node.get('title'),
                    "node_id": node_id,
                    "file_name": att.get('file_name'),
                    "file_path": att.get('file_path'),
                    "file_type": att.get('file_type'),
                    "uploaded_at": att.get('uploaded_at')
                })
        
        return {
            "success": True,
            "total_attachments_found": len(all_attachments),
            "attachments": all_attachments,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in debug attachments endpoint: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.get("/debug/concept-mappings", summary="Debug: View semantic understanding settings")
async def debug_concept_mappings():
    """Debug endpoint to view semantic search settings and thresholds."""
    from app.services.rag import get_rag_service
    
    try:
        rag_service = get_rag_service()
        
        return {
            "success": True,
            "semantic_understanding": {
                "strategy": "Dynamic embedding-based semantic search",
                "description": "Uses embeddings to automatically understand synonyms, context, and related concepts without hardcoded lists",
                "how_it_works": [
                    "1. Query is embedded into high-dimensional space using the embedding model",
                    "2. Embedding model understands semantic meaning (synonyms, context, relationships)",
                    "3. Database search finds semantically similar content",
                    "4. Lower similarity threshold catches related concepts (not just exact matches)"
                ]
            },
            "search_settings": {
                "top_k_results": rag_service.top_k,
                "exact_match_threshold": rag_service.similarity_threshold,
                "semantic_understanding_threshold": rag_service.semantic_threshold,
                "threshold_explanation": f"Semantic threshold ({rag_service.semantic_threshold:.2f}) is lower than exact match threshold ({rag_service.similarity_threshold:.2f}) to catch synonyms and related concepts"
            },
            "examples": {
                "example_1": {
                    "query": "What is reliance between teams?",
                    "understood_as": "Searching semantically for content about trust, dependency, confidence, mutual reliance",
                    "why": "Embeddings capture semantic similarity between 'reliance' and 'trust'"
                },
                "example_2": {
                    "query": "Tell me about accountability",
                    "understood_as": "Searching for content about responsibility, ownership, accountability",
                    "why": "Embedding model understands these terms are semantically related"
                }
            },
            "scaling": "Automatically scales with database growth - no manual concept list maintenance needed",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in debug concept mappings endpoint: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.get("/debug/query-expansion/{query:path}", summary="Debug: Test semantic understanding")
async def debug_query_expansion(query: str):
    """Debug endpoint to test semantic understanding of a query."""
    from app.services.rag import get_rag_service
    from app.services.embedding import get_embedding_service
    
    try:
        rag_service = get_rag_service()
        embedding_service = get_embedding_service()
        
        # Get embedding for the query
        query_embedding = embedding_service.embed_text(query)
        
        # Perform a small search to see what the model finds
        results = rag_service.db.search_by_similarity(query_embedding, limit=5)
        
        return {
            "success": True,
            "query": query,
            "semantic_search_results": [
                {
                    "title": r.get('title'),
                    "similarity": r.get('similarity'),
                    "meets_semantic_threshold": r.get('similarity', 0) >= rag_service.semantic_threshold
                }
                for r in results
            ],
            "explanation": "The embedding model found these semantically related results. Even if they don't contain the exact word from your query, they are semantically related.",
            "threshold": {
                "semantic_threshold": rag_service.semantic_threshold,
                "exact_match_threshold": rag_service.similarity_threshold
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in debug query expansion endpoint: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


