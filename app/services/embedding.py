"""
Embedding service using OpenAI API.
"""

from typing import List
from loguru import logger
from app.core.config import settings

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class EmbeddingService:
    """Manages text embeddings using OpenAI API."""
    
    def __init__(self):
        """Initialize embedding service."""
        self.client = None
        self.model = settings.EMBEDDING_MODEL
        self.dimension = settings.EMBEDDING_DIMENSION
        self._init_client()
    
    def _init_client(self):
        """Initialize OpenAI client."""
        if not OPENAI_AVAILABLE:
            logger.warning("OpenAI library not available. Install with: pip install openai")
            return
        
        try:
            if not settings.OPENAI_API_KEY:
                logger.warning("OPENAI_API_KEY not provided in environment")
                return
            
            self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
            logger.info(f"OpenAI client initialized. Model: {self.model}, Dimension: {self.dimension}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        try:
            if not self.client:
                logger.warning("OpenAI client not initialized")
                return [0.0] * self.dimension
            
            if not isinstance(text, str) or not text.strip():
                logger.warning("Empty text provided for embedding")
                return [0.0] * self.dimension
            
            # Truncate to reasonable length
            text = text[:2000]
            
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            
            embedding = response.data[0].embedding
            
            # Ensure correct dimension
            if len(embedding) > self.dimension:
                embedding = embedding[:self.dimension]
            elif len(embedding) < self.dimension:
                embedding = embedding + [0.0] * (self.dimension - len(embedding))
            
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return [0.0] * self.dimension
    
    def embed_texts(self, texts: List[str], batch_size: int = 10) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        try:
            if not self.client:
                logger.warning("OpenAI client not initialized")
                return [[0.0] * self.dimension] * len(texts)
            
            embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                cleaned_batch = [
                    (t[:2000] if isinstance(t, str) and t.strip() else "")
                    for t in batch
                ]
                
                response = self.client.embeddings.create(
                    input=cleaned_batch,
                    model=self.model
                )
                
                for item in response.data:
                    embedding = item.embedding
                    
                    if len(embedding) > self.dimension:
                        embedding = embedding[:self.dimension]
                    elif len(embedding) < self.dimension:
                        embedding = embedding + [0.0] * (self.dimension - len(embedding))
                    
                    embeddings.append(embedding)
            
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return [[0.0] * self.dimension] * len(texts)


# Global embedding instance
_embedding_instance: EmbeddingService = None


def get_embedding_service() -> EmbeddingService:
    """Get or create the global embedding service instance."""
    global _embedding_instance
    if _embedding_instance is None:
        _embedding_instance = EmbeddingService()
    return _embedding_instance
