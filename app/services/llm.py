"""
LLM service using Groq API.
"""

from typing import Optional
from loguru import logger
from app.core.config import settings

try:
    import groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


class LLMService:
    """Groq LLM client for generating responses."""
    
    def __init__(self):
        """Initialize LLM service."""
        self.client = None
        self.model = settings.LLM_MODEL
        self.temperature = settings.LLM_TEMPERATURE
        self.max_tokens = settings.LLM_MAX_TOKENS
        self._init_client()
    
    def _init_client(self):
        """Initialize Groq client."""
        if not GROQ_AVAILABLE:
            logger.warning("Groq library not available. Install with: pip install groq")
            return
        
        try:
            if not settings.GROQ_API_KEY:
                logger.warning("GROQ_API_KEY not provided in environment")
                return
            
            # Initialize Groq client - only pass api_key
            self.client = groq.Groq(api_key=settings.GROQ_API_KEY)
            logger.info(f"Groq LLM client initialized. Model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
            # Service degrades gracefully - client stays None
    
    def generate_response(self, prompt: str, context: str = "", temperature: Optional[float] = None) -> str:
        """
        Generate a response using Groq API.
        
        Args:
            prompt: The user's question
            context: Knowledge base context
            temperature: Temperature for generation (uses default if None)
        
        Returns:
            Generated response
        """
        try:
            if not self.client:
                logger.warning("Groq client not initialized")
                return "LLM service not available. Please check your GROQ_API_KEY configuration."
            
            temp = temperature if temperature is not None else self.temperature
            
            system_message = """You are an expert assistant for a company knowledge base. Your role is to provide accurate, helpful responses based ONLY on the information provided in the knowledge base context.

CRITICAL RULES:
1. Answer ONLY based on the knowledge base context provided - do not add external knowledge or assumptions
2. If the context contains the information needed, use it exactly as provided
3. Always cite the source knowledge base sections when referencing information
4. Do NOT describe what images contain - images are provided separately to the user interface for visual reference only
5. Do NOT use phrases like "As shown in the image", "The diagram shows", "illustrated below", etc. - let the images speak for themselves
6. Do NOT add assumptions about visual content - the user can see images in their own interface
7. Reference images only if explicitly mentioned in the knowledge base text itself

ABOUT SEMANTIC UNDERSTANDING:
- The search system automatically finds semantically related content using embeddings
- When a user asks about "reliance", the system may return content about "trust" (semantically similar)
- When they ask about "accountability", it may find content about "ownership" or "responsibility"
- Your context is already filtered to relevant semantic matches - use it as provided
- You don't need to map synonyms - the system has already done that for you

For greetings or casual questions (hello, hi, how are you, etc.):
- You can respond naturally without requiring knowledge base context
- Keep the response brief and friendly

For knowledge base questions:
- Use ONLY the exact information provided in the context
- The context may use different terminology than the user's question - that's OK, it's semantically related
- Do not fabricate details or make assumptions
- If the context is insufficient, clearly state what information is available instead
- Be literal and precise - do not embellish with interpretations

Examples:
- USER ASKS: "What is reliance between teams?"
- CONTEXT PROVIDED: Discusses "trust" between teams
- RIGHT: "Trust - which refers to the reliance between managers and teams - is built through clear mission definition, communication, escalation, and feedback."

Remember: The system has used semantic search to find relevant content. Your job is to use that context accurately."""
            
            user_message = f"""Based ONLY on the knowledge base context provided below, answer the user's question.

CONTEXT FROM KNOWLEDGE BASE:
{context if context else "No relevant context found in the knowledge base."}

---

USER QUESTION: {prompt}

INSTRUCTIONS:
1. Use ONLY the information from the context above
2. Do NOT add external information or make assumptions
3. Do NOT describe what images contain - they are displayed separately
4. Do NOT use phrases like "as shown in the image" or "the diagram shows"
5. If the context doesn't have the answer, clearly state what information is available instead
6. Be direct and accurate - cite the context exactly as provided

Answer now:"""
            
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                model=self.model,
                temperature=temp,
                max_tokens=self.max_tokens,
            )
            
            response = chat_completion.choices[0].message.content
            logger.info("Response generated successfully via Groq")
            return response
        
        except Exception as e:
            logger.error(f"Error generating response with Groq: {e}")
            return f"Error generating response: {str(e)}"


# Global LLM instance
_llm_instance: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Get or create the global LLM service instance."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLMService()
    return _llm_instance
