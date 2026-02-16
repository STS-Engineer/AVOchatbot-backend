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
    
    def generate_response(
        self,
        prompt: str,
        context: str = "",
        temperature: Optional[float] = None,
        conversation_history: Optional[list[dict[str, str]]] = None,
        has_kb_context: bool = False,
    ) -> str:
        """
        Flexible response generation: Use KB when available, conversation when not.
        
        Args:
            prompt: The user's question
            context: Knowledge base context (empty if not needed)
            temperature: Temperature for generation
            conversation_history: Recent conversation
            has_kb_context: Whether KB context was provided
        
        Returns:
            Generated response
        """
        try:
            if not self.client:
                logger.warning("Groq client not initialized")
                return "LLM service not available. Please check your GROQ_API_KEY configuration."
            
            temp = temperature if temperature is not None else self.temperature
            
            # Flexible system message - less prescriptive
            system_message = """You are an intelligent assistant for an organization's knowledge base chatbot.

Your behavior adapts based on available context:

**When KB context is provided:**
- Use the knowledge base information as primary source
- Cite sources when referencing specific information
- Stay faithful to the content provided
- Don't describe images - they're shown separately in the UI

**When NO KB context (conversation mode):**
- Use conversation history to answer follow-up questions
- Provide helpful, accurate responses based on what was previously discussed
- Elaborate on topics already covered in the conversation
- For new topics without KB: provide general guidance or ask clarifying questions

**General guidelines:**
- Be conversational and natural
- Use conversation history to understand references ("it", "that", "those")
- Don't repeat information unnecessarily
- If unsure, ask for clarification
- For greetings/small talk, respond naturally

**Image handling:**
- Never describe image contents
- Don't use phrases like "as shown in the image"
- Images are displayed separately

You can answer any question - prioritize KB when available, use conversation and knowledge otherwise."""

            # Build conversation context
            history_block = self._format_conversation_history(conversation_history)
            
            # Adaptive user message based on context availability
            if has_kb_context and context:
                user_message = f"""CONVERSATION HISTORY:
{history_block}

KNOWLEDGE BASE CONTEXT:
{context}

---

USER QUESTION: {prompt}

Provide a helpful response using the KB context above. Cite sources when relevant.

Answer:"""
            else:
                # No KB context - use conversation mode
                user_message = f"""CONVERSATION HISTORY:
{history_block}

---

USER QUESTION: {prompt}

Provide a helpful response based on the conversation history. If this is a follow-up question, elaborate on what was previously discussed. For new topics without KB context, provide general guidance or ask clarifying questions.

Answer:"""
            
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
            mode = "KB" if has_kb_context else "Conversation"
            logger.info(f"Response generated via Groq ({mode} mode)")
            return response
        
        except Exception as e:
            logger.error(f"Error generating response with Groq: {e}")
            return f"Error generating response: {str(e)}"

    def _format_conversation_history(self, history: Optional[list[dict[str, str]]]) -> str:
        if not history:
            return "No prior messages."

        lines = []
        for item in history:
            role = (item.get("role") or "").strip().upper() or "UNKNOWN"
            content = (item.get("content") or "").strip()
            if not content:
                continue
            lines.append(f"{role}: {content}")

        return "\n".join(lines) if lines else "No prior messages."

    def translate_to_english(self, text: str) -> Optional[str]:
        """Translate user text to English for retrieval."""
        if not text or not text.strip():
            return None

        if not self.client:
            logger.warning("Groq client not initialized; skipping translation")
            return None

        try:
            system_message = (
                "You are a translation engine. Translate the user's text to English. "
                "Return only the translation, with no extra commentary."
            )

            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": text}
                ],
                model=self.model,
                temperature=0.0,
                max_tokens=min(512, self.max_tokens),
            )

            translation = chat_completion.choices[0].message.content
            if not translation:
                return None
            return translation.strip()
        except Exception as e:
            logger.warning(f"Translation failed: {e}")
            return None

    def classify_domain(self, text: str, candidates: list[str]) -> Optional[str]:
        """Pick the best matching domain title from candidates."""
        if not text or not text.strip() or not candidates:
            return None

        if not self.client:
            logger.warning("Groq client not initialized; skipping domain classification")
            return None

        try:
            system_message = (
                "You are a classifier. Choose the single best matching domain title from the list. "
                "If none match, return NONE. Return only the exact title or NONE."
            )

            options = "\n".join([f"- {title}" for title in candidates])
            user_message = f"User query: {text}\n\nDomain titles:\n{options}"

            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                model=self.model,
                temperature=0.0,
                max_tokens=min(256, self.max_tokens),
            )

            choice = chat_completion.choices[0].message.content
            if not choice:
                return None
            choice = choice.strip()
            if choice.upper() == "NONE":
                return None
            return choice
        except Exception as e:
            logger.warning(f"Domain classification failed: {e}")
            return None

    def _call_groq(self, prompt: str, max_tokens: int = 100, temperature: float = 0.0) -> Optional[str]:
        """Generic method to call Groq for simple tasks."""
        if not self.client:
            logger.warning("Groq client not initialized")
            return None

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                temperature=temperature,
                max_tokens=min(max_tokens, self.max_tokens),
            )

            response = chat_completion.choices[0].message.content
            if response:
                logger.debug(f"_call_groq response: '{response[:100]}'")
            else:
                logger.warning(f"_call_groq got None/empty response")
                logger.warning(f"Prompt: '{prompt[:200]}'")
                logger.warning(f"Model: {self.model}, max_tokens: {max_tokens}, temp: {temperature}")
            return response.strip() if response else None
        except Exception as e:
            logger.error(f"Groq call failed: {e}", exc_info=True)
            logger.error(f"Prompt was: '{prompt[:200]}'")
            return None


# Global LLM instance
_llm_instance: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Get or create the global LLM service instance."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLMService()
    return _llm_instance
