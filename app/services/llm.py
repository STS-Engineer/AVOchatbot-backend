"""
LLM service supporting Groq and OpenAI providers.
"""

from typing import Optional
from loguru import logger
from app.core.config import settings

try:
    import groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class LLMService:
    """LLM client for generating responses."""
    
    def __init__(self):
        """Initialize LLM service."""
        self.client = None
        self.provider = (settings.LLM_PROVIDER or "groq").strip().lower()
        self.model = settings.LLM_MODEL if self.provider == "groq" else settings.OPENAI_LLM_MODEL
        self.temperature = settings.LLM_TEMPERATURE
        self.max_tokens = settings.LLM_MAX_TOKENS
        self._init_client()
    
    def _init_client(self):
        """Initialize provider-specific LLM client."""
        if self.provider == "openai":
            if not OPENAI_AVAILABLE:
                logger.warning("OpenAI library not available. Install with: pip install openai")
                return
            try:
                if not settings.OPENAI_API_KEY:
                    logger.warning("OPENAI_API_KEY not provided in environment")
                    return
                self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
                logger.info(f"OpenAI LLM client initialized. Model: {self.model}")
                return
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                return

        # Default to Groq when provider is unknown or set to groq
        if self.provider not in {"groq", "openai"}:
            logger.warning(f"Unknown LLM_PROVIDER '{self.provider}', defaulting to groq")
            self.provider = "groq"
            self.model = settings.LLM_MODEL

        if not GROQ_AVAILABLE:
            logger.warning("Groq library not available. Install with: pip install groq")
            return
        
        try:
            if not settings.GROQ_API_KEY:
                logger.warning("GROQ_API_KEY not provided in environment")
                return
            self.client = groq.Groq(api_key=settings.GROQ_API_KEY)
            logger.info(f"Groq LLM client initialized. Model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
    
    def generate_response(
        self,
        prompt: str,
        context: str = "",
        temperature: Optional[float] = None,
        conversation_history: Optional[list[dict[str, str]]] = None,
        has_kb_context: bool = False,
        has_file_context: bool = False,
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
                logger.warning("LLM client not initialized")
                return "LLM service not available. Please check your LLM provider configuration."
            
            temp = temperature if temperature is not None else self.temperature
            
            # Flexible system message - less prescriptive
            system_message = """
You are an intelligent assistant for an organization's knowledge base chatbot.

Your behavior adapts based on available context:

**When KB context is provided:**
- Use the knowledge base information as primary source
- Cite sources when referencing specific information
- Stay faithful to the content provided
- Don't describe images - they're shown separately in the UI

**When uploaded file context is provided:**
- Prioritize uploaded file analysis for the user's answer
- Treat uploaded file context as user-provided source of truth for this turn
- Only use KB context as secondary support unless user explicitly asks for comparison with KB/policies
- If the uploaded file context indicates unreadable/empty content, clearly say what could not be read

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

**Formatting instructions:**
- Use Markdown or HTML for your answers to make them visually clear and engaging.
- Use tables, lists, headings, and bold text for structure.
- Add color (using HTML style or Markdown where supported) for emphasis.
- Prefer concise, readable formatting.
- If the answer involves a process or steps, use numbered lists or tables.

You can answer any question - prioritize KB when available, use conversation and knowledge otherwise.
"""

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

Provide a helpful response using the context above.
If uploaded file context exists, focus the answer on that file first.

Answer:"""
            else:
                # No KB context - use conversation mode
                user_message = f"""CONVERSATION HISTORY:
{history_block}

---

USER QUESTION: {prompt}

Provide a helpful response based on the conversation history. If this is a follow-up question, elaborate on what was previously discussed. For new topics without KB context, provide general guidance or ask clarifying questions.

Answer:"""
            
            response = self._chat_completion(
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=temp,
                max_tokens=self.max_tokens,
            )
            if not response:
                return "I could not generate a response right now."
            mode = "KB" if has_kb_context else "Conversation"
            logger.info(f"Response generated via {self.provider} ({mode} mode)")
            return response
        
        except Exception as e:
            logger.error(f"Error generating response with LLM provider: {e}")
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
            logger.warning("LLM client not initialized; skipping translation")
            return None

        try:
            system_message = (
                "You are a translation engine. Translate the user's text to English. "
                "Return only the translation, with no extra commentary."
            )

            translation = self._chat_completion(
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": text}
                ],
                temperature=0.0,
                max_tokens=min(512, self.max_tokens),
            )
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
            logger.warning("LLM client not initialized; skipping domain classification")
            return None

        try:
            system_message = (
                "You are a classifier. Choose the single best matching domain title from the list. "
                "If none match, return NONE. Return only the exact title or NONE."
            )

            options = "\n".join([f"- {title}" for title in candidates])
            user_message = f"User query: {text}\n\nDomain titles:\n{options}"

            choice = self._chat_completion(
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.0,
                max_tokens=min(256, self.max_tokens),
            )
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
        """Backward-compatible helper used by existing code paths."""
        return self._call_llm(prompt=prompt, max_tokens=max_tokens, temperature=temperature)

    def _call_llm(self, prompt: str, max_tokens: int = 100, temperature: float = 0.0) -> Optional[str]:
        """Generic method to call the configured LLM provider for simple tasks."""
        if not self.client:
            logger.warning("LLM client not initialized")
            return None

        try:
            response = self._chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=min(max_tokens, self.max_tokens),
            )
            if response:
                logger.debug(f"_call_llm response: '{response[:100]}'")
            else:
                logger.warning(f"_call_llm got None/empty response")
                logger.warning(f"Prompt: '{prompt[:200]}'")
                logger.warning(f"Model: {self.model}, max_tokens: {max_tokens}, temp: {temperature}")
            return response.strip() if response else None
        except Exception as e:
            logger.error(f"LLM call failed: {e}", exc_info=True)
            logger.error(f"Prompt was: '{prompt[:200]}'")
            return None

    def summarize_uploaded_text(self, file_name: str, extracted_text: str) -> Optional[str]:
        """Generate a concise analysis for uploaded document text."""
        if not extracted_text.strip():
            return None

        prompt = f"""You are analyzing a user-uploaded file for a support chatbot.

File name: {file_name}

Task:
1) Summarize the most important content in 5-8 bullet points.
2) Identify key entities (names, dates, amounts, IDs) if present.
3) Mention any risks, inconsistencies, or missing information.

Return concise markdown.

Document text:
{extracted_text[:12000]}
"""
        return self._call_llm(prompt=prompt, max_tokens=700, temperature=0.2)

    def _chat_completion(self, messages: list[dict[str, str]], temperature: float, max_tokens: int) -> Optional[str]:
        """Execute chat completion against configured provider."""
        if not self.client:
            return None

        if self.provider == "openai":
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return completion.choices[0].message.content

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return completion.choices[0].message.content


# Global LLM instance
_llm_instance: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Get or create the global LLM service instance."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLMService()
    return _llm_instance
