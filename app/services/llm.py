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
        allow_generic_guidance: bool = False,
        conversation_history: Optional[list[dict[str, str]]] = None,
        last_assistant_response: Optional[str] = None,
        is_followup: bool = False,
    ) -> str:
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
            
            system_message = """You are an expert assistant for a company knowledge base. Your role is to provide accurate, helpful responses grounded in the knowledge base context when it is available.

CRITICAL RULES:
1. When relevant knowledge base context is available, prioritize it and stay faithful to it
2. If the context contains the information needed, use it as provided and you may rephrase or elaborate for clarity
3. Always cite the source knowledge base titles when referencing information
4. Do NOT describe what images contain - images are provided separately to the user interface for visual reference only
5. Do NOT use phrases like "As shown in the image", "The diagram shows", "illustrated below", etc. - let the images speak for themselves
6. Do NOT add assumptions about visual content - the user can see images in their own interface
7. Reference images only if explicitly mentioned in the knowledge base text itself
8. You may use conversation history to resolve references (e.g., "that policy") and maintain continuity

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
- Use the information provided in the context and keep the same meaning
- The context may use different terminology than the user's question - that's OK, it's semantically related
- You may explain concepts in your own words as long as you do not add new facts beyond the context
- If the context is insufficient, ask a clarifying question or explain what you can with what is available
- Be precise and grounded

For general questions or when no context is available:
- Respond conversationally and helpfully
- Ask 1-2 clarifying questions when needed
- Offer brief, practical guidance without claiming specific company policy
- Add a short note at the end: "Note: No KB context found for this response."

Examples:
- USER ASKS: "What is reliance between teams?"
- CONTEXT PROVIDED: Discusses "trust" between teams
- RIGHT: "Trust - which refers to the reliance between managers and teams - is built through clear mission definition, communication, escalation, and feedback."

Remember: The system has used semantic search to find relevant content. Your job is to use that context accurately."""

            if allow_generic_guidance:
                system_message += """

GENERAL GUIDANCE:
- If the context is empty or clearly insufficient, you may provide brief, generic professional guidance (2-4 sentences).
- Keep it practical and neutral (e.g., communicate calmly, focus on facts, ask for a 1:1).
- Avoid legal/HR determinations and do not invent company policy.
"""
            
            history_block = self._format_conversation_history(conversation_history)
            last_assistant_block = (last_assistant_response or "").strip() or "(none)"

            followup_instruction = ""
            if is_followup:
                followup_instruction = (
                    "\nFOLLOW-UP RESPONSE RULES:\n"
                    "- Do not repeat the previous answer\n"
                    "- Add new details or angles grounded in the context\n"
                    "- If the context is empty, elaborate using the prior assistant response and conversation history without adding new facts\n"
                    "- Do not ask what topic the user means; answer based on the prior assistant response\n"
                    "- Do not ask clarifying questions about which document or step the user refers to\n"
                )

            followup_override = ""
            if is_followup and last_assistant_response and last_assistant_response != "(none)":
                followup_override = (
                    "\nFOLLOW-UP OVERRIDE:\n"
                    "- If context is empty, do NOT ask clarifying questions; use the last assistant response and conversation history as your grounding.\n"
                )

            instructions = [
                "1. If context is available, use it and stay faithful to it",
                "2. You may explain or rephrase for clarity, but do NOT add new facts beyond the context",
                "3. Do NOT describe what images contain - they are displayed separately",
                "4. Do NOT use phrases like \"as shown in the image\" or \"the diagram shows\"",
                "5. If the context is empty, respond conversationally and ask 1-2 clarifying questions",
                "6. If the context doesn't have the answer, clearly state what information is available instead",
                "7. If there is no KB context, append: \"Note: No KB context found for this response.\"",
                "8. Be direct and accurate - cite the context titles exactly as provided",
            ]

            if is_followup:
                instructions[4] = (
                    "5. If the context is empty, answer using the last assistant response and conversation history; "
                    "do NOT ask clarifying questions"
                )

            instructions_block = "\n".join(instructions)

            user_message = f"""Based ONLY on the knowledge base context provided below, answer the user's question.

CONVERSATION HISTORY (context only, not a knowledge source):
{history_block}

LAST ASSISTANT RESPONSE (for follow-ups):
{last_assistant_block}

CONTEXT FROM KNOWLEDGE BASE:
{context if context else "(no context available)"}

---

USER QUESTION: {prompt}

INSTRUCTIONS:
{instructions_block}
{followup_instruction}
{followup_override}

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
