"""
Chat service - main business logic orchestrating RAG and LLM.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from loguru import logger
from app.services.rag import get_rag_service
from app.services.llm import get_llm_service


class ChatService:
    """Main chat service orchestrating RAG and LLM."""
    
    def __init__(self):
        """Initialize chat service."""
        self.rag_service = get_rag_service()
        self.llm_service = get_llm_service()
        self.conversation_history_by_id: Dict[str, List[Dict[str, Any]]] = {}
        self.last_context_by_id: Dict[str, Optional[str]] = {}
        self.last_context_items_by_id: Dict[str, List[Dict[str, Any]]] = {}
        logger.info("Chat Service initialized")
    
    def process_message(
        self,
        message: str,
        top_k: Optional[int] = None,
        include_context: bool = True,
        conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a user message and return AI response with context.
        
        Args:
            message: User's query/message
            top_k: Number of context items to retrieve
            include_context: Whether to include context in response
        
        Returns:
            Dictionary with success status, message, context, and metadata
        """
        try:
            logger.info(f"Processing message: {message[:100]}...")
            
            conversation_key = self._normalize_conversation_id(conversation_id)
            history = self._get_history_list(conversation_key)

            # Store user message in history
            history.append({
                "role": "user",
                "content": message,
                "timestamp": datetime.now()
            })
            
            context_items: List[Dict[str, Any]] = []
            formatted_context = ""
            used_context_items: List[Dict[str, Any]] = []
            used_formatted_context = ""
            
            # Check if this is a follow-up using LLM (language-agnostic)
            is_followup = self._is_followup(message, history)
            
            # Prefer new retrieval for follow-ups using prior topic context
            last_context_items = self.last_context_items_by_id.get(conversation_key, [])
            last_context = self.last_context_by_id.get(conversation_key)

            if is_followup:
                # Get prior context for rewriting
                prior_user_query = self._get_last_user_query(history, exclude_latest=True)
                prior_assistant = self._get_last_assistant_response(history)
                
                if prior_user_query and prior_assistant:
                    logger.info(f"Follow-up detected for: '{message[:80]}'")
                    # Use LLM to rewrite query with context (language-agnostic)
                    combined_query = self._rewrite_with_context(message, prior_user_query, prior_assistant)
                    logger.info(f"Rewritten query: '{combined_query[:120]}'")
                else:
                    logger.warning("Follow-up detected but context incomplete - using original query")
                    combined_query = message
                
                formatted_context, context_items = self.rag_service.retrieve_context(combined_query, k=top_k)

                if context_items:
                    used_context_items = context_items
                    used_formatted_context = formatted_context
                    self.last_context_items_by_id[conversation_key] = context_items
                    self.last_context_by_id[conversation_key] = formatted_context
                    logger.info("Retrieved new context for follow-up message")
                elif last_context_items:
                    used_context_items = last_context_items
                    used_formatted_context = last_context or ""
                    logger.info("Fallback to previous context for follow-up message")
            else:
                logger.info(f"New query (not a follow-up): '{message[:80]}'")
                # Retrieve context from knowledge base for new queries
                formatted_context, context_items = self.rag_service.retrieve_context(message, k=top_k)

                used_context_items = context_items
                used_formatted_context = formatted_context

                # Update last context only when new context is found
                if context_items:
                    self.last_context_items_by_id[conversation_key] = context_items
                    self.last_context_by_id[conversation_key] = formatted_context
            
            # Generate response using LLM
            history_for_prompt = self._get_history_window(history, exclude_latest=True)
            last_assistant_response = self._get_last_assistant_response(history_for_prompt)
            response = self.llm_service.generate_response(
                message,
                context=used_formatted_context,
                allow_generic_guidance=True,
                conversation_history=history_for_prompt,
                last_assistant_response=last_assistant_response,
                is_followup=is_followup,
            )
            
            # Store assistant response in history
            history.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now(),
                "context_count": len(used_context_items)
            })
            
            result = {
                "success": True,
                "message": response,
                "context": used_formatted_context if include_context and used_formatted_context else None,
                "context_items": used_context_items if include_context and used_context_items else None,
                "context_count": len(used_context_items),
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Message processed successfully. Context items: {len(used_context_items)}")
            return result
        
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "An error occurred while processing your message. Please try again.",
                "timestamp": datetime.now().isoformat()
            }

    def edit_message_and_respond(
        self,
        message_index: int,
        new_message: str,
        top_k: Optional[int] = None,
        include_context: bool = True,
        conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Edit a prior user message, truncate history, and regenerate the assistant response.
        """
        try:
            logger.info(f"Editing message at index {message_index}: {new_message[:100]}...")

            conversation_key = self._normalize_conversation_id(conversation_id)
            history = self._get_history_list(conversation_key)

            if message_index < 0 or message_index >= len(history):
                raise ValueError("Message index out of range")

            if history[message_index].get("role") != "user":
                raise ValueError("Only user messages can be edited")

            # Update the user message and truncate any following messages
            history[message_index]["content"] = new_message
            history[message_index]["timestamp"] = datetime.now()
            del history[message_index + 1 :]

            # Reset cached context for this conversation
            self.last_context_by_id.pop(conversation_key, None)
            self.last_context_items_by_id.pop(conversation_key, None)

            context_items: List[Dict[str, Any]] = []
            formatted_context = ""
            used_context_items: List[Dict[str, Any]] = []
            used_formatted_context = ""
            is_followup = self._is_followup(new_message)

            if is_followup:
                prior_user_query = self._get_last_user_query(history, exclude_latest=True)
                combined_query = self._build_followup_query(prior_user_query, new_message)
                formatted_context, context_items = self.rag_service.retrieve_context(combined_query, k=top_k)

                if context_items:
                    used_context_items = context_items
                    used_formatted_context = formatted_context
                    self.last_context_items_by_id[conversation_key] = context_items
                    self.last_context_by_id[conversation_key] = formatted_context
                    logger.info("Retrieved new context for edited follow-up message")
            else:
                formatted_context, context_items = self.rag_service.retrieve_context(new_message, k=top_k)

                used_context_items = context_items
                used_formatted_context = formatted_context

                if context_items:
                    self.last_context_items_by_id[conversation_key] = context_items
                    self.last_context_by_id[conversation_key] = formatted_context

            history_for_prompt = self._get_history_window(history, exclude_latest=True)
            last_assistant_response = self._get_last_assistant_response(history_for_prompt)
            response = self.llm_service.generate_response(
                new_message,
                context=used_formatted_context,
                allow_generic_guidance=True,
                conversation_history=history_for_prompt,
                last_assistant_response=last_assistant_response,
                is_followup=is_followup,
            )

            history.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now(),
                "context_count": len(used_context_items)
            })

            result = {
                "success": True,
                "message": response,
                "context": used_formatted_context if include_context and used_formatted_context else None,
                "context_items": used_context_items if include_context and used_context_items else None,
                "context_count": len(used_context_items),
                "timestamp": datetime.now().isoformat()
            }

            logger.info(f"Edited message processed successfully. Context items: {len(used_context_items)}")
            return result

        except Exception as e:
            logger.error(f"Error editing message: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "An error occurred while editing your message. Please try again.",
                "timestamp": datetime.now().isoformat()
            }
    
    def _is_greeting(self, message: str) -> bool:
        """
        Detect if the message is a simple greeting.
        
        Args:
            message: User's message
        
        Returns:
            True if the message is a greeting, False otherwise
        """
        greetings = [
            "hello", "hi", "hey", "greetings", "hola", "bonjour",
            "yo", "sup", "howdy", "morning", "afternoon", "evening",
            "good morning", "good afternoon", "good evening",
            "what's up", "whats up", "g'day", "gday"
        ]

        message_lower = message.strip().lower()
        if not message_lower:
            return False

        import re
        clean_message = re.sub(r"[^a-z0-9\s]", " ", message_lower).strip()
        if not clean_message:
            return False

        # Exact or short greeting (e.g., "hi", "hello team", "good morning")
        if clean_message in greetings:
            return True

        tokens = clean_message.split()
        if not tokens:
            return False

        simple_greetings = {"hello", "hi", "hey", "yo", "sup", "greetings", "howdy","Bonjour"}
        if tokens[0] in simple_greetings and len(tokens) <= 4:
            non_greeting_tokens = {
                "can", "could", "please", "help", "what", "why", "how",
                "where", "when", "who", "tell", "explain", "show", "need"
            }
            if any(token in non_greeting_tokens for token in tokens[1:]):
                return False
            return True

        if "good morning" in clean_message or "good afternoon" in clean_message or "good evening" in clean_message:
            return len(tokens) <= 4

        if "what's up" in clean_message or "whats up" in clean_message:
            return len(tokens) <= 4

        return False

    def _greeting_response(self) -> str:
        """Return a friendly greeting response."""
        return (
            "Hello! How can I help you today? "
            "You can ask about policies, procedures, or any internal guidance."
        )

    def _is_thanks(self, message: str) -> bool:
        """Detect a short thank-you message."""
        message_lower = message.strip().lower()
        if not message_lower:
            return False

        import re
        clean_message = re.sub(r"[^a-z0-9\s]", " ", message_lower).strip()
        if not clean_message:
            return False

        tokens = clean_message.split()
        if not tokens:
            return False

        thanks_tokens = {
            "thanks", "thank", "thankyou", "thx", "ty", "merci", "thanks!",
        }

        if clean_message in {"thank you", "thanks", "thanks a lot", "thank you!", "merci"}:
            return True

        if tokens[0] in thanks_tokens and len(tokens) <= 4:
            return True

        if "thank you" in clean_message and len(tokens) <= 6:
            return True

        return False

    def _thanks_response(self) -> str:
        """Return a brief thank-you response."""
        return "Thank you!"

    def _is_vague_advice_request(self, message: str) -> bool:
        """Detect short, vague requests for advice without a topic."""
        message_lower = message.strip().lower()
        if not message_lower:
            return False

        import re
        clean_message = re.sub(r"[^a-z0-9\s]", " ", message_lower).strip()
        if not clean_message:
            return False

        tokens = clean_message.split()
        if len(tokens) > 6:
            return False

        vague_phrases = {
            "i need your advice",
            "need advice",
            "i need advice",
            "i want advice",
            "can you advise",
            "can you give advice",
            "advice please",
            "need help",
            "i need help",
        }

        if clean_message in vague_phrases:
            return True

        if "advice" in tokens and len(tokens) <= 4:
            return True

        return False

    def _is_interpersonal(self, message: str) -> bool:
        """Detect interpersonal workplace situations."""
        message_lower = message.strip().lower()
        if not message_lower:
            return False

        keywords = [
            "colleague", "coworker", "co-worker", "manager", "team", "conflict",
            "sarcastic", "rude", "disrespect", "behavior", "harassment",
            "collegue", "collègue", "equipe", "équipe", "conflit", "comportement",
            "sarcastique", "difficile", "tendu", "tension"
        ]

        return any(word in message_lower for word in keywords)

    def _is_followup(self, message: str, history: List[Dict[str, Any]]) -> bool:
        """
        Use LLM to determine if message is a follow-up requiring prior context.
        Language-agnostic approach - works for French, English, or any language.
        """
        # Need at least one prior exchange to have a follow-up
        if not history or len(history) < 2:
            return False
        
        # Get recent conversation context (last 2 exchanges max)
        recent_history = history[-4:] if len(history) >= 4 else history
        context = "\n".join([
            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg.get('content', '')[:200]}"
            for msg in recent_history
        ])
        
        prompt = f"""Given this conversation:
{context}

New user message: "{message}"

Is this new message a follow-up question that needs the previous context to be understood, or is it a completely new topic?

Answer only "FOLLOWUP" or "NEW_TOPIC"."""
        
        try:
            response = self.llm_service._call_groq(prompt, max_tokens=20, temperature=0.0)
            is_followup = response and "FOLLOWUP" in response.upper()
            logger.debug(f"LLM follow-up detection: '{message[:60]}' -> {response} -> {is_followup}")
            return is_followup
        except Exception as e:
            logger.error(f"Follow-up detection failed: {e}", exc_info=True)
            # Fallback: treat very short messages as potential follow-ups
            return len(message.split()) <= 5

    def _get_last_user_query(
        self,
        history: List[Dict[str, Any]],
        exclude_latest: bool = False,
    ) -> Optional[str]:
        """Return the most recent substantive user query."""
        if not history:
            return None

        items = history[:-1] if exclude_latest else history
        for item in reversed(items):
            if item.get("role") != "user":
                continue
            content = (item.get("content") or "").strip()
            if not content:
                continue
            if self._is_greeting(content) or self._is_thanks(content):
                continue
            return content
        return None

    def _get_last_assistant_response(self, history: List[Dict[str, Any]]) -> Optional[str]:
        """Return the most recent assistant response."""
        for item in reversed(history):
            if item.get("role") == "assistant":
                return (item.get("content") or "").strip()
        return None
    
    def _rewrite_with_context(self, query: str, prior_query: str, prior_response: str) -> str:
        """
        Use LLM to rewrite follow-up query as standalone query.
        Language-agnostic - works for any language (French, English, etc.).
        """
        # Truncate prior response to avoid token limits
        prior_response_short = prior_response[:500] if prior_response else ""
        
        prompt = f"""Previous user question: "{prior_query}"
Previous assistant response: "{prior_response_short}"

New user question: "{query}"

Rewrite the new question as a complete, standalone question that includes all necessary context from the previous exchange. The rewritten question should be understandable without seeing the conversation history. Keep the same language as the user's question.

Rewritten question:"""
        
        try:
            response = self.llm_service._call_groq(prompt, max_tokens=200, temperature=0.3)
            if response:
                logger.debug(f"Query rewritten: '{query[:50]}' -> '{response[:80]}'")
                return response
        except Exception as e:
            logger.error(f"Query rewriting failed: {e}", exc_info=True)
        
        # Fallback: simple concatenation
        return f"{prior_query}. Follow-up: {query}"
    
    def get_history(
        self,
        limit: int = 50,
        offset: int = 0,
        conversation_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history with pagination.
        
        Args:
            limit: Maximum number of messages to return
            offset: Number of messages to skip
        
        Returns:
            List of conversation messages
        """
        try:
            conversation_key = self._normalize_conversation_id(conversation_id)
            history = self._get_history_list(conversation_key)

            start = max(0, len(history) - limit - offset)
            end = max(0, len(history) - offset)

            history_slice = history[start:end]
            
            # Convert datetime objects to ISO format strings
            for msg in history_slice:
                if isinstance(msg.get('timestamp'), datetime):
                    msg['timestamp'] = msg['timestamp'].isoformat()
            
            return history_slice
        except Exception as e:
            logger.error(f"Error retrieving history: {e}")
            return []
    
    def clear_history(self, conversation_id: Optional[str] = None) -> bool:
        """Clear conversation history."""
        try:
            conversation_key = self._normalize_conversation_id(conversation_id)
            if conversation_id is None:
                self.conversation_history_by_id = {}
                self.last_context_by_id = {}
                self.last_context_items_by_id = {}
                logger.info("All conversation history cleared")
                return True

            self.conversation_history_by_id.pop(conversation_key, None)
            self.last_context_by_id.pop(conversation_key, None)
            self.last_context_items_by_id.pop(conversation_key, None)
            logger.info(f"Conversation history cleared for {conversation_key}")
            return True
        except Exception as e:
            logger.error(f"Error clearing history: {e}")
            return False
    
    def get_history_count(self, conversation_id: Optional[str] = None) -> int:
        """Get total number of messages in history."""
        conversation_key = self._normalize_conversation_id(conversation_id)
        return len(self._get_history_list(conversation_key))

    def _normalize_conversation_id(self, conversation_id: Optional[str]) -> str:
        cleaned = (conversation_id or "").strip()
        return cleaned or "default"

    def _get_history_list(self, conversation_id: str) -> List[Dict[str, Any]]:
        if conversation_id not in self.conversation_history_by_id:
            self.conversation_history_by_id[conversation_id] = []
        return self.conversation_history_by_id[conversation_id]

    def _get_history_window(
        self,
        history: List[Dict[str, Any]],
        max_messages: int = 12,
        exclude_latest: bool = False,
    ) -> List[Dict[str, Any]]:
        if not history:
            return []

        window = history[:-1] if exclude_latest and len(history) > 1 else history
        return window[-max_messages:]

    def _get_last_assistant_response(self, history: List[Dict[str, Any]]) -> Optional[str]:
        if not history:
            return None

        for item in reversed(history):
            if item.get("role") == "assistant":
                content = (item.get("content") or "").strip()
                return content or None
        return None


# Global chat service instance
_chat_service: Optional[ChatService] = None


def get_chat_service() -> ChatService:
    """Get or create the global chat service instance."""
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService()
    return _chat_service
