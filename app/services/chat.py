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
        self.conversation_history: List[Dict[str, Any]] = []
        self.last_context: Optional[str] = None
        self.last_context_items: List[Dict[str, Any]] = []
        logger.info("Chat Service initialized")
    
    def process_message(
        self,
        message: str,
        top_k: Optional[int] = None,
        include_context: bool = True
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
            
            # Store user message in history
            self.conversation_history.append({
                "role": "user",
                "content": message,
                "timestamp": datetime.now()
            })
            
            # Check if this is a greeting - don't use RAG for simple greetings
            is_greeting = self._is_greeting(message)
            context_items: List[Dict[str, Any]] = []
            formatted_context = ""
            used_context_items: List[Dict[str, Any]] = []
            used_formatted_context = ""

            if is_greeting:
                response = self._greeting_response()
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now(),
                    "context_count": 0
                })

                return {
                    "success": True,
                    "message": response,
                    "context": None,
                    "context_items": None,
                    "context_count": 0,
                    "timestamp": datetime.now().isoformat()
                }
            
            if not is_greeting:
                is_followup = self._is_followup(message)

                # Prefer previous context for follow-ups to keep the same topic
                if is_followup and self.last_context_items:
                    used_context_items = self.last_context_items
                    used_formatted_context = self.last_context or ""
                    logger.info("Using previous context for follow-up message")
                else:
                    # Retrieve context from knowledge base only for non-follow-up queries
                    formatted_context, context_items = self.rag_service.retrieve_context(message, k=top_k)

                    used_context_items = context_items
                    used_formatted_context = formatted_context

                    # Update last context only when new context is found
                    if context_items:
                        self.last_context_items = context_items
                        self.last_context = formatted_context
            
            # Generate response using LLM
            allow_generic_guidance = not used_context_items and self._is_interpersonal(message)
            response = self.llm_service.generate_response(
                message,
                context=used_formatted_context,
                allow_generic_guidance=allow_generic_guidance,
            )
            
            # Store assistant response in history
            self.conversation_history.append({
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

        simple_greetings = {"hello", "hi", "hey", "yo", "sup", "greetings", "howdy"}
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

    def _is_followup(self, message: str) -> bool:
        """Detect if the message is a short follow-up like 'tell me more'."""
        message_lower = message.strip().lower()
        if not message_lower:
            return False

        import re
        cleaned = re.sub(r"[^a-z0-9\s]", " ", message_lower).strip()
        tokens = cleaned.split()

        if len(tokens) <= 6:
            followup_markers = {
                "more", "continue", "details", "detail", "elaborate", "expand",
                "explain", "tell", "talk", "about", "it", "that", "this",
                "ok", "okay", "please", "encore", "plus", "suite", "continue",
                "ca", "ce", "cela", "explique", "peux"
            }
            if any(token in followup_markers for token in tokens):
                return True

        if re.search(r"(tell\s+me\s+more|talk\s+more|more\s+about|continue|elaborate)", cleaned):
            return True

        if re.search(r"(plus\s+de\s+details|dis\s+m\s+en\s+plus|parle\s+plus|explique\s+plus)", cleaned):
            return True

        return False
    
    def get_history(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get conversation history with pagination.
        
        Args:
            limit: Maximum number of messages to return
            offset: Number of messages to skip
        
        Returns:
            List of conversation messages
        """
        try:
            start = max(0, len(self.conversation_history) - limit - offset)
            end = max(0, len(self.conversation_history) - offset)
            
            history = self.conversation_history[start:end]
            
            # Convert datetime objects to ISO format strings
            for msg in history:
                if isinstance(msg.get('timestamp'), datetime):
                    msg['timestamp'] = msg['timestamp'].isoformat()
            
            return history
        except Exception as e:
            logger.error(f"Error retrieving history: {e}")
            return []
    
    def clear_history(self) -> bool:
        """Clear conversation history."""
        try:
            self.conversation_history = []
            logger.info("Conversation history cleared")
            return True
        except Exception as e:
            logger.error(f"Error clearing history: {e}")
            return False
    
    def get_history_count(self) -> int:
        """Get total number of messages in history."""
        return len(self.conversation_history)


# Global chat service instance
_chat_service: Optional[ChatService] = None


def get_chat_service() -> ChatService:
    """Get or create the global chat service instance."""
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService()
    return _chat_service
