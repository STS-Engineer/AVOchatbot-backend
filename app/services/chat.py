"""
Chat service - main business logic orchestrating RAG and LLM.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from loguru import logger
from app.services.rag import get_rag_service
from app.services.llm import get_llm_service
from app.services.embedding import get_embedding_service


class ChatService:
    """Main chat service orchestrating RAG and LLM."""
    
    def __init__(self):
        """Initialize chat service."""
        self.rag_service = get_rag_service()
        self.llm_service = get_llm_service()
        self.embedding_service = get_embedding_service()
        self.conversation_history_by_id: Dict[str, List[Dict[str, Any]]] = {}
        # Cache context per conversation, but allow using conversation history when KB returns nothing
        self.last_context_by_id: Dict[str, Optional[str]] = {}
        self.last_context_items_by_id: Dict[str, List[Dict[str, Any]]] = {}
        logger.info("Chat Service initialized with intelligent KB routing")
    
    def process_message(
        self,
        message: str,
        top_k: Optional[int] = None,
        include_context: bool = True,
        conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Intelligent message processing: Let LLM decide when to search KB.
        Flow: Check conversation context → Ask LLM if KB needed → Search if needed → Generate response
        
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
            
            # Get conversation context
            history_for_prompt = self._get_history_window(history, exclude_latest=True)
            last_context = self.last_context_by_id.get(conversation_key, "")
            last_context_items = self.last_context_items_by_id.get(conversation_key, [])
            
            # STEP 1: Let LLM decide if we need to search KB
            needs_kb_search = self._should_search_kb(message, history_for_prompt, last_context)
            
            used_context_items: List[Dict[str, Any]] = []
            used_formatted_context = ""
            search_query = message
            
            # STEP 2: Search KB only if needed
            if needs_kb_search:
                logger.info(f"KB search needed for: '{message[:80]}'")
                
                # For follow-up questions, enhance query with conversation context
                if len(history_for_prompt) > 0:
                    enhanced_query = self._enhance_query_with_context(message, history_for_prompt)
                    if enhanced_query and enhanced_query != message:
                        search_query = enhanced_query
                        logger.info(f"Enhanced query: '{search_query[:120]}'")
                
                # Search KB
                formatted_context, context_items = self.rag_service.retrieve_context(search_query, k=top_k)
                
                if context_items:
                    used_context_items = context_items
                    used_formatted_context = formatted_context
                    # Cache for potential reuse
                    self.last_context_items_by_id[conversation_key] = context_items
                    self.last_context_by_id[conversation_key] = formatted_context
                    logger.info(f"Retrieved {len(context_items)} KB items")
                else:
                    logger.info("No KB results found - will use conversation context")
                    # Don't fail - let LLM use conversation history
            else:
                logger.info(f"KB search not needed - using conversation context: '{message[:80]}'")
                # LLM can answer from conversation history without KB
            
            # STEP 3: Generate response with available context
            response = self.llm_service.generate_response(
                message,
                context=used_formatted_context,
                conversation_history=history_for_prompt,
                has_kb_context=bool(used_context_items),
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
            
            logger.info(f"Message processed. KB items: {len(used_context_items)}")
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
        Edit a prior user message, truncate history, and regenerate response.
        Uses the same intelligent flow as process_message.
        """
        try:
            logger.info(f"Editing message at index {message_index}: {new_message[:100]}...")

            conversation_key = self._normalize_conversation_id(conversation_id)
            history = self._get_history_list(conversation_key)
            
            # Find all user message indices
            user_message_indices = [i for i, msg in enumerate(history) if msg.get('role') == 'user']
            
            if not user_message_indices:
                raise ValueError("No user messages to edit")
            
            if message_index < 0 or message_index >= len(user_message_indices):
                raise ValueError(f"Message index {message_index} out of range (history has {len(user_message_indices)} user messages)")
            
            # Get the actual history index and update message
            actual_history_index = user_message_indices[message_index]
            history[actual_history_index]["content"] = new_message
            history[actual_history_index]["timestamp"] = datetime.now()
            
            # Truncate everything after this message
            del history[actual_history_index + 1 :]

            # Clear cached context
            self.last_context_by_id.pop(conversation_key, None)
            self.last_context_items_by_id.pop(conversation_key, None)

            # Get conversation context
            history_for_prompt = self._get_history_window(history, exclude_latest=True)
            last_context = self.last_context_by_id.get(conversation_key, "")
            
            # Use same intelligent processing as regular messages
            needs_kb_search = self._should_search_kb(new_message, history_for_prompt, last_context)
            
            used_context_items: List[Dict[str, Any]] = []
            used_formatted_context = ""
            search_query = new_message
            
            if needs_kb_search:
                logger.info(f"KB search needed for edited message")
                
                # Enhance query if there's conversation context
                if len(history_for_prompt) > 0:
                    enhanced_query = self._enhance_query_with_context(new_message, history_for_prompt)
                    if enhanced_query and enhanced_query != new_message:
                        search_query = enhanced_query
                
                formatted_context, context_items = self.rag_service.retrieve_context(search_query, k=top_k)
                
                if context_items:
                    used_context_items = context_items
                    used_formatted_context = formatted_context
                    self.last_context_items_by_id[conversation_key] = context_items
                    self.last_context_by_id[conversation_key] = formatted_context
            else:
                logger.info(f"KB search not needed - using conversation context")

            # Generate response
            response = self.llm_service.generate_response(
                new_message,
                context=used_formatted_context,
                conversation_history=history_for_prompt,
                has_kb_context=bool(used_context_items),
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

            logger.info(f"Edited message processed. KB items: {len(used_context_items)}")
            return result

        except Exception as e:
            logger.error(f"Error editing message: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "An error occurred while editing your message. Please try again.",
                "timestamp": datetime.now().isoformat()
            }
    
    def _should_search_kb(self, message: str, history: List[Dict[str, Any]], last_context: str) -> bool:
        """
        Intelligent KB search decision: Let LLM decide if we need KB or can use conversation.
        Reduces hard-coding and makes the system more flexible.
        
        Returns:
            True if KB search needed, False if conversation context sufficient
        """
        # Simple heuristics for obvious cases (minimal hard-coding)
        message_lower = message.strip().lower()
        
        # Very short/casual messages probably don't need KB
        if len(message.split()) <= 2 and not any(q in message_lower for q in ['what', 'how', 'why', 'when', 'where', 'who']):
            return False
        
        # If no conversation history, search KB
        if not history or len(history) < 2:
            return True
        
        # Let LLM decide based on conversation context
        history_summary = self._format_recent_history(history[-4:])  # Last 2 exchanges
        
        prompt = f"""Analyze if this question needs knowledge base search or can be answered from conversation.

Conversation history:
{history_summary}

New question: "{message}"

Decision criteria:
- If question is NEW topic → KB_SEARCH
- If question is about SAME topic (follow-up, clarification, elaboration) → USE_CONVERSATION
- If asking for more details on what was just discussed → USE_CONVERSATION
- If question references "this", "that", "it", "these" → USE_CONVERSATION
- If completely different subject → KB_SEARCH

Respond with ONLY:
- "KB_SEARCH" if need to search knowledge base
- "USE_CONVERSATION" if can answer from conversation history"""
        
        try:
            response = self.llm_service._call_groq(prompt, max_tokens=20, temperature=0.0)
            if response and "USE_CONVERSATION" in response.upper():
                logger.info(f"LLM decision: Use conversation context (no KB search)")
                return False
            logger.info(f"LLM decision: Search KB")
            return True
        except Exception as e:
            logger.warning(f"KB decision failed, defaulting to search: {e}")
            return True  # Safe default
    
    def _enhance_query_with_context(self, message: str, history: List[Dict[str, Any]]) -> str:
        """
        Enhance vague follow-up queries with conversation context.
        Only when needed - minimal intervention.
        """
        # Get recent context
        recent = history[-2:] if len(history) >= 2 else history
        if not recent:
            return message
        
        history_summary = self._format_recent_history(recent)
        
        prompt = f"""Rewrite this follow-up question to be standalone and searchable.

Recent conversation:
{history_summary}

Follow-up question: "{message}"

Task: Rewrite to include the topic from conversation. Keep it concise (max 15 words).

Rewritten query:"""
        
        try:
            response = self.llm_service._call_groq(prompt, max_tokens=50, temperature=0.2)
            if response and len(response.strip()) > 0:
                return response.strip()
        except Exception as e:
            logger.warning(f"Query enhancement failed: {e}")
        
        return message
    
    def _format_recent_history(self, history: List[Dict[str, Any]]) -> str:
        """Format recent conversation for LLM context."""
        if not history:
            return "(no history)"
        
        lines = []
        for msg in history:
            role = msg.get('role', '').upper()
            content = msg.get('content', '').strip()
            if content:
                # Truncate long messages
                content = content[:200] + "..." if len(content) > 200 else content
                lines.append(f"{role}: {content}")
        
        return "\n".join(lines) if lines else "(no history)"

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
