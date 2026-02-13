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
                    logger.info(f"Retrieved {len(context_items)} new context items for follow-up message")
                else:
                    # Fallback: try with just the prior query's topics
                    logger.info("No results with rewritten query, retrying with prior query as fallback")
                    if prior_user_query:
                        formatted_context, context_items = self.rag_service.retrieve_context(prior_user_query, k=top_k)
                        if context_items:
                            used_context_items = context_items
                            used_formatted_context = formatted_context
                            logger.info(f"Retrieved {len(context_items)} context items using prior query")
                        elif last_context_items:
                            # Last resort: use cached context
                            used_context_items = last_context_items
                            used_formatted_context = last_context or ""
                            logger.info("Using cached context from previous exchange")
                    elif last_context_items:
                        used_context_items = last_context_items
                        used_formatted_context = last_context or ""
                        logger.info("Using cached context (no prior query available)")
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
        Intelligently handles message indices by finding user messages only.
        """
        try:
            logger.info(f"Editing message at index {message_index}: {new_message[:100]}...")

            conversation_key = self._normalize_conversation_id(conversation_id)
            history = self._get_history_list(conversation_key)

            # Detailed logging for debugging
            logger.info(f"Edit request - conversation_id: {conversation_id}, normalized: {conversation_key}")
            logger.info(f"Edit request - message_index: {message_index}, history length: {len(history)}")
            logger.info(f"History roles: {[msg.get('role') for msg in history]}")
            
            # Find all user message indices
            user_message_indices = [i for i, msg in enumerate(history) if msg.get('role') == 'user']
            
            if not user_message_indices:
                logger.error("No user messages found in history")
                raise ValueError("No user messages to edit")
            
            # Map frontend index (user message number) to backend history index
            if message_index < 0 or message_index >= len(user_message_indices):
                logger.error(f"User message index {message_index} out of range (found {len(user_message_indices)} user messages)")
                raise ValueError(f"Message index {message_index} out of range (history has {len(user_message_indices)} user messages)")
            
            # Get the actual history index of the user message to edit
            actual_history_index = user_message_indices[message_index]
            logger.info(f"Mapping user message index {message_index} to history index {actual_history_index}")

            # Update the user message and truncate any following messages
            logger.info(f"Updating message at history index {actual_history_index} and truncating {len(history) - actual_history_index - 1} following messages")
            history[actual_history_index]["content"] = new_message
            history[actual_history_index]["timestamp"] = datetime.now()
            del history[actual_history_index + 1 :]

            # Reset cached context for this conversation
            self.last_context_by_id.pop(conversation_key, None)
            self.last_context_items_by_id.pop(conversation_key, None)

            context_items: List[Dict[str, Any]] = []
            formatted_context = ""
            used_context_items: List[Dict[str, Any]] = []
            used_formatted_context = ""
            
            # Check if this is a follow-up after edit
            is_followup = self._is_followup(new_message, history)

            if is_followup:
                # Get context from previous exchange for better query rewriting
                prior_user_query = self._get_last_user_query(history, exclude_latest=True)
                prior_assistant = self._get_last_assistant_response(history)
                
                if prior_user_query and prior_assistant:
                    logger.info(f"Follow-up detected after edit: '{new_message[:80]}'")
                    # Use intelligent rewriting with full context
                    combined_query = self._rewrite_with_context(new_message, prior_user_query, prior_assistant)
                    logger.info(f"Rewritten query for edit: '{combined_query[:120]}'")
                else:
                    logger.warning("Follow-up detected but context incomplete - using original query")
                    combined_query = new_message
                
                formatted_context, context_items = self.rag_service.retrieve_context(combined_query, k=top_k)

                if context_items:
                    used_context_items = context_items
                    used_formatted_context = formatted_context
                    self.last_context_items_by_id[conversation_key] = context_items
                    self.last_context_by_id[conversation_key] = formatted_context
                    logger.info(f"Retrieved {len(context_items)} context items for edited follow-up message")
                else:
                    # Fallback: try with just the prior query's topics
                    if prior_user_query:
                        logger.info("Retrying with prior query as fallback")
                        formatted_context, context_items = self.rag_service.retrieve_context(prior_user_query, k=top_k)
                        if context_items:
                            used_context_items = context_items
                            used_formatted_context = formatted_context
            else:
                # New query - retrieve fresh context
                logger.info(f"New query after edit (not a follow-up): '{new_message[:80]}'")
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
        Intelligent LLM-based follow-up detection using topic extraction and semantic analysis.
        No pattern matching - pure AI understanding of conversation continuity.
        """
        # Need at least one prior exchange to have a follow-up
        if not history or len(history) < 2:
            return False
        
        # Get the last user-assistant exchange for topic analysis
        last_user_msg = self._get_last_user_query(history, exclude_latest=True)
        last_assistant_msg = self._get_last_assistant_response(history)
        
        if not last_user_msg or not last_assistant_msg:
            logger.debug("No prior exchange found for follow-up detection")
            return False
        
        # Use LLM to extract the main topic from previous conversation and compare
        prompt = f"""Analyze this conversation to determine if the new message is a follow-up.

Previous exchange:
User: "{last_user_msg}"
Assistant: "{last_assistant_msg[:400]}"

New user message: "{message}"

Task: Determine if the new message is about the SAME TOPIC as the previous exchange or a DIFFERENT TOPIC.

Analysis criteria:
1. Topic Continuity: Does the new message discuss the same subject, concept, or domain?
   - Example: Previous about "SBA loans" → New asks "what are the steps" → SAME TOPIC (SBA)
   - Example: Previous about "leave policy" → New asks "what is health insurance" → DIFFERENT TOPIC

2. Semantic Relationship: Is the new question conceptually related to what was just discussed?
   - Asking for more details, steps, examples, clarifications → Usually SAME TOPIC
   - Introducing a new subject matter entirely → DIFFERENT TOPIC

3. Contextual Dependency: Does the new message need the previous context to be understood?
   - Uses references like "it", "that", "these steps" → SAME TOPIC
   - Self-contained question on different subject → DIFFERENT TOPIC

Be intelligent about topic boundaries:
- "What is X?" followed by "What about Y?" -> If X and Y are related concepts (e.g., both insurance types), treat as SAME TOPIC
- If Y is completely unrelated to X (different business domain), treat as DIFFERENT TOPIC

Respond with ONLY one of these exact phrases:
- "SAME_TOPIC" if the new message continues the previous conversation
- "DIFFERENT_TOPIC" if the new message introduces a new subject"""
        
        try:
            # Increased max_tokens to give model more room to respond reliably
            response = self.llm_service._call_groq(prompt, max_tokens=50, temperature=0.0)
            
            # Handle None or empty responses gracefully
            if not response or not response.strip():
                logger.warning(f"LLM returned empty response for follow-up detection, using fallback")
                return self._is_followup_by_similarity(message, last_user_msg, last_assistant_msg)
            
            is_followup = "SAME_TOPIC" in response.upper()
            
            logger.info(f"Intelligent follow-up detection: '{message[:60]}' | LLM: {response.strip()} | Result: {'FOLLOWUP' if is_followup else 'NEW_QUERY'}")
            return is_followup
            
        except Exception as e:
            logger.error(f"LLM-based follow-up detection failed: {e}", exc_info=True)
            # Fallback: Use semantic similarity with embeddings
            return self._is_followup_by_similarity(message, last_user_msg, last_assistant_msg)

    def _is_followup_by_similarity(
        self, 
        message: str, 
        last_user_msg: str, 
        last_assistant_msg: str
    ) -> bool:
        """
        Fallback: Use embedding similarity to detect follow-ups when LLM fails.
        Compares semantic similarity between current and previous messages.
        Also uses linguistic heuristics for generic follow-up phrases.
        """
        # First check: Is the message a generic continuation phrase?
        message_lower = message.lower().strip()
        generic_followup_patterns = [
            "what are the", "how does", "can you explain", "tell me more",
            "what about the", "and the", "more details", "steps", "process",
            "quelles sont les", "comment", "expliquez", "étapes", "processus",
            "et les", "plus de détails"
        ]
        
        # If message is generic AND short, it's likely a follow-up
        is_generic = any(pattern in message_lower for pattern in generic_followup_patterns)
        is_short = len(message.split()) <= 10
        
        if is_generic and is_short:
            logger.info(f"Generic follow-up phrase detected: '{message[:60]}' -> FOLLOWUP")
            return True
        
        try:
            # Embed current message and previous context
            current_embedding = self.embedding_service.embed_text(message)
            # Combine previous user and assistant for context
            previous_context = f"{last_user_msg} {last_assistant_msg[:300]}"
            previous_embedding = self.embedding_service.embed_text(previous_context)
            
            # Calculate cosine similarity
            import numpy as np
            current_vec = np.array(current_embedding)
            previous_vec = np.array(previous_embedding)
            
            similarity = np.dot(current_vec, previous_vec) / (
                np.linalg.norm(current_vec) * np.linalg.norm(previous_vec)
            )
            
            # Adaptive threshold: lower for generic questions
            base_threshold = 0.50  # Lowered from 0.65 to catch more follow-ups
            threshold = 0.35 if is_generic else base_threshold
            is_followup = similarity > threshold
            
            logger.info(f"Similarity-based follow-up detection: {similarity:.3f} (threshold: {threshold}, generic: {is_generic}) -> {'FOLLOWUP' if is_followup else 'NEW_QUERY'}")
            return is_followup
            
        except Exception as e:
            logger.error(f"Similarity-based follow-up detection failed: {e}", exc_info=True)
            # Last resort: generic questions are likely follow-ups if recent context exists
            if is_generic:
                logger.debug(f"Fallback: treating generic question as FOLLOWUP")
                return True
            # Very short messages might be follow-ups
            is_followup = len(message.split()) <= 5
            logger.debug(f"Final fallback: message length {len(message.split())} words -> {'FOLLOWUP' if is_followup else 'NEW_QUERY'}")
            return is_followup
    
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
        Use LLM to rewrite follow-up query as standalone query with topic preservation.
        Language-agnostic - works for any language (French, English, etc.).
        Intelligently extracts topics and maintains conversation context.
        """
        # Truncate prior response but keep enough context
        prior_response_short = prior_response[:800] if prior_response else ""
        
        prompt = f"""Conversation context:
User previously asked: "{prior_query}"
Assistant responded about: "{prior_response_short}"

New user question: "{query}"

Task: Rewrite the new question to be a complete, standalone search query that:
1. Preserves the EXACT TOPIC from the previous conversation
2. Incorporates the specific aspect the user is now asking about
3. Maintains the same language as the user's question
4. Is optimized for searching a knowledge base

Important: Keep the topic focus from the previous exchange. If the previous conversation was about "SBA", the rewritten query must also be about "SBA".

Rewritten standalone search query:"""
        
        try:
            response = self.llm_service._call_groq(prompt, max_tokens=250, temperature=0.2)
            if response:
                rewritten = response.strip()
                logger.info(f"Query rewritten: '{query[:60]}' -> '{rewritten[:100]}'")
                
                # Validate the rewrite maintains topic relevance
                # Extract key terms from prior query
                prior_terms = set(prior_query.lower().split())
                rewritten_terms = set(rewritten.lower().split())
                
                # Check if rewrite shares some key terms with prior context
                if len(prior_terms & rewritten_terms) > 0 or len(rewritten.split()) > len(query.split()):
                    return rewritten
                else:
                    logger.warning(f"Rewrite lost context, using enhanced fallback")
                    # Better fallback that preserves topic
                    return f"{prior_query} - {query}"
            
        except Exception as e:
            logger.error(f"Query rewriting failed: {e}", exc_info=True)
        
        # Enhanced fallback: preserve topic from prior query
        logger.info("Using fallback query combining prior and current")
        return f"{prior_query} - {query}"
    
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
