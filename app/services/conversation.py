"""
Conversation service - Database persistence for conversations and messages.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy import text
from loguru import logger
from app.services.database import get_database


class ConversationService:
    """Handles conversation and message persistence."""
    
    def __init__(self):
        """Initialize conversation service."""
        self.db = get_database()
        logger.info("Conversation service initialized")
    
    def create_conversation(self, user_id: str, title: Optional[str] = None) -> Optional[str]:
        """
        Create a new conversation for a user.
        Ensures user exists in knowledge_DB.users before creating conversation.
        """
        try:
            # Pre-conversation check: ensure user exists in users table
            from app.services.user import get_user_service
            user_service = get_user_service()
            user = user_service.get_user_by_id(user_id)
            if not user:
                logger.error(f"Cannot create conversation: user_id {user_id} not found in knowledge_DB.users.")
                return None
            with self.db.get_session() as session:
                result = session.execute(
                    text("""
                    INSERT INTO conversations (user_id, title, created_at, updated_at)
                    VALUES (:user_id, :title, :created_at, :updated_at)
                    RETURNING id
                    """),
                    {
                        "user_id": user_id,
                        "title": title or "New Conversation",
                        "created_at": datetime.utcnow(),
                        "updated_at": datetime.utcnow()
                    }
                )
                session.commit()
                conversation = result.fetchone()
                if conversation:
                    conversation_id = str(conversation[0])
                    logger.info(f"Conversation created: {conversation_id} for user {user_id}")
                    return conversation_id
                return None
        except Exception as e:
            logger.error(f"Error creating conversation: {e}")
            return None
        except Exception as e:
            logger.error(f"Error creating conversation: {e}")
            return None
    
    def get_conversation(self, conversation_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a conversation by ID (validates ownership).
        
        Args:
            conversation_id: Conversation ID
            user_id: User ID (for ownership validation)
        
        Returns:
            Conversation data if found and owned by user, None otherwise
        """
        try:
            with self.db.get_session() as session:
                result = session.execute(
                    text("""
                    SELECT id, user_id, title, created_at, updated_at, is_archived
                    FROM conversations
                    WHERE id = :conversation_id AND user_id = :user_id
                    """),
                    {"conversation_id": conversation_id, "user_id": user_id}
                )
                conversation = result.fetchone()
                return dict(conversation._mapping) if conversation else None
        
        except Exception as e:
            logger.error(f"Error getting conversation: {e}")
            return None
    
    def list_conversations(self, user_id: str, include_archived: bool = False) -> List[Dict[str, Any]]:
        """
        List all conversations for a user.
        
        Args:
            user_id: User ID
            include_archived: Whether to include archived conversations
        
        Returns:
            List of conversations
        """
        try:
            with self.db.get_session() as session:
                query = """
                SELECT id, title, created_at, updated_at, is_archived,
                       (SELECT COUNT(*) FROM messages WHERE conversation_id = conversations.id) as message_count
                FROM conversations
                WHERE user_id = :user_id
                """
                
                if not include_archived:
                    query += " AND is_archived = FALSE"
                
                query += " ORDER BY updated_at DESC"
                
                result = session.execute(text(query), {"user_id": user_id})
                conversations = [dict(row._mapping) for row in result.fetchall()]
                
                logger.info(f"Retrieved {len(conversations)} conversations for user {user_id}")
                return conversations
        
        except Exception as e:
            logger.error(f"Error listing conversations: {e}")
            return []
    
    def update_conversation_title(self, conversation_id: str, user_id: str, title: str) -> bool:
        """Update conversation title."""
        try:
            with self.db.get_session() as session:
                session.execute(
                    text("""
                    UPDATE conversations
                    SET title = :title, updated_at = :updated_at
                    WHERE id = :conversation_id AND user_id = :user_id
                    """),
                    {
                        "conversation_id": conversation_id,
                        "user_id": user_id,
                        "title": title,
                        "updated_at": datetime.utcnow()
                    }
                )
                session.commit()
                return True
        
        except Exception as e:
            logger.error(f"Error updating conversation title: {e}")
            return False
    
    def delete_conversation(self, conversation_id: str, user_id: str) -> bool:
        """Delete a conversation (cascade deletes messages)."""
        try:
            with self.db.get_session() as session:
                session.execute(
                    text("""
                    DELETE FROM conversations
                    WHERE id = :conversation_id AND user_id = :user_id
                    """),
                    {"conversation_id": conversation_id, "user_id": user_id}
                )
                session.commit()
                logger.info(f"Conversation deleted: {conversation_id}")
                return True
        
        except Exception as e:
            logger.error(f"Error deleting conversation: {e}")
            return False
    
    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        context_used: Optional[str] = None,
        context_count: int = 0
    ) -> Optional[str]:
        """
        Add a message to a conversation.
        
        Args:
            conversation_id: Conversation ID
            role: Message role ('user' or 'assistant')
            content: Message content
            context_used: Optional context that was used (for assistant messages)
            context_count: Number of context items used
        
        Returns:
            message_id if created successfully, None otherwise
        """
        try:
            with self.db.get_session() as session:
                result = session.execute(
                    text("""
                    INSERT INTO messages (conversation_id, role, content, context_used, context_count, created_at)
                    VALUES (:conversation_id, :role, :content, :context_used, :context_count, :created_at)
                    RETURNING id
                    """),
                    {
                        "conversation_id": conversation_id,
                        "role": role,
                        "content": content,
                        "context_used": context_used,
                        "context_count": context_count,
                        "created_at": datetime.utcnow()
                    }
                )
                
                # Update conversation's updated_at timestamp
                session.execute(
                    text("""
                    UPDATE conversations
                    SET updated_at = :updated_at
                    WHERE id = :conversation_id
                    """),
                    {"conversation_id": conversation_id, "updated_at": datetime.utcnow()}
                )
                
                session.commit()
                
                message = result.fetchone()
                if message:
                    message_id = str(message[0])
                    logger.debug(f"Message added: {message_id} to conversation {conversation_id}")
                    return message_id
                
                return None
        
        except Exception as e:
            logger.error(f"Error adding message: {e}")
            return None
    
    def get_messages(
        self,
        conversation_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get messages for a conversation.
        
        Args:
            conversation_id: Conversation ID
            limit: Maximum number of messages to retrieve
            offset: Offset for pagination
        
        Returns:
            List of messages
        """
        try:
            with self.db.get_session() as session:
                result = session.execute(
                    text("""
                    SELECT id, role, content, context_used, context_count, created_at, edited_at, is_edited
                    FROM messages
                    WHERE conversation_id = :conversation_id
                    ORDER BY created_at ASC
                    LIMIT :limit OFFSET :offset
                    """),
                    {
                        "conversation_id": conversation_id,
                        "limit": limit,
                        "offset": offset
                    }
                )
                
                messages = [dict(row._mapping) for row in result.fetchall()]
                return messages
        
        except Exception as e:
            logger.error(f"Error getting messages: {e}")
            return []
    
    def get_message_count(self, conversation_id: str) -> int:
        """Get total message count for a conversation."""
        try:
            with self.db.get_session() as session:
                result = session.execute(
                    text("""
                    SELECT COUNT(*) FROM messages
                    WHERE conversation_id = :conversation_id
                    """),
                    {"conversation_id": conversation_id}
                )
                count = result.fetchone()
                return count[0] if count else 0
        
        except Exception as e:
            logger.error(f"Error getting message count: {e}")
            return 0
    
    def update_message(self, message_id: str, content: str) -> bool:
        """Update a message's content."""
        try:
            with self.db.get_session() as session:
                session.execute(
                    text("""
                    UPDATE messages
                    SET content = :content, edited_at = :edited_at, is_edited = TRUE
                    WHERE id = :message_id
                    """),
                    {
                        "message_id": message_id,
                        "content": content,
                        "edited_at": datetime.utcnow()
                    }
                )
                session.commit()
                return True
        
        except Exception as e:
            logger.error(f"Error updating message: {e}")
            return False
    
    def delete_messages_after(self, conversation_id: str, message_id: str) -> bool:
        """Delete all messages after a specific message (for edit functionality)."""
        try:
            with self.db.get_session() as session:
                # Get the timestamp of the message to edit
                result = session.execute(
                    text("""
                    SELECT created_at FROM messages
                    WHERE id = :message_id
                    """),
                    {"message_id": message_id}
                )
                message = result.fetchone()
                
                if not message:
                    return False
                
                message_time = message[0]
                
                # Delete all messages after this timestamp
                session.execute(
                    text("""
                    DELETE FROM messages
                    WHERE conversation_id = :conversation_id
                    AND created_at > :message_time
                    """),
                    {"conversation_id": conversation_id, "message_time": message_time}
                )
                session.commit()
                return True
        
        except Exception as e:
            logger.error(f"Error deleting messages after: {e}")
            return False
    
    def store_message_context_items(
        self,
        message_id: str,
        context_items: List[Dict[str, Any]]
    ) -> bool:
        """Store context items that were used for a message."""
        try:
            with self.db.get_session() as session:
                for idx, item in enumerate(context_items):
                    similarity = item.get("similarity")
                    logger.debug(f"Storing context item {idx}: id={item.get('id')}, similarity={similarity}, item_keys={list(item.keys())}")
                    session.execute(
                        text("""
                        INSERT INTO message_context_items (message_id, node_id, similarity_score, position)
                        VALUES (:message_id, :node_id, :similarity_score, :position)
                        """),
                        {
                            "message_id": message_id,
                            "node_id": item.get("id"),
                            "similarity_score": similarity,
                            "position": idx
                        }
                    )
                session.commit()
                return True
        
        except Exception as e:
            logger.error(f"Error storing message context items: {e}")
            return False


# Singleton instance
_conversation_service = None


def get_conversation_service() -> ConversationService:
    """Get or create conversation service instance."""
    global _conversation_service
    if _conversation_service is None:
        _conversation_service = ConversationService()
    return _conversation_service
