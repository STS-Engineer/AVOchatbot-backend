"""
Database connection and queries service.
"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
from typing import List, Dict, Any, Optional
from loguru import logger
from app.core.config import settings


class DatabaseService:
    """Handles database connections and queries."""
    def __init__(self, database_url: str = None, sslmode: str = None, db_host: str = None, db_port: int = None, db_name: str = None):
        """Initialize database connection. Defaults to main knowledge base DB."""
        self.engine = None
        self.SessionLocal = None
        self._database_url = database_url or settings.database_url
        self._sslmode = sslmode or settings.DB_SSLMODE
        self._db_host = db_host or settings.DB_HOST
        self._db_port = db_port or settings.DB_PORT
        self._db_name = db_name or settings.DB_NAME
        self._init_connection()

    def _init_connection(self):
        """Initialize the database engine."""
        try:
            connect_args = {
                "sslmode": self._sslmode,
                "connect_timeout": 10
            }
            self.engine = create_engine(
                self._database_url,
                connect_args=connect_args,
                echo=False,
                pool_pre_ping=True,
                pool_size=5,
                max_overflow=10
            )
            self.SessionLocal = sessionmaker(bind=self.engine)
            logger.info(f"Database connection initialized: {self._db_host}:{self._db_port}/{self._db_name}")
        except Exception as e:
            logger.error(f"Failed to initialize database connection: {e}")
            raise
    
    @contextmanager
    def get_session(self):
        """Context manager for database sessions."""
        session = self.SessionLocal()
        try:
            yield session
        finally:
            session.close()
    
    def test_connection(self) -> bool:
        """Test if the database connection is working."""
        try:
            with self.get_session() as session:
                result = session.execute(text("SELECT 1"))
                return result.fetchone() is not None
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def search_by_similarity(self, embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """Search knowledge nodes by vector similarity."""
        try:
            with self.get_session() as session:
                # Convert embedding list to PostgreSQL vector format
                embedding_str = "[" + ",".join(map(str, embedding)) + "]"
                
                result = session.execute(
                    text("""
                    SELECT 
                        id, 
                        parent_id, 
                        title, 
                        slug, 
                        node_type, 
                        structured_data,
                        (1 - (embedding <=> :embedding)) as similarity
                    FROM public.knowledge_node
                    WHERE status = 'draft'
                    ORDER BY embedding <=> :embedding
                    LIMIT :limit
                    """),
                    {
                        "embedding": embedding_str,
                        "limit": limit
                    }
                )
                rows = result.fetchall()
                return [dict(row._mapping) if hasattr(row, '_mapping') else dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error searching by similarity: {e}")
            return []
    
    def get_node_by_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a knowledge node by its ID."""
        try:
            with self.get_session() as session:
                result = session.execute(
                    text("""
                    SELECT * FROM public.knowledge_node 
                    WHERE id = :node_id
                    """),
                    {"node_id": node_id}
                )
                row = result.fetchone()
                if row:
                    return dict(row._mapping) if hasattr(row, '_mapping') else dict(row)
                return None
        except Exception as e:
            logger.error(f"Error fetching knowledge node: {e}")
            return None
    
    def search_by_title(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search knowledge nodes by title."""
        try:
            with self.get_session() as session:
                result = session.execute(
                    text("""
                    SELECT 
                        id, 
                        parent_id, 
                        title, 
                        slug, 
                        node_type, 
                        structured_data
                    FROM public.knowledge_node
                    WHERE status = 'draft' 
                    AND (LOWER(title) LIKE LOWER(:query) 
                         OR LOWER(slug) LIKE LOWER(:query))
                    LIMIT :limit
                    """),
                    {
                        "query": f"%{query}%",
                        "limit": limit
                    }
                )
                rows = result.fetchall()
                return [dict(row._mapping) if hasattr(row, '_mapping') else dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error searching by title: {e}")
            return []
    
    def search_by_keyword(self, keyword: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search knowledge nodes by keyword in title or structured_data."""
        try:
            with self.get_session() as session:
                keyword_pattern = f"%{keyword}%"
                result = session.execute(
                    text("""
                    SELECT 
                        id, 
                        parent_id, 
                        title, 
                        slug, 
                        node_type, 
                        structured_data
                    FROM public.knowledge_node
                    WHERE status = 'draft' 
                    AND (LOWER(title) LIKE LOWER(:keyword)
                         OR LOWER(structured_data::text) LIKE LOWER(:keyword))
                    LIMIT :limit
                    """),
                    {
                        "keyword": keyword_pattern,
                        "limit": limit
                    }
                )
                rows = result.fetchall()
                return [dict(row._mapping) if hasattr(row, '_mapping') else dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error searching by keyword: {e}")
            return []
    
    def get_child_nodes(self, parent_id: str) -> List[Dict[str, Any]]:
        """Get all child nodes for a parent node."""
        try:
            with self.get_session() as session:
                result = session.execute(
                    text("""
                    SELECT 
                        id, 
                        parent_id, 
                        title, 
                        slug, 
                        node_type, 
                        structured_data
                    FROM public.knowledge_node
                    WHERE parent_id = :parent_id
                    AND status = 'draft'
                    """),
                    {"parent_id": parent_id}
                )
                rows = result.fetchall()
                return [dict(row._mapping) if hasattr(row, '_mapping') else dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error fetching child nodes: {e}")
            return []

    def get_root_nodes(self) -> List[Dict[str, Any]]:
        """Get all root nodes (domains) with no parent."""
        try:
            with self.get_session() as session:
                result = session.execute(
                    text("""
                    SELECT
                        id,
                        parent_id,
                        title,
                        slug,
                        node_type,
                        structured_data
                    FROM public.knowledge_node
                    WHERE parent_id IS NULL
                    AND status = 'draft'
                    """)
                )
                rows = result.fetchall()
                return [dict(row._mapping) if hasattr(row, '_mapping') else dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error fetching root nodes: {e}")
            return []
    
    def get_attachments_for_node(self, node_id: str) -> List[Dict[str, Any]]:
        """Get all attachments for a knowledge node."""
        try:
            with self.get_session() as session:
                result = session.execute(
                    text("""
                    SELECT 
                        id, 
                        node_id, 
                        file_name, 
                        file_type, 
                        file_path,
                        uploaded_at
                    FROM public.knowledge_attachment
                    WHERE node_id = :node_id
                    ORDER BY uploaded_at DESC
                    """),
                    {"node_id": node_id}
                )
                rows = result.fetchall()
                attachments = []
                for row in rows:
                    att = dict(row._mapping) if hasattr(row, '_mapping') else dict(row)
                    # Convert UUID to string
                    if 'id' in att and hasattr(att['id'], '__str__'):
                        att['id'] = str(att['id'])
                    if 'node_id' in att and hasattr(att['node_id'], '__str__'):
                        att['node_id'] = str(att['node_id'])
                    # Convert datetime to ISO format string
                    if 'uploaded_at' in att and hasattr(att['uploaded_at'], 'isoformat'):
                        att['uploaded_at'] = att['uploaded_at'].isoformat()
                    attachments.append(att)
                return attachments
        except Exception as e:
            logger.error(f"Error fetching attachments: {e}")
            return []


# Global database instance
_db_instance: Optional[DatabaseService] = None

# Global central users database instance
_users_db_instance: Optional[DatabaseService] = None


def get_database() -> DatabaseService:
    """Get or create the global database service instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseService()
    return _db_instance

def get_users_database() -> DatabaseService:
    """Get or create the global central users database service instance (chatbots_users DB)."""
    global _users_db_instance
    if _users_db_instance is None:
        _users_db_instance = DatabaseService(
            database_url=settings.users_database_url,
            sslmode=settings.USERS_DB_SSLMODE,
            db_host=settings.USERS_DB_HOST,
            db_port=settings.USERS_DB_PORT,
            db_name=settings.USERS_DB_NAME
        )
    return _users_db_instance
