"""
RAG (Retrieval Augmented Generation) service.
"""

from typing import List, Dict, Any, Tuple, Optional
from loguru import logger
from app.core.config import settings
from app.services.database import get_database
from app.services.embedding import get_embedding_service
from app.services.llm import get_llm_service


class RAGService:
    """Retrieval Augmented Generation service."""
    
    def __init__(self, top_k: int = 8, similarity_threshold: float = 0.2):
        """Initialize RAG service."""
        self.db = get_database()
        self.embedding_service = get_embedding_service()
        self.llm_service = get_llm_service()
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        # For semantic search, use a slightly lower threshold to catch synonyms and related concepts
        self.semantic_threshold = max(0.15, similarity_threshold - 0.05)
        logger.info(f"RAG Service initialized with top_k={top_k}, threshold={similarity_threshold}, semantic_threshold={self.semantic_threshold}")
    
    def retrieve_context(self, query: str, k: Optional[int] = None) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Retrieve relevant context using intelligent semantic search.
        The embedding model handles synonym and context understanding automatically.
        
        Returns:
            Tuple of (formatted_context, context_items)
        """
        try:
            k = k or self.top_k
            logger.info(f"Retrieving context for: '{query}'")
            
            # Build query variants (original + translated if needed)
            query_variants = [query]
            translated_query = None
            if self._should_translate(query):
                translated_query = self.llm_service.translate_to_english(query)
                if translated_query and translated_query.lower().strip() != query.lower().strip():
                    query_variants.append(translated_query)
                    logger.info(f"Added translated query for retrieval: '{translated_query[:120]}'")
            
            logger.info(f"Using embedding model for semantic understanding: '{query}'")
            
            all_results = {}  # Store by ID to avoid duplicates
            match_scores = {}  # Track match quality
            
            # STRATEGY 1: Exact/Title Match (highest priority)
            logger.debug("[STRATEGY 1] Searching for title match...")
            for variant in query_variants:
                clean_query = self._clean_query(variant)
                title_results = self.db.search_by_title(clean_query, limit=k*2)
                for result in title_results:
                    result_id = result.get('id')
                    if result_id not in all_results:
                        all_results[result_id] = result
                    match_scores[result_id] = max(match_scores.get(result_id, 0), 100)
                    logger.debug(f"Title match: {result.get('title')}")
            
            # If exact match found, use it
            if all_results:
                sorted_results = sorted(
                    all_results.values(),
                    key=lambda x: match_scores.get(x.get('id'), 0),
                    reverse=True
                )[:k]

                sorted_results = self._expand_with_children(sorted_results, match_scores, k)
                
                enriched_results = [self._enrich_result(r) for r in sorted_results]
                
                # Add similarity score to enriched results (1.0 for title matches)
                for result in enriched_results:
                    result_id = result.get('id')
                    score = match_scores.get(result_id, 100)
                    result['similarity'] = 1.0
                
                formatted_context = self._format_context(enriched_results)
                logger.info(f"Retrieved {len(enriched_results)} title matches")
                return formatted_context, enriched_results
            
            # STRATEGY 2: Semantic Similarity Search (uses embeddings for synonyms and context)
            logger.debug("[STRATEGY 2] Semantic search using embeddings (handles synonyms automatically)...")
            try:
                for variant in query_variants:
                    # Embed each variant (original + translated)
                    query_embedding = self.embedding_service.embed_text(variant)

                    similarity_results = self.db.search_by_similarity(query_embedding, limit=k*3)

                    for result in similarity_results:
                        result_id = result.get('id')
                        similarity = result.get('similarity', 0)

                        # Use semantic_threshold which is slightly lower to catch related concepts
                        if similarity >= self.semantic_threshold:
                            all_results[result_id] = result
                            match_scores[result_id] = max(match_scores.get(result_id, 0), 60 + (similarity * 20))
                            logger.debug(f"Semantic match: {result.get('title')} (similarity: {similarity:.2%})")

                            # Log high-quality matches for understanding what the model found
                            if similarity >= 0.7:
                                logger.info(f"Strong semantic match: {result.get('title')} (similarity: {similarity:.2%})")
            except Exception as e:
                logger.warning(f"Error in semantic search: {e}")
            
            
            
            # Sort by score and return top k
            if all_results:
                sorted_results = sorted(
                    all_results.values(),
                    key=lambda x: match_scores.get(x.get('id'), 0),
                    reverse=True
                )[:k]

                sorted_results = self._expand_with_children(sorted_results, match_scores, k)
                
                enriched_results = [self._enrich_result(r) for r in sorted_results]
                
                # Add similarity score to enriched results
                for i, result in enumerate(enriched_results):
                    result_id = result.get('id')
                    # Normalize score to 0-1 range (was 40-100)
                    score = match_scores.get(result_id, 0)
                    result['similarity'] = min(1.0, score / 100.0)
                
                formatted_context = self._format_context(enriched_results)
                logger.info(f"Retrieved {len(enriched_results)} semantic matches")
                return formatted_context, enriched_results
            
            # STRATEGY 3: Keyword Search (last resort, for exact term matching in text)
            logger.debug("[STRATEGY 3] Searching by keyword...")
            for variant in query_variants:
                clean_query = self._clean_query(variant)
                main_term = clean_query.split()[-1] if clean_query else variant.lower()
                keyword_results = self.db.search_by_keyword(main_term, limit=k*2)
                for result in keyword_results:
                    result_id = result.get('id')
                    if result_id not in all_results:
                        all_results[result_id] = result
                        match_scores[result_id] = 40
                        logger.debug(f"Keyword match: {result.get('title')}")
            
            # No results found, try domain classification fallback
            if not all_results:
                fallback_items = self._fallback_to_domain(query)
                for item in fallback_items:
                    item_id = item.get('id')
                    if item_id:
                        all_results[item_id] = item
                        match_scores[item_id] = 75

            if not all_results:
                logger.warning(f"No relevant context found for: {query}")
                return "", []
            
            sorted_results = sorted(
                all_results.values(),
                key=lambda x: match_scores.get(x.get('id'), 0),
                reverse=True
            )[:k]

            sorted_results = self._expand_with_children(sorted_results, match_scores, k)
            
            enriched_results = [self._enrich_result(r) for r in sorted_results]
            
            # Add similarity score to enriched results
            for i, result in enumerate(enriched_results):
                result_id = result.get('id')
                # Normalize score to 0-1 range (was 40-100)
                score = match_scores.get(result_id, 0)
                result['similarity'] = min(1.0, score / 100.0)
            
            formatted_context = self._format_context(enriched_results)
            logger.info(f"Retrieved {len(enriched_results)} keyword matches")
            return formatted_context, enriched_results
        
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return "", []

    def _expand_with_children(
        self,
        results: List[Dict[str, Any]],
        match_scores: Dict[str, float],
        k: int
    ) -> List[Dict[str, Any]]:
        """Add child nodes for matched parents to enrich context."""
        if not results:
            return results

        seen_ids = {item.get('id') for item in results if item.get('id')}
        parent_ids = [item.get('id') for item in results if item.get('id')]

        for parent_id in parent_ids:
            child_nodes = self.db.get_child_nodes(parent_id)
            if not child_nodes:
                continue

            parent_score = match_scores.get(parent_id, 60)
            child_score = max(10, parent_score - 15)

            for child in child_nodes:
                child_id = child.get('id')
                if not child_id or child_id in seen_ids:
                    continue
                match_scores[child_id] = child_score
                results.append(child)
                seen_ids.add(child_id)

        return results

    def _fallback_to_domain(self, query: str) -> List[Dict[str, Any]]:
        """Fallback: classify query to a domain and return it if found."""
        root_nodes = self.db.get_root_nodes()
        if not root_nodes:
            return []

        candidates = [node.get('title') for node in root_nodes if node.get('title')]
        if not candidates:
            return []

        selected = self.llm_service.classify_domain(query, candidates)
        if not selected:
            return []

        selected_lower = selected.strip().lower()
        for node in root_nodes:
            if (node.get('title') or '').strip().lower() == selected_lower:
                logger.info(f"Domain fallback matched: {node.get('title')}")
                return [node]

        # Fallback to title search if exact match not found
        title_results = self.db.search_by_title(selected, limit=1)
        if title_results:
            logger.info(f"Domain fallback matched via title search: {title_results[0].get('title')}")
            return [title_results[0]]

        return []

    def _clean_query(self, query: str) -> str:
        """Normalize query for title/keyword search."""
        import re
        clean_query = re.sub(r'^(what\s+is|who\s+is|explain|tell\s+me|describe)\s+', '', query.lower(), flags=re.IGNORECASE)
        clean_query = re.sub(r'\?$', '', clean_query).strip()
        return clean_query

    def _should_translate(self, query: str) -> bool:
        """Detect non-English queries (lightweight heuristic)."""
        if not query:
            return False

        lowered = query.lower()
        french_markers = [
            "bonjour", "merci", "retard", "paiement", "equipe", "équipe", "difficile",
            "sarcastique", "situation", "problème", "probleme", "client", "facture",
            "responsabilité", "responsabilite", "confiance", "valeurs"
        ]
        if any(marker in lowered for marker in french_markers):
            return True

        # Detect common French diacritics
        if any(ch in lowered for ch in "àâäçéèêëîïôöùûüÿœ"): 
            return True

        return False
    
    def _enrich_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich result with additional information including attachments."""
        try:
            # Convert UUID fields to strings
            from uuid import UUID
            if 'id' in result and isinstance(result['id'], UUID):
                result['id'] = str(result['id'])
            if 'parent_id' in result and isinstance(result['parent_id'], UUID):
                result['parent_id'] = str(result['parent_id'])
            
            # Extract content from structured data if available
            structured_data = result.get('structured_data', {})
            if isinstance(structured_data, dict):
                explanation = structured_data.get('explanation', '')
                if explanation:
                    result['content'] = explanation
            
            # MIME type mapping
            mime_types = {
                '.pdf': 'application/pdf',
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.gif': 'image/gif',
                '.bmp': 'image/bmp',
                '.svg': 'image/svg+xml',
                '.webp': 'image/webp',
                '.doc': 'application/msword',
                '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                '.xls': 'application/vnd.ms-excel',
                '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                '.txt': 'text/plain',
                '.csv': 'text/csv',
                '.zip': 'application/zip',
            }
            
            def get_mime_type(file_path: str) -> str:
                """Detect MIME type from file extension."""
                if not file_path:
                    return 'unknown'
                ext = '.' + file_path.lower().split('.')[-1]
                return mime_types.get(ext, 'unknown')
            
            # Fetch attachments for this node and child nodes
            node_id = result.get('id')
            all_attachments = []
            
            if node_id:
                # Get attachments for this node
                attachments = self.db.get_attachments_for_node(node_id)
                # Fix MIME types and ensure proper structure
                for att in attachments:
                    # Determine MIME type
                    if att.get('file_type') == 'unknown' or not att.get('file_type'):
                        att['file_type'] = get_mime_type(att.get('file_path', ''))
                    
                    # Ensure file_path is properly formatted
                    file_path = att.get('file_path', '')
                    if file_path and not file_path.startswith('uploads/'):
                        # Add uploads prefix if missing (for backward compatibility)
                        if not file_path.startswith('/'):
                            att['file_path'] = f'uploads/{file_path}'
                    
                    # Ensure all required fields exist
                    if not att.get('id'):
                        att['id'] = att.get('node_id', 'unknown')
                    if not att.get('parent_node_title'):
                        att['parent_node_title'] = result.get('title')
                    if not att.get('parent_node_type'):
                        att['parent_node_type'] = result.get('node_type')
                
                all_attachments.extend(attachments)
                if attachments:
                    logger.info(f"Found {len(attachments)} attachments for node {node_id}")
                    for att in attachments:
                        logger.debug(f"  Attachment: {att.get('file_name')} ({att.get('file_type')})")
                
                # Do not include child node attachments here.
                # Child items are returned separately and will include their own attachments.
                
                # Sort attachments: images first, then by filename
                def sort_attachments(att):
                    is_image = att.get('file_type', '').startswith('image/')
                    return (not is_image, att.get('file_name', '').lower())
                
                # Deduplicate attachments by file_path to avoid showing the same file twice
                seen_files = set()
                unique_attachments = []
                for att in all_attachments:
                    file_path = att.get('file_path', '') or att.get('file_name', '')
                    file_key = file_path.lower().strip()
                    if file_key and file_key not in seen_files:
                        seen_files.add(file_key)
                        unique_attachments.append(att)
                    elif not file_key:
                        # If no file path, include it anyway
                        unique_attachments.append(att)
                
                unique_attachments.sort(key=sort_attachments)
                result['attachments'] = unique_attachments
                
                if not all_attachments:
                    logger.debug(f"No attachments found for node {node_id} or its children")
            
            return result
        except Exception as e:
            logger.warning(f"Error enriching result: {e}")
            return result
    
    def _format_context(self, items: List[Dict[str, Any]]) -> str:
        """Format context items into readable text optimized for LLM understanding."""
        if not items:
            return "No relevant context found."
        
        formatted = []
        formatted.append("KNOWLEDGE BASE CONTEXT:")
        formatted.append("=" * 50)
        
        for i, item in enumerate(items, 1):
            title = item.get('title', 'Unknown')
            node_type = item.get('node_type', '').upper()
            content = item.get('content', '')
            attachments = item.get('attachments', [])
            similarity = item.get('similarity', 0)
            
            # Header with source information (no section numbering)
            section = f"\n{title}"
            if node_type:
                section += f" ({node_type})"
            if similarity > 0:
                section += f" [Match: {similarity:.0%}]"
            
            formatted.append(section)
            formatted.append("-" * 40)
            
            # Main content
            if content:
                formatted.append(content)
            
            # Separate images and other files
            if attachments:
                images = [a for a in attachments if a.get('file_type', '').startswith('image/')]
                files = [a for a in attachments if not a.get('file_type', '').startswith('image/')]
                
                # Include image references with metadata
                if images:
                    formatted.append("\n📎 Visual Content (Images):")
                    for img in images:
                        file_name = img.get('file_name', 'image')
                        file_type = img.get('file_type', 'image/png')
                        parent_title = img.get('parent_node_title')
                        ref = f"  • {file_name} ({file_type})"
                        if parent_title:
                            ref += f" [Source: {parent_title}]"
                        formatted.append(ref)
                    formatted.append("[Note: These images will be displayed alongside this response]")
                
                # Include other file references
                if files:
                    formatted.append("\n📄 Document Attachments:")
                    for att in files:
                        file_name = att.get('file_name', 'Unknown')
                        file_type = att.get('file_type', 'unknown')
                        ref = f"  • {file_name} ({file_type})"
                        formatted.append(ref)
        
        formatted.append("\n" + "=" * 50)
        return "\n".join(formatted)


# Global RAG instance
_rag_instance: Optional[RAGService] = None


def get_rag_service(top_k: Optional[int] = None, threshold: Optional[float] = None) -> RAGService:
    """Get or create the global RAG service instance."""
    global _rag_instance
    if _rag_instance is None:
        top_k = top_k or settings.TOP_K_RESULTS
        threshold = threshold or settings.SIMILARITY_THRESHOLD
        _rag_instance = RAGService(top_k=top_k, similarity_threshold=threshold)
    return _rag_instance
