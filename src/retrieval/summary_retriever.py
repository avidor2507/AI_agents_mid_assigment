"""
Summary index retriever implementation.

This module implements SummaryRetriever which:
- Queries the SummaryIndex (ChromaDB collection)
- Returns relevant summaries at chunk, section, or document level
- Supports metadata filtering (e.g., by summary_level)
- Uses semantic search via embeddings

Implements RetrieverInterface for Dependency Inversion.
"""

from typing import List, Dict, Any, Optional
import re

from llama_index.embeddings.openai import OpenAIEmbedding
from src.retrieval.base_retriever import RetrieverInterface
from src.config.settings import config
from src.config.constants import IndexType
from src.utils.exceptions import RetrievalError
from src.utils.logger import logger


class SummaryRetriever(RetrieverInterface):
    """
    Retriever for querying the Summary Index.
    
    This retriever is designed for high-level, timeline-based queries that
    benefit from summarized information rather than raw chunks. It queries
    the summary index which contains:
    - Chunk-level summaries
    - Section-level summaries
    - Document-level summaries
    
    Use this retriever when:
    - Query asks for overviews, timelines, or high-level information
    - Need broad context rather than specific details
    - Query is about "what happened" rather than specific facts
    
    Responsibilities:
    - Query SummaryIndex using semantic search
    - Return relevant summaries with metadata
    - Support filtering by summary_level (chunk/section/document)
    """
    
    def __init__(self, collection, embedding_model: Optional[str] = None):
        """
        Initialize SummaryRetriever.
        
        Args:
            collection: ChromaDB collection for summary index
            embedding_model: Optional embedding model name (defaults to config)
        
        Raises:
            RetrievalError: If collection is None or invalid
        """
        if collection is None:
            raise RetrievalError("Summary collection cannot be None")
        
        self.collection = collection
        self.embedding_model = embedding_model or config.EMBEDDING_MODEL
        self.logger = logger
        
        # Initialize embedding function
        try:
            self.embedding_fn = OpenAIEmbedding(
                model_name=self.embedding_model,
                api_key=config.OPENAI_API_KEY,
            )
            self.logger.info(f"SummaryRetriever initialized with model: {self.embedding_model}")
        except Exception as e:
            error_msg = f"Failed to initialize embedding model: {str(e)}"
            self.logger.error(error_msg)
            raise RetrievalError(error_msg) from e
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        use_section_rerank: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant summaries from the Summary Index.
        
        Args:
            query: Search query string
            top_k: Number of results to return (defaults to config.TOP_K_RESULTS)
            filters: Optional metadata filters, e.g.:
                - {"summary_level": "document"} - Only document-level summaries
                - {"section_id": "section_3"} - Only summaries from section 3
        
        Returns:
            List[Dict[str, Any]]: Retrieved summaries with:
                - text: str - Summary text
                - metadata: Dict - Summary metadata (level, section_id, etc.)
                - score: float - Relevance score
                - id: str - Summary ID
        
        Raises:
            RetrievalError: If retrieval fails
        """
        if not self.validate_query(query):
            raise RetrievalError("Invalid query: query must be a non-empty string")
        
        top_k = top_k or config.TOP_K_RESULTS
        
        try:
            self.logger.debug(f"Retrieving summaries for query: '{query[:50]}...' (top_k={top_k})")
            
            # If section reranking is enabled, check if query explicitly mentions a section.
            # If so, add it as a filter to narrow the search to that section only.
            where_clause = filters.copy() if filters else {}
            if use_section_rerank:
                section_ids = self._extract_section_ids(query)
                if section_ids and len(section_ids) == 1:
                    # User explicitly asked for ONE specific section - filter to that section
                    where_clause["section_id"] = section_ids[0]
                    self.logger.debug(
                        f"Filtering summaries to section_id={section_ids[0]} based on explicit query reference"
                    )
                    # Since we're filtering to one section, we don't need reranking anymore
                    # but we'll keep it enabled in case there are multiple summaries from that section
                elif section_ids and len(section_ids) > 1:
                    # Multiple sections mentioned - use OR logic or just rerank
                    # For simplicity, we'll just rerank (no filter)
                    self.logger.debug(
                        f"Multiple sections mentioned in query: {section_ids}; will rerank instead of filter"
                    )
            
            # Generate query embedding
            query_embedding = self.embedding_fn.get_query_embedding(query)
            
            # Decide how many candidates to fetch BEFORE optional reranking.
            n_candidates = top_k
            if use_section_rerank and not where_clause.get("section_id"):
                # Use a wider pool when we plan to rerank by section (but not when filtering)
                n_candidates = top_k * 4
            
            # Query ChromaDB collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_candidates,
                where=where_clause if where_clause else None,
                include=["metadatas", "documents", "distances"],
            )
            
            # Format results
            formatted_results: List[Dict[str, Any]] = []
            if results and results.get("ids") and len(results["ids"][0]) > 0:
                for i in range(len(results["ids"][0])):
                    result = {
                        "id": results["ids"][0][i],
                        "text": results["documents"][0][i] if results.get("documents") else "",
                        "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                        "score": 1.0 - results["distances"][0][i] if results.get("distances") else 0.0,
                    }
                    formatted_results.append(result)

            # Optionally apply section-aware reranking similar to HierarchicalRetriever
            if use_section_rerank and formatted_results:
                self.logger.debug("Applying section-aware reranking to summary results")
                formatted_results = self._rerank_by_section(
                    query,
                    formatted_results,
                    top_k=None,
                )

            final_results = formatted_results[:top_k]
            
            self.logger.info(f"Retrieved {len(final_results)} summaries for query")
            return self.format_results(final_results, include_metadata=True)
        
        except Exception as e:
            error_msg = f"Error retrieving summaries: {str(e)}"
            self.logger.error(error_msg)
            raise RetrievalError(error_msg) from e
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about this retriever.
        
        Returns:
            Dict[str, Any]: Retriever metadata
        """
        return {
            "retriever_type": "SummaryRetriever",
            "index_type": IndexType.SUMMARY.value,
            "collection_name": getattr(self.collection, "name", "summary_index"),
            "embedding_model": self.embedding_model,
            "capabilities": [
                "semantic_search",
                "summary_level_filtering",
                "metadata_filtering",
                "section_reranking",
            ],
        }

    # ------------------------------------------------------------------
    # Section-aware reranking helpers (similar to HierarchicalRetriever)
    # ------------------------------------------------------------------
    def _extract_section_ids(self, query: str) -> List[str]:
        """
        Extract explicit section references from the query.

        We look for patterns like:
        - 'section 3'
        - 'Section 16'

        and convert them to normalized section_ids:
        - 'section_3'
        - 'section_16'
        """
        pattern = r"\bsection\s+(\d+)\b"
        matches = re.findall(pattern, query, flags=re.IGNORECASE)
        section_ids = [f"section_{m}" for m in matches]

        # Deduplicate while preserving order
        seen = set()
        unique_section_ids: List[str] = []
        for sid in section_ids:
            if sid not in seen:
                seen.add(sid)
                unique_section_ids.append(sid)
        return unique_section_ids

    def _rerank_by_section(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank summary results by matching section_id metadata with the query.

        This mirrors the section-aware reranking used in HierarchicalRetriever,
        but operates on summary-level results.
        """
        if not results:
            return results

        section_ids = self._extract_section_ids(query)
        if not section_ids:
            self.logger.debug(
                "SummaryRetriever._rerank_by_section: no section ids in query; skipping rerank"
            )
            return results[:top_k] if top_k else results

        self.logger.debug(
            f"SummaryRetriever._rerank_by_section: found section ids in query: {section_ids}"
        )

        reranked: List[Dict[str, Any]] = []
        for res in results:
            meta = res.get("metadata", {}) or {}
            chunk_section_id = meta.get("section_id") or meta.get("section")

            match_count = 1 if chunk_section_id in section_ids else 0
            base_score = res.get("score", 0.0)

            # Boost summaries whose section_id matches the query reference.
            adjusted_score = base_score + match_count * 2.0

            new_res = dict(res)
            new_res["section_match"] = bool(match_count)
            new_res["section_reranked_score"] = adjusted_score
            reranked.append(new_res)

        reranked.sort(
            key=lambda r: r.get("section_reranked_score", r.get("score", 0.0)),
            reverse=True,
        )

        if top_k:
            reranked = reranked[:top_k]

        return reranked

