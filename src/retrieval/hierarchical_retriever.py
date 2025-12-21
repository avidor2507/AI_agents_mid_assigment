"""
Hierarchical index retriever implementation.

This module implements HierarchicalRetriever which:
- Queries the HierarchicalIndex (ChromaDB collection)
- Starts with small chunks for fine-grained retrieval
- Supports auto-merging to larger chunks when more context is needed
- Filters by chunk level (small/medium/large), section, timestamp
- Uses semantic search via embeddings

Implements RetrieverInterface and uses AutoMergingRetriever for merging logic.
"""

from typing import List, Dict, Any, Optional
import re
from llama_index.embeddings.openai import OpenAIEmbedding
from src.retrieval.base_retriever import RetrieverInterface
from src.retrieval.auto_merging_retriever import AutoMergingRetriever
from src.config.settings import config
from src.config.constants import IndexType, ChunkSize
from src.utils.exceptions import RetrievalError
from src.utils.logger import logger


class HierarchicalRetriever(RetrieverInterface):
    """
    Retriever for querying the Hierarchical Index with auto-merging support.
    
    This retriever is designed for precise, factual queries that need to
    find specific information. It:
    1. Starts with small chunks for fine-grained search
    2. Can auto-merge adjacent chunks when more context is needed
    3. Supports filtering by level, section, timestamp
    
    Use this retriever when:
    - Query asks for specific facts, dates, amounts, names
    - Need precise information (needle-in-haystack queries)
    - Query requires deep search through document
    
    Responsibilities:
    - Query HierarchicalIndex starting with small chunks
    - Delegate merging logic to AutoMergingRetriever
    - Return merged results with full metadata
    """
    
    def __init__(
        self,
        collection,
        embedding_model: Optional[str] = None,
        enable_auto_merge: bool = True,
    ):
        """
        Initialize HierarchicalRetriever.
        
        Args:
            collection: ChromaDB collection for hierarchical index
            embedding_model: Optional embedding model name (defaults to config)
            enable_auto_merge: Whether to enable auto-merging of adjacent chunks
        
        Raises:
            RetrievalError: If collection is None or invalid
        """
        if collection is None:
            raise RetrievalError("Hierarchical collection cannot be None")
        
        self.collection = collection
        self.embedding_model = embedding_model or config.EMBEDDING_MODEL
        self.enable_auto_merge = enable_auto_merge
        self.logger = logger
        
        # Initialize embedding function
        try:
            self.embedding_fn = OpenAIEmbedding(
                model_name=self.embedding_model,
                api_key=config.OPENAI_API_KEY,
            )
            self.logger.info(
                f"HierarchicalRetriever initialized with model: {self.embedding_model}, "
                f"auto_merge: {enable_auto_merge}"
            )
        except Exception as e:
            error_msg = f"Failed to initialize embedding model: {str(e)}"
            self.logger.error(error_msg)
            raise RetrievalError(error_msg) from e
        
        # Initialize auto-merging retriever if enabled
        self.auto_merger = None
        if self.enable_auto_merge:
            self.auto_merger = AutoMergingRetriever(collection, self.embedding_fn)
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        start_level: str = ChunkSize.SMALL.value,
        use_time_rerank: bool = False,
        use_section_rerank: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks from the Hierarchical Index.
        
        Args:
            query: Search query string
            top_k: Number of results to return (defaults to config.TOP_K_RESULTS)
            filters: Optional metadata filters, e.g.:
                - {"level": "small"} - Only small chunks
                - {"section_id": "section_3"} - Only chunks from section 3
                - {"timestamp": "2025-03-03"} - Only chunks with this timestamp
            start_level: Starting chunk level ("small", "medium", or "large")
                        Defaults to "small" for fine-grained search
            use_time_rerank: If True, apply time-aware reranking based on explicit
                             time/date tokens in the query (e.g. 08:11:02, 03 March 2025).
            use_section_rerank: If True, apply section-aware reranking when the query
                                explicitly references sections (e.g. "section 3").
        
        Returns:
            List[Dict[str, Any]]: Retrieved chunks with:
                - text: str - Chunk text (possibly merged)
                - metadata: Dict - Chunk metadata (level, section_id, etc.)
                - score: float - Relevance score
                - id: str - Chunk ID
                - merged: bool - Whether this result was merged from multiple chunks
        
        Raises:
            RetrievalError: If retrieval fails
        """
        if not self.validate_query(query):
            raise RetrievalError("Invalid query: query must be a non-empty string")
        
        top_k = top_k or config.TOP_K_RESULTS
        
        try:
            self.logger.debug(
                f"Retrieving hierarchical chunks for query: '{query[:50]}...' "
                f"(top_k={top_k}, start_level={start_level})"
            )
            
            # Prepare filters: start with small chunks by default
            query_filters = filters.copy() if filters else {}
            if "level" not in query_filters:
                query_filters["level"] = start_level
            
            # Generate query embedding
            query_embedding = self.embedding_fn.get_query_embedding(query)
            
            # Decide how many candidates to fetch BEFORE reranking/merging.
            # When we plan to rerank (time/section), we want a wider candidate pool.
            if use_time_rerank or use_section_rerank:
                candidate_multiplier = 4
            else:
                candidate_multiplier = 2
            n_candidates = top_k * candidate_multiplier
            print(f"n_candidates: {n_candidates}")
            
            # Query ChromaDB collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_candidates,
                where=query_filters,
                include=["metadatas", "documents", "distances"],
            )
            
            # Format initial results
            formatted_results = []
            if results and results.get("ids") and len(results["ids"][0]) > 0:
                for i in range(len(results["ids"][0])):
                    result = {
                        "id": results["ids"][0][i],
                        "text": results["documents"][0][i] if results.get("documents") else "",
                        "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                        "score": 1.0 - results["distances"][0][i] if results.get("distances") else 0.0,
                        "merged": False,
                    }
                    formatted_results.append(result)
            
            # Optionally apply reranking based on time/section metadata
            # BEFORE auto-merging and final top_k truncation.
            if use_time_rerank and formatted_results:
                self.logger.debug("Applying time-aware reranking to hierarchical results")
                formatted_results = self._rerank_by_time(
                    query,
                    formatted_results,
                    top_k=None,
                )
            
            if use_section_rerank and formatted_results:
                self.logger.debug("Applying section-aware reranking to hierarchical results")
                formatted_results = self._rerank_by_section(
                    query,
                    formatted_results,
                    top_k=None,
                )
            
            # Limit to top_k results first, then apply auto-merging only on the
            # final set of results we intend to return. This ensures we merge
            # only the chunks that are actually going back to the caller.
            final_results = formatted_results[:top_k]

            # Apply auto-merging if enabled
            if self.enable_auto_merge and self.auto_merger and len(final_results) > 0:
                self.logger.debug("Applying auto-merging to final hierarchical results")
                final_results = self.auto_merger.merge_chunks(
                    final_results,
                    max_results=top_k,
                )
            
            self.logger.info(
                f"Retrieved {len(final_results)} hierarchical chunks "
                f"({'with' if self.enable_auto_merge else 'without'} auto-merging)"
            )
            return self.format_results(final_results, include_metadata=True)
        
        except Exception as e:
            error_msg = f"Error retrieving hierarchical chunks: {str(e)}"
            self.logger.error(error_msg)
            raise RetrievalError(error_msg) from e
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about this retriever.
        
        Returns:
            Dict[str, Any]: Retriever metadata
        """
        return {
            "retriever_type": "HierarchicalRetriever",
            "index_type": IndexType.HIERARCHICAL.value,
            "collection_name": getattr(self.collection, "name", "hierarchical_index"),
            "embedding_model": self.embedding_model,
            "auto_merge_enabled": self.enable_auto_merge,
            "capabilities": [
                "semantic_search",
                "level_filtering",
                "section_filtering",
                "timestamp_filtering",
                "auto_merging" if self.enable_auto_merge else "no_auto_merging",
            ],
        }

    # ------------------------------------------------------------------
    # Time-aware reranking helpers
    # ------------------------------------------------------------------
    def _extract_time_tokens(self, query: str) -> List[str]:
        """
        Extract explicit time/date tokens from the query.

        We look for:
        - Times like 8:11:02, 08:18:41, 08:20:05, 8:20:31
        - Dates like 03 March 2025
        """
        # HH:MM or HH:MM:SS (single or double digit hour)
        time_pattern = r"\b\d{1,2}:\d{2}(?::\d{2})?\b"
        # Simple "DD Month YYYY" pattern (e.g. 03 March 2025)
        date_pattern = r"\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}\b"

        times = re.findall(time_pattern, query)
        dates = re.findall(date_pattern, query)

        tokens = list({t.strip() for t in (times + dates) if t.strip()})
        return tokens

    def _rerank_by_time(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank retrieval results by matching timestamps/dates between query and chunks.

        This is useful for timeline-style queries where the user mentions specific
        times/dates and we want chunks containing those exact values to be promoted.

        Args:
            query: Original user query
            results: List of results from retrieve() (already formatted)
            top_k: Number of results to keep after reranking (None = keep all)

        Returns:
            List[Dict[str, Any]]: Reranked results (same structure as input)
        """
        if not results:
            return results

        time_tokens = self._extract_time_tokens(query)
        if not time_tokens:
            # No explicit time/date patterns in query; return as-is
            self.logger.debug(
                "rerank_by_time: no time/date tokens found in query; skipping rerank"
            )
            return results[:top_k] if top_k else results

        self.logger.debug(
            f"rerank_by_time: found time/date tokens in query: {time_tokens}"
        )

        reranked: List[Dict[str, Any]] = []
        for res in results:
            text = res.get("text", "") or ""
            meta = res.get("metadata", {}) or {}
            chunk_text = meta.get("chunk_text", "")

            combined = f"{text} {chunk_text}"

            # Count how many of the explicit query time tokens appear in this chunk
            match_count = sum(
                1 for token in time_tokens if token in combined
            )

            base_score = res.get("score", 0.0)

            # Strongly boost chunks that contain explicit time tokens from query.
            # We add +1.0 per match so a chunk containing several of the
            # specified times will dominate others.
            adjusted_score = base_score + match_count * 1.0

            new_res = dict(res)
            new_res["time_match_count"] = match_count
            new_res["time_reranked_score"] = adjusted_score
            reranked.append(new_res)

        # Sort by adjusted score (falling back to original score)
        reranked.sort(
            key=lambda r: r.get("time_reranked_score", r.get("score", 0.0)),
            reverse=True,
        )

        if top_k:
            reranked = reranked[:top_k]

        return reranked

    # ------------------------------------------------------------------
    # Section-aware reranking helpers
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
        # e.g. "section 3", "Section 16"
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
        Rerank retrieval results by matching section_id metadata with the query.

        This is useful when the user explicitly mentions a section, e.g.:
        - "In section 3, what happened at ...?"
        - "According to Section 16 ..."

        In that case, we want chunks whose metadata.section_id matches the referenced
        section(s) to be promoted.

        Args:
            query: Original user query
            results: List of results from retrieve() (already formatted)
            top_k: Number of results to keep after reranking (None = keep all)

        Returns:
            List[Dict[str, Any]]: Reranked results (same structure as input)
        """
        if not results:
            return results

        section_ids = self._extract_section_ids(query)
        if not section_ids:
            # No explicit section references; return as-is
            self.logger.debug(
                "rerank_by_section: no section ids found in query; skipping rerank"
            )
            return results[:top_k] if top_k else results

        self.logger.debug(
            f"rerank_by_section: found section ids in query: {section_ids}"
        )

        reranked: List[Dict[str, Any]] = []
        for res in results:
            meta = res.get("metadata", {}) or {}
            chunk_section_id = meta.get("section_id") or meta.get("section")

            # Count matches across all referenced sections
            match_count = 1 if chunk_section_id in section_ids else 0

            base_score = res.get("score", 0.0)

            # Strongly boost chunks whose section_id matches the query reference.
            # We add +2.0 if the chunk is from a referenced section.
            adjusted_score = base_score + match_count * 2.0

            new_res = dict(res)
            new_res["section_match"] = bool(match_count)
            new_res["section_reranked_score"] = adjusted_score
            reranked.append(new_res)

        # Sort by adjusted score (falling back to original score)
        reranked.sort(
            key=lambda r: r.get("section_reranked_score", r.get("score", 0.0)),
            reverse=True,
        )

        if top_k:
            reranked = reranked[:top_k]

        return reranked