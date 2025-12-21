"""
Auto-merging retriever for hierarchical chunks.

This module implements AutoMergingRetriever which:
- Merges adjacent chunks from the same section when context is needed
- Preserves context boundaries (doesn't merge across sections)
- Uses chunk metadata (section_id, position_index) to identify adjacent chunks
- Can merge small chunks into medium/large chunks when appropriate

This is used by HierarchicalRetriever to provide broader context when needed.
"""

from typing import List, Dict, Any, Optional
from llama_index.embeddings.openai import OpenAIEmbedding
from src.utils.logger import logger


class AutoMergingRetriever:
    """
    Handles automatic merging of adjacent chunks for broader context.

    Auto-merging is useful when:
    - Initial retrieval returns small chunks that are too fragmented
    - Query needs more context than a single small chunk provides
    - Adjacent chunks are from the same section (preserve boundaries)

    Merging logic:
    1. Groups chunks by section_id
    2. Within each section, merges chunks that are adjacent (based on position_index)
    3. Preserves metadata from the first chunk
    4. Combines text with proper spacing

    This class is used internally by HierarchicalRetriever.
    """

    def __init__(self, collection, embedding_fn: Optional[OpenAIEmbedding] = None):
        """
        Initialize AutoMergingRetriever.

        Args:
            collection: ChromaDB collection (for potential re-querying if needed)
            embedding_fn: Optional embedding function (for similarity checks)
        """
        self.collection = collection
        self.embedding_fn = embedding_fn
        self.logger = logger

    def merge_chunks(
        self,
        chunks: List[Dict[str, Any]],
        max_results: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Merge adjacent chunks that belong to the same section.

        Args:
            chunks: List of chunk dictionaries with metadata
            max_results: Maximum number of merged results to return

        Returns:
            List[Dict[str, Any]]: Merged chunks with:
                - text: Combined text from merged chunks
                - metadata: Metadata from first chunk
                - score: Average score of merged chunks
                - merged: True
                - merged_count: Number of chunks that were merged
        """
        if not chunks:
            return []

        # Group chunks by section_id
        sections: Dict[str, List[Dict[str, Any]]] = {}
        for chunk in chunks:
            section_id = chunk.get("metadata", {}).get("section_id", "unknown")
            sections.setdefault(section_id, []).append(chunk)

        merged_results: List[Dict[str, Any]] = []

        # Process each section independently
        for section_id, section_chunks in sections.items():
            # Sort chunks by position_index within the section
            section_chunks.sort(
                key=lambda c: c.get("metadata", {}).get("position_index", 0)
            )

            # Merge adjacent chunks in this section (structure-based, ignore score threshold)
            merged_section_chunks = self._merge_adjacent_in_section(section_chunks)
            merged_results.extend(merged_section_chunks)

        # Sort merged results by score (descending)
        merged_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)

        # Limit to max_results if specified
        if max_results:
            merged_results = merged_results[:max_results]

        self.logger.debug(
            f"Merged {len(chunks)} chunks into {len(merged_results)} merged results"
        )
        return merged_results

    def _merge_adjacent_in_section(
        self,
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Merge adjacent chunks within a single section.

        Args:
            chunks: Chunks from the same section, sorted by position_index

        Returns:
            List[Dict[str, Any]]: Merged chunks
        """
        if not chunks:
            return []

        merged: List[Dict[str, Any]] = []
        i = 0

        while i < len(chunks):
            current_chunk = chunks[i]

            # Always try to merge with adjacent chunks from the same section,
            # based purely on structural adjacency (ignore score thresholds).
            merged_chunks = [current_chunk]
            current_pos = current_chunk.get("metadata", {}).get("position_index", 0)
            total_score = current_chunk.get("score", 0.0)
            j = i + 1

            # Look for adjacent chunks to merge
            while j < len(chunks):
                next_chunk = chunks[j]
                next_pos = next_chunk.get("metadata", {}).get("position_index", 0)
                next_score = next_chunk.get("score", 0.0)

                # Check if chunks are adjacent (position difference <= 1)
                # and from same section
                if (
                    next_pos - current_pos <= 1
                    and next_chunk.get("metadata", {}).get("section_id")
                    == current_chunk.get("metadata", {}).get("section_id")
                ):
                    merged_chunks.append(next_chunk)
                    total_score += next_score
                    current_pos = next_pos
                    j += 1
                else:
                    break

            # Create merged result if we merged more than one chunk
            if len(merged_chunks) > 1:
                merged_result = self._create_merged_chunk(merged_chunks, total_score)
                merged.append(merged_result)
            else:
                merged.append(current_chunk)

            i = j

        return merged
    
    def _create_merged_chunk(
        self,
        chunks: List[Dict[str, Any]],
        total_score: float,
    ) -> Dict[str, Any]:
        """
        Create a single merged chunk from multiple chunks.
        
        Args:
            chunks: List of chunks to merge
            total_score: Sum of scores from all chunks
        
        Returns:
            Dict[str, Any]: Merged chunk dictionary
        """
        # Use metadata from first chunk
        first_metadata = chunks[0].get("metadata", {}).copy()
        
        # Combine text from all chunks
        combined_text = "\n\n".join(
            chunk.get("text", "") for chunk in chunks if chunk.get("text")
        )
        
        # Average score
        avg_score = total_score / len(chunks) if chunks else 0.0
        
        return {
            "id": chunks[0].get("id", ""),
            "text": combined_text,
            "metadata": first_metadata,
            "score": avg_score,
            "merged": True,
            "merged_count": len(chunks),
            "merged_ids": [chunk.get("id", "") for chunk in chunks],
        }

