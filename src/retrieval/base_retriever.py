"""
Base retriever interface and abstract base class.

This module defines the RetrieverInterface using Dependency Inversion Principle.
All retrievers must implement this interface to ensure consistent behavior
and allow for easy swapping of retrieval strategies.

Uses Dependency Inversion: High-level modules (agents) depend on abstractions
(RetrieverInterface) rather than concrete implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from src.utils.logger import logger


class RetrieverInterface(ABC):
    """
    Abstract interface for all retrievers (Dependency Inversion Principle).
    
    This interface defines the contract that all retrievers must follow.
    By depending on this abstraction rather than concrete implementations,
    we enable:
    - Easy swapping of retrieval strategies
    - Consistent behavior across all retrievers
    - Testability with mock retrievers
    - Extension with new retrieval methods
    
    All retrievers must implement:
    - retrieve(): Query the index and return relevant results
    - get_metadata(): Return metadata about the retriever
    """
    
    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents/chunks from the index.
        
        Args:
            query: Search query string
            top_k: Number of results to return (None = use default)
            filters: Optional metadata filters (e.g., {"level": "small", "section_id": "section_1"})
        
        Returns:
            List[Dict[str, Any]]: List of retrieved results, each containing:
                - text: str - The retrieved text content
                - metadata: Dict[str, Any] - Associated metadata
                - score: float - Relevance score (if available)
                - id: str - Unique identifier (if available)
        
        Raises:
            RetrievalError: If retrieval fails
        """
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about this retriever.
        
        Returns:
            Dict[str, Any]: Metadata including:
                - retriever_type: str - Type of retriever
                - index_type: str - Type of index being queried
                - collection_name: str - Name of ChromaDB collection
                - capabilities: List[str] - List of supported features
        """
        pass
    
    def validate_query(self, query: str) -> bool:
        """
        Validate that a query is acceptable for retrieval.
        
        Args:
            query: Query string to validate
        
        Returns:
            bool: True if query is valid, False otherwise
        """
        if not query or not isinstance(query, str):
            return False
        if len(query.strip()) == 0:
            return False
        return True
    
    def format_results(
        self,
        results: List[Dict[str, Any]],
        include_metadata: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Format retrieval results into a consistent structure.
        
        Args:
            results: Raw results from ChromaDB query
            include_metadata: Whether to include full metadata in results
        
        Returns:
            List[Dict[str, Any]]: Formatted results with consistent structure
        """
        formatted = []
        for result in results:
            formatted_result = {
                "text": result.get("text", result.get("chunk_text", "")),
                "score": result.get("score", 0.0),
                "id": result.get("id", ""),
            }
            if include_metadata:
                formatted_result["metadata"] = result.get("metadata", {})
            formatted.append(formatted_result)
        return formatted

