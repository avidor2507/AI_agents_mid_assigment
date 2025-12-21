"""
Retrieval system module.

This module provides retrievers for querying the indexed data:
- SummaryRetriever: For high-level summary queries
- HierarchicalRetriever: For precise factual queries with auto-merging
- AutoMergingRetriever: Handles merging of adjacent chunks
"""

from src.retrieval.base_retriever import RetrieverInterface
from src.retrieval.summary_retriever import SummaryRetriever
from src.retrieval.hierarchical_retriever import HierarchicalRetriever
from src.retrieval.auto_merging_retriever import AutoMergingRetriever

__all__ = [
    "RetrieverInterface",
    "SummaryRetriever",
    "HierarchicalRetriever",
    "AutoMergingRetriever",
]

