"""
Constants and enumerations for the Insurance Claim Timeline Retrieval System.

This module defines all application-wide constants including:
- Chunk size constants
- Agent types
- Index types
- Evaluation metrics
- Error codes
"""

from enum import Enum


# ============================================================================
# CHUNK SIZE CONSTANTS
# ============================================================================

class ChunkSize(Enum):
    """Enumeration for chunk size levels in the hierarchical structure."""
    SMALL = "small"      # Fine-grained chunks: ~100-200 tokens
    MEDIUM = "medium"    # Balanced chunks: ~500-800 tokens
    LARGE = "large"      # Broad context chunks: ~1500-2000 tokens


# Token size ranges for each chunk level (approximate)
CHUNK_SIZE_TOKENS = {
    ChunkSize.SMALL: (100, 200),
    ChunkSize.MEDIUM: (500, 800),
    ChunkSize.LARGE: (1500, 2000)
}

# Chunk overlap percentage (20% overlap between adjacent chunks)
CHUNK_OVERLAP_PERCENTAGE = 0.2


# ============================================================================
# AGENT TYPES
# ============================================================================

class AgentType(Enum):
    """Enumeration for different agent types in the system."""
    MANAGER = "manager"                      # Router/Manager agent
    SUMMARIZATION_EXPERT = "summarization"   # Summarization expert agent
    NEEDLE_IN_HAYSTACK = "needle"            # Needle-in-haystack agent


# ============================================================================
# INDEX TYPES
# ============================================================================

class IndexType(Enum):
    """Enumeration for different index types."""
    HIERARCHICAL = "hierarchical"  # Multi-level hierarchical index
    SUMMARY = "summary"            # Summary index (MapReduce)


# ============================================================================
# EVALUATION METRICS
# ============================================================================

class EvaluationMetric(Enum):
    """Enumeration for evaluation metrics used in LLM-as-a-judge."""
    ANSWER_CORRECTNESS = "answer_correctness"    # Does answer match ground truth?
    CONTEXT_RELEVANCY = "context_relevancy"      # Did agent use correct index and relevant segments?
    CONTEXT_RECALL = "context_recall"            # Did system retrieve the correct chunk(s)?


# ============================================================================
# ERROR CODES
# ============================================================================

class ErrorCode(Enum):
    """Enumeration for error codes."""
    INDEXING_ERROR = "INDEXING_ERROR"
    RETRIEVAL_ERROR = "RETRIEVAL_ERROR"
    AGENT_ERROR = "AGENT_ERROR"
    MCP_TOOL_ERROR = "MCP_TOOL_ERROR"
    EVALUATION_ERROR = "EVALUATION_ERROR"
    PDF_LOADING_ERROR = "PDF_LOADING_ERROR"
    CHUNKING_ERROR = "CHUNKING_ERROR"


# ============================================================================
# METADATA KEYS
# ============================================================================

# Metadata keys used in ChromaDB collections
METADATA_KEYS = {
    "chunk_id": "chunk_id",
    "parent_id": "parent_id",
    "level": "level",                    # small, medium, or large
    "section": "section",                # Section name/ID
    "timestamp": "timestamp",            # Event timestamp
    "chunk_text": "chunk_text",          # Full chunk text
    "position_index": "position_index",  # Position in document
    "document_id": "document_id",        # Parent document ID
    "section_id": "section_id",          # Parent section ID
    "claim_id": "claim_id",              # Claim ID for the whole file
    "summary_level": "summary_level",    # For summary index: chunk, section, or document
}


# ============================================================================
# CHROMADB COLLECTION NAMES
# ============================================================================

HIERARCHICAL_COLLECTION_NAME = "hierarchical_index"
SUMMARY_COLLECTION_NAME = "summary_index"

