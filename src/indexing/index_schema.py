"""
Index schema definitions and metadata structures.

This module defines the metadata schemas used in ChromaDB collections,
including field definitions, validation functions, and collection names.
"""

from typing import Dict, Any, List
from src.config.constants import (
    METADATA_KEYS,
    HIERARCHICAL_COLLECTION_NAME,
    SUMMARY_COLLECTION_NAME,
    ChunkSize
)


def get_hierarchical_metadata_keys() -> List[str]:
    """
    Get list of metadata keys for hierarchical index.
    
    Returns:
        List[str]: List of metadata key names
    """
    return [
        METADATA_KEYS["chunk_id"],
        METADATA_KEYS["parent_id"],
        METADATA_KEYS["level"],
        METADATA_KEYS["section"],
        METADATA_KEYS["timestamp"],
        METADATA_KEYS["chunk_text"],
        METADATA_KEYS["position_index"],
        METADATA_KEYS["document_id"],
        METADATA_KEYS["section_id"],
        METADATA_KEYS["claim_id"],
    ]


def get_summary_metadata_keys() -> List[str]:
    """
    Get list of metadata keys for summary index.
    
    Returns:
        List[str]: List of metadata key names
    """
    return [
        METADATA_KEYS["chunk_id"],
        METADATA_KEYS["summary_level"],  # chunk, section, or document
        METADATA_KEYS["section_id"],
        METADATA_KEYS["document_id"],
        METADATA_KEYS["claim_id"],
        METADATA_KEYS["chunk_text"],  # Summary text
    ]


def validate_hierarchical_metadata(metadata: Dict[str, Any]) -> bool:
    """
    Validate metadata for hierarchical index chunk.
    
    Args:
        metadata: Metadata dictionary to validate
    
    Returns:
        bool: True if metadata is valid, False otherwise
    """
    required_keys = [
        METADATA_KEYS["chunk_id"],
        METADATA_KEYS["level"],
        METADATA_KEYS["document_id"],
    ]
    
    for key in required_keys:
        if key not in metadata:
            return False
    
    # Validate level value
    level = metadata.get(METADATA_KEYS["level"])
    if level not in [cs.value for cs in ChunkSize]:
        return False
    
    return True


def validate_summary_metadata(metadata: Dict[str, Any]) -> bool:
    """
    Validate metadata for summary index.
    
    Args:
        metadata: Metadata dictionary to validate
    
    Returns:
        bool: True if metadata is valid, False otherwise
    """
    required_keys = [
        METADATA_KEYS["chunk_id"],
        METADATA_KEYS["summary_level"],
    ]
    
    for key in required_keys:
        if key not in metadata:
            return False
    
    # Validate summary_level value
    summary_level = metadata.get(METADATA_KEYS["summary_level"])
    if summary_level not in ["chunk", "section", "document"]:
        return False
    
    return True


def create_hierarchical_metadata(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create metadata dictionary for hierarchical index from chunk data.
    
    Args:
        chunk: Chunk dictionary with hierarchical information
    
    Returns:
        Dict[str, Any]: Metadata dictionary for ChromaDB
    """
    metadata = {
        METADATA_KEYS["chunk_id"]: chunk.get("chunk_id", ""),
        METADATA_KEYS["level"]: chunk.get("level", ""),
        METADATA_KEYS["document_id"]: chunk.get("document_id", ""),
        METADATA_KEYS["chunk_text"]: chunk.get("text", ""),
    }
    
    # Add optional fields if present
    if "parent_id" in chunk:
        metadata[METADATA_KEYS["parent_id"]] = chunk["parent_id"]
    if "section_id" in chunk:
        metadata[METADATA_KEYS["section_id"]] = chunk["section_id"]
        metadata[METADATA_KEYS["section"]] = chunk.get("section_id", "")
    if "claim_id" in chunk:
        metadata[METADATA_KEYS["claim_id"]] = chunk["claim_id"]
    if "chunk_index" in chunk:
        metadata[METADATA_KEYS["position_index"]] = chunk["chunk_index"]
    if "timestamp" in chunk.get("metadata", {}):
        metadata[METADATA_KEYS["timestamp"]] = chunk["metadata"]["timestamp"]
    
    return metadata


def create_summary_metadata(summary_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create metadata dictionary for summary index from summary data.
    
    Args:
        summary_data: Summary data dictionary
    
    Returns:
        Dict[str, Any]: Metadata dictionary for ChromaDB
    """
    metadata = {
        METADATA_KEYS["chunk_id"]: summary_data.get("summary_id", ""),
        METADATA_KEYS["summary_level"]: summary_data.get("summary_level", ""),
        METADATA_KEYS["chunk_text"]: summary_data.get("summary_text", ""),
    }
    
    # Add optional fields if present
    if "section_id" in summary_data:
        metadata[METADATA_KEYS["section_id"]] = summary_data["section_id"]
    if "document_id" in summary_data:
        metadata[METADATA_KEYS["document_id"]] = summary_data["document_id"]
    if "claim_id" in summary_data:
        metadata[METADATA_KEYS["claim_id"]] = summary_data["claim_id"]
    
    return metadata

