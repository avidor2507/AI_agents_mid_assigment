"""
Helper utility functions.

This module contains common utility functions used throughout the application:
- Text processing utilities
- Date/time parsing
- Validation functions
- Common transformations
"""

import re
from datetime import datetime
from typing import Optional, List, Dict, Any
from dateutil import parser as date_parser


def parse_timestamp(text: str) -> Optional[datetime]:
    """
    Parse a timestamp from text.
    
    Attempts to extract and parse various timestamp formats from text.
    Uses dateutil.parser for flexible date/time parsing.
    
    Args:
        text: Text containing a timestamp
    
    Returns:
        Optional[datetime]: Parsed datetime object, or None if parsing fails
    
    Example:
        >>> parse_timestamp("Event occurred on 2024-12-10 at 14:30:00")
        datetime.datetime(2024, 12, 10, 14, 30, 0)
    """
    # Try to find common timestamp patterns
    patterns = [
        r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}',  # YYYY-MM-DD HH:MM:SS
        r'\d{4}-\d{2}-\d{2}',                       # YYYY-MM-DD
        r'\d{2}/\d{2}/\d{4}',                       # MM/DD/YYYY
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return date_parser.parse(match.group())
            except (ValueError, date_parser.ParserError):
                continue
    
    # Try parsing the entire text
    try:
        return date_parser.parse(text, fuzzy=True)
    except (ValueError, date_parser.ParserError):
        return None


def extract_entities(text: str, entity_types: List[str] = None) -> Dict[str, List[str]]:
    """
    Extract entities from text using simple pattern matching.
    
    This is a basic implementation. For production, consider using
    more sophisticated NLP libraries like spaCy or NER models.
    
    Args:
        text: Text to extract entities from
        entity_types: List of entity types to extract (e.g., ['date', 'money', 'person'])
    
    Returns:
        Dict[str, List[str]]: Dictionary mapping entity types to lists of found entities
    
    Example:
        >>> extract_entities("Patient John Doe, cost $1500.50")
        {'person': ['John Doe'], 'money': ['$1500.50']}
    """
    if entity_types is None:
        entity_types = ['date', 'money', 'person', 'email', 'phone']
    
    entities = {entity_type: [] for entity_type in entity_types}
    
    # Simple pattern matching (can be enhanced with regex or NLP models)
    # Money pattern
    if 'money' in entity_types:
        money_pattern = r'\$\d+(?:\.\d{2})?'
        entities['money'] = re.findall(money_pattern, text)
    
    # Email pattern
    if 'email' in entity_types:
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        entities['email'] = re.findall(email_pattern, text)
    
    # Phone pattern (US format)
    if 'phone' in entity_types:
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        entities['phone'] = re.findall(phone_pattern, text)
    
    return entities


def normalize_text(text: str) -> str:
    """
    Normalize text for processing.
    
    Performs basic text normalization:
    - Removes extra whitespace
    - Normalizes line breaks
    - Trims leading/trailing whitespace
    
    Args:
        text: Text to normalize
    
    Returns:
        str: Normalized text
    
    Example:
        >>> normalize_text("  Hello   world  \\n\\n")
        "Hello world"
    """
    # Remove extra whitespace and normalize line breaks
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n', text)
    return text.strip()


def calculate_overlap_tokens(text1: str, text2: str, tokenizer) -> int:
    """
    Calculate the number of overlapping tokens between two texts.
    
    Args:
        text1: First text
        text2: Second text
        tokenizer: Tokenizer function (e.g., from tiktoken)
    
    Returns:
        int: Number of overlapping tokens
    """
    tokens1 = set(tokenizer.encode(text1))
    tokens2 = set(tokenizer.encode(text2))
    return len(tokens1.intersection(tokens2))


def validate_chunk_size(text: str, min_tokens: int, max_tokens: int, tokenizer) -> bool:
    """
    Validate that a chunk's token count is within the specified range.
    
    Args:
        text: Text to validate
        min_tokens: Minimum allowed tokens
        max_tokens: Maximum allowed tokens
        tokenizer: Tokenizer function
    
    Returns:
        bool: True if chunk size is within range, False otherwise
    """
    token_count = len(tokenizer.encode(text))
    return min_tokens <= token_count <= max_tokens


def merge_chunks(chunks: List[Dict[str, Any]], preserve_order: bool = True) -> str:
    """
    Merge multiple chunks into a single text.
    
    Args:
        chunks: List of chunk dictionaries, each containing at least 'text' key
        preserve_order: Whether to preserve the original order of chunks
    
    Returns:
        str: Merged text
    """
    if not chunks:
        return ""
    
    if preserve_order:
        # Sort by position_index if available
        sorted_chunks = sorted(
            chunks,
            key=lambda x: x.get('position_index', 0)
        )
    else:
        sorted_chunks = chunks
    
    texts = [chunk.get('text', '') for chunk in sorted_chunks]
    return '\n\n'.join(texts)

