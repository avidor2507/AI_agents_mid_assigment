import re
import warnings
from datetime import datetime
from dateutil import parser
from typing import Union


def get_date_diff(date1: Union[str, datetime], date2: Union[str, datetime]) -> str:
    """
    MCP tool: Calculate absolute time difference between two dates/times.

    Accepts:
    - ISO dates
    - Slash dates
    - Datetime strings
    - Datetime objects

    Returns:
    - Human-readable deterministic string
    """

    try:
        dt1 = _normalize_datetime(date1)
        dt2 = _normalize_datetime(date2)
    except Exception as e:
        return f"invalid date format: {e}"

    delta = abs(dt2 - dt1)

    days = delta.days
    seconds = delta.seconds

    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    return (
        f"{days} days, "
        f"{hours} hours, "
        f"{minutes} minutes, "
        f"{seconds} seconds"
    )


def _preprocess_date_string(text: str) -> str:
    """
    Preprocess date string to remove abbreviations that cause timezone warnings.
    
    Args:
        text: Date string to preprocess
        
    Returns:
        Preprocessed date string
    """
    # Common abbreviations that cause timezone warnings
    problematic_abbreviations = [
        r'\bETA\b',      # Estimated Time of Arrival
        r'\bFNOL\b',     # First Notice of Loss
        r'\bGP\b',       # General Practitioner or other insurance term
    ]
    
    preprocessed = text
    for pattern in problematic_abbreviations:
        preprocessed = re.sub(pattern, ' ', preprocessed, flags=re.IGNORECASE)
    
    return preprocessed


def _normalize_datetime(value: Union[str, datetime]) -> datetime:
    """
    Normalize various date/time inputs into a datetime object.
    """

    if isinstance(value, datetime):
        return value

    if not isinstance(value, str):
        raise TypeError(f"Unsupported type: {type(value)}")

    # Try common date patterns first (no fuzzy needed)
    patterns = [
        r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}',  # YYYY-MM-DD HH:MM:SS
        r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}',         # YYYY-MM-DD HH:MM
        r'\d{4}-\d{2}-\d{2}',                       # YYYY-MM-DD
        r'\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2}',   # MM/DD/YYYY HH:MM:SS
        r'\d{2}/\d{2}/\d{4}',                       # MM/DD/YYYY
    ]
    
    for pattern in patterns:
        match = re.search(pattern, value)
        if match:
            try:
                # Parse matched pattern directly (no fuzzy needed)
                return parser.parse(match.group(), fuzzy=False)
            except (ValueError, parser.ParserError):
                continue
    
    # Preprocess and use fuzzy parsing as fallback for LLM outputs
    preprocessed_value = _preprocess_date_string(value)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning, module='dateutil')
        return parser.parse(preprocessed_value, fuzzy=True)