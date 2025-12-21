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


def _normalize_datetime(value: Union[str, datetime]) -> datetime:
    """
    Normalize various date/time inputs into a datetime object.
    """

    if isinstance(value, datetime):
        return value

    if not isinstance(value, str):
        raise TypeError(f"Unsupported type: {type(value)}")

    # Allow fuzzy parsing for LLM outputs
    return parser.parse(value, fuzzy=True)