"""
TestCase dataclass for evaluation.

Defines the structure of a test case used in evaluation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TestCase:
    """
    Represents a single test case for evaluation.
    
    Attributes:
        query: The user query to test
        expected_answer: The expected answer string
        expected_context: List of expected context chunks/sections that should be retrieved
        ground_truth: Additional ground truth information (e.g., metadata, timestamps, etc.)
        category: Optional category label (e.g., "needle", "summarization", "timeline")
        description: Optional description of what this test case validates
    """
    query: str
    expected_answer: str
    expected_context: List[str] = field(default_factory=list)
    ground_truth: Dict[str, Any] = field(default_factory=dict)
    category: Optional[str] = None
    description: Optional[str] = None
    
    def __post_init__(self):
        """Validate test case data."""
        if not self.query or not self.query.strip():
            raise ValueError("TestCase query cannot be empty")
        if not self.expected_answer or not self.expected_answer.strip():
            raise ValueError("TestCase expected_answer cannot be empty")

