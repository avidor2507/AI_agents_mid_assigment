"""
Evaluation system for the Insurance Claim Timeline Retrieval System.

This package provides LLM-as-a-judge evaluation capabilities including:
- TestCase dataclass for defining test cases
- JudgeEvaluator for scoring responses on multiple metrics
- TestSuite for running and evaluating test cases
"""

from src.evaluation.test_case import TestCase
from src.evaluation.judge_evaluator import JudgeEvaluator
from src.evaluation.test_suite import TestSuite
from src.evaluation.test_cases import get_test_cases

__all__ = ["TestCase", "JudgeEvaluator", "TestSuite", "get_test_cases"]

