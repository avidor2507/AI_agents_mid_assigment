"""
Evaluation system for the Insurance Claim Timeline Retrieval System.

This package provides LLM-as-a-judge evaluation capabilities including:
- EvalCase dataclass for defining test cases
- JudgeEvaluator for scoring responses on multiple metrics
- EvalSuite for running and evaluating test cases
- EvalResult dataclass for evaluation results
"""

from src.evaluation.eval_case import EvalCase
from src.evaluation.judge_evaluator import JudgeEvaluator
from src.evaluation.eval_suite import EvalSuite, EvalResult
from src.evaluation.eval_cases import get_test_cases

__all__ = ["EvalCase", "JudgeEvaluator", "EvalSuite", "EvalResult", "get_test_cases"]

