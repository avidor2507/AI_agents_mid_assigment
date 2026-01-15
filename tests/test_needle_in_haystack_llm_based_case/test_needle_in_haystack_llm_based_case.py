"""LLM-based tests for NeedleInHaystackAgent."""

import pytest
from src.evaluation.eval_case import EvalCase
from tests.test_needle_in_haystack_llm_based_case.data import TEST_CASES
from src.helpers.agent_helper import assert_llm_based_query


@pytest.mark.parametrize("test_case", TEST_CASES, ids=lambda tc: tc.query[:50] + "..." if len(tc.query) > 50 else tc.query)
def test_needle_in_haystack_llm_based_case(orchestrator, logger, test_case: EvalCase):
    """Test LLM-based cases for NeedleInHaystackAgent.
    
    Each test case runs as a separate test instance.
    
    Args:
        orchestrator: OrchestratorSystem fixture from conftest
        logger: Logger fixture from conftest
        test_case: EvalCase object from data.py TEST_CASES
    """
    assert_llm_based_query(orchestrator, test_case, 0.8, logger)

