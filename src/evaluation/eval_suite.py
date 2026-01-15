"""
EvalSuite implementation for running and evaluating test cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.agents.orchestrator_system import OrchestratorSystem
from src.config.constants import EvaluationMetric
from src.evaluation.judge_evaluator import JudgeEvaluator
from src.evaluation.eval_case import EvalCase
from src.utils.exceptions import EvaluationError
from src.utils.logger import logger


@dataclass
class EvalResult:
    """
    Result object containing evaluation results for a single test case.
    
    Attributes:
        query: The query that was evaluated
        expected_answer: The expected answer
        answer: The actual answer from the system
        category: Category of the test case
        description: Description of the test case
        answer_correctness: Score for answer correctness (0.0-1.0)
        context_relevancy: Score for context relevancy (0.0-1.0)
        context_recall: Score for context recall (0.0-1.0) or None if not evaluated
        retrieved_context_count: Number of retrieved context chunks
    """
    query: str
    expected_answer: str
    answer: str
    category: Optional[str] = None
    description: Optional[str] = None
    answer_correctness: float = 0.0
    context_relevancy: float = 0.0
    context_recall: Optional[float] = None
    retrieved_context_count: int = 0


class EvalSuite:
    """
    Test suite for evaluating individual test cases.
    
    Evaluates a single test case at a time, running queries through the system,
    evaluating responses, and returning evaluation results.
    """
    
    def __init__(
        self,
        orchestrator: Optional[OrchestratorSystem] = None,
        evaluator: Optional[JudgeEvaluator] = None,
    ) -> None:
        """
        Initialize the test suite.
        
        Args:
            orchestrator: OrchestratorSystem instance (created if not provided)
            evaluator: JudgeEvaluator instance (created if not provided)
        """
        self.logger = logger
        self.orchestrator = orchestrator or OrchestratorSystem()
        self.evaluator = evaluator or JudgeEvaluator()
    
    def evaluate(
        self,
        test_case: EvalCase,
        get_retrieval_context: bool = True,
    ) -> EvalResult:
        """
        Evaluate a single test case and return the result.
        
        Args:
            test_case: EvalCase to evaluate
            get_retrieval_context: If True, get full agent response with retrieval context
            
        Returns:
            EvalResult object containing evaluation results
        """
        self.logger.info(f"Evaluating test case: {test_case.query[:60]}...")
        
        try:
            # Get answer from orchestrator
            answer = self.orchestrator.handle_query(test_case.query)
            
            # Ensure answer is a string (handle AIMessage objects that might slip through)
            if hasattr(answer, 'content'):
                answer = str(answer.content) if answer.content else ""
            else:
                answer = str(answer) if answer else ""
            
            # Get retrieval context if needed (for context evaluation metrics)
            retrieved_context: List[Dict[str, Any]] = []
            if get_retrieval_context:
                # Get routing decision to know which agent handled it
                routing_response = self.orchestrator.router_agent.handle_query(test_case.query)
                routing_decision = routing_response.get("routing_decision", {})
                agent_type_value = routing_decision.get("primary_agent_type", "")
                
                # Get full agent response with retrieval info
                if agent_type_value == "needle":
                    agent_response = self.orchestrator.needle_agent.handle_query(test_case.query)
                else:
                    agent_response = self.orchestrator.summarization_agent.handle_query(test_case.query)
                
                # Extract retrieval context
                retrieval_info = agent_response.get("retrieval", {})
                retrieved_context = retrieval_info.get("results", [])
            else:
                # If we don't need retrieval context, create empty list
                retrieved_context = []
            
            # Evaluate on all metrics
            answer_correctness: float = 0.0
            context_relevancy: float = 0.0
            context_recall: Optional[float] = None
            
            # Answer Correctness
            try:
                answer_correctness = self.evaluator.evaluate(
                    metric=EvaluationMetric.ANSWER_CORRECTNESS,
                    query=test_case.query,
                    answer=answer,
                    retrieved_context=retrieved_context,
                    expected_answer=test_case.expected_answer,
                    ground_truth=test_case.ground_truth,
                )
            except Exception as e:
                self.logger.error(f"Error evaluating answer correctness: {e}")
                answer_correctness = 0.0
            
            # Context Relevancy
            try:
                context_relevancy = self.evaluator.evaluate(
                    metric=EvaluationMetric.CONTEXT_RELEVANCY,
                    query=test_case.query,
                    answer=answer,
                    retrieved_context=retrieved_context,
                )
            except Exception as e:
                self.logger.error(f"Error evaluating context relevancy: {e}")
                context_relevancy = 0.0
            
            # Context Recall (only if expected_context is provided)
            if test_case.expected_context:
                try:
                    context_recall = self.evaluator.evaluate(
                        metric=EvaluationMetric.CONTEXT_RECALL,
                        query=test_case.query,
                        answer=answer,
                        retrieved_context=retrieved_context,
                        expected_context=test_case.expected_context,
                    )
                except Exception as e:
                    self.logger.error(f"Error evaluating context recall: {e}")
                    context_recall = 0.0
            
            # Create and return result object
            result = EvalResult(
                query=test_case.query,
                expected_answer=test_case.expected_answer,
                answer=answer,
                category=test_case.category,
                description=test_case.description,
                answer_correctness=answer_correctness,
                context_relevancy=context_relevancy,
                context_recall=context_recall,
                retrieved_context_count=len(retrieved_context),
            )
            
            self.logger.info(
                f"Evaluation completed. Scores: correctness={answer_correctness:.3f}, "
                f"relevancy={context_relevancy:.3f}, recall={context_recall if context_recall is not None else 'N/A'}"
            )
            return result
            
        except Exception as e:
            self.logger.error(f"Error evaluating test case: {e}")
            raise EvaluationError(f"Failed to evaluate test case '{test_case.query[:50]}...': {e}") from e
    
    def evaluate_average(
        self,
        test_case: EvalCase,
        num_runs: int = 10,
        get_retrieval_context: bool = True,
    ) -> EvalResult:
        """
        Evaluate a test case multiple times and return the average of all scores.
        
        This function runs the evaluation multiple times to get more stable/reliable scores
        by averaging the results. The answer and retrieval context are generated once,
        but the LLM judge evaluation is run multiple times and averaged.
        
        Args:
            test_case: EvalCase to evaluate
            num_runs: Number of times to run the evaluation (default: 5)
            get_retrieval_context: If True, get full agent response with retrieval context
            
        Returns:
            EvalResult object containing averaged evaluation results
            
        Example:
            If num_runs=5 and scores are [1.0, 1.0, 0.5, 1.0, 1.0], the average will be 0.9
        """
        if num_runs < 1:
            raise ValueError("num_runs must be at least 1")
        
        self.logger.info(
            f"Evaluating test case {num_runs} times for averaging: {test_case.query[:60]}..."
        )
        
        try:
            # Get answer and retrieval context once (these don't change between runs)
            answer = self.orchestrator.handle_query(test_case.query)
            
            # Ensure answer is a string (handle AIMessage objects that might slip through)
            if hasattr(answer, 'content'):
                answer = str(answer.content) if answer.content else ""
            else:
                answer = str(answer) if answer else ""
            
            # Get retrieval context if needed (for context evaluation metrics)
            retrieved_context: List[Dict[str, Any]] = []
            if get_retrieval_context:
                # Get routing decision to know which agent handled it
                routing_response = self.orchestrator.router_agent.handle_query(test_case.query)
                routing_decision = routing_response.get("routing_decision", {})
                agent_type_value = routing_decision.get("primary_agent_type", "")
                
                # Get full agent response with retrieval info
                if agent_type_value == "needle":
                    agent_response = self.orchestrator.needle_agent.handle_query(test_case.query)
                else:
                    agent_response = self.orchestrator.summarization_agent.handle_query(test_case.query)
                
                # Extract retrieval context
                retrieval_info = agent_response.get("retrieval", {})
                retrieved_context = retrieval_info.get("results", [])
            else:
                # If we don't need retrieval context, create empty list
                retrieved_context = []
            
            # Collect scores from all runs
            answer_correctness_scores: List[float] = []
            context_relevancy_scores: List[float] = []
            context_recall_scores: List[float] = []
            
            # Run evaluation multiple times
            for run_num in range(1, num_runs + 1):
                self.logger.info(f"Evaluation run {run_num}/{num_runs}")
                
                # Answer Correctness
                try:
                    answer_correctness = self.evaluator.evaluate(
                        metric=EvaluationMetric.ANSWER_CORRECTNESS,
                        query=test_case.query,
                        answer=answer,
                        retrieved_context=retrieved_context,
                        expected_answer=test_case.expected_answer,
                        ground_truth=test_case.ground_truth,
                    )
                    answer_correctness_scores.append(answer_correctness)
                except Exception as e:
                    self.logger.error(f"Error evaluating answer correctness in run {run_num}: {e}")
                    answer_correctness_scores.append(0.0)
                
                # Context Relevancy
                try:
                    context_relevancy = self.evaluator.evaluate(
                        metric=EvaluationMetric.CONTEXT_RELEVANCY,
                        query=test_case.query,
                        answer=answer,
                        retrieved_context=retrieved_context,
                    )
                    context_relevancy_scores.append(context_relevancy)
                except Exception as e:
                    self.logger.error(f"Error evaluating context relevancy in run {run_num}: {e}")
                    context_relevancy_scores.append(0.0)
                
                # Context Recall (only if expected_context is provided)
                if test_case.expected_context:
                    try:
                        context_recall = self.evaluator.evaluate(
                            metric=EvaluationMetric.CONTEXT_RECALL,
                            query=test_case.query,
                            answer=answer,
                            retrieved_context=retrieved_context,
                            expected_context=test_case.expected_context,
                        )
                        context_recall_scores.append(context_recall)
                    except Exception as e:
                        self.logger.error(f"Error evaluating context recall in run {run_num}: {e}")
                        context_recall_scores.append(0.0)
            
            # Calculate averages
            avg_answer_correctness = (
                sum(answer_correctness_scores) / len(answer_correctness_scores)
                if answer_correctness_scores else 0.0
            )
            avg_context_relevancy = (
                sum(context_relevancy_scores) / len(context_relevancy_scores)
                if context_relevancy_scores else 0.0
            )
            avg_context_recall = (
                sum(context_recall_scores) / len(context_recall_scores)
                if context_recall_scores else None
            )
            
            # Create and return result object with averaged scores
            result = EvalResult(
                query=test_case.query,
                expected_answer=test_case.expected_answer,
                answer=answer,
                category=test_case.category,
                description=test_case.description,
                answer_correctness=avg_answer_correctness,
                context_relevancy=avg_context_relevancy,
                context_recall=avg_context_recall,
                retrieved_context_count=len(retrieved_context),
            )
            
            self.logger.info(
                f"Average evaluation completed ({num_runs} runs). "
                f"Average scores: correctness={avg_answer_correctness:.3f}, "
                f"relevancy={avg_context_relevancy:.3f}, "
                f"recall={avg_context_recall if avg_context_recall is not None else 'N/A'}"
            )
            return result
            
        except Exception as e:
            self.logger.error(f"Error in average evaluation: {e}")
            raise EvaluationError(
                f"Failed to evaluate test case '{test_case.query[:50]}...' with averaging: {e}"
            ) from e

