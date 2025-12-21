"""
TestSuite implementation for running and evaluating test cases.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.agents.orchestrator_system import OrchestratorSystem
from src.config.constants import EvaluationMetric
from src.config.settings import config
from src.evaluation.judge_evaluator import JudgeEvaluator
from src.evaluation.test_case import TestCase
from src.utils.exceptions import EvaluationError
from src.utils.logger import logger


class TestSuite:
    """
    Test suite for running and evaluating test cases.
    
    Loads test cases, runs queries through the system, evaluates responses,
    and generates evaluation reports.
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
        self.test_cases: List[TestCase] = []
        self.results: List[Dict[str, Any]] = []
    
    def load_test_cases(self, test_cases: List[TestCase]) -> None:
        """
        Load test cases into the test suite.
        
        Args:
            test_cases: List of TestCase objects
        """
        self.test_cases = test_cases
        self.logger.info(f"Loaded {len(self.test_cases)} test cases")
    
    def run_test_case(
        self,
        test_case: TestCase,
        get_retrieval_context: bool = True,
    ) -> Dict[str, Any]:
        """
        Run a single test case and evaluate the response.
        
        Args:
            test_case: TestCase to run
            get_retrieval_context: If True, get full agent response with retrieval context
            
        Returns:
            Dictionary containing test results and scores
        """
        self.logger.info(f"Running test case: {test_case.query[:60]}...")
        
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
            scores: Dict[str, float] = {}
            
            # Answer Correctness
            try:
                scores["answer_correctness"] = self.evaluator.evaluate(
                    metric=EvaluationMetric.ANSWER_CORRECTNESS,
                    query=test_case.query,
                    answer=answer,
                    retrieved_context=retrieved_context,
                    expected_answer=test_case.expected_answer,
                    ground_truth=test_case.ground_truth,
                )
            except Exception as e:
                self.logger.error(f"Error evaluating answer correctness: {e}")
                scores["answer_correctness"] = 0.0
            
            # Context Relevancy
            try:
                scores["context_relevancy"] = self.evaluator.evaluate(
                    metric=EvaluationMetric.CONTEXT_RELEVANCY,
                    query=test_case.query,
                    answer=answer,
                    retrieved_context=retrieved_context,
                )
            except Exception as e:
                self.logger.error(f"Error evaluating context relevancy: {e}")
                scores["context_relevancy"] = 0.0
            
            # Context Recall (only if expected_context is provided)
            if test_case.expected_context:
                try:
                    scores["context_recall"] = self.evaluator.evaluate(
                        metric=EvaluationMetric.CONTEXT_RECALL,
                        query=test_case.query,
                        answer=answer,
                        retrieved_context=retrieved_context,
                        expected_context=test_case.expected_context,
                    )
                except Exception as e:
                    self.logger.error(f"Error evaluating context recall: {e}")
                    scores["context_recall"] = 0.0
            else:
                scores["context_recall"] = None  # Not evaluated if no expected context
            
            # Build result dictionary
            result: Dict[str, Any] = {
                "test_case": {
                    "query": test_case.query,
                    "expected_answer": test_case.expected_answer,
                    "category": test_case.category,
                    "description": test_case.description,
                },
                "response": {
                    "answer": answer,
                    "retrieved_context_count": len(retrieved_context),
                },
                "scores": scores,
            }
            
            self.logger.info(
                f"Test case completed. Scores: {scores}"
            )
            return result
            
        except Exception as e:
            self.logger.error(f"Error running test case: {e}")
            raise EvaluationError(f"Failed to run test case '{test_case.query[:50]}...': {e}") from e
    
    def run_all(self, get_retrieval_context: bool = True) -> List[Dict[str, Any]]:
        """
        Run all test cases and collect results.
        
        Args:
            get_retrieval_context: If True, get full agent response with retrieval context
            
        Returns:
            List of result dictionaries
        """
        if not self.test_cases:
            raise EvaluationError("No test cases loaded. Call load_test_cases() first.")
        
        self.logger.info(f"Running {len(self.test_cases)} test cases...")
        self.results = []
        
        for i, test_case in enumerate(self.test_cases, start=1):
            self.logger.info(f"Test case {i}/{len(self.test_cases)}")
            try:
                result = self.run_test_case(test_case, get_retrieval_context=get_retrieval_context)
                self.results.append(result)
            except Exception as e:
                self.logger.error(f"Test case {i} failed: {e}")
                # Add error result
                self.results.append({
                    "test_case": {
                        "query": test_case.query,
                        "expected_answer": test_case.expected_answer,
                        "category": test_case.category,
                    },
                    "error": str(e),
                    "scores": {},
                })
        
        self.logger.info(f"Completed running all test cases. {len(self.results)} results collected.")
        return self.results
    
    def generate_report(
        self,
        output_file: Optional[Path] = None,
        include_details: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate an evaluation report from the results.
        
        Args:
            output_file: Path to save the report JSON file (optional)
            include_details: If True, include detailed results for each test case
            
        Returns:
            Dictionary containing the evaluation report
        """
        if not self.results:
            raise EvaluationError("No results available. Run test cases first using run_all().")
        
        # Calculate aggregate statistics
        all_scores: Dict[str, List[float]] = {
            "answer_correctness": [],
            "context_relevancy": [],
            "context_recall": [],
        }
        
        for result in self.results:
            scores = result.get("scores", {})
            for metric in all_scores:
                if scores.get(metric) is not None:
                    all_scores[metric].append(scores[metric])
        
        # Calculate averages
        averages: Dict[str, Optional[float]] = {}
        for metric, scores_list in all_scores.items():
            if scores_list:
                averages[metric] = sum(scores_list) / len(scores_list)
            else:
                averages[metric] = None
        
        # Count by category
        category_counts: Dict[str, int] = {}
        for result in self.results:
            category = result.get("test_case", {}).get("category", "unknown")
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Build report
        report: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_test_cases": len(self.results),
                "average_scores": averages,
                "category_distribution": category_counts,
            },
        }
        
        if include_details:
            report["detailed_results"] = self.results
        
        # Save to file if specified
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Evaluation report saved to: {output_file}")
        
        return report
    
    def print_summary(self) -> None:
        """Print a summary of the evaluation results."""
        if not self.results:
            print("No results available. Run test cases first.")
            return
        
        report = self.generate_report(include_details=False)
        summary = report["summary"]
        
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Total Test Cases: {summary['total_test_cases']}")
        print("\nAverage Scores:")
        for metric, score in summary["average_scores"].items():
            if score is not None:
                print(f"  {metric}: {score:.3f}")
            else:
                print(f"  {metric}: N/A")
        print("\nCategory Distribution:")
        for category, count in summary["category_distribution"].items():
            print(f"  {category}: {count}")
        print("=" * 60 + "\n")

