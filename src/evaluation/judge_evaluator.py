"""
JudgeEvaluator implementation using LLM-as-a-judge pattern.

Evaluates system responses on multiple metrics:
- Answer Correctness: Does the answer match the expected answer?
- Context Relevancy: Are the retrieved chunks relevant to the query?
- Context Recall: Were the expected context chunks retrieved?
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.config.constants import EvaluationMetric
from src.config.settings import config
from src.utils.exceptions import EvaluationError
from src.utils.logger import logger


class JudgeEvaluator:
    """
    LLM-based evaluator using the "LLM-as-a-judge" pattern.
    
    Uses a separate LLM model to evaluate responses on multiple metrics.
    Each metric is scored on a 0-1 scale.
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """
        Initialize the judge evaluator.
        
        Args:
            model: LLM model to use for judging (defaults to JUDGE_LLM_MODEL from config)
            api_key: OpenAI API key (defaults to OPENAI_API_KEY from config)
        """
        self.logger = logger
        model_name = model or config.JUDGE_LLM_MODEL
        key = api_key or config.OPENAI_API_KEY
        
        if not key:
            raise EvaluationError(
                "OPENAI_API_KEY is required for JudgeEvaluator. "
                "Set it in environment variables or pass it as api_key."
            )
        
        try:
            self._llm = ChatOpenAI(
                model=model_name,
                api_key=key,
            )
            self.logger.info(f"JudgeEvaluator initialized with model: {model_name}")
        except Exception as e:
            raise EvaluationError(f"Failed to initialize LLM for JudgeEvaluator: {e}") from e
    
    def evaluate(
        self,
        metric: EvaluationMetric,
        query: str,
        answer: str,
        retrieved_context: List[Dict[str, Any]],
        expected_answer: Optional[str] = None,
        expected_context: Optional[List[str]] = None,
        ground_truth: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Evaluate a response on a specific metric.
        
        Args:
            metric: The evaluation metric to use
            query: The original user query
            answer: The system's answer
            retrieved_context: List of retrieved chunks (dicts with 'text' and 'metadata')
            expected_answer: Expected answer (for ANSWER_CORRECTNESS)
            expected_context: Expected context chunks (for CONTEXT_RECALL)
            ground_truth: Additional ground truth information
            
        Returns:
            Score between 0.0 and 1.0
        """
        if metric == EvaluationMetric.ANSWER_CORRECTNESS:
            return self._evaluate_answer_correctness(
                query=query,
                answer=answer,
                expected_answer=expected_answer or "",
            )
        elif metric == EvaluationMetric.CONTEXT_RELEVANCY:
            return self._evaluate_context_relevancy(
                query=query,
                retrieved_context=retrieved_context,
            )
        elif metric == EvaluationMetric.CONTEXT_RECALL:
            return self._evaluate_context_recall(
                query=query,
                retrieved_context=retrieved_context,
                expected_context=expected_context or [],
            )
        else:
            raise EvaluationError(f"Unknown evaluation metric: {metric}")
    
    def _call_llm(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Call the LLM with error handling."""
        try:
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=prompt))
            
            response = self._llm.invoke(messages)
            
            # Extract text content from AIMessage object
            if hasattr(response, 'content'):
                return str(response.content) if response.content else ""
            return str(response)
        except Exception as e:
            raise EvaluationError(f"LLM call failed in JudgeEvaluator: {e}") from e
    
    def _parse_score(self, llm_output: str) -> float:
        """
        Parse a score from LLM output.
        
        Looks for a number between 0 and 1 in the output.
        """
        import re
        
        # Try to find a number between 0 and 1
        patterns = [
            r"\b(0\.\d+)\b",  # 0.0 to 0.9
            r"\b(1\.0)\b",    # 1.0
            r"\b(0)\b",       # 0
            r"\b(1)\b",       # 1
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, llm_output)
            if matches:
                try:
                    score = float(matches[0])
                    return max(0.0, min(1.0, score))  # Clamp to [0, 1]
                except ValueError:
                    continue
        
        # If no number found, try to extract from common phrases
        output_lower = llm_output.lower()
        if "correct" in output_lower or "accurate" in output_lower or "yes" in output_lower:
            if "not" not in output_lower and "incorrect" not in output_lower:
                return 1.0
        if "incorrect" in output_lower or "wrong" in output_lower or "no" in output_lower:
            return 0.0
        
        # Default to 0.5 if we can't parse
        self.logger.warning(f"Could not parse score from LLM output: {llm_output[:100]}")
        return 0.5
    
    def _evaluate_answer_correctness(
        self,
        query: str,
        answer: str,
        expected_answer: str,
    ) -> float:
        """
        Evaluate if the answer is correct compared to expected answer.
        
        Score: 1.0 if answer is correct, 0.0 if incorrect, 0.0-1.0 for partial matches.
        """
        system_prompt = (
            "You are an evaluator for an insurance claim document Q&A system.\n\n"
            "Your task is to determine if a system's answer correctly answers the user's query "
            "compared to the expected answer.\n\n"
            "EVALUATION CRITERIA:\n"
            "- Score 1.0 if the answer is factually correct and fully addresses the query\n"
            "- Score 0.0 if the answer is factually incorrect or does not address the query\n"
            "- Score between 0.0-1.0 for partial correctness (e.g., correct but incomplete)\n\n"
            "IMPORTANT:\n"
            "- Consider semantic equivalence (same meaning, different wording is OK)\n"
            "- For exact values (amounts, IDs, dates), they must match exactly\n"
            "- For descriptive answers, focus on factual correctness, not wording\n"
            "- Respond with ONLY a number between 0.0 and 1.0 (e.g., '0.8' or '1.0')\n"
        )
        
        prompt = (
            "Query: {query}\n\n"
            "Expected Answer: {expected_answer}\n\n"
            "System Answer: {answer}\n\n"
            "Score the correctness of the system's answer (0.0 to 1.0):"
        ).format(
            query=query,
            expected_answer=expected_answer,
            answer=answer,
        )
        
        try:
            output = self._call_llm(prompt, system_prompt=system_prompt)
            score = self._parse_score(output)
            self.logger.debug(f"Answer correctness score: {score} for query: {query[:50]}...")
            return score
        except Exception as e:
            self.logger.error(f"Error evaluating answer correctness: {e}")
            raise EvaluationError(f"Failed to evaluate answer correctness: {e}") from e
    
    def _evaluate_context_relevancy(
        self,
        query: str,
        retrieved_context: List[Dict[str, Any]],
    ) -> float:
        """
        Evaluate if the retrieved context is relevant to the query.
        
        Score: Average relevancy of retrieved chunks (0.0-1.0 per chunk, then averaged).
        """
        if not retrieved_context:
            return 0.0
        
        system_prompt = (
            "You are an evaluator for an insurance claim document retrieval system.\n\n"
            "Your task is to determine if retrieved context chunks are relevant to the user's query.\n\n"
            "EVALUATION CRITERIA:\n"
            "- Score 1.0 if the chunk is highly relevant and directly addresses the query\n"
            "- Score 0.5 if the chunk is somewhat relevant but not directly related\n"
            "- Score 0.0 if the chunk is irrelevant to the query\n\n"
            "Respond with ONLY a number between 0.0 and 1.0 for each chunk.\n"
        )
        
        scores = []
        for i, chunk in enumerate(retrieved_context):
            chunk_text = chunk.get("text", "")
            if not chunk_text:
                scores.append(0.0)
                continue
            
            prompt = (
                "Query: {query}\n\n"
                "Retrieved Context Chunk {idx}:\n{chunk_text}\n\n"
                "How relevant is this chunk to the query? (0.0 to 1.0):"
            ).format(
                query=query,
                idx=i + 1,
                chunk_text=chunk_text[:500],  # Limit chunk text for prompt
            )
            
            try:
                output = self._call_llm(prompt, system_prompt=system_prompt)
                score = self._parse_score(output)
                scores.append(score)
            except Exception as e:
                self.logger.warning(f"Error evaluating chunk {i} relevancy: {e}")
                scores.append(0.5)  # Default to neutral if evaluation fails
        
        avg_score = sum(scores) / len(scores) if scores else 0.0
        self.logger.debug(f"Context relevancy score: {avg_score:.3f} ({len(retrieved_context)} chunks)")
        return avg_score
    
    def _evaluate_context_recall(
        self,
        query: str,
        retrieved_context: List[Dict[str, Any]],
        expected_context: List[str],
    ) -> float:
        """
        Evaluate if the expected context chunks were retrieved.
        
        Score: Proportion of expected chunks that were retrieved (0.0-1.0).
        """
        if not expected_context:
            # If no expected context specified, cannot evaluate recall
            return 1.0  # Return perfect score to avoid penalizing
        
        if not retrieved_context:
            return 0.0
        
        # Extract text from retrieved chunks
        retrieved_texts = [chunk.get("text", "").strip() for chunk in retrieved_context]
        
        system_prompt = (
            "You are an evaluator for an insurance claim document retrieval system.\n\n"
            "Your task is to determine if expected context chunks were retrieved by the system.\n\n"
            "EVALUATION CRITERIA:\n"
            "- For each expected chunk, check if a similar/containing chunk exists in retrieved chunks\n"
            "- Consider semantic similarity (same meaning, different wording is OK)\n"
            "- Consider partial matches (if retrieved chunk contains expected content)\n"
            "- Respond with the number of expected chunks that were found (e.g., '2 out of 3' or '3/3')\n"
        )
        
        # For each expected chunk, check if it's in retrieved chunks
        found_count = 0
        for expected_text in expected_context:
            if not expected_text.strip():
                continue
            
            prompt = (
                "Query: {query}\n\n"
                "Expected Context Chunk:\n{expected_text}\n\n"
                "Retrieved Context Chunks:\n{retrieved_texts}\n\n"
                "Was the expected chunk found in the retrieved chunks? "
                "Answer 'YES' or 'NO' (consider semantic similarity and partial matches):"
            ).format(
                query=query,
                expected_text=expected_text[:300],
                retrieved_texts="\n\n---\n\n".join(
                    [f"Chunk {i+1}:\n{t[:300]}" for i, t in enumerate(retrieved_texts)]
                ),
            )
            
            try:
                output = self._call_llm(prompt, system_prompt=system_prompt).strip().upper()
                if "YES" in output or ("FOUND" in output and "NOT" not in output):
                    found_count += 1
            except Exception as e:
                self.logger.warning(f"Error checking if expected chunk was found: {e}")
                # On error, don't count as found
        
        recall_score = found_count / len(expected_context) if expected_context else 1.0
        self.logger.debug(
            f"Context recall score: {recall_score:.3f} "
            f"({found_count}/{len(expected_context)} expected chunks found)"
        )
        return recall_score

