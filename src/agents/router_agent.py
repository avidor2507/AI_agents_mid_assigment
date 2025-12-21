"""
RouterAgent implementation.

Responsibilities:
- Inspect incoming queries and decide which agent/index should handle them
  (Summary vs Hierarchical).
- Use a "model as a function" for routing decisions when possible (LLM-based),
  with a deterministic rule-based fallback.
- Provide structured routing metadata for logging and debugging.

Note:
The RouterAgent itself does not talk to the indices directly. It focuses on
classification/routing and can be used by OrchestratorSystem to choose the
appropriate specialist agent.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.agents.base_agent import BaseAgent
from src.config.constants import AgentType
from src.utils.exceptions import AgentError


class RouterAgent(BaseAgent):
    """
    Router agent.

    This agent does NOT answer user questions directly. Instead, it returns a
    routing decision describing which specialist agent should handle the query.
    """

    def __init__(
        self,
        use_llm_routing: bool = True,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        # Initialize BaseAgent with ROUTER/MANAGER type (BaseAgent handles LLM initialization)
        super().__init__(agent_type=AgentType.MANAGER, model=model, api_key=api_key)
        # Check if LLM is available for routing decisions
        self._use_llm_routing = bool(use_llm_routing and self._llm is not None)

    def handle_query(self, query: str) -> Dict[str, Any]:
        """
        Return a routing decision for the given query.

        The decision object is JSON-friendly and can be logged or passed to
        the orchestrator system.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("RouterAgent received an invalid/empty query")

        self.logger.info(f"[RouterAgent] Routing query: {query[:100]}...")

        decision = self.route(query)

        response: Dict[str, Any] = {
            "agent_type": self.get_agent_type().value,
            "agent_name": "RouterAgent",
            "original_query": query,
            "routing_decision": decision,
        }
        self.logger.info(
            f"[RouterAgent] Routed query to {decision['primary_agent_type']} "
            f"(source={decision['source']}, confidence={decision['confidence']:.2f})"
        )
        return response

    # ------------------------------------------------------------------
    # Routing logic
    # ------------------------------------------------------------------
    def route(self, query: str) -> Dict[str, Any]:
        """
        Compute a routing decision for the query.

        Returns:
            Dict with keys:
            - primary_agent_type: AgentType value ("summarization" or "needle")
            - source: "llm" or "rule_based"
            - confidence: float in [0, 1]
            - raw_output: raw LLM output (if applicable)
        """
        query = (query or "").strip()
        if not query:
            # Default to hierarchical agent when in doubt
            return self._build_decision(
                primary_agent_type=AgentType.NEEDLE_IN_HAYSTACK,
                source="rule_based",
                confidence=0.5,
                raw_output="empty_query",
            )

        if self._use_llm_routing:
            decision = self._llm_route(query)
        else:
            decision = self._rule_based_route(query)

        return decision

    def _build_decision(
        self,
        primary_agent_type: AgentType,
        source: str,
        confidence: float,
        raw_output: str,
    ) -> Dict[str, Any]:
        return {
            "primary_agent_type": primary_agent_type.value,
            "source": source,
            "confidence": float(confidence),
            "raw_output": raw_output,
        }

    def _llm_route(self, query: str) -> Dict[str, Any]:
        """
        Use an LLM as a "model function" to choose summary vs needle agent.

        The model is instructed to respond with a single token: 'summary' or
        'needle'. If parsing fails, we fall back to rule-based routing.
        """
        system_prompt = (
            "You are a routing agent for an insurance-claim RAG system.\n\n"
            "Your task is to decide which specialist agent should handle the user's query.\n\n"
            "AVAILABLE ROUTES:\n"
            "- 'needle' → for precise factual questions that require exact answers.\n"
            "- 'summary' → for requests that ask to summarize, explain, or provide overviews.\n\n"
            "ROUTING RULES:\n"
            "- Respond with EXACTLY one word: 'needle' or 'summary'.\n"
            "- Do NOT include explanations, punctuation, or additional text.\n"
            "- Do NOT answer the user's question.\n"
            "- Do NOT ask clarifying questions.\n\n"
            "CLASSIFICATION GUIDELINES:\n"
            "Use 'needle' if the query asks for:\n"
            "- Exact values (amounts, dates, times)\n"
            "- Identifiers (IDs, registration numbers, policy numbers)\n"
            "- Specific events or single facts\n"
            "- Yes/No factual confirmation\n\n"
            "Use 'summary' if the query asks for:\n"
            "- Summaries of sections or documents\n"
            "- Overviews, explanations, or descriptions\n"
            "- Timelines or multi-step processes\n"
            "- General understanding of content\n\n"
            "If the query is ambiguous, default to:\n"
            "'needle'"
        )

        prompt = (
            "Decide which specialist agent should handle the following user query.\n\n"
            "User query:\n"
            f"{query}\n\n"
            "Respond with exactly one word:\n"
            "'needle'\n"
            "or\n"
            "'summary'\n"
        )

        try:
            # Use BaseAgent's _call_llm with system_prompt for consistent error handling
            raw_output = self._call_llm(prompt=prompt, system_prompt=system_prompt)
            answer_lower = raw_output.strip().lower()

            if "summary" in answer_lower and "needle" not in answer_lower:
                target = AgentType.SUMMARIZATION_EXPERT
            elif "needle" in answer_lower and "summary" not in answer_lower:
                target = AgentType.NEEDLE_IN_HAYSTACK
            else:
                # Ambiguous or unexpected output: fall back to rules
                self.logger.warning(
                    f"RouterAgent LLM routing produced ambiguous output: {raw_output!r}; "
                    "falling back to rule-based routing."
                )
                return self._rule_based_route(query)

            confidence = 0.9  # Heuristic; could be refined
            return self._build_decision(
                primary_agent_type=target,
                source="llm",
                confidence=confidence,
                raw_output=raw_output,
            )
        except AgentError as e:
            # LLM call failed (AgentError from _call_llm) - fall back gracefully
            self.logger.warning(
                f"RouterAgent LLM routing failed: {e}; "
                "falling back to rule-based routing."
            )
            return self._rule_based_route(query, raw_output_override=str(e))
        except Exception as e:
            # Catch any other unexpected errors
            self.logger.warning(
                f"RouterAgent LLM routing failed with unexpected error: {e}; "
                "falling back to rule-based routing."
            )
            return self._rule_based_route(query, raw_output_override=str(e))

    def _rule_based_route(
        self,
        query: str,
        raw_output_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Simple deterministic routing based on keyword heuristics.

        This keeps tests stable and avoids external dependencies when the
        LLM is not available.
        """
        q = query.lower()

        summary_keywords = [
            "overview",
            "summary",
            "summarize",
            "high-level",
            "high level",
            "timeline",
            "what happened overall",
            "from incident to resolution",
            "big picture",
            "in general",
        ]

        needle_keywords = [
            "exact",
            "registration",
            "policy number",
            "reference number",
            "claim number",
            "driver name",
            "what time",
            "time did",
            "timestamp",
            "amount",
            "£",
            "section ",
        ]

        if any(kw in q for kw in summary_keywords):
            primary = AgentType.SUMMARIZATION_EXPERT
            confidence = 0.8
        elif any(kw in q for kw in needle_keywords):
            primary = AgentType.NEEDLE_IN_HAYSTACK
            confidence = 0.8
        else:
            # Default: for ambiguous queries, prefer summary first
            primary = AgentType.SUMMARIZATION_EXPERT
            confidence = 0.6

        return self._build_decision(
            primary_agent_type=primary,
            source="rule_based",
            confidence=confidence,
            raw_output=raw_output_override or "",
        )


