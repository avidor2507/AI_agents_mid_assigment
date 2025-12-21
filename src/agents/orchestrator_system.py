"""
OrchestratorSystem implementation.

This component wires the agent architecture together:

1. Initializes all agents in __init__:
   - RouterAgent (router)
   - SummarizationExpertAgent
   - NeedleInHaystackAgent
2. Exposes a single handle_query() method that:
   - Sends the query to the RouterAgent for routing (summary vs needle)
   - Forwards the query to the chosen specialist agent
   - Returns the specialist agent's answer together with routing metadata.
"""

from __future__ import annotations

from src.agents.router_agent import RouterAgent
from src.agents.summarization_agent import SummarizationExpertAgent
from src.agents.needle_in_haystack_agent import NeedleInHaystackAgent
from src.config.constants import AgentType
from src.utils.logger import logger


class OrchestratorSystem:
    """
    High-level orchestration system for the agents.
    """

    def __init__(self) -> None:
        self.logger = logger

        # Initialize all agents here (single source of truth)
        self.router_agent = RouterAgent()
        self.summarization_agent = SummarizationExpertAgent()
        self.needle_agent = NeedleInHaystackAgent()

        self.logger.info("OrchestratorSystem initialized with Router, Summary, and Needle agents.")

    def handle_query(self, query: str) -> str:
        """
        Execute the full routing + answering chain:
        Router (RouterAgent) -> Specialist agent -> Answer.

        Returns:
            The answer string from the specialist agent.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string")

        self.logger.info("[OrchestratorSystem] Received query for processing.")

        # Step 1: Get routing decision from RouterAgent
        routing_response = self.router_agent.handle_query(query)
        routing_decision = routing_response.get("routing_decision", {})
        primary_agent_value = routing_decision.get(
            "primary_agent_type",
            AgentType.SUMMARIZATION_EXPERT.value,
        )

        try:
            primary_agent_type = AgentType(primary_agent_value)
        except ValueError:
            self.logger.warning(
                f"[OrchestratorSystem] Unknown primary_agent_type={primary_agent_value!r}; "
                f"defaulting to summarization."
            )
            primary_agent_type = AgentType.SUMMARIZATION_EXPERT

        # Step 2: Forward query to the selected specialist agent
        if primary_agent_type == AgentType.NEEDLE_IN_HAYSTACK:
            agent = self.needle_agent
        else:
            agent = self.summarization_agent

        agent_response = agent.handle_query(query)

        # Step 3: Extract and return the answer string
        answer = agent_response.get("answer", "")
        
        self.logger.info(
            f"[OrchestratorSystem] Query handled by agent_type={primary_agent_type.value}"
        )
        return answer


