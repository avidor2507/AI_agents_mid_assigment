"""
Agent architecture module.

This package defines the Agent layer that sits on top of the retrieval system.
It provides:

- AgentInterface: Minimal contract for all agents.
- BaseAgent: Shared LLM setup and helper for sending prompts.
- RouterAgent: Router agent that classifies queries (summary vs needle).
- SummarizationExpertAgent: Uses SummaryRetriever internally for high-level/timeline questions.
- NeedleInHaystackAgent: Uses HierarchicalRetriever / AutoMerging internally for deep factual search.
- OrchestratorSystem: High-level coordinator (router -> specialist agent -> answer).
"""

from src.agents.base_agent import AgentInterface, BaseAgent
from src.agents.router_agent import RouterAgent
from src.agents.summarization_agent import SummarizationExpertAgent
from src.agents.needle_in_haystack_agent import NeedleInHaystackAgent
from src.agents.orchestrator_system import OrchestratorSystem

__all__ = [
    "AgentInterface",
    "BaseAgent",
    "RouterAgent",
    "SummarizationExpertAgent",
    "NeedleInHaystackAgent",
    "OrchestratorSystem",
]


