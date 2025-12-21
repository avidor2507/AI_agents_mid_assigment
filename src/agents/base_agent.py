"""
Base agent definitions for the Agent Architecture (Phase 5).

This module defines:
- AgentInterface: minimal common interface for all agents.
- BaseAgent: shared LLM setup and helper for sending prompts.

Concrete agents (e.g., SummarizationExpertAgent, NeedleInHaystackAgent)
inherit from BaseAgent and:
- Own their specific retriever internally (Summary vs Hierarchical)
- Implement their own handle_query() logic that:
  - Uses the retriever to get relevant context
  - Calls the BaseAgent's LLM helper to generate the final answer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Callable
from src.config.constants import AgentType
from src.config.settings import config
from src.utils.logger import logger
from src.utils.exceptions import AgentError
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)

class AgentInterface(ABC):
    """
    Minimal interface for all agents in the system.

    Agents expose their type and a single handle_query() method.
    """

    @abstractmethod
    def get_agent_type(self) -> AgentType:
        """Return the AgentType enum value for this agent."""

    @abstractmethod
    def handle_query(self, query: str) -> Dict[str, Any]:
        """
        Handle a user query and return a structured response.

        The response is a dictionary to keep it easy to serialize/log in the
        test scripts (e.g., JSON in results/logs/agents).
        """


class BaseAgent(AgentInterface, ABC):
    """
    Abstract base class for agents.

    Responsibilities:
    - Store the AgentType
    - Initialize an LLM client (required)
    - Provide a private helper for sending prompts to the LLM

    Concrete agents focus on:
    - Building the right retrieval query
    - Constructing the LLM prompt from (user question + retrieved context)
    - Returning a JSON-friendly response object
    """

    def __init__(
        self,
        agent_type: AgentType,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        tools: Optional[List[callable]] = None,
    ) -> None:
        self._agent_type = agent_type
        self.logger = logger
        self._llm = None
        self._tools = tools

        model_name = model or config.LLM_MODEL
        key = api_key or config.OPENAI_API_KEY

        try:
            self._llm = ChatOpenAI(
                model=model_name,
                api_key=key,
            )
            if self._tools:
                self._llm = self._llm.bind_tools(self._tools)
            self.logger.info(
                f"[{agent_type.value}] Initialized LLM model for agent: {model_name}"
            )
        except Exception as e:  # pragma: no cover - network/credentials dependent
            raise AgentError(
                f"Failed to initialize LLM for agent '{agent_type.value}': {e}"
            ) from e

    # ------------------------------------------------------------------
    # AgentInterface implementation
    # ------------------------------------------------------------------
    def get_agent_type(self) -> AgentType:
        return self._agent_type

    # ------------------------------------------------------------------
    # LLM helper
    # ------------------------------------------------------------------
    def _call_llm(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self._llm:
            raise AgentError("LLM client is not initialized")

        messages = []

        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))

        messages.append(HumanMessage(content=prompt))

        # Safety: prevent infinite tool loops
        max_tool_iterations = 5
        iteration = 0

        while iteration < max_tool_iterations:
            iteration += 1

            response = self._llm.invoke(messages)

            # If the model wants to call tools
            if response.tool_calls:
                messages.append(response)

                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_id = tool_call["id"]

                    # Find the tool
                    tool_fn = next(
                        (t for t in (self._tools or []) if t.__name__ == tool_name),
                        None,
                    )

                    if tool_fn is None:
                        # Tool not found → hard failure or controlled fallback
                        messages.append(
                            ToolMessage(
                                content="no tool found",
                                tool_call_id=tool_id,
                            )
                        )
                        continue

                    try:
                        result = tool_fn(**tool_args)
                    except Exception as e:
                        result = f"tool execution failed: {e}"

                    messages.append(
                        ToolMessage(
                            content=str(result),
                            tool_call_id=tool_id,
                        )
                    )

                # Continue loop to let LLM react to tool output
                continue

            # No tool calls → final answer
            # Extract text content from AIMessage object
            if hasattr(response, 'content'):
                return str(response.content) if response.content else ""
            # Fallback: convert to string if content attribute doesn't exist
            return str(response)

        # Safety fallback
        raise AgentError(
            f"Exceeded max tool iterations for agent '{self._agent_type.value}'"
        )
