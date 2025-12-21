"""
SummarizationExpertAgent implementation.

This agent:
- Inherits from BaseAgent
- Owns its own SummaryRetriever internally
- Uses the LLM (via BaseAgent) to generate high-level/timeline answers

It is best suited for:
- "What is the overall timeline of the claim?"
- "Summarize what happened during the incident"
- "Give me a high-level overview of the claim"
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.agents.base_agent import BaseAgent
from src.config.constants import AgentType
from src.indexing.index_manager import IndexManager
from src.retrieval.summary_retriever import SummaryRetriever
from src.utils.exceptions import RetrievalError
from src.mcp.time_diff_tool import get_date_diff


class SummarizationExpertAgent(BaseAgent):
    """
    Agent specialized in high-level/timeline questions.

    This agent:
    - Loads the Summary index via IndexManager
    - Uses SummaryRetriever to get relevant summaries
    - Builds a prompt and calls the LLM to generate a concise answer
    """

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None) -> None:
        super().__init__(agent_type=AgentType.SUMMARIZATION_EXPERT, model=model, api_key=api_key, tools=[get_date_diff])
        self._retriever = self._create_retriever()

    def _create_retriever(self) -> SummaryRetriever:
        index_manager = IndexManager()
        index_manager.initialize()
        indices_loaded = index_manager.load_indices()
        if not indices_loaded:
            raise RetrievalError(
                "Summary index is not loaded. Please run test_indexing.py first to build indices."
            )

        collection = index_manager.get_summary_collection()
        if collection is None:
            raise RetrievalError("Summary index collection is not available.")

        return SummaryRetriever(collection)

    # ------------------------------------------------------------------
    # AgentInterface implementation
    # ------------------------------------------------------------------
    def handle_query(self, query: str) -> Dict[str, Any]:
        q = (query or "").strip()
        if not q:
            raise ValueError("Query must be a non-empty string")

        self.logger.info("[SummarizationExpertAgent] Handling query...")

        # Retrieve high-level summaries
        try:
            results = self._retriever.retrieve(q, top_k=5, use_section_rerank=True)
        except Exception as e:
            raise RetrievalError(f"Error retrieving summaries for agent: {e}") from e

        # Build context block for the LLM
        if results:
            context_lines: List[str] = []
            for i, item in enumerate(results, start=1):
                text = (item.get("text") or "").strip()
                meta = item.get("metadata") or {}
                summary_level = meta.get("summary_level", "unknown")
                section_id = meta.get("section_id") or meta.get("section")

                prefix = f"[{i}] (level={summary_level}"
                if section_id:
                    prefix += f", section={section_id}"
                prefix += ") "
                context_lines.append(prefix + text)

            context_block = "\n\n".join(context_lines)
        else:
            context_block = "No relevant summaries were found in the Summary index."

        system_prompt = (
            "You are a summarization assistant for insurance claim documents.\n\n"
            "Your task is to produce a concise, accurate summary of ONLY the provided content.\n"
            "This is a summarization task, not a question-answering task.\n\n"
            "STRICT RULES:\n"
            "- For time difference questions, use the 'get_date_diff' tool to get the time difference. if you can't find the tool return 'can't calculate time difference'\n"
            "- Use ONLY the information explicitly present in the provided context.\n"  
            "- Do NOT introduce information from outside the context.\n"
            "- Do NOT infer, assume, or generalize beyond what is stated.\n"
            "- Do NOT reference or summarize content from other sections or documents.\n"
            "- Do NOT include procedural commentary, lifecycle overviews, or conclusions unless they appear in the context.\n\n"
            "SCOPE CONTROL:\n"
            "- If the user requests a specific section, summarize ONLY that section.\n"
            "- Treat section boundaries as hard constraints.\n\n"
            "OUTPUT GUIDELINES:\n"
            "- Be concise and factual.\n"
            "- Preserve key facts, figures, names, dates, and identifiers.\n"
            "- Do NOT add headings unless they appear in the context.\n"
            "- Do NOT include opinions, recommendations, or analysis.\n\n"
            "If the provided context does not contain enough information to summarize,\n"
            "respond exactly with:\n"
            'Insufficient information in the provided context.'
        )

        prompt = (
            "Summarize the following content using ONLY the context provided.\n\n"

            "User request:\n"
            f"{q}\n\n"

            "Context:\n"
            f"{context_block}\n\n"

            "Provide a concise summary.\n"
             "if answer is 'can't calculate time difference' ay this as response as it."
            "If the context does not contain enough information to fulfill the request,\n"
            "respond with:\n"
            "'Insufficient information in the provided context.'"
        )

        answer_text = self._call_llm(prompt=prompt, system_prompt=system_prompt)

        return {
            "agent_type": self.get_agent_type().value,
            "agent_name": "SummarizationExpertAgent",
            "query": query,
            "answer": answer_text,
            "retrieval": {
                "result_count": len(results),
                "results": results,
            },
        }


