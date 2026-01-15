"""
NeedleInHaystackAgent implementation.

This agent:
- Inherits from BaseAgent
- Owns its own HierarchicalRetriever (with auto-merge) internally
- Uses the LLM (via BaseAgent) to answer precise factual questions

Examples:
- "What is the exact registration number of the insured vehicle?"
- "What time did the collision occur on March 3rd?"
- "What was the policy excess amount?"
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.agents.base_agent import BaseAgent
from src.config.constants import AgentType, ChunkSize
from src.indexing.index_manager import IndexManager
from src.retrieval.hierarchical_retriever import HierarchicalRetriever
from src.utils.exceptions import RetrievalError
from src.mcp.time_diff_tool import get_date_diff

class NeedleInHaystackAgent(BaseAgent):
    """
    Agent specialized in precise factual queries.

    This agent:
    - Loads the Hierarchical index via IndexManager
    - Uses HierarchicalRetriever with auto-merging enabled
    - Leverages time/section-aware reranking for better precision
    - Calls the LLM to generate a concise, factual answer
    """

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None) -> None:
        super().__init__(agent_type=AgentType.NEEDLE_IN_HAYSTACK, model=model, api_key=api_key, tools=[get_date_diff])
        self._retriever = self._create_retriever()

    def _create_retriever(self) -> HierarchicalRetriever:
        index_manager = IndexManager()
        index_manager.initialize()
        indices_loaded = index_manager.load_indices()
        if not indices_loaded:
            raise RetrievalError(
                "Hierarchical index is not loaded. Please run test_indexing.py first to build indices."
            )

        collection = index_manager.get_hierarchical_collection()
        if collection is None:
            raise RetrievalError("Hierarchical index collection is not available.")

        return HierarchicalRetriever(collection, enable_auto_merge=True)

    # ------------------------------------------------------------------
    # AgentInterface implementation
    # ------------------------------------------------------------------
    def handle_query(self, query: str) -> Dict[str, Any]:
        q = (query or "").strip()
        if not q:
            raise ValueError("Query must be a non-empty string")

        self.logger.info("[NeedleInHaystackAgent] Handling query...")

        # Retrieve precise chunks from the hierarchical index
        try:
            results = self._retriever.retrieve(
                q,
                top_k=10,
                filters=None,
                start_level=ChunkSize.SMALL.value,
                use_time_rerank=True,
                use_section_rerank=True,
            )
        except Exception as e:
            raise RetrievalError(f"Error retrieving hierarchical chunks for agent: {e}") from e

        if results:
            context_lines: List[str] = []
            for i, item in enumerate(results, start=1):
                text = (item.get("text") or "").strip()
                meta = item.get("metadata") or {}

                section_id = meta.get("section_id") or meta.get("section")
                level = meta.get("level")
                timestamp = meta.get("timestamp")

                meta_parts = []
                if level:
                    meta_parts.append(f"level={level}")
                if section_id:
                    meta_parts.append(f"section={section_id}")
                if timestamp:
                    meta_parts.append(f"timestamp={timestamp}")

                meta_str = ", ".join(meta_parts) if meta_parts else "no-metadata"
                prefix = f"[{i}] ({meta_str}) "
                context_lines.append(prefix + text)

            context_block = "\n\n".join(context_lines)
        else:
            context_block = (
                "No precise matching chunks were found in the Hierarchical index "
                "for this question."
            )

        system_prompt = (
            "You are a precision question-answering assistant for insurance claim documents.\n\n"
            "Your task is to answer a single, specific factual question using ONLY the provided context.\n"
            "This is not a summarization task.\n\n"
            "STRICT RULES:\n"
            "- For time difference questions, use the 'get_date_diff' tool to get the time difference. if you can't find the tool return 'can't calculate time difference'\n"
            "- Use ONLY information explicitly present in the context.\n"
            "- Do NOT infer, assume, generalize, or add background knowledge.\n"
            "- Do NOT include information from outside the context.\n"
            "- If the answer is not fully supported by the context, respond exactly with:\n"
            "'Not found in the provided context.'\n\n"
            "ANSWERING GUIDELINES:\n"
            "- Be concise and factual.\n"
            "- Prefer exact wording, values, names, timestamps, and identifiers as written.\n"
            "- Do not restate the question.\n"
            "- Do not add explanations, commentary, or formatting.\n"    
            "- Do not merge information from unrelated sections unless explicitly required.\n\n"
            "You must treat the context as the single source of truth."
        )

        prompt = (
            "Answer the following factual question using ONLY the context below.\n\n"
            "Question:\n"
            f"{q}\n\n"
            "Context:\n"
            f"{context_block}\n\n"
            "Provide a single, precise answer.\n"
            "if answer is 'can't calculate time difference' ay this as response as it."
            "If the answer is not explicitly stated in the context, say:\n"
            "'Not found in the provided context.'"
        )

        answer_text = self._call_llm(prompt=prompt, system_prompt=system_prompt)

        return {
            "agent_type": self.get_agent_type().value,
            "agent_name": "NeedleInHaystackAgent",
            "query": query,
            "answer": answer_text,
            "retrieval": {
                "result_count": len(results),
                "results": results,
            },
        }


