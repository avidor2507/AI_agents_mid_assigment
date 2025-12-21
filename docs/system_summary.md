# Insurance Claim Timeline Retrieval System - 1-Page Summary

## System Overview

A Retrieval-Augmented Generation (RAG) system that processes insurance claim PDFs and provides accurate, context-aware answers using hierarchical indexing, intelligent agent routing, and LLM-based evaluation.

## Architecture

**Dual-Agent System with Intelligent Routing:**
- **RouterAgent**: Classifies queries using LLM (fallback: rule-based) and routes to specialist agents
- **SummarizationExpertAgent**: Handles high-level queries (summaries, timelines) using Summary Index
- **NeedleInHaystackAgent**: Handles precise factual queries using Hierarchical Index with auto-merging

**Three-Level Hierarchical Chunking:**
- Small (100-200 tokens): Precise facts
- Medium (500-800 tokens): Balanced context
- Large (1500-2000 tokens): Broad summaries
- 20% overlap ensures boundary information preservation

**Dual Index Architecture:**
- **Hierarchical Index**: Stores chunks at 3 granularities with metadata (section, timestamp, position)
- **Summary Index**: Stores summaries at chunk, section, and document levels
- Both use ChromaDB with OpenAI embeddings (text-embedding-3-small)

**Advanced Retrieval:**
- Auto-merging of adjacent chunks for broader context
- Time-aware reranking (prioritizes chunks with matching timestamps)
- Section-aware reranking (prioritizes explicitly mentioned sections)
- Direct section filtering when single section requested

## Main Results

**Evaluation Metrics (LLM-as-a-Judge):**
- Answer Correctness: Measures factual accuracy against expected answers
- Context Relevancy: Measures retrieval quality (average relevancy of retrieved chunks)
- Context Recall: Measures retrieval completeness (proportion of expected chunks retrieved)

**System Capabilities:**
- Handles diverse query types: exact values, summaries, timelines, section-specific queries
- Intelligent routing ensures appropriate agent handles each query type
- Reranking and filtering improve retrieval accuracy
- Auto-merging provides broader context when needed

## MCP Integration

**Model Context Protocol Tools:**
- **time_diff_tool**: Calculates time differences between dates/timestamps
- Supports multiple date formats (ISO, slash dates, datetime strings)
- Returns human-readable time differences
- Bound to agents via BaseAgent for timeline-related queries

## Key Features

✅ Multi-level hierarchical chunking with parent-child relationships  
✅ Semantic search with OpenAI embeddings  
✅ LLM-based agent routing with rule-based fallback  
✅ Specialized retrievers with reranking and filtering  
✅ Auto-merging for adaptive context window  
✅ LLM-as-a-judge evaluation framework  
✅ Section and time-aware retrieval optimization  
✅ MCP tool integration for specialized operations  

## Technology Stack

- **LLM**: OpenAI GPT-4o-mini (ChatOpenAI from langchain)
- **Embeddings**: OpenAI text-embedding-3-small
- **Vector DB**: ChromaDB
- **Chunking**: tiktoken for token counting, intelligent text splitting
- **Evaluation**: LLM-as-a-judge with GPT-4o-mini

## Usage

1. Place PDF in `data/raw/claim.pdf`
2. Run `python main.py`
3. Interactive CLI for queries
4. Type `eval` for evaluation mode

## Limitations

- PDF parsing quality depends on document structure
- Semantic search may retrieve similar but irrelevant chunks (mitigated by reranking)
- LLM dependency introduces hallucination risk (mitigated by strict prompts)
- Fixed chunk sizes may not fit all content types perfectly

