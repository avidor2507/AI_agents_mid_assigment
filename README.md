# Insurance Claim Timeline Retrieval System

A Retrieval-Augmented Generation (RAG) system for answering questions about insurance claim documents using hierarchical indexing, intelligent agent routing, and LLM-as-a-judge evaluation.

## Table of Contents

- [System Overview](#system-overview)
- [Architecture](#architecture)
- [Data Segmentation & Chunking](#data-segmentation--chunking)
- [Index Schemas](#index-schemas)
- [Agent Design](#agent-design)
- [MCP Integration](#mcp-integration)
- [Evaluation Methodology](#evaluation-methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Limitations & Trade-offs](#limitations--trade-offs)

## System Overview

This system processes insurance claim PDFs and provides accurate, context-aware answers to queries about claim details. It uses:

- **Hierarchical Chunking**: Multi-level text segmentation (small, medium, large chunks)
- **Dual Index Architecture**: Separate indices for precise facts and high-level summaries
- **Intelligent Agent Routing**: LLM-based router that selects the appropriate specialist agent
- **RAG Pipeline**: Semantic search + LLM generation with context
- **Auto-Merging Retrieval**: Automatically combines adjacent chunks for broader context
- **Time & Section-Aware Reranking**: Prioritizes chunks based on temporal and section relevance
- **LLM-as-a-Judge Evaluation**: Automated evaluation of system performance

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Query                               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │  RouterAgent   │
                    │  (LLM Routing) │
                    └───────┬────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
    ┌──────────────────┐      ┌──────────────────────┐
    │Summarization     │      │NeedleInHaystack      │
    │ExpertAgent       │      │Agent                 │
    └────────┬─────────┘      └──────────┬───────────┘
             │                           │
             ▼                           ▼
    ┌──────────────────┐      ┌──────────────────────┐
    │SummaryRetriever  │      │HierarchicalRetriever │
    └────────┬─────────┘      └──────────┬───────────┘
             │                           │
             ▼                           ▼
    ┌──────────────────┐      ┌──────────────────────┐
    │Summary Index     │      │Hierarchical Index    │
    │(ChromaDB)        │      │(ChromaDB)            │
    └──────────────────┘      └──────────────────────┘
```

### Key Components

1. **OrchestratorSystem**: Coordinates the routing and execution flow
2. **RouterAgent**: Classifies queries and routes to specialist agents
3. **SummarizationExpertAgent**: Handles high-level/timeline queries
4. **NeedleInHaystackAgent**: Handles precise factual queries
5. **IndexManager**: Singleton managing index lifecycle (build, load, cache)
6. **HierarchicalRetriever**: Retrieves with auto-merging and reranking
7. **SummaryRetriever**: Retrieves summaries with section-aware filtering
8. **JudgeEvaluator**: Evaluates system responses using LLM-as-a-judge

## Data Segmentation & Chunking

### Chunking Strategy

The system uses a **hierarchical chunking approach** with three levels:

#### Chunk Sizes

- **Small**: 100-200 tokens (~150 tokens default)
  - Fine-grained chunks for precise factual queries
  - Best for extracting exact values, IDs, timestamps

- **Medium**: 500-800 tokens (~650 tokens default)
  - Balanced context for moderate complexity queries
  - Good for understanding relationships within sections

- **Large**: 1500-2000 tokens (~1750 tokens default)
  - Broad context for comprehensive understanding
  - Used for high-level summaries and overviews

#### Overlap Strategy

- **20% overlap** between adjacent chunks
- Ensures important information at chunk boundaries isn't lost
- Preserves context continuity across chunk boundaries

#### Hierarchical Structure

Chunks are organized hierarchically:
- Small chunks are children of medium chunks
- Medium chunks are children of large chunks
- Maintains parent-child relationships for structural retrieval

### Chunking Rationale

1. **Multiple Granularities**: Different query types require different context sizes
   - Exact values need small, focused chunks
   - Summaries need broader context from large chunks

2. **Auto-Merging**: Adjacent small chunks can be merged automatically when they share:
   - Same section_id
   - Sequential position_index
   - This provides broader context without manual chunk size selection

3. **Intelligent Splitting**: Chunks are split at natural boundaries:
   - Paragraph boundaries (primary)
   - Sentence boundaries (secondary)
   - Preserves semantic coherence

## Index Schemas

### Hierarchical Index

Stores chunks at three granularity levels with metadata:

**Metadata Fields**:
- `chunk_id`: Unique identifier (e.g., `section_1_small_0`)
- `parent_id`: Parent chunk ID (for hierarchical structure)
- `level`: Chunk size level (`small`, `medium`, `large`)
- `section_id`: Section identifier (e.g., `section_1`)
- `section`: Alias for section_id
- `timestamp`: Event timestamp (if applicable)
- `chunk_text`: Full chunk text
- `position_index`: Position within section
- `document_id`: Parent document ID
- `claim_id`: Claim identifier

**Use Case**: Precise factual retrieval, exact value extraction

### Summary Index

Stores summaries at multiple levels:

**Metadata Fields**:
- `chunk_id`: Summary identifier (e.g., `summary_section_1`)
- `summary_level`: Level of summary (`chunk`, `section`, `document`)
- `section_id`: Section identifier (for section summaries)
- `document_id`: Document identifier
- `claim_id`: Claim identifier
- `chunk_text`: Summary text content

**Summary Levels**:
- **Chunk-level**: Summaries of individual chunks
- **Section-level**: Summaries of entire sections
- **Document-level**: Summaries of complete documents

**Use Case**: High-level queries, timelines, overviews

### Vector Store

Both indices use **ChromaDB** for:
- Semantic similarity search using OpenAI embeddings (`text-embedding-3-small`)
- Persistent storage on disk
- Metadata filtering capabilities

## Agent Design

### BaseAgent

All agents inherit from `BaseAgent` which provides:
- LLM initialization (ChatOpenAI from langchain)
- `_call_llm()` helper method with system prompt support
- Tool binding support (for MCP integration)
- Error handling with `AgentError` exceptions

### RouterAgent

**Purpose**: Classify queries and route to appropriate specialist agent

**Routing Strategy**:
1. **LLM-based routing** (primary): Uses GPT-4o-mini to classify query
   - Prompt: System prompt with routing guidelines
   - Response: "needle" or "summary"
   - Confidence: 0.9 when LLM routing succeeds

2. **Rule-based routing** (fallback): Keyword-based heuristics
   - Summary keywords: "overview", "summary", "timeline", "high-level"
   - Needle keywords: "exact", "registration", "amount", "timestamp"
   - Confidence: 0.6-0.8

**System Prompt Structure**:
```
- Role: Routing agent for insurance-claim RAG system
- Task: Decide which specialist agent should handle the query
- Routes: 'needle' (precise facts) or 'summary' (overviews)
- Rules: Respond with exactly one word
- Guidelines: Classification criteria for each route type
```

### SummarizationExpertAgent

**Purpose**: Handle high-level queries, summaries, timelines

**Retriever**: `SummaryRetriever`
- Retrieves from Summary Index
- Uses section-aware reranking
- Applies direct section filtering when single section mentioned

**Prompt Structure**:
```
System Prompt:
- Role: Summarization assistant for insurance claim documents
- Task: Produce concise, accurate summary of ONLY provided content
- Rules: Use ONLY information from context, no inference beyond context
- Scope: If user requests specific section, summarize ONLY that section
- Output: Concise, factual, preserve key facts and figures

User Prompt:
- User request: {query}
- Context: {retrieved summaries}
- Instructions: Provide summary or "Insufficient information" if needed
```

### NeedleInHaystackAgent

**Purpose**: Handle precise factual queries requiring exact answers

**Retriever**: `HierarchicalRetriever`
- Retrieves from Hierarchical Index
- Starts at small chunk level for precision
- Uses time-aware and section-aware reranking
- Applies auto-merging for broader context when needed

**Prompt Structure**:
```
System Prompt:
- Role: Precision question-answering assistant
- Task: Answer single, specific factual question using ONLY context
- Rules: Use ONLY explicit information, no inference or background knowledge
- Answer Guidelines: Be concise, factual, prefer exact wording/values
- If not in context: Respond "Not found in the provided context."

User Prompt:
- Question: {query}
- Context: {retrieved chunks with metadata}
- Instructions: Provide single, precise answer
```

## MCP Integration

### Model Context Protocol (MCP) Tools

The system integrates MCP tools for specialized operations:

#### Time Difference Tool (`time_diff_tool.py`)

**Purpose**: Calculate time differences between dates/timestamps

**Function**: `get_date_diff(date1, date2)`

**Features**:
- Accepts multiple date formats (ISO, slash dates, datetime strings)
- Returns human-readable time difference
- Format: "{days} days, {hours} hours, {minutes} minutes, {seconds} seconds"

**Usage**:
- Agents can use this tool when queries involve time calculations
- Bound to agents via `BaseAgent` tool support
- Helps answer timeline questions accurately

**Example**:
```python
from src.mcp.time_diff_tool import get_date_diff

result = get_date_diff("2025-03-03 08:20:05", "2025-03-10 14:30:00")
# Returns: "7 days, 6 hours, 9 minutes, 55 seconds"
```

**Integration Point**: Tools are bound to agents via `ChatOpenAI.bind_tools()` in `BaseAgent.__init__`

## Evaluation Methodology

### LLM-as-a-Judge Approach

The system uses an LLM evaluator (`JudgeEvaluator`) to assess response quality on three metrics:

#### Metrics

1. **Answer Correctness** (0.0 - 1.0)
   - Evaluates if the system's answer correctly addresses the query
   - Compares against expected answer (semantic equivalence considered)
   - For exact values (amounts, IDs), requires exact match
   - For descriptive answers, focuses on factual correctness

2. **Context Relevancy** (0.0 - 1.0)
   - Evaluates if retrieved chunks are relevant to the query
   - Average relevancy score across all retrieved chunks
   - Measures retrieval quality independent of answer quality

3. **Context Recall** (0.0 - 1.0)
   - Evaluates if expected context chunks were retrieved
   - Proportion of expected chunks found in retrieved results
   - Measures retrieval completeness

### Evaluation Process

1. **Test Suite**: 8 diverse test cases covering:
   - High-level summaries (2)
   - Precise factual queries (3)
   - Needle-in-haystack queries (2)
   - Timeline questions (1)

2. **Execution**:
   - Each test case runs through full system pipeline
   - Retrieval context captured for evaluation
   - All three metrics evaluated per test case

3. **Reporting**:
   - Per-test-case scores
   - Aggregate statistics (averages, category distribution)
   - Detailed JSON report saved to `results/evaluation/`

### Example Evaluation Results

See `results/evaluation/evaluation_report_*.json` for detailed evaluation reports.

## Installation

### Prerequisites

- **Python 3.12** (required)
- OpenAI API key

### Setup

1. **Clone the repository** (or navigate to project directory)

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set environment variables**:
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_api_key_here

# Optional: Override defaults
LLM_MODEL=gpt-4o-mini
JUDGE_LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
SMALL_CHUNK_SIZE=150
MEDIUM_CHUNK_SIZE=650
LARGE_CHUNK_SIZE=1750
CHUNK_OVERLAP=0.2
```

4. **Place PDF file**:
   - Place your insurance claim PDF in `data/raw/claim.pdf`

5. **Run the system**:
```bash
python main.py
```

The system will:
- Load and parse the PDF
- Create hierarchical chunks
- Build indices (if they don't exist)
- Start the interactive CLI

## Usage

### Example Query/Answer Pairs

**Summary Queries**:

1. **Query**: Summarize Section 1 of the insurance claim.  
   **Answer**: The policyholder has a comprehensive private motor insurance policy that includes full own-damage protection, third-party liability, uninsured driver coverage, personal injury benefits, and legal expense protection, which was active at the time of the incident. The collision excess is £650, which may be recovered through subrogation. The insured vehicle is a 2022 BMW 320i M Sport (registration LK22 RWT), with a mileage of 18,462 miles at the time of the incident, and it was reported to be in good condition with no prior damage.

2. **Query**: Provide a high-level summary of how the collision occurred.  
   **Answer**: The collision occurred when the insured vehicle, stopped at a junction with a green light, was struck by a third-party vehicle (a 2018 Vauxhall Insignia) driven by Thomas Ellison, who ran a red light and was speeding. This led to a significant first impact on the driver-side front quarter of the insured vehicle, followed by additional minor collisions as other vehicles failed to stop in time. The incident was classified as a third-party liability event, with the assessment revealing the vehicle was repairable and the medical impact as minor to moderate. The policyholder cooperated throughout the claim process, with no concerns raised about claim integrity.

3. **Query**: Summarize the final outcome and status of the insurance claim.  
   **Answer**: The insurance claim has been fully resolved, with all key elements addressed and a complete evidentiary chain documented. Outstanding recoveries are being monitored and scheduled, and the claim file is ready for archival storage, pending the receipt of final recovery funds. The claim process adhered to service standards and regulatory expectations, progressing smoothly from notification to settlement administration. There have been no disputes reported, and the ongoing status remains pending while additional recovery efforts are monitored.

**Needle Queries**:

1. **Query**: What was the total claim exposure amount?  
   **Answer**: £22,625.20

2. **Query**: At which junction did the collision occur?  
   **Answer**: The collision occurred at the signal-controlled junction of Euston Road and Judd Street, London NW1.

3. **Query**: What was the registration number of the insured vehicle?  
   **Answer**: LK22 RWT

### Interactive CLI

Run `python main.py` to start the interactive interface:

```
==============================================
 Insurance Claim Assistant
==============================================
Type your question about the claim.
Type 'eval' or 'evaluation' for evaluation
Type 'exit' or 'quit' to exit.
==============================================
>>> 
```

### Query Examples

**High-level Summary**:
```
>>> Give me a high-level summary of what happened in this insurance claim
```

**Precise Factual Query**:
```
>>> What is the exact registration number of the insured vehicle?
```

**Timeline Query**:
```
>>> What is the overall timeline of the insurance claim from incident to resolution?
```

**Section-Specific**:
```
>>> Please summarize section 1 content
```

**Exact Value**:
```
>>> What was the total claim exposure amount?
```

### Evaluation Mode

Type `eval`, `evaluation`, or `e` to run the evaluation suite:

```
>>> eval
```

This will:
- Run all 8 test cases
- Evaluate on all three metrics
- Generate evaluation report in `results/evaluation/`
- Display summary statistics

### Programmatic Usage

```python
from src.agents.orchestrator_system import OrchestratorSystem
from src.indexing.index_manager import IndexManager

# Initialize
index_manager = IndexManager()
index_manager.initialize()
index_manager.load_indices()

# Create orchestrator
orchestrator = OrchestratorSystem()

# Query
answer = orchestrator.handle_query("What is the registration number?")
print(answer)
```

## Limitations & Trade-offs

### Limitations

1. **PDF Parsing**: Quality depends on PDF structure
   - May struggle with scanned PDFs (requires OCR)
   - Complex layouts may affect section detection

2. **Chunking**: Fixed chunk sizes may not fit all content types
   - Very long paragraphs may still be split awkwardly
   - Highly structured content benefits more than free-form text

3. **Retrieval**: Semantic search has limitations
   - May retrieve semantically similar but irrelevant chunks
   - Requires good embedding model quality
   - Reranking helps but isn't perfect

4. **LLM Dependency**: All answers depend on LLM quality
   - Hallucination risk (mitigated by strict prompts)
   - Token limits affect context window size
   - API costs scale with usage

5. **Evaluation**: LLM-as-a-judge has its own limitations
   - Judge LLM may have biases
   - Evaluation consistency varies
   - Requires expected answers for correctness metric

### Trade-offs

1. **Hierarchy Depth**: Three levels balance precision vs. context
   - More levels = more storage and complexity
   - Fewer levels = less flexibility

2. **Chunk Overlap**: 20% overlap balances redundancy vs. coverage
   - More overlap = better boundary coverage, more storage
   - Less overlap = risk of losing boundary information

3. **Reranking**: Time and section reranking improve accuracy but add latency
   - More sophisticated reranking = slower queries
   - Simple semantic search = faster but less accurate

4. **Agent Routing**: LLM routing is more accurate but slower
   - LLM routing = better decisions, API call overhead
   - Rule-based = fast, but less nuanced

5. **Auto-Merging**: Improves context but can reduce precision
   - Merging = broader context, may include irrelevant info
   - No merging = precise chunks, may miss broader context

## Project Structure

```
.
├── main.py                 # CLI entry point
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (create this)
├── data/
│   ├── raw/               # Input PDF files
│   ├── processed/         # Processed chunks
│   └── indices/           # ChromaDB indices
├── results/
│   ├── evaluation/        # Evaluation reports
│   └── logs/              # Application logs
├── src/
│   ├── agents/            # Agent implementations
│   ├── config/            # Configuration and constants
│   ├── data/              # PDF loading and chunking
│   ├── evaluation/        # Evaluation system
│   ├── indexing/          # Index building and management
│   ├── mcp/               # MCP tools
│   ├── retrieval/         # Retrieval strategies
│   └── utils/             # Utilities and helpers
└── project/               # Project documentation
```

## Contributing

This is a research/educational project. For questions or improvements, please refer to the project documentation in `project/`.

## License

[Specify license if applicable]

