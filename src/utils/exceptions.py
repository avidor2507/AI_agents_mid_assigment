"""
Custom exception classes for the Insurance Claim Timeline Retrieval System.

This module defines a hierarchy of custom exceptions following best practices:
- Base exception class for the application
- Specific exception types for different error scenarios
- Allows for fine-grained error handling throughout the system
"""


class InsuranceClaimError(Exception):
    """
    Base exception class for all application-specific errors.
    
    All custom exceptions in this application inherit from this base class,
    allowing for catching all application errors with a single exception type
    while still maintaining specificity when needed.
    """
    pass


class PDFLoadingError(InsuranceClaimError):
    """
    Raised when there's an error loading or parsing a PDF file.
    
    This exception is raised by PDFLoader when:
    - PDF file cannot be found
    - PDF file is corrupted
    - Text extraction fails
    - Document structure parsing fails
    """
    pass


class ChunkingError(InsuranceClaimError):
    """
    Raised when there's an error during the chunking process.
    
    This exception is raised by chunkers when:
    - Text cannot be split into chunks
    - Chunk size constraints cannot be met
    - Hierarchical structure cannot be created
    - Metadata extraction fails
    """
    pass


class IndexingError(InsuranceClaimError):
    """
    Raised when there's an error during the indexing process.
    
    This exception is raised by indexers when:
    - ChromaDB connection fails
    - Embedding generation fails
    - Index persistence fails
    - Metadata storage fails
    """
    pass


class RetrievalError(InsuranceClaimError):
    """
    Raised when there's an error during the retrieval process.
    
    This exception is raised by retrievers when:
    - Query embedding generation fails
    - Vector similarity search fails
    - Results cannot be retrieved from index
    - Auto-merging logic fails
    """
    pass


class AgentError(InsuranceClaimError):
    """
    Raised when there's an error in agent processing.
    
    This exception is raised by agents when:
    - Agent routing fails
    - LLM calls fail
    - Answer generation fails
    - Agent initialization fails
    """
    pass


class MCPToolError(InsuranceClaimError):
    """
    Raised when there's an error with MCP (Model Context Protocol) tools.
    
    This exception is raised by MCP tools when:
    - Tool invocation fails
    - Tool parameters are invalid
    - Tool execution fails
    - Tool registry errors occur
    """
    pass


class EvaluationError(InsuranceClaimError):
    """
    Raised when there's an error during evaluation.
    
    This exception is raised by evaluators when:
    - Test case loading fails
    - LLM-as-a-judge calls fail
    - Evaluation metrics cannot be calculated
    - Report generation fails
    """
    pass


class ConfigurationError(InsuranceClaimError):
    """
    Raised when there's a configuration error.
    
    This exception is raised when:
    - Required environment variables are missing
    - Configuration values are invalid
    - Paths cannot be resolved
    """
    pass

