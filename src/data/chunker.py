"""
Chunking strategies and hierarchical chunking implementation.

This module implements:
- ChunkingStrategy interface (Strategy Pattern)
- SmallChunkStrategy, MediumChunkStrategy, LargeChunkStrategy
- HierarchicalChunker for multi-level chunk structure
- Chunk overlap logic (20% overlap between chunks)
- Metadata preservation and parent-child relationships

Follows Strategy Pattern for different chunking strategies and
Single Responsibility Principle for each chunker class.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import tiktoken
from src.config.constants import ChunkSize, CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_PERCENTAGE
from src.config.settings import config
from src.utils.exceptions import ChunkingError
from src.utils.logger import logger
from src.utils.helpers import normalize_text, validate_chunk_size


class ChunkingStrategy(ABC):
    """
    Abstract base class for chunking strategies (Strategy Pattern).
    
    This defines the interface that all chunking strategies must implement.
    Different strategies (Small, Medium, Large) can be swapped at runtime
    without changing the client code.
    
    Strategy Pattern allows:
    - Easy addition of new chunking strategies
    - Runtime selection of chunking strategy
    - Separation of chunking algorithm from the chunking context
    """
    
    def __init__(self, chunk_size: ChunkSize):
        """
        Initialize the chunking strategy.
        
        Args:
            chunk_size: The chunk size level (SMALL, MEDIUM, or LARGE)
        """
        self.chunk_size = chunk_size
        self.min_tokens, self.max_tokens = CHUNK_SIZE_TOKENS[chunk_size]
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT tokenizer
        self.logger = logger
    
    @abstractmethod
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Chunk text into segments according to the strategy.
        
        This is the main method that each concrete strategy must implement.
        Each strategy will have different logic for splitting text while
        respecting the token limits.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to chunks
        
        Returns:
            List[Dict[str, Any]]: List of chunk dictionaries, each containing:
                - text: Chunk text
                - tokens: Number of tokens
                - start_char: Character position where chunk starts
                - end_char: Character position where chunk ends
                - metadata: Any additional metadata
        """
        pass
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using the tokenizer."""
        return len(self.tokenizer.encode(text))
    
    def _split_text_intelligent(self, text: str) -> List[str]:
        """
        Split text intelligently at sentence or paragraph boundaries.
        
        Attempts to split at natural boundaries (paragraphs, then sentences)
        rather than arbitrary character positions. This preserves semantic
        coherence within chunks.
        
        Args:
            text: Text to split
        
        Returns:
            List[str]: List of text segments
        """
        # First try splitting by paragraphs (double newline)
        paragraphs = text.split('\n\n')
        
        # If paragraphs are still too large, split by sentences
        segments = []
        for para in paragraphs:
            # Split by sentence endings (. ! ? followed by space)
            sentences = re.split(r'([.!?]\s+)', para)
            # Recombine sentences with their punctuation
            sentences = [sentences[i] + (sentences[i+1] if i+1 < len(sentences) else '')
                        for i in range(0, len(sentences), 2)]
            
            segments.extend(sentences)
        
        # Filter out empty segments
        return [seg.strip() for seg in segments if seg.strip()]


class SmallChunkStrategy(ChunkingStrategy):
    """
    Fine-grained chunking strategy (~100-200 tokens).
    
    This strategy creates small, precise chunks suitable for:
    - High-precision factual retrieval
    - Needle-in-haystack queries
    - Detailed information extraction
    
    Small chunks allow for very specific retrieval but may lack context.
    """
    
    def __init__(self):
        """Initialize small chunk strategy."""
        super().__init__(ChunkSize.SMALL)
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Chunk text into small segments (100-200 tokens).
        
        Strategy:
        - Split at sentence boundaries when possible
        - Target chunk size: ~150 tokens (middle of range)
        - Apply overlap between chunks
        
        Args:
            text: Text to chunk
            metadata: Optional metadata
        
        Returns:
            List[Dict[str, Any]]: List of small chunks
        """
        text = normalize_text(text)
        chunks = []
        segments = self._split_text_intelligent(text)
        
        current_chunk = []
        current_tokens = 0
        target_tokens = (self.min_tokens + self.max_tokens) // 2  # ~150 tokens
        
        for segment in segments:
            segment_tokens = self._count_tokens(segment)
            
            # If adding this segment exceeds max tokens, finalize current chunk
            if current_tokens + segment_tokens > self.max_tokens and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(self._create_chunk(chunk_text, len(chunks), metadata))
                
                # Start new chunk with overlap (last 20% of previous chunk)
                overlap_text = self._get_overlap_text(chunk_text)
                current_chunk = [overlap_text] if overlap_text else []
                current_tokens = self._count_tokens(overlap_text) if overlap_text else 0
            
            # Add segment to current chunk
            current_chunk.append(segment)
            current_tokens += segment_tokens
            
            # If we've reached target size, consider finalizing chunk
            if current_tokens >= target_tokens:
                chunk_text = ' '.join(current_chunk)
                chunks.append(self._create_chunk(chunk_text, len(chunks), metadata))
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(chunk_text)
                current_chunk = [overlap_text] if overlap_text else []
                current_tokens = self._count_tokens(overlap_text) if overlap_text else 0
        
        # Add final chunk if exists
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(self._create_chunk(chunk_text, len(chunks), metadata))
        
        self.logger.debug(f"Created {len(chunks)} small chunks from text")
        return chunks
    
    def _get_overlap_text(self, text: str) -> str:
        """
        Get the last portion of text for overlap (20%).
        
        Args:
            text: Full chunk text
        
        Returns:
            str: Overlap portion of text
        """
        tokens = self.tokenizer.encode(text)
        overlap_tokens = int(len(tokens) * CHUNK_OVERLAP_PERCENTAGE)
        if overlap_tokens > 0:
            overlap_token_ids = tokens[-overlap_tokens:]
            return self.tokenizer.decode(overlap_token_ids)
        return ""
    
    def _create_chunk(self, text: str, index: int, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a chunk dictionary with metadata."""
        return {
            "text": text,
            "tokens": self._count_tokens(text),
            "chunk_index": index,
            "level": ChunkSize.SMALL.value,
            "metadata": metadata or {},
        }


class MediumChunkStrategy(ChunkingStrategy):
    """
    Balanced chunking strategy (~500-800 tokens).
    
    This strategy creates medium-sized chunks suitable for:
    - Balanced reasoning and context
    - General queries
    - Contextual information retrieval
    
    Medium chunks provide a good balance between specificity and context.
    """
    
    def __init__(self):
        """Initialize medium chunk strategy."""
        super().__init__(ChunkSize.MEDIUM)
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Chunk text into medium segments (500-800 tokens).
        
        Strategy:
        - Split at paragraph boundaries when possible
        - Target chunk size: ~650 tokens (middle of range)
        - Apply overlap between chunks
        
        Args:
            text: Text to chunk
            metadata: Optional metadata
        
        Returns:
            List[Dict[str, Any]]: List of medium chunks
        """
        text = normalize_text(text)
        chunks = []
        paragraphs = text.split('\n\n')
        
        current_chunk = []
        current_tokens = 0
        target_tokens = (self.min_tokens + self.max_tokens) // 2  # ~650 tokens
        
        for para in paragraphs:
            para_tokens = self._count_tokens(para)
            
            # If paragraph alone exceeds max, split it further
            if para_tokens > self.max_tokens:
                # Split paragraph into sentences
                sentences = re.split(r'([.!?]\s+)', para)
                sentences = [sentences[i] + (sentences[i+1] if i+1 < len(sentences) else '')
                            for i in range(0, len(sentences), 2)]
                
                for sentence in sentences:
                    sentence_tokens = self._count_tokens(sentence)
                    if current_tokens + sentence_tokens > self.max_tokens and current_chunk:
                        chunk_text = '\n\n'.join(current_chunk)
                        chunks.append(self._create_chunk(chunk_text, len(chunks), metadata))
                        overlap_text = self._get_overlap_text(chunk_text)
                        current_chunk = [overlap_text] if overlap_text else []
                        current_tokens = self._count_tokens(overlap_text) if overlap_text else 0
                    
                    current_chunk.append(sentence)
                    current_tokens += sentence_tokens
            else:
                # Normal paragraph processing
                if current_tokens + para_tokens > self.max_tokens and current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunks.append(self._create_chunk(chunk_text, len(chunks), metadata))
                    overlap_text = self._get_overlap_text(chunk_text)
                    current_chunk = [overlap_text] if overlap_text else []
                    current_tokens = self._count_tokens(overlap_text) if overlap_text else 0
                
                current_chunk.append(para)
                current_tokens += para_tokens
            
            # If we've reached target size, finalize chunk
            if current_tokens >= target_tokens:
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append(self._create_chunk(chunk_text, len(chunks), metadata))
                overlap_text = self._get_overlap_text(chunk_text)
                current_chunk = [overlap_text] if overlap_text else []
                current_tokens = self._count_tokens(overlap_text) if overlap_text else 0
        
        # Add final chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append(self._create_chunk(chunk_text, len(chunks), metadata))
        
        self.logger.debug(f"Created {len(chunks)} medium chunks from text")
        return chunks
    
    def _get_overlap_text(self, text: str) -> str:
        """Get the last 20% of text for overlap."""
        tokens = self.tokenizer.encode(text)
        overlap_tokens = int(len(tokens) * CHUNK_OVERLAP_PERCENTAGE)
        if overlap_tokens > 0:
            overlap_token_ids = tokens[-overlap_tokens:]
            return self.tokenizer.decode(overlap_token_ids)
        return ""
    
    def _create_chunk(self, text: str, index: int, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a chunk dictionary with metadata."""
        return {
            "text": text,
            "tokens": self._count_tokens(text),
            "chunk_index": index,
            "level": ChunkSize.MEDIUM.value,
            "metadata": metadata or {},
        }


class LargeChunkStrategy(ChunkingStrategy):
    """
    Broad context chunking strategy (~1500-2000 tokens).
    
    This strategy creates large chunks suitable for:
    - Broader context reconstruction
    - Summary generation
    - Comprehensive understanding
    
    Large chunks provide maximum context but may reduce precision.
    """
    
    def __init__(self):
        """Initialize large chunk strategy."""
        super().__init__(ChunkSize.LARGE)
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Chunk text into large segments (1500-2000 tokens).
        
        Strategy:
        - Split at section boundaries when possible
        - Target chunk size: ~1750 tokens (middle of range)
        - Apply overlap between chunks
        
        Args:
            text: Text to chunk
            metadata: Optional metadata
        
        Returns:
            List[Dict[str, Any]]: List of large chunks
        """
        text = normalize_text(text)
        chunks = []
        
        # Split by sections (double newline + header pattern)
        sections = re.split(r'\n\n+(?=[A-Z][^\n]{0,80}\n)', text)
        if len(sections) == 1:
            # No clear sections, split by paragraphs
            sections = text.split('\n\n')
        
        current_chunk = []
        current_tokens = 0
        target_tokens = (self.min_tokens + self.max_tokens) // 2  # ~1750 tokens
        
        for section in sections:
            section_tokens = self._count_tokens(section)
            
            # If section is too large, split it further
            if section_tokens > self.max_tokens:
                # Split by paragraphs
                paragraphs = section.split('\n\n')
                for para in paragraphs:
                    para_tokens = self._count_tokens(para)
                    if current_tokens + para_tokens > self.max_tokens and current_chunk:
                        chunk_text = '\n\n'.join(current_chunk)
                        chunks.append(self._create_chunk(chunk_text, len(chunks), metadata))
                        overlap_text = self._get_overlap_text(chunk_text)
                        current_chunk = [overlap_text] if overlap_text else []
                        current_tokens = self._count_tokens(overlap_text) if overlap_text else 0
                    
                    current_chunk.append(para)
                    current_tokens += para_tokens
            else:
                # Normal section processing
                if current_tokens + section_tokens > self.max_tokens and current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunks.append(self._create_chunk(chunk_text, len(chunks), metadata))
                    overlap_text = self._get_overlap_text(chunk_text)
                    current_chunk = [overlap_text] if overlap_text else []
                    current_tokens = self._count_tokens(overlap_text) if overlap_text else 0
                
                current_chunk.append(section)
                current_tokens += section_tokens
            
            # If we've reached target size, finalize chunk
            if current_tokens >= target_tokens:
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append(self._create_chunk(chunk_text, len(chunks), metadata))
                overlap_text = self._get_overlap_text(chunk_text)
                current_chunk = [overlap_text] if overlap_text else []
                current_tokens = self._count_tokens(overlap_text) if overlap_text else 0
        
        # Add final chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append(self._create_chunk(chunk_text, len(chunks), metadata))
        
        self.logger.debug(f"Created {len(chunks)} large chunks from text")
        return chunks
    
    def _get_overlap_text(self, text: str) -> str:
        """Get the last 20% of text for overlap."""
        tokens = self.tokenizer.encode(text)
        overlap_tokens = int(len(tokens) * CHUNK_OVERLAP_PERCENTAGE)
        if overlap_tokens > 0:
            overlap_token_ids = tokens[-overlap_tokens:]
            return self.tokenizer.decode(overlap_token_ids)
        return ""
    
    def _create_chunk(self, text: str, index: int, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a chunk dictionary with metadata."""
        return {
            "text": text,
            "tokens": self._count_tokens(text),
            "chunk_index": index,
            "level": ChunkSize.LARGE.value,
            "metadata": metadata or {},
        }


class HierarchicalChunker:
    """
    Hierarchical chunker that creates multi-level chunk structure.
    
    Creates chunks at multiple granularity levels (small, medium, large)
    and preserves hierarchical relationships:
    
    Claim
      └── Document
          └── Section
              └── Chunk (Small/Medium/Large)
    
    Responsibilities:
    - Apply multiple chunking strategies to the same text
    - Preserve parent-child relationships
    - Store metadata at each level
    - Maintain hierarchical structure for Auto-Merging Retriever
    """
    
    def __init__(self):
        """Initialize the hierarchical chunker."""
        self.small_strategy = SmallChunkStrategy()
        self.medium_strategy = MediumChunkStrategy()
        self.large_strategy = LargeChunkStrategy()
        self.logger = logger
    
    def chunk_document(
        self,
        document,
        document_id: str = "document_1",
        claim_id: str = "claim_1"
    ) -> Dict[str, Any]:
        """
        Chunk a document hierarchically at multiple levels.
        
        Creates chunks at small, medium, and large granularities while
        preserving the hierarchical structure and relationships.
        
        Args:
            document: Document object (from PDFLoader)
            document_id: Unique identifier for the document
            claim_id: Unique identifier for the claim
        
        Returns:
            Dict[str, Any]: Hierarchical chunk structure containing:
                - claim_id: Claim identifier
                - document_id: Document identifier
                - sections: List of sections with chunks at all levels
                - chunks: Flat list of all chunks with hierarchy metadata
        """
        try:
            self.logger.info(f"Chunking document {document_id} hierarchically")
            
            hierarchical_structure = {
                "claim_id": claim_id,
                "document_id": document_id,
                "sections": [],
                "chunks": {
                    "small": [],
                    "medium": [],
                    "large": [],
                },
                "metadata": document.metadata,
            }
            
            # Process each section in the document
            sections = document.sections if document.sections else [{"text": document.text, "section_id": "section_1"}]
            
            section_index = 0
            for section_data in sections:
                section_text = section_data.get("text", "")
                section_id = section_data.get("section_id", f"section_{section_index + 1}")
                
                if not section_text.strip():
                    continue
                
                section_index += 1
                
                # Create section metadata
                section_metadata = {
                    "section_id": section_id,
                    "section_number": section_index,
                    "header": section_data.get("header", ""),
                    "document_id": document_id,
                    "claim_id": claim_id,
                    "parent_id": document_id,
                }
                
                # Chunk section at all three levels
                small_chunks = self.small_strategy.chunk_text(section_text, section_metadata)
                medium_chunks = self.medium_strategy.chunk_text(section_text, section_metadata)
                large_chunks = self.large_strategy.chunk_text(section_text, section_metadata)
                
                # Add hierarchical metadata to chunks
                for i, chunk in enumerate(small_chunks):
                    chunk["chunk_id"] = f"{section_id}_small_{i}"
                    chunk["parent_id"] = section_id
                    chunk["section_id"] = section_id
                    chunk["document_id"] = document_id
                    chunk["claim_id"] = claim_id
                    hierarchical_structure["chunks"]["small"].append(chunk)
                
                for i, chunk in enumerate(medium_chunks):
                    chunk["chunk_id"] = f"{section_id}_medium_{i}"
                    chunk["parent_id"] = section_id
                    chunk["section_id"] = section_id
                    chunk["document_id"] = document_id
                    chunk["claim_id"] = claim_id
                    hierarchical_structure["chunks"]["medium"].append(chunk)
                
                for i, chunk in enumerate(large_chunks):
                    chunk["chunk_id"] = f"{section_id}_large_{i}"
                    chunk["parent_id"] = section_id
                    chunk["section_id"] = section_id
                    chunk["document_id"] = document_id
                    chunk["claim_id"] = claim_id
                    hierarchical_structure["chunks"]["large"].append(chunk)
                
                # Store section with reference to its chunks
                hierarchical_structure["sections"].append({
                    **section_metadata,
                    "chunk_counts": {
                        "small": len(small_chunks),
                        "medium": len(medium_chunks),
                        "large": len(large_chunks),
                    }
                })
            
            self.logger.info(
                f"Created hierarchical chunks: "
                f"{len(hierarchical_structure['chunks']['small'])} small, "
                f"{len(hierarchical_structure['chunks']['medium'])} medium, "
                f"{len(hierarchical_structure['chunks']['large'])} large"
            )
            
            return hierarchical_structure
        
        except Exception as e:
            error_msg = f"Error chunking document {document_id}: {str(e)}"
            self.logger.error(error_msg)
            raise ChunkingError(error_msg) from e


# Import re for regex operations
import re

