"""
Summary index builder implementation using MapReduce strategy.

This module implements SummaryIndexer which:
- Generates chunk-level summaries (Map phase)
- Generates section-level summaries (Reduce phase 1)
- Generates document-level summary (Reduce phase 2)
- Indexes summaries in ChromaDB
- Persists index to disk

MapReduce Strategy:
- Map: Each chunk is summarized independently
- Reduce: Summaries are combined at section level, then document level

Extends BaseIndexer and implements the template method steps.
"""

from typing import List, Dict, Any
from pathlib import Path
import chromadb
from chromadb.config import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import Document as LlamaIndexDocument
from src.indexing.base_indexer import BaseIndexer
from src.indexing.index_schema import (
    create_summary_metadata,
    validate_summary_metadata,
    get_summary_metadata_keys,
)
from src.config.constants import SUMMARY_COLLECTION_NAME
from src.config.settings import config
from src.utils.exceptions import IndexingError
from src.utils.logger import logger


class SummaryIndexer(BaseIndexer):
    """
    Indexer for summary index using MapReduce strategy.
    
    This indexer creates summaries at multiple levels:
    1. Chunk-level summaries (Map phase)
    2. Section-level summaries (Reduce phase 1)
    3. Document-level summary (Reduce phase 2)
    
    The summaries are indexed in ChromaDB for fast retrieval of
    high-level information and timeline-oriented queries.
    
    Responsibilities:
    - MapReduce summarization strategy
    - Generate chunk-level summaries
    - Generate section-level summaries (reduce from chunks)
    - Generate document-level summaries (reduce from sections)
    - Store in separate ChromaDB collection
    """
    
    def __init__(self, persist_directory: Path = None):
        """
        Initialize the summary indexer.
        
        Args:
            persist_directory: Directory to persist the ChromaDB index
        """
        persist_dir = persist_directory or config.SUMMARY_INDEX_DIR
        super().__init__(SUMMARY_COLLECTION_NAME, persist_dir)
        self.chroma_client = None
        self.embedding_function = None
        self.llm = None
    
    def initialize_index(self):
        """
        Initialize the ChromaDB collection for summary index.
        
        This method:
        1. Creates or connects to ChromaDB client
        2. Creates or gets the collection
        3. Initializes embedding model and LLM
        
        Raises:
            IndexingError: If initialization fails
        """
        try:
            self.logger.info("Initializing summary index")
            
            # Initialize ChromaDB client with persistence
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Initialize embedding model (OpenAI)
            self.embedding_function = OpenAIEmbedding(
                model_name=config.EMBEDDING_MODEL,
                api_key=config.OPENAI_API_KEY
            )
            
            # Initialize LLM for summarization
            self.llm = OpenAI(
                model=config.LLM_MODEL,
                api_key=config.OPENAI_API_KEY
            )
            
            # Create or get collection
            try:
                self.collection = self.chroma_client.get_collection(
                    name=self.collection_name
                )
                self.logger.info(f"Loaded existing collection '{self.collection_name}'")
            except Exception:
                # Collection doesn't exist, create it
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Summary index with chunk, section, and document summaries"}
                )
                self.logger.info(f"Created new collection '{self.collection_name}'")
            
            self.logger.info("Summary index initialized successfully")
        
        except Exception as e:
            error_msg = f"Error initializing summary index: {str(e)}"
            self.logger.error(error_msg)
            raise IndexingError(error_msg) from e
    
    def prepare_data(self, hierarchical_structure: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Prepare summary data using MapReduce strategy.
        
        MapReduce Process:
        1. Map: Generate summaries for each chunk
        2. Reduce (Level 1): Combine chunk summaries into section summaries
        3. Reduce (Level 2): Combine section summaries into document summary
        
        Args:
            hierarchical_structure: Hierarchical chunk structure
        
        Returns:
            List[Dict[str, Any]]: List of summaries at all levels
        
        Raises:
            IndexingError: If data preparation fails
        """
        try:
            self.logger.info("Preparing summary data using MapReduce strategy")
            
            summaries = []
            chunks = hierarchical_structure.get("chunks", {})
            sections = hierarchical_structure.get("sections", [])
            
            # MAP PHASE: Generate chunk-level summaries
            self.logger.info("Map Phase: Generating chunk-level summaries")
            chunk_summaries = {}
            
            # Process small chunks for summarization (they have the finest granularity)
            small_chunks = chunks.get("small", [])
            for chunk in small_chunks:
                chunk_id = chunk.get("chunk_id", "")
                chunk_text = chunk.get("text", "")
                section_id = chunk.get("section_id", "")
                
                # Generate summary for chunk
                chunk_summary = self._generate_chunk_summary(chunk_text)
                
                chunk_summaries[chunk_id] = {
                    "summary_id": f"summary_{chunk_id}",
                    "summary_level": "chunk",
                    "summary_text": chunk_summary,
                    "section_id": section_id,
                    "document_id": chunk.get("document_id", ""),
                    "claim_id": chunk.get("claim_id", ""),
                    "original_chunk_id": chunk_id,
                }
            
            self.logger.info(f"Generated {len(chunk_summaries)} chunk-level summaries")
            
            # Add chunk summaries to results
            summaries.extend(chunk_summaries.values())
            
            # REDUCE PHASE 1: Generate section-level summaries
            self.logger.info("Reduce Phase 1: Generating section-level summaries")
            section_summaries = {}
            
            for section in sections:
                section_id = section.get("section_id", "")
                
                # Get all chunk summaries for this section
                section_chunk_summaries = [
                    cs for cs in chunk_summaries.values()
                    if cs.get("section_id") == section_id
                ]
                
                if not section_chunk_summaries:
                    continue
                
                # Combine chunk summaries
                combined_chunk_summaries = "\n\n".join(
                    [cs["summary_text"] for cs in section_chunk_summaries]
                )
                
                # Generate section-level summary
                section_summary = self._generate_section_summary(
                    combined_chunk_summaries,
                    section.get("header", "")
                )
                
                section_summaries[section_id] = {
                    "summary_id": f"summary_section_{section_id}",
                    "summary_level": "section",
                    "summary_text": section_summary,
                    "section_id": section_id,
                    "document_id": hierarchical_structure.get("document_id", ""),
                    "claim_id": hierarchical_structure.get("claim_id", ""),
                }
            
            self.logger.info(f"Generated {len(section_summaries)} section-level summaries")
            
            # Add section summaries to results
            summaries.extend(section_summaries.values())
            
            # REDUCE PHASE 2: Generate document-level summary
            self.logger.info("Reduce Phase 2: Generating document-level summary")
            
            if section_summaries:
                # Combine all section summaries
                combined_section_summaries = "\n\n".join(
                    [ss["summary_text"] for ss in section_summaries.values()]
                )
                
                # Generate document-level summary
                document_summary = self._generate_document_summary(
                    combined_section_summaries,
                    hierarchical_structure.get("metadata", {})
                )
                
                doc_summary = {
                    "summary_id": f"summary_document_{hierarchical_structure.get('document_id', 'doc')}",
                    "summary_level": "document",
                    "summary_text": document_summary,
                    "document_id": hierarchical_structure.get("document_id", ""),
                    "claim_id": hierarchical_structure.get("claim_id", ""),
                }
                
                summaries.append(doc_summary)
                self.logger.info("Generated document-level summary")
            
            self.logger.info(f"Prepared {len(summaries)} summaries for indexing")
            return summaries
        
        except Exception as e:
            error_msg = f"Error preparing summary data: {str(e)}"
            self.logger.error(error_msg)
            raise IndexingError(error_msg) from e
    
    def _generate_chunk_summary(self, chunk_text: str) -> str:
        """
        Generate a summary for a single chunk (Map phase).
        
        Args:
            chunk_text: Text content of the chunk
        
        Returns:
            str: Summary of the chunk
        """
        prompt = f"""Summarize the following text chunk from an insurance claim document.
Focus on key facts, dates, amounts, and events.

Text chunk:
{chunk_text}

Summary:"""
        
        try:
            response = self.llm.complete(prompt)
            return response.text.strip()
        except Exception as e:
            self.logger.warning(f"Error generating chunk summary: {str(e)}")
            # Return truncated version as fallback
            return chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text
    
    def _generate_section_summary(self, combined_chunk_summaries: str, section_header: str) -> str:
        """
        Generate a summary for a section from chunk summaries (Reduce phase 1).
        
        Args:
            combined_chunk_summaries: Combined summaries of all chunks in the section
            section_header: Header/title of the section
        
        Returns:
            str: Summary of the section
        """
        prompt = f"""Create a comprehensive summary for the following section of an insurance claim document.
Combine the individual chunk summaries into a coherent section overview.
Include a timeline of events, key entities, and important details.

Section: {section_header}

Chunk summaries:
{combined_chunk_summaries}

Section summary:"""
        
        try:
            response = self.llm.complete(prompt)
            return response.text.strip()
        except Exception as e:
            self.logger.warning(f"Error generating section summary: {str(e)}")
            return combined_chunk_summaries[:500] + "..." if len(combined_chunk_summaries) > 500 else combined_chunk_summaries
    
    def _generate_document_summary(self, combined_section_summaries: str, metadata: Dict[str, Any]) -> str:
        """
        Generate a document-level summary from section summaries (Reduce phase 2).
        
        Args:
            combined_section_summaries: Combined summaries of all sections
            metadata: Document metadata (claim_id, timestamps, etc.)
        
        Returns:
            str: Document-level summary
        """
        claim_id = metadata.get("claim_id", "Unknown")
        timestamps = metadata.get("timestamps", [])
        first_date = metadata.get("first_timestamp", "")
        last_date = metadata.get("last_timestamp", "")
        
        prompt = f"""Create a comprehensive document-level summary for this insurance claim.
Combine all section summaries into a high-level overview.

Claim ID: {claim_id}
Time Period: {first_date} to {last_date}

Section summaries:
{combined_section_summaries}

Document summary (include overall timeline, major events, key entities, total costs, and claim status):"""
        
        try:
            response = self.llm.complete(prompt)
            return response.text.strip()
        except Exception as e:
            self.logger.warning(f"Error generating document summary: {str(e)}")
            return combined_section_summaries[:1000] + "..." if len(combined_section_summaries) > 1000 else combined_section_summaries
    
    def store_in_index(self, summaries: List[Dict[str, Any]]):
        """
        Store summaries in the ChromaDB summary index.
        
        Args:
            summaries: List of summary dictionaries
        
        Raises:
            IndexingError: If storage fails
        """
        try:
            if not summaries:
                self.logger.warning("No summaries to store in summary index")
                return
            
            self.logger.info(f"Storing {len(summaries)} summaries in summary index")
            
            # Extract texts, ids, and metadata
            texts = [summary["summary_text"] for summary in summaries]
            ids = [summary["summary_id"] for summary in summaries]
            metadatas = [create_summary_metadata(summary) for summary in summaries]
            
            # Generate embeddings
            self.logger.info("Generating embeddings for summaries")
            embeddings = []
            
            batch_size = 100
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self.embedding_function.get_text_embedding_batch(batch_texts)
                embeddings.extend(batch_embeddings)
                self.logger.debug(f"Generated embeddings for summary batch {i//batch_size + 1}")
            
            # Store in ChromaDB
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=texts
            )
            
            self.logger.info(f"Successfully stored {len(summaries)} summaries in summary index")
        
        except Exception as e:
            error_msg = f"Error storing summaries in summary index: {str(e)}"
            self.logger.error(error_msg)
            raise IndexingError(error_msg) from e
    
    def get_collection(self):
        """
        Get the ChromaDB collection.
        
        Returns:
            Collection: ChromaDB collection object
        """
        return self.collection

