"""
Hierarchical index builder implementation.

This module implements HierarchicalIndexer which:
- Creates ChromaDB collection for hierarchical index
- Indexes chunks at multiple granularity levels (small, medium, large)
- Stores parent-child relationships in metadata
- Supports Auto-Merging Retriever structure
- Persists index to disk

Extends BaseIndexer and implements the template method steps.
"""

from typing import List, Dict, Any
from pathlib import Path
import chromadb
from chromadb.config import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Document as LlamaIndexDocument
from src.indexing.base_indexer import BaseIndexer
from src.indexing.index_schema import (
    create_hierarchical_metadata,
    validate_hierarchical_metadata,
    get_hierarchical_metadata_keys,
)
from src.config.constants import HIERARCHICAL_COLLECTION_NAME
from src.config.settings import config
from src.utils.exceptions import IndexingError
from src.utils.logger import logger


class HierarchicalIndexer(BaseIndexer):
    """
    Indexer for hierarchical multi-level index using ChromaDB.
    
    This indexer creates and manages a hierarchical index that stores chunks
    at multiple granularity levels (small, medium, large). The hierarchical
    structure enables:
    - Fine-grained retrieval starting with small chunks
    - Auto-merging to larger chunks when more context is needed
    - Metadata filtering by level, section, timestamp
    - Parent-child relationship traversal
    
    Responsibilities:
    - Build multi-level index with ChromaDB
    - Store chunks with metadata (level, parent_id, timestamp, section)
    - Support Auto-Merging Retriever structure
    - Persist index to disk
    """
    
    def __init__(self, persist_directory: Path = None):
        """
        Initialize the hierarchical indexer.
        
        Args:
            persist_directory: Directory to persist the ChromaDB index
        """
        persist_dir = persist_directory or config.HIERARCHICAL_INDEX_DIR
        super().__init__(HIERARCHICAL_COLLECTION_NAME, persist_dir)
        self.chroma_client = None
        self.embedding_function = None
    
    def initialize_index(self):
        """
        Initialize the ChromaDB collection for hierarchical index.
        
        This method:
        1. Creates or connects to ChromaDB client
        2. Creates or gets the collection
        3. Initializes the embedding model
        
        Raises:
            IndexingError: If initialization fails
        """
        try:
            self.logger.info("Initializing hierarchical index")
            
            # Initialize ChromaDB client with persistence
            # Note: We use PersistentClient to save to disk
            # For Windows compatibility, ensure path exists
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
            
            # Create or get collection
            # ChromaDB will create the collection if it doesn't exist
            try:
                self.collection = self.chroma_client.get_collection(
                    name=self.collection_name
                )
                self.logger.info(f"Loaded existing collection '{self.collection_name}'")
            except Exception:
                # Collection doesn't exist, create it
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Hierarchical index with small, medium, and large chunks"}
                )
                self.logger.info(f"Created new collection '{self.collection_name}'")
            
            self.logger.info("Hierarchical index initialized successfully")
        
        except Exception as e:
            error_msg = f"Error initializing hierarchical index: {str(e)}"
            self.logger.error(error_msg)
            raise IndexingError(error_msg) from e
    
    def prepare_data(self, hierarchical_structure: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Prepare hierarchical chunk data for indexing.
        
        This method takes the hierarchical structure from HierarchicalChunker
        and prepares all chunks (small, medium, large) for indexing.
        
        Args:
            hierarchical_structure: Hierarchical chunk structure from chunker
        
        Returns:
            List[Dict[str, Any]]: List of prepared items, each containing:
                - text: Chunk text
                - metadata: Metadata dictionary
                - id: Unique chunk ID
        
        Raises:
            IndexingError: If data preparation fails
        """
        try:
            self.logger.info("Preparing hierarchical chunks for indexing")
            
            items = []
            chunks = hierarchical_structure.get("chunks", {})
            
            # Process chunks at all three levels
            for level in ["small", "medium", "large"]:
                level_chunks = chunks.get(level, [])
                
                for chunk in level_chunks:
                    # Create metadata using schema helper
                    metadata = create_hierarchical_metadata(chunk)
                    
                    # Validate metadata
                    if not validate_hierarchical_metadata(metadata):
                        self.logger.warning(f"Invalid metadata for chunk {chunk.get('chunk_id')}, skipping")
                        continue
                    
                    # Create item for indexing
                    item = {
                        "id": chunk.get("chunk_id", ""),
                        "text": chunk.get("text", ""),
                        "metadata": metadata,
                    }
                    
                    items.append(item)
            
            self.logger.info(f"Prepared {len(items)} chunks for indexing")
            return items
        
        except Exception as e:
            error_msg = f"Error preparing hierarchical data: {str(e)}"
            self.logger.error(error_msg)
            raise IndexingError(error_msg) from e
    
    def store_in_index(self, items: List[Dict[str, Any]]):
        """
        Store items in the ChromaDB hierarchical index.
        
        This method:
        1. Generates embeddings for all chunk texts
        2. Stores chunks with embeddings and metadata in ChromaDB
        3. Preserves all hierarchical relationships
        
        Args:
            items: List of items to store
        
        Raises:
            IndexingError: If storage fails
        """
        try:
            if not items:
                self.logger.warning("No items to store in hierarchical index")
                return
            
            self.logger.info(f"Storing {len(items)} chunks in hierarchical index")
            
            # Extract texts, ids, and metadata
            texts = [item["text"] for item in items]
            ids = [item["id"] for item in items]
            metadatas = [item["metadata"] for item in items]
            
            # Generate embeddings using OpenAI
            # Note: ChromaDB can generate embeddings automatically, but we use
            # OpenAI embeddings explicitly for consistency with llama-index
            self.logger.info("Generating embeddings for chunks")
            embeddings = []
            
            # Batch embeddings for efficiency
            batch_size = 100
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self.embedding_function.get_text_embedding_batch(batch_texts)
                embeddings.extend(batch_embeddings)
                self.logger.debug(f"Generated embeddings for batch {i//batch_size + 1}")
            
            # Store in ChromaDB collection
            # ChromaDB expects:
            # - ids: List of unique IDs
            # - embeddings: List of embedding vectors (or None if using ChromaDB's embedding function)
            # - metadatas: List of metadata dictionaries
            # - documents: List of text documents
            
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=texts
            )
            
            self.logger.info(f"Successfully stored {len(items)} chunks in hierarchical index")
        
        except Exception as e:
            error_msg = f"Error storing items in hierarchical index: {str(e)}"
            self.logger.error(error_msg)
            raise IndexingError(error_msg) from e
    
    def get_collection(self):
        """
        Get the ChromaDB collection.
        
        Returns:
            Collection: ChromaDB collection object
        """
        return self.collection

