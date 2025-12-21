"""
Abstract base class for all indexers (Template Method Pattern).

This module defines the BaseIndexer abstract class that provides:
- Common interface for all indexers
- Template methods for indexing workflow
- Common functionality (embedding generation, metadata handling)
- Enforces consistent indexer behavior

Template Method Pattern allows:
- Common workflow definition in base class
- Specific steps implemented by subclasses
- Consistent indexing behavior across different index types
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path
from src.utils.exceptions import IndexingError
from src.utils.logger import logger


class BaseIndexer(ABC):
    """
    Abstract base class for all indexers (Template Method Pattern).
    
    This class defines the template method pattern for indexing:
    1. Initialize index (abstract)
    2. Prepare data (abstract)
    3. Generate embeddings (common)
    4. Store in index (abstract)
    5. Persist index (common)
    
    Subclasses implement specific steps while inheriting common functionality.
    """
    
    def __init__(self, collection_name: str, persist_directory: Path = None):
        """
        Initialize the base indexer.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the index
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.logger = logger
        self.collection = None
        self.embedding_model = None
    
    @abstractmethod
    def initialize_index(self):
        """
        Initialize the index (create or load existing).
        
        This method must be implemented by subclasses to:
        - Set up ChromaDB collection
        - Configure embedding model
        - Load existing index if available
        
        Raises:
            IndexingError: If index initialization fails
        """
        pass
    
    @abstractmethod
    def prepare_data(self, data: Any) -> List[Dict[str, Any]]:
        """
        Prepare data for indexing.
        
        This method transforms input data into the format needed for indexing.
        Each subclass implements its own preparation logic.
        
        Args:
            data: Input data (varies by indexer type)
        
        Returns:
            List[Dict[str, Any]]: Prepared data items for indexing
        
        Raises:
            IndexingError: If data preparation fails
        """
        pass
    
    @abstractmethod
    def store_in_index(self, items: List[Dict[str, Any]]):
        """
        Store items in the index.
        
        This method handles the actual storage of items in ChromaDB.
        Each subclass implements its own storage logic.
        
        Args:
            items: List of items to store, each containing text and metadata
        
        Raises:
            IndexingError: If storage fails
        """
        pass
    
    def persist_index(self):
        """
        Persist the index to disk (common functionality).
        
        This method is shared across all indexers as the persistence
        mechanism is the same (ChromaDB handles it automatically when
        using PersistentClient).
        
        Raises:
            IndexingError: If persistence fails
        """
        try:
            self.logger.info(f"Persisting index '{self.collection_name}' to {self.persist_directory}")
            # ChromaDB automatically persists when using PersistentClient
            # This method is here for logging and future extension
            self.logger.info(f"Index '{self.collection_name}' persisted successfully")
        except Exception as e:
            error_msg = f"Error persisting index '{self.collection_name}': {str(e)}"
            self.logger.error(error_msg)
            raise IndexingError(error_msg) from e
    
    def build_index(self, data: Any) -> bool:
        """
        Template method that defines the indexing workflow.
        
        This is the template method that defines the steps:
        1. Initialize index
        2. Prepare data
        3. Store in index
        4. Persist index
        
        Args:
            data: Input data to index
        
        Returns:
            bool: True if indexing succeeded, False otherwise
        
        Raises:
            IndexingError: If any step in the workflow fails
        """
        try:
            self.logger.info(f"Building index '{self.collection_name}'")
            
            # Step 1: Initialize index
            self.initialize_index()
            
            # Step 2: Prepare data
            prepared_items = self.prepare_data(data)
            self.logger.info(f"Prepared {len(prepared_items)} items for indexing")
            
            # Step 3: Store in index
            if prepared_items:
                self.store_in_index(prepared_items)
                self.logger.info(f"Stored {len(prepared_items)} items in index")
            
            # Step 4: Persist index
            self.persist_index()
            
            self.logger.info(f"Successfully built index '{self.collection_name}'")
            return True
        
        except Exception as e:
            error_msg = f"Error building index '{self.collection_name}': {str(e)}"
            self.logger.error(error_msg)
            raise IndexingError(error_msg) from e
    
    def load_index(self) -> bool:
        """
        Load an existing index.
        
        This method attempts to load an existing index from disk.
        Implementation depends on the specific indexer type.
        
        Returns:
            bool: True if index loaded successfully, False otherwise
        """
        try:
            self.logger.info(f"Loading index '{self.collection_name}' from {self.persist_directory}")
            self.initialize_index()  # This will load existing if available
            self.logger.info(f"Successfully loaded index '{self.collection_name}'")
            return True
        except Exception as e:
            self.logger.warning(f"Could not load index '{self.collection_name}': {str(e)}")
            return False

