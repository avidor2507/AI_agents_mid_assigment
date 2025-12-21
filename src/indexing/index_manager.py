"""
Index management and coordination module (Factory Pattern).

This module implements IndexManager which:
- Coordinates and manages all indexing operations
- Provides a unified interface to both indices
- Handles index loading and building
- Manages index lifecycle

Uses Factory Pattern to create and manage indexer instances.
"""

from typing import Optional, Dict, Any
from pathlib import Path
from src.indexing.hierarchical_indexer import HierarchicalIndexer
from src.indexing.summary_indexer import SummaryIndexer
from src.config.settings import config
from src.utils.exceptions import IndexingError, ConfigurationError
from src.utils.logger import logger


class IndexManager:
    """
    Manager for coordinating indexing operations (Factory Pattern + Singleton).

    This class acts as a factory and coordinator for all indexing operations.
    It manages:
    - Hierarchical indexer
    - Summary indexer
    - Index loading and building
    - Unified access to both indices

    Factory Pattern allows:
    - Centralized index creation and management
    - Easy extension with new index types
    - Consistent interface for index operations

    Singleton behavior:
    - All callers receive the same IndexManager instance
    - Expensive initialization and index loading happen at most once
    """

    _instance: Optional["IndexManager"] = None

    def __new__(cls, *args, **kwargs):
        """
        Ensure only one IndexManager instance exists (Singleton).
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """
        Initialize the index manager (idempotent).

        This constructor may be called multiple times due to the Singleton
        pattern, so we guard initialization with an internal flag.
        """
        if getattr(self, "_constructed", False):
            return

        self._constructed = True
        self.logger = logger
        self.hierarchical_indexer: Optional[HierarchicalIndexer] = None
        self.summary_indexer: Optional[SummaryIndexer] = None
        self._initialized = False
        # Caches whether indices have already been loaded from disk
        self._indices_loaded: bool = False
    
    def initialize(self):
        """
        Initialize both indexers.
        
        This method creates instances of both indexers and prepares them
        for use. It should be called before any indexing operations.
        """
        try:
            self.logger.info("Initializing IndexManager")
            
            # Validate configuration
            if not config.validate():
                raise ConfigurationError("Invalid configuration. Check API keys and settings.")
            
            # Create indexer instances (Factory Pattern)
            self.hierarchical_indexer = HierarchicalIndexer()
            self.summary_indexer = SummaryIndexer()
            
            self._initialized = True
            self.logger.info("IndexManager initialized successfully")
        
        except Exception as e:
            error_msg = f"Error initializing IndexManager: {str(e)}"
            self.logger.error(error_msg)
            raise IndexingError(error_msg) from e
    
    def load_indices(self) -> bool:
        """
        Load existing indices if they exist.
        
        This method attempts to load both hierarchical and summary indices
        from disk. If indices don't exist, they can be built later.
        
        Returns:
            bool: True if at least one index was loaded, False otherwise
        """
        if not self._initialized:
            self.initialize()

        # If we've already successfully loaded indices once, reuse them.
        if self._indices_loaded:
            return True

        loaded_any = False

        try:
            # Try loading hierarchical index
            if self.hierarchical_indexer:
                if self.hierarchical_indexer.load_index():
                    self.logger.info("Hierarchical index loaded successfully")
                    loaded_any = True
                else:
                    self.logger.info("Hierarchical index does not exist yet")

            # Try loading summary index
            if self.summary_indexer:
                if self.summary_indexer.load_index():
                    self.logger.info("Summary index loaded successfully")
                    loaded_any = True
                else:
                    self.logger.info("Summary index does not exist yet")

        except Exception as e:
            self.logger.warning(f"Error loading indices: {str(e)}")
            loaded_any = False

        # Cache the fact that indices have been loaded (at least one).
        self._indices_loaded = loaded_any
        return loaded_any
    
    def build_indices(self, hierarchical_structure: Dict[str, Any]) -> bool:
        """
        Build both indices from hierarchical chunk structure.
        
        This method coordinates the building of both indices:
        1. Builds hierarchical index from chunks
        2. Builds summary index using MapReduce strategy
        
        Args:
            hierarchical_structure: Hierarchical chunk structure from chunker
        
        Returns:
            bool: True if both indices built successfully, False otherwise
        
        Raises:
            IndexingError: If index building fails
        """
        if not self._initialized:
            self.initialize()
        
        try:
            self.logger.info("Building indices from hierarchical structure")
            
            # Build hierarchical index
            if self.hierarchical_indexer:
                self.logger.info("Building hierarchical index...")
                self.hierarchical_indexer.build_index(hierarchical_structure)
                self.logger.info("Hierarchical index built successfully")
            
            # Build summary index
            if self.summary_indexer:
                self.logger.info("Building summary index...")
                self.summary_indexer.build_index(hierarchical_structure)
                self.logger.info("Summary index built successfully")
            
            self.logger.info("All indices built successfully")
            return True
        
        except Exception as e:
            error_msg = f"Error building indices: {str(e)}"
            self.logger.error(error_msg)
            raise IndexingError(error_msg) from e
    
    def get_hierarchical_indexer(self) -> Optional[HierarchicalIndexer]:
        """
        Get the hierarchical indexer instance.
        
        Returns:
            Optional[HierarchicalIndexer]: Hierarchical indexer, or None if not initialized
        """
        if not self._initialized:
            self.initialize()
        return self.hierarchical_indexer
    
    def get_summary_indexer(self) -> Optional[SummaryIndexer]:
        """
        Get the summary indexer instance.
        
        Returns:
            Optional[SummaryIndexer]: Summary indexer, or None if not initialized
        """
        if not self._initialized:
            self.initialize()
        return self.summary_indexer
    
    def get_hierarchical_collection(self):
        """
        Get the ChromaDB collection for hierarchical index.
        
        Returns:
            Collection: ChromaDB collection, or None if not available
        """
        # Ensure indexer is initialized
        if not self._initialized:
            self.initialize()
        
        # Ensure indices are loaded (this will set up the collection)
        if not self._indices_loaded:
            self.load_indices()
        
        if self.hierarchical_indexer:
            collection = self.hierarchical_indexer.get_collection()
            if collection is None:
                # Collection might not be set if load_index() failed
                # Try to load it explicitly
                if self.hierarchical_indexer.load_index():
                    collection = self.hierarchical_indexer.get_collection()
            return collection
        return None
    
    def get_summary_collection(self):
        """
        Get the ChromaDB collection for summary index.
        
        Returns:
            Collection: ChromaDB collection, or None if not available
        """
        # Ensure indexer is initialized
        if not self._initialized:
            self.initialize()
        
        # Ensure indices are loaded (this will set up the collection)
        if not self._indices_loaded:
            self.load_indices()
        
        if self.summary_indexer:
            collection = self.summary_indexer.get_collection()
            if collection is None:
                # Collection might not be set if load_index() failed
                # Try to load it explicitly
                if self.summary_indexer.load_index():
                    collection = self.summary_indexer.get_collection()
            return collection
        return None
    
    def check_indices_exist(self) -> bool:
        """
        Check if indices exist on disk.
        
        Returns:
            Dict[str, bool]: Dictionary indicating which indices exist:
                - hierarchical: True if hierarchical index exists
                - summary: True if summary index exists
        """
        result = {
            "hierarchical": False,
            "summary": False,
        }
        
        # Check if index directories exist and contain collections
        hierarchical_path = config.HIERARCHICAL_INDEX_DIR
        summary_path = config.SUMMARY_INDEX_DIR
        
        if hierarchical_path.exists():
            # Check if ChromaDB collection exists (look for chroma.sqlite3 or similar)
            collection_files = list(hierarchical_path.glob("*"))
            if not len(collection_files) > 0:
                return False
        
        if summary_path.exists():
            collection_files = list(summary_path.glob("*"))
            if not len(collection_files) > 0:
                return False        
        return True
    
    def rebuild_indices(self, hierarchical_structure: Dict[str, Any]) -> bool:
        """
        Rebuild indices from scratch (delete existing and rebuild).
        
        This method:
        1. Deletes existing indices
        2. Rebuilds both indices from the hierarchical structure
        
        Args:
            hierarchical_structure: Hierarchical chunk structure
        
        Returns:
            bool: True if rebuild successful
        """
        try:
            self.logger.info("Rebuilding indices from scratch")
            
            # Delete existing indices
            import shutil
            if config.HIERARCHICAL_INDEX_DIR.exists():
                shutil.rmtree(config.HIERARCHICAL_INDEX_DIR)
                self.logger.info("Deleted existing hierarchical index")
            
            if config.SUMMARY_INDEX_DIR.exists():
                shutil.rmtree(config.SUMMARY_INDEX_DIR)
                self.logger.info("Deleted existing summary index")
            
            # Reinitialize indexers
            self.initialize()
            
            # Build indices
            return self.build_indices(hierarchical_structure)
        
        except Exception as e:
            error_msg = f"Error rebuilding indices: {str(e)}"
            self.logger.error(error_msg)
            raise IndexingError(error_msg) from e

