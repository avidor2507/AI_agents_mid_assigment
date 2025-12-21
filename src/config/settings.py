"""
Application-wide configuration management.

This module loads configuration from environment variables and provides
a singleton Config object that can be accessed throughout the application.
Uses Singleton pattern to ensure consistent configuration access.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


class Config:
    """
    Singleton configuration class for application settings.
    
    This class follows the Singleton pattern to ensure only one instance
    of configuration exists throughout the application lifecycle.
    """
    
    _instance: Optional['Config'] = None
    _initialized: bool = False
    
    def __new__(cls):
        """Singleton pattern: return the same instance if already created."""
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize configuration (only once due to Singleton pattern)."""
        if self._initialized:
            return
        
        # ====================================================================
        # API KEYS
        # ====================================================================
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        self.ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
        
        # ====================================================================
        # PATHS
        # ====================================================================
        # Get the project root directory (parent of src/)
        project_root = Path(__file__).parent.parent.parent
        
        self.PROJECT_ROOT = project_root
        self.DATA_DIR = project_root / "data"
        self.RAW_DATA_DIR = self.DATA_DIR / "raw"
        self.PROCESSED_DATA_DIR = self.DATA_DIR / "processed"
        self.CHUNKS_DIR = self.PROCESSED_DATA_DIR / "chunks"
        self.INDICES_DIR = self.DATA_DIR / "indices"
        self.HIERARCHICAL_INDEX_DIR = self.INDICES_DIR / "hierarchical_index"
        self.SUMMARY_INDEX_DIR = self.INDICES_DIR / "summary_index"
        self.RESULTS_DIR = project_root / "results"
        self.LOGS_DIR = self.RESULTS_DIR / "logs"
        
        # Ensure directories exist
        self._ensure_directories()
        
        # ====================================================================
        # CHUNKING SETTINGS
        # ====================================================================
        # Chunk sizes (tokens) - can be overridden by environment variables
        self.SMALL_CHUNK_SIZE = int(os.getenv("SMALL_CHUNK_SIZE", "150"))
        self.MEDIUM_CHUNK_SIZE = int(os.getenv("MEDIUM_CHUNK_SIZE", "650"))
        self.LARGE_CHUNK_SIZE = int(os.getenv("LARGE_CHUNK_SIZE", "1750"))
        self.CHUNK_OVERLAP = float(os.getenv("CHUNK_OVERLAP", "0.2"))  # 20% overlap
        
        # ====================================================================
        # EMBEDDING & LLM SETTINGS
        # ====================================================================
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.JUDGE_LLM_MODEL = os.getenv("JUDGE_LLM_MODEL", "gpt-4o")  # For evaluation
        
        # ====================================================================
        # INDEXING SETTINGS
        # ====================================================================
        self.TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))  # Number of results to retrieve
        
        # ====================================================================
        # LOGGING SETTINGS
        # ====================================================================
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        self.LOG_FILE = self.LOGS_DIR / "app.log"
        
        # ====================================================================
        # CHROMADB SETTINGS
        # ====================================================================
        self.CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(self.INDICES_DIR))
        
        # Mark as initialized
        self._initialized = True
    
    def _ensure_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.DATA_DIR,
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.CHUNKS_DIR,
            self.INDICES_DIR,
            self.HIERARCHICAL_INDEX_DIR,
            self.SUMMARY_INDEX_DIR,
            self.RESULTS_DIR,
            self.LOGS_DIR,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def validate(self) -> bool:
        """
        Validate that required configuration is present.
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        if not self.OPENAI_API_KEY:
            print("WARNING: OPENAI_API_KEY not set in environment variables")
            return False
        return True


# Global singleton instance
config = Config()

