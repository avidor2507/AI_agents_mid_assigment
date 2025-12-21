"""
Centralized logging configuration.

This module sets up application-wide logging with both file and console handlers.
Logs are formatted consistently and saved to a log file for debugging and auditing.
"""

import logging
import sys
from pathlib import Path
from src.config.settings import config


def setup_logger(name: str = "InsuranceClaimSystem", level: str = None) -> logging.Logger:
    """
    Set up and configure the application logger.
    
    This function creates a logger with:
    - Console handler (streams to stdout)
    - File handler (writes to app.log)
    - Consistent formatting with timestamps, log level, and messages
    
    Args:
        name: Name of the logger (default: "InsuranceClaimSystem")
        level: Logging level (default: from config, or INFO)
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Get log level from config or use default
    log_level = level or config.LOG_LEVEL
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)
    
    # Avoid duplicate handlers if logger already configured
    if logger.handlers:
        return logger
    
    # Create formatters
    # Detailed format for file handler
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Simpler format for console handler
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler (streams to stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (writes to log file)
    # Ensure log directory exists
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(config.LOG_FILE, encoding='utf-8')
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    return logger


# Create default logger instance
logger = setup_logger()

