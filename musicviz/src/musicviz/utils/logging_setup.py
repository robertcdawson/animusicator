#!/usr/bin/env python3
"""
Logging setup module for Animusicator.

This module configures application-wide logging with proper formatting,
file rotation, and multiple handlers for both console and file output.
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from typing import Dict, Any, Optional

from .config_loader import get_config

# Default log directory
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'logs')

# Log levels map
LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}


def setup_logging(log_dir: Optional[str] = None, 
                 console_level: str = 'INFO',
                 file_level: str = 'DEBUG',
                 max_size: int = 10 * 1024 * 1024,  # 10 MB
                 backup_count: int = 3) -> None:
    """
    Set up logging for the application.
    
    Args:
        log_dir: Directory to store log files (default: app/logs)
        console_level: Logging level for console output
        file_level: Logging level for file output
        max_size: Maximum size of log file before rotation
        backup_count: Number of backup log files to keep
    """
    # Try to load from config if available
    try:
        config = get_config()
        
        # Get logging config
        log_config = config.get('logging', {})
        
        # Override with config values if available
        log_dir = log_dir or os.path.dirname(log_config.get('file', os.path.join(LOG_DIR, 'animusicator.log')))
        console_level = log_config.get('console_level', console_level)
        file_level = log_config.get('file_level', file_level)
        max_size = log_config.get('max_size', max_size)
        backup_count = log_config.get('backup_count', backup_count)
    except Exception as e:
        # If config loading fails, use defaults
        print(f"Warning: Could not load logging config: {e}")
    
    # Create log directory if it doesn't exist
    log_dir = log_dir or LOG_DIR
    os.makedirs(log_dir, exist_ok=True)
    
    # Get log file path
    log_file = os.path.join(log_dir, 'animusicator.log')
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all levels
    
    # Remove existing handlers (to avoid duplicates on reconfiguration)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Determine log levels
    console_level = LOG_LEVELS.get(console_level.upper(), logging.INFO)
    file_level = LOG_LEVELS.get(file_level.upper(), logging.DEBUG)
    
    # Create formatters
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Create file handler with rotation
    try:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_size,
            backupCount=backup_count
        )
        file_handler.setLevel(file_level)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    except Exception as e:
        # If file handler creation fails, log to console
        root_logger.error(f"Could not create log file {log_file}: {e}")
    
    # Set a more conservative level for third-party modules
    for logger_name in ['matplotlib', 'PIL', 'PyQt5']:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    # Log setup completion
    root_logger.info(f"Logging initialized (console: {logging.getLevelName(console_level)}, file: {logging.getLevelName(file_level)})")
    root_logger.debug(f"Log file: {log_file}")


# Convenience function to set up structured log messages
def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


# Test code
if __name__ == "__main__":
    # Set up logging
    setup_logging(console_level='DEBUG')
    
    # Get a logger
    logger = get_logger(__name__)
    
    # Test logging
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test exception logging
    try:
        raise ValueError("This is a test exception")
    except ValueError as e:
        logger.exception("An exception occurred")
    
    print("Logging test complete. Check the logs directory for the log file.") 