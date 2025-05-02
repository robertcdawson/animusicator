"""
Logging utilities for Animusicator.

This module provides a centralized logging setup with configurable verbosity,
log rotation, and optional console output.
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Optional, Dict, Any, List, Union, Tuple

# Try to import optional dependencies
try:
    from rich.logging import RichHandler
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    from loguru import logger as loguru_logger
    LOGURU_AVAILABLE = True
except ImportError:
    LOGURU_AVAILABLE = False


# Define log levels
class LogLevel:
    """Log level constants."""
    
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class Logger:
    """
    Centralized logging utility.
    
    Provides configurable logging to file with optional console output.
    """
    
    def __init__(self, 
                name: str = "musicviz",
                log_dir: Optional[str] = None,
                level: int = LogLevel.INFO,
                console: bool = True,
                rich_console: bool = True,
                max_size: int = 10 * 1024 * 1024,  # 10 MB
                backup_count: int = 5):
        """
        Initialize the logger.
        
        Args:
            name: Logger name
            log_dir: Directory to store log files
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            console: Whether to log to console
            rich_console: Whether to use rich formatting (if available)
            max_size: Maximum log file size in bytes
            backup_count: Number of backup files to keep
        """
        self.name = name
        self.level = level
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False
        
        # Clear existing handlers (in case logger was previously configured)
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # If log_dir is not specified, use default platform-specific location
        if log_dir is None:
            if sys.platform == "darwin":  # macOS
                log_dir = os.path.expanduser("~/Library/Logs/Animusicator")
            elif sys.platform == "win32":  # Windows
                log_dir = os.path.join(os.environ.get("APPDATA", ""), "Animusicator", "logs")
            else:  # Linux/Unix
                log_dir = os.path.expanduser("~/.local/share/animusicator/logs")
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Generate log file path with date
        date_str = datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(log_dir, f"{name}_{date_str}.log")
        
        # Create file handler with rotation
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_size,
            backupCount=backup_count
        )
        
        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Add console handler if requested
        if console:
            if rich_console and RICH_AVAILABLE:
                # Rich console formatting
                console_handler = RichHandler(
                    rich_tracebacks=True,
                    omit_repeated_times=False,
                    show_path=False,
                    markup=True
                )
            else:
                # Standard console handler
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(formatter)
            
            self.logger.addHandler(console_handler)
        
        self.debug(f"Logger initialized (name: {name}, level: {level}, file: {log_file})")
        
        # Store configuration
        self.config = {
            "name": name,
            "level": level,
            "log_dir": log_dir,
            "log_file": log_file,
            "console": console,
            "rich_console": rich_console and RICH_AVAILABLE,
            "max_size": max_size,
            "backup_count": backup_count
        }
    
    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        """
        Log a debug message.
        
        Args:
            message: Message to log
            *args: Additional arguments for string formatting
            **kwargs: Additional keyword arguments for logger
        """
        self.logger.debug(message, *args, **kwargs)
    
    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        """
        Log an info message.
        
        Args:
            message: Message to log
            *args: Additional arguments for string formatting
            **kwargs: Additional keyword arguments for logger
        """
        self.logger.info(message, *args, **kwargs)
    
    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        """
        Log a warning message.
        
        Args:
            message: Message to log
            *args: Additional arguments for string formatting
            **kwargs: Additional keyword arguments for logger
        """
        self.logger.warning(message, *args, **kwargs)
    
    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        """
        Log an error message.
        
        Args:
            message: Message to log
            *args: Additional arguments for string formatting
            **kwargs: Additional keyword arguments for logger
        """
        self.logger.error(message, *args, **kwargs)
    
    def critical(self, message: str, *args: Any, **kwargs: Any) -> None:
        """
        Log a critical message.
        
        Args:
            message: Message to log
            *args: Additional arguments for string formatting
            **kwargs: Additional keyword arguments for logger
        """
        self.logger.critical(message, *args, **kwargs)
    
    def set_level(self, level: int) -> None:
        """
        Set the logging level.
        
        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.level = level
        self.logger.setLevel(level)
        self.config["level"] = level
        
        # Update handlers
        for handler in self.logger.handlers:
            handler.setLevel(level)
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get logger configuration.
        
        Returns:
            Dictionary with logger configuration
        """
        return self.config.copy()


# Global logger instance
_logger: Optional[Logger] = None


def get_logger(name: str = "musicviz", level: int = LogLevel.INFO) -> Logger:
    """
    Get the global logger instance.
    
    Args:
        name: Logger name
        level: Log level
        
    Returns:
        Logger instance
    """
    global _logger
    if _logger is None:
        _logger = Logger(name=name, level=level)
    return _logger


# Convenience functions for global logger
def debug(message: str, *args: Any, **kwargs: Any) -> None:
    """
    Log a debug message with the global logger.
    
    Args:
        message: Message to log
        *args: Additional arguments for string formatting
        **kwargs: Additional keyword arguments for logger
    """
    logger = get_logger()
    logger.debug(message, *args, **kwargs)


def info(message: str, *args: Any, **kwargs: Any) -> None:
    """
    Log an info message with the global logger.
    
    Args:
        message: Message to log
        *args: Additional arguments for string formatting
        **kwargs: Additional keyword arguments for logger
    """
    logger = get_logger()
    logger.info(message, *args, **kwargs)


def warning(message: str, *args: Any, **kwargs: Any) -> None:
    """
    Log a warning message with the global logger.
    
    Args:
        message: Message to log
        *args: Additional arguments for string formatting
        **kwargs: Additional keyword arguments for logger
    """
    logger = get_logger()
    logger.warning(message, *args, **kwargs)


def error(message: str, *args: Any, **kwargs: Any) -> None:
    """
    Log an error message with the global logger.
    
    Args:
        message: Message to log
        *args: Additional arguments for string formatting
        **kwargs: Additional keyword arguments for logger
    """
    logger = get_logger()
    logger.error(message, *args, **kwargs)


def critical(message: str, *args: Any, **kwargs: Any) -> None:
    """
    Log a critical message with the global logger.
    
    Args:
        message: Message to log
        *args: Additional arguments for string formatting
        **kwargs: Additional keyword arguments for logger
    """
    logger = get_logger()
    logger.critical(message, *args, **kwargs) 