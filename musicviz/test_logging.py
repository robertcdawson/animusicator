#!/usr/bin/env python3
"""
Simple test script to verify loguru logging functionality.
"""

import sys
import os
from loguru import logger

def test_logging():
    """Test loguru logging functionality."""
    # Get package version
    print(f"Testing loguru logging functionality")
    
    # Create a log directory if it doesn't exist
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "test.log")
    
    # Remove existing log file if it exists
    if os.path.exists(log_file):
        os.remove(log_file)
    
    # Configure logger to write to console and file
    logger.remove()  # Remove default handler
    logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    logger.add(log_file, rotation="10 MB", format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}")
    
    # Log messages at different levels
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.success("This is a success message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Try using exception logging
    try:
        x = 1 / 0
    except Exception as e:
        logger.exception(f"An exception occurred: {e}")
    
    # Check if log file was created
    if os.path.exists(log_file):
        print(f"Log file was created successfully at {log_file}")
        print("Contents of the log file:")
        with open(log_file, 'r') as f:
            print(f.read())
    else:
        print(f"Error: Log file was not created at {log_file}")
    
    print("Loguru is working correctly!")

if __name__ == "__main__":
    test_logging() 