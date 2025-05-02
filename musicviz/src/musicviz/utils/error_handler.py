#!/usr/bin/env python3
"""
Error handling utilities for Animusicator.

This module provides standardized error handling, reporting, and custom exceptions
for application-specific error conditions.
"""

import logging
import traceback
import functools
import enum
from typing import Dict, List, Any, Optional, Callable, TypeVar, Union, Type

# Setup optional Sentry SDK for error reporting
try:
    import sentry_sdk
    SENTRY_AVAILABLE = True
except ImportError:
    SENTRY_AVAILABLE = False

# Setup logging
logger = logging.getLogger(__name__)

# Type variables for decorator
T = TypeVar('T')
R = TypeVar('R')


# Define error categories
class ErrorCategory(enum.Enum):
    """Categories of errors for filtering and handling."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    GENERAL = "general"
    AUDIO = "audio"
    GPU = "gpu"
    UI = "ui"


# Custom exception classes
class AudioDeviceError(Exception):
    """Exception raised for audio device issues."""
    
    def __init__(self, message: str, device_name: Optional[str] = None):
        """
        Initialize audio device error.
        
        Args:
            message: Error message
            device_name: Name of the problematic device
        """
        self.device_name = device_name
        self.message = message
        super().__init__(f"{message} (Device: {device_name})")


class ShaderCompilationError(Exception):
    """Exception raised for shader compilation failures."""
    
    def __init__(self, message: str, shader_type: str = "unknown", log: str = ""):
        """
        Initialize shader compilation error.
        
        Args:
            message: Error message
            shader_type: Type of shader (vertex, fragment, etc.)
            log: Compilation log with detailed errors
        """
        self.shader_type = shader_type
        self.log = log
        self.message = message
        super().__init__(f"{message} (Shader type: {shader_type}): {log}")


class GPUNotAvailableError(Exception):
    """Exception raised when GPU functionality is not available."""
    
    def __init__(self, message: str, fallback: Optional[str] = None):
        """
        Initialize GPU not available error.
        
        Args:
            message: Error message
            fallback: Fallback method being used instead
        """
        self.fallback = fallback
        self.message = message
        msg = f"{message}"
        if fallback:
            msg += f" (Fallback: {fallback})"
        super().__init__(msg)


class ErrorHandler:
    """
    Centralized error handler for application-wide error management.
    
    Handles tracking, reporting, and formatting of errors throughout the application.
    """
    
    def __init__(self, max_errors: int = 100, enable_reporting: bool = False):
        """
        Initialize error handler.
        
        Args:
            max_errors: Maximum number of errors to track
            enable_reporting: Whether to enable external error reporting
        """
        self.errors: List[Dict[str, Any]] = []
        self.max_errors = max_errors
        self.is_reporting_enabled = enable_reporting and SENTRY_AVAILABLE
        self.callbacks = {}
        
        if self.is_reporting_enabled:
            logger.info("External error reporting is enabled")
    
    def register_callback(self, callback: Callable, categories: Optional[List[Union[str, ErrorCategory]]] = None) -> None:
        """
        Register a callback function for specific error categories.
        
        Args:
            callback: Function to call when an error of the specified category occurs
            categories: List of categories to register for, or None for all categories
        """
        if categories is None:
            # Register for all categories
            categories = [cat.value for cat in ErrorCategory]
        else:
            # Convert any Enum values to their string representation
            categories = [cat.value if isinstance(cat, ErrorCategory) else cat for cat in categories]
        
        for category in categories:
            if category not in self.callbacks:
                self.callbacks[category] = []
            if callback not in self.callbacks[category]:
                self.callbacks[category].append(callback)
    
    def handle_error(self, error: Exception, message: str = None, context: Any = None, category: str = "general") -> None:
        """
        Handle an error.
        
        Args:
            error: The exception to handle
            message: Optional error message
            context: Optional context information
            category: Category of the error for filtering
        """
        # Get traceback info
        tb = traceback.format_exc()
        
        # Use the message from the error if none provided
        if message is None:
            message = str(error)
        
        # Log the error
        logger.error(f"Error in {context}: {error}\n{tb}")
        
        # Add to error list
        error_info = {
            "error": error,
            "message": message,
            "context": context,
            "traceback": tb,
            "category": category,
            "timestamp": None  # In a real implementation, this would be a datetime
        }
        
        self.errors.append(error_info)
        
        # Trim error list if necessary
        if len(self.errors) > self.max_errors:
            self.errors = self.errors[len(self.errors) - self.max_errors:]
            
        # Report error if enabled
        if self.is_reporting_enabled:
            import sentry_sdk
            sentry_sdk.capture_exception(error)
        
        # Invoke callbacks
        self._invoke_callbacks(error, message, context, category)
    
    def _invoke_callbacks(self, error: Exception, message: str, context: Any, category: str) -> None:
        """
        Invoke registered callbacks for an error.
        
        Args:
            error: The error that occurred
            message: Error message
            context: Error context
            category: Error category
        """
        if category in self.callbacks:
            for callback in self.callbacks[category]:
                try:
                    callback(error, message, context)
                except Exception as e:
                    logger.error(f"Error in error callback: {e}")
        
        # Also invoke general callbacks
        if "general" in self.callbacks and category != "general":
            for callback in self.callbacks["general"]:
                try:
                    callback(error, message, context)
                except Exception as e:
                    logger.error(f"Error in general error callback: {e}")
    
    def clear_errors(self) -> None:
        """Clear all tracked errors."""
        self.errors = []
    
    def get_recent_errors(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most recent errors.
        
        Args:
            count: Number of errors to return
            
        Returns:
            List of error dictionaries
        """
        return self.errors[-count:] if self.errors else []
    
    def has_errors(self) -> bool:
        """
        Check if there are any errors.
        
        Returns:
            True if there are errors, False otherwise
        """
        return len(self.errors) > 0
    
    def format_error(self, error_info: Dict[str, Any]) -> str:
        """
        Format an error for display.
        
        Args:
            error_info: Error information dictionary
            
        Returns:
            Formatted error string
        """
        error = error_info["error"]
        context = error_info["context"]
        tb = error_info["traceback"]
        
        return f"Error in {context}: {error.__class__.__name__}: {str(error)}\n{tb}"
    
    def get_formatted_errors(self, count: int = 10) -> List[str]:
        """
        Get formatted error strings.
        
        Args:
            count: Number of errors to return
            
        Returns:
            List of formatted error strings
        """
        recent_errors = self.get_recent_errors(count)
        return [self.format_error(error_info) for error_info in recent_errors]
    
    def get_error_summary(self) -> Dict[str, Any]:
        """
        Get a summary of errors by type.
        
        Returns:
            Dictionary with error summary information
        """
        if not self.errors:
            return {"total": 0, "types": {}}
            
        # Count errors by type
        error_counts = {}
        for error_info in self.errors:
            error_type = error_info["error"].__class__.__name__
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
        return {
            "total": len(self.errors),
            "types": error_counts
        }
    
    def get_errors_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get errors of a specific category.
        
        Args:
            category: Category to filter by
            
        Returns:
            List of error dictionaries in the specified category
        """
        return [error for error in self.errors if error["category"] == category]


def track_exceptions(context: str, fallback_value: Any = None) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """
    Decorator to track exceptions in a function.
    
    Args:
        context: Context for the error
        fallback_value: Value to return if an exception occurs
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> R:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Exception in {context}: {e}")
                if fallback_value is None:
                    raise
                return fallback_value
        return wrapper
    return decorator


# Singleton instance for global access
_error_handler_instance = None

def get_error_handler() -> ErrorHandler:
    """
    Get singleton ErrorHandler instance.
    
    Returns:
        Shared ErrorHandler instance
    """
    global _error_handler_instance
    if _error_handler_instance is None:
        _error_handler_instance = ErrorHandler()
    return _error_handler_instance


def handle_exception(func):
    """
    Decorator for handling exceptions in a function.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function that handles exceptions
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_handler = get_error_handler()
            error_handler.handle_error(
                e,
                context=f"Function: {func.__name__}",
                category=ErrorCategory.CRITICAL.value
            )
            # Re-raise exception after handling it
            raise
    return wrapper


# Example usage
if __name__ == "__main__":
    # Setup basic logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Get error handler
    handler = ErrorHandler()
    
    # Register a callback
    def on_audio_error(exception, message, context):
        print(f"Audio Error: {message}")
    
    handler.register_callback(on_audio_error, [ErrorCategory.AUDIO])
    
    # Test error handling
    try:
        # Simulate an error
        raise ValueError("Sample rate must be positive")
    except Exception as e:
        # Handle the error
        handler.handle_error(
            e,
            message="Failed to initialize audio device",
            category=ErrorCategory.AUDIO,
            context={"device": "default", "sample_rate": -100}
        )
    
    # Test decorator
    @track_exceptions("process_audio", fallback_value=None)
    def process_audio():
        # Simulate an error
        raise RuntimeError("Audio processing failed")
    
    try:
        process_audio()
    except:
        pass
        
    # Print recent errors
    for error in handler.get_recent_errors():
        print(f"Recent error: {error}")
        
    # Print error counts
    print("Error counts:")
    for category, count in handler.get_error_summary().items():
        if count > 0:
            print(f"  {category}: {count}") 