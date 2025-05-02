"""
Utility modules for Animusicator.

This package provides various utilities used throughout the application.
"""

# Expose key logger components
try:
    from .logger import (
        get_logger,
        LogLevel,
        debug,
        info,
        warning,
        error,
        critical
    )
except ImportError:
    # Create placeholder if logger not available
    LogLevel = type('LogLevel', (), {
        'DEBUG': 10, 'INFO': 20, 'WARNING': 30, 'ERROR': 40, 'CRITICAL': 50
    })
    
    # Placeholder functions
    def get_logger(*args, **kwargs):
        return None
        
    def debug(*args, **kwargs): 
        pass
        
    def info(*args, **kwargs): 
        pass
        
    def warning(*args, **kwargs): 
        pass
        
    def error(*args, **kwargs): 
        pass
        
    def critical(*args, **kwargs): 
        pass

# Expose key profiler components
try:
    from .profiler import (
        get_profiler,
        ProfilerMetricType,
        profile_time
    )
except ImportError:
    # Create placeholder if profiler not available
    ProfilerMetricType = type('ProfilerMetricType', (), {
        'TIME': 'time',
        'MEMORY': 'memory',
        'CPU': 'cpu',
        'FPS': 'fps',
        'AUDIO_BUFFER': 'buffer',
        'GPU': 'gpu',
        'LATENCY': 'latency'
    })
    
    # Placeholder functions
    def get_profiler(*args, **kwargs):
        return None
        
    def profile_time(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
