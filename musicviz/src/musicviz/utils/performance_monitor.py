#!/usr/bin/env python3
"""
Performance monitoring utility for tracking system resources and processing metrics.

This module provides tools to monitor CPU usage, memory consumption, frame rate,
audio buffer health, and processing latency.
"""

import time
import logging
import threading
from collections import deque
import platform
from typing import Dict, List, Any, Optional, Deque, Tuple

# Set up logging
logger = logging.getLogger(__name__)

# Try to import psutil, but don't fail if not available
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    logger.warning("psutil not available, system monitoring will be limited")
    HAS_PSUTIL = False


class PerformanceMonitor:
    """
    Monitor system performance and application metrics.
    
    This class collects and reports on various performance metrics like
    CPU usage, memory consumption, frame rates, and audio processing latency.
    """
    
    def __init__(self, window_size: int = 120):
        """
        Initialize performance monitor.
        
        Args:
            window_size: Number of samples to keep in the history (default: 120)
        """
        self.window_size = window_size
        
        # System metrics
        self.cpu_usage: Deque[float] = deque(maxlen=window_size)
        self.memory_usage: Deque[float] = deque(maxlen=window_size)
        self.memory_available: Deque[float] = deque(maxlen=window_size)
        
        # Application metrics
        self.frame_times: Deque[float] = deque(maxlen=window_size)
        self.audio_buffer_health: Deque[float] = deque(maxlen=window_size)
        self.audio_latency: Deque[float] = deque(maxlen=window_size)
        
        # Audio processing metrics
        self.audio_processing_time: Deque[float] = deque(maxlen=window_size)
        self.feature_extraction_time: Deque[float] = deque(maxlen=window_size)
        
        # Timing references
        self.last_frame_time = time.time()
        self.monitoring_interval = 1.0  # seconds
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Collect system info once
        self.system_info = self._get_system_info()
        
    def start_monitoring(self):
        """Start continuous background monitoring."""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("Performance monitoring started")
        
    def stop_monitoring(self):
        """Stop continuous background monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
            self.monitor_thread = None
        logger.info("Performance monitoring stopped")
            
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.is_monitoring:
            try:
                self.update_system_metrics()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
    
    def update_system_metrics(self):
        """Update system metrics (CPU, memory)."""
        if not HAS_PSUTIL:
            return
            
        try:
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            self.cpu_usage.append(cpu_percent)
            
            # Get memory usage
            memory = psutil.virtual_memory()
            self.memory_usage.append(memory.percent)
            self.memory_available.append(memory.available / (1024 * 1024))  # MB
            
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")
    
    def update_frame_time(self):
        """Update frame rendering time metric."""
        current_time = time.time()
        dt = current_time - self.last_frame_time
        self.last_frame_time = current_time
        
        # Add to history
        self.frame_times.append(dt)
        
        return dt
        
    def get_fps(self) -> float:
        """
        Get the current frames per second.
        
        Returns:
            float: Current FPS
        """
        if not self.frame_times:
            return 0.0
            
        # Calculate average of recent frame times (last 30)
        recent_times = list(self.frame_times)[-30:]
        avg_time = sum(recent_times) / len(recent_times)
        
        # Avoid division by zero
        if avg_time <= 0:
            return 0.0
            
        return 1.0 / avg_time
    
    def record_audio_buffer_health(self, buffer_size: int, max_size: int):
        """
        Record audio buffer health.
        
        Args:
            buffer_size: Current buffer size
            max_size: Maximum buffer size
        """
        # Calculate health as percentage of buffer used
        health = (buffer_size / max_size) * 100.0
        self.audio_buffer_health.append(health)
    
    def record_audio_latency(self, latency_ms: float):
        """
        Record audio processing latency.
        
        Args:
            latency_ms: Latency in milliseconds
        """
        self.audio_latency.append(latency_ms)
    
    def record_audio_processing_time(self, processing_time_ms: float):
        """
        Record audio processing time.
        
        Args:
            processing_time_ms: Processing time in milliseconds
        """
        self.audio_processing_time.append(processing_time_ms)
    
    def record_feature_extraction_time(self, extraction_time_ms: float):
        """
        Record feature extraction time.
        
        Args:
            extraction_time_ms: Extraction time in milliseconds
        """
        self.feature_extraction_time.append(extraction_time_ms)
    
    def get_performance_data(self) -> Dict[str, Any]:
        """
        Get a comprehensive report of all performance metrics.
        
        Returns:
            dict: Performance metrics
        """
        return {
            "system": self._get_system_metrics(),
            "rendering": self._get_rendering_metrics(),
            "audio": self._get_audio_metrics(),
            "info": self.system_info
        }
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """
        Get system resource metrics.
        
        Returns:
            dict: System metrics
        """
        metrics = {
            "cpu": {
                "current": self.cpu_usage[-1] if self.cpu_usage else 0.0,
                "avg": self._calculate_average(self.cpu_usage),
                "max": max(self.cpu_usage) if self.cpu_usage else 0.0
            },
            "memory": {
                "percent": {
                    "current": self.memory_usage[-1] if self.memory_usage else 0.0,
                    "avg": self._calculate_average(self.memory_usage),
                    "max": max(self.memory_usage) if self.memory_usage else 0.0
                },
                "available_mb": {
                    "current": self.memory_available[-1] if self.memory_available else 0.0,
                    "avg": self._calculate_average(self.memory_available),
                    "min": min(self.memory_available) if self.memory_available else 0.0
                }
            }
        }
        
        # Add GPU info if available
        if HAS_PSUTIL:
            try:
                # Try to get detailed process info for this process
                process = psutil.Process()
                metrics["process"] = {
                    "cpu_percent": process.cpu_percent(interval=None),
                    "memory_percent": process.memory_percent(),
                    "memory_mb": process.memory_info().rss / (1024 * 1024),
                    "threads": len(process.threads()),
                    "open_files": len(process.open_files())
                }
            except Exception as e:
                logger.debug(f"Could not get detailed process info: {e}")
        
        return metrics
    
    def _get_rendering_metrics(self) -> Dict[str, Any]:
        """
        Get rendering performance metrics.
        
        Returns:
            dict: Rendering metrics
        """
        fps = self.get_fps()
        
        return {
            "fps": {
                "current": fps,
                "stable": fps >= 58.0  # Consider stable if near 60 FPS
            },
            "frame_time_ms": {
                "current": (self.frame_times[-1] * 1000.0) if self.frame_times else 0.0,
                "avg": self._calculate_average(self.frame_times) * 1000.0,
                "max": max(self.frame_times) * 1000.0 if self.frame_times else 0.0
            }
        }
    
    def _get_audio_metrics(self) -> Dict[str, Any]:
        """
        Get audio processing metrics.
        
        Returns:
            dict: Audio metrics
        """
        return {
            "buffer_health_percent": {
                "current": self.audio_buffer_health[-1] if self.audio_buffer_health else 0.0,
                "avg": self._calculate_average(self.audio_buffer_health),
                "min": min(self.audio_buffer_health) if self.audio_buffer_health else 0.0
            },
            "latency_ms": {
                "current": self.audio_latency[-1] if self.audio_latency else 0.0,
                "avg": self._calculate_average(self.audio_latency),
                "max": max(self.audio_latency) if self.audio_latency else 0.0
            },
            "processing_time_ms": {
                "current": self.audio_processing_time[-1] if self.audio_processing_time else 0.0,
                "avg": self._calculate_average(self.audio_processing_time),
                "max": max(self.audio_processing_time) if self.audio_processing_time else 0.0
            },
            "feature_extraction_ms": {
                "current": self.feature_extraction_time[-1] if self.feature_extraction_time else 0.0,
                "avg": self._calculate_average(self.feature_extraction_time),
                "max": max(self.feature_extraction_time) if self.feature_extraction_time else 0.0
            }
        }
    
    def _get_system_info(self) -> Dict[str, Any]:
        """
        Get system information.
        
        Returns:
            dict: System information
        """
        info = {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "processor": platform.processor()
        }
        
        if HAS_PSUTIL:
            try:
                # Get CPU info
                cpu_count = psutil.cpu_count(logical=False)
                cpu_count_logical = psutil.cpu_count(logical=True)
                
                # Get memory info
                memory = psutil.virtual_memory()
                
                info.update({
                    "cpu": {
                        "physical_cores": cpu_count,
                        "logical_cores": cpu_count_logical,
                        "frequency_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else "N/A"
                    },
                    "memory": {
                        "total_gb": memory.total / (1024**3),
                        "type": "virtual"
                    }
                })
            except Exception as e:
                logger.debug(f"Could not get detailed system info: {e}")
                
        return info
    
    def _calculate_average(self, values: Deque[float]) -> float:
        """
        Calculate average of values in a deque.
        
        Args:
            values: Deque of values
            
        Returns:
            float: Average value
        """
        if not values:
            return 0.0
            
        return sum(values) / len(values)


# Create a singleton instance
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """
    Get the performance monitor singleton.
    
    Returns:
        PerformanceMonitor: Performance monitor instance
    """
    global _performance_monitor
    
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
        
    return _performance_monitor


# Context manager for timing code blocks
class TimingContext:
    """
    Context manager for timing code execution.
    
    Example:
        ```
        with TimingContext("feature_extraction") as tc:
            # Code to time
            features = extract_features(data)
            
        # tc.elapsed_ms contains elapsed time in milliseconds
        ```
    """
    
    def __init__(self, operation_name: str, record_in_monitor: bool = True):
        """
        Initialize timing context.
        
        Args:
            operation_name: Name of operation being timed
            record_in_monitor: Whether to record timing in performance monitor
        """
        self.operation_name = operation_name
        self.start_time = 0.0
        self.elapsed_ms = 0.0
        self.record_in_monitor = record_in_monitor
        
    def __enter__(self):
        """Enter context manager and start timing."""
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and stop timing."""
        end_time = time.time()
        self.elapsed_ms = (end_time - self.start_time) * 1000.0
        
        if self.record_in_monitor:
            monitor = get_performance_monitor()
            
            # Record in appropriate metric based on operation name
            if "audio" in self.operation_name.lower():
                monitor.record_audio_processing_time(self.elapsed_ms)
            elif "feature" in self.operation_name.lower():
                monitor.record_feature_extraction_time(self.elapsed_ms)
        
        logger.debug(f"{self.operation_name} took {self.elapsed_ms:.2f} ms") 