#!/usr/bin/env python3
"""
System resource monitoring utility.

This module provides tools to monitor system resources like CPU, memory,
disk usage, and network activity using psutil.
"""

import logging
import threading
import time
from typing import Dict, Any, List, Optional, Tuple
from collections import deque

# Set up logging
logger = logging.getLogger(__name__)

# Try to import psutil, but don't fail if not available
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    logger.warning("psutil not available, system monitoring will be limited")
    HAS_PSUTIL = False


class SystemMonitor:
    """
    System resource monitor using psutil.
    
    This class provides methods to monitor system resources like CPU, memory,
    disk usage, and network activity, with support for continuous monitoring
    in a background thread.
    """
    
    def __init__(self, interval: float = 1.0, history_size: int = 60):
        """
        Initialize system monitor.
        
        Args:
            interval: Monitoring interval in seconds
            history_size: Number of data points to keep in history
        """
        self.interval = interval
        self.history_size = history_size
        
        # Initialize history containers
        self.cpu_percent: deque = deque(maxlen=history_size)
        self.memory_percent: deque = deque(maxlen=history_size)
        self.memory_available: deque = deque(maxlen=history_size)
        self.disk_usage: deque = deque(maxlen=history_size)
        self.swap_percent: deque = deque(maxlen=history_size)
        self.network_sent: deque = deque(maxlen=history_size)
        self.network_recv: deque = deque(maxlen=history_size)
        
        # Process-specific metrics (for this process)
        self.process_cpu: deque = deque(maxlen=history_size)
        self.process_memory: deque = deque(maxlen=history_size)
        self.process_threads: deque = deque(maxlen=history_size)
        self.process_fds: deque = deque(maxlen=history_size)
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Get current process if psutil is available
        self.process = psutil.Process() if HAS_PSUTIL else None
        
        # Get initial network counters
        self.last_net_io = self._get_network_counters()
        
        # Check if monitoring is possible
        self.can_monitor = HAS_PSUTIL
        
        # Get system info once
        self.system_info = self._get_system_info()
    
    def start_monitoring(self):
        """Start continuous monitoring in a background thread."""
        if not self.can_monitor:
            logger.warning("System monitoring not available (psutil not installed)")
            return
            
        if self.is_monitoring:
            return  # Already monitoring
            
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True  # Daemon thread will exit when main program exits
        )
        self.monitor_thread.start()
        logger.info(f"System monitoring started (interval: {self.interval}s)")
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        if not self.is_monitoring:
            return
            
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
            self.monitor_thread = None
        logger.info("System monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.is_monitoring:
            try:
                self.update_all()
                time.sleep(self.interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
    
    def update_all(self):
        """Update all system metrics."""
        if not self.can_monitor:
            return
            
        try:
            # Update system metrics
            self._update_cpu()
            self._update_memory()
            self._update_disk()
            self._update_network()
            
            # Update process metrics
            self._update_process()
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")
    
    def _update_cpu(self):
        """Update CPU metrics."""
        if not HAS_PSUTIL:
            return
            
        try:
            # Get CPU usage percentage across all cores
            cpu_percent = psutil.cpu_percent(interval=0)
            self.cpu_percent.append(cpu_percent)
        except Exception as e:
            logger.error(f"Error updating CPU metrics: {e}")
    
    def _update_memory(self):
        """Update memory metrics."""
        if not HAS_PSUTIL:
            return
            
        try:
            # Get virtual memory info
            memory = psutil.virtual_memory()
            self.memory_percent.append(memory.percent)
            self.memory_available.append(memory.available / (1024 * 1024))  # MB
            
            # Get swap memory info
            swap = psutil.swap_memory()
            self.swap_percent.append(swap.percent)
        except Exception as e:
            logger.error(f"Error updating memory metrics: {e}")
    
    def _update_disk(self):
        """Update disk usage metrics."""
        if not HAS_PSUTIL:
            return
            
        try:
            # Get disk usage for system drive
            disk = psutil.disk_usage('/')
            self.disk_usage.append(disk.percent)
        except Exception as e:
            logger.error(f"Error updating disk metrics: {e}")
    
    def _update_network(self):
        """Update network metrics."""
        if not HAS_PSUTIL:
            return
            
        try:
            # Get network IO counters
            net_io = self._get_network_counters()
            
            if self.last_net_io:
                # Calculate bytes sent/received since last update
                bytes_sent = net_io[0] - self.last_net_io[0]
                bytes_recv = net_io[1] - self.last_net_io[1]
                
                # Convert to KB/s
                sent_kbs = bytes_sent / 1024 / self.interval
                recv_kbs = bytes_recv / 1024 / self.interval
                
                self.network_sent.append(sent_kbs)
                self.network_recv.append(recv_kbs)
            
            # Update last counters
            self.last_net_io = net_io
        except Exception as e:
            logger.error(f"Error updating network metrics: {e}")
    
    def _update_process(self):
        """Update process-specific metrics."""
        if not HAS_PSUTIL or not self.process:
            return
            
        try:
            # Get CPU usage for this process
            self.process_cpu.append(self.process.cpu_percent(interval=0))
            
            # Get memory usage for this process
            memory_info = self.process.memory_info()
            self.process_memory.append(memory_info.rss / (1024 * 1024))  # MB
            
            # Get thread count
            self.process_threads.append(len(self.process.threads()))
            
            # Get file descriptor count (unix) or handle count (windows)
            if hasattr(self.process, 'num_fds'):
                self.process_fds.append(self.process.num_fds())
            elif hasattr(self.process, 'num_handles'):
                self.process_fds.append(self.process.num_handles())
            else:
                pass  # Not supported on this platform
        except Exception as e:
            logger.error(f"Error updating process metrics: {e}")
    
    def _get_network_counters(self) -> Tuple[int, int]:
        """
        Get network IO counters.
        
        Returns:
            tuple: (bytes_sent, bytes_received)
        """
        if not HAS_PSUTIL:
            return (0, 0)
            
        try:
            net_io = psutil.net_io_counters()
            return (net_io.bytes_sent, net_io.bytes_recv)
        except Exception as e:
            logger.error(f"Error getting network counters: {e}")
            return (0, 0)
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """
        Get current system metrics.
        
        Returns:
            dict: System metrics
        """
        if not self.can_monitor:
            return {"error": "System monitoring not available"}
            
        # Get latest metrics
        metrics = {
            "cpu": {
                "percent": self.cpu_percent[-1] if self.cpu_percent else 0.0,
                "avg_percent": self._calculate_average(self.cpu_percent),
                "count": psutil.cpu_count() if HAS_PSUTIL else 0
            },
            "memory": {
                "percent": self.memory_percent[-1] if self.memory_percent else 0.0,
                "available_mb": self.memory_available[-1] if self.memory_available else 0.0,
                "avg_percent": self._calculate_average(self.memory_percent)
            },
            "swap": {
                "percent": self.swap_percent[-1] if self.swap_percent else 0.0
            },
            "disk": {
                "percent": self.disk_usage[-1] if self.disk_usage else 0.0
            },
            "network": {
                "sent_kbs": self.network_sent[-1] if self.network_sent else 0.0,
                "recv_kbs": self.network_recv[-1] if self.network_recv else 0.0
            },
            "process": {
                "cpu_percent": self.process_cpu[-1] if self.process_cpu else 0.0,
                "memory_mb": self.process_memory[-1] if self.process_memory else 0.0,
                "threads": self.process_threads[-1] if self.process_threads else 0,
                "file_descriptors": self.process_fds[-1] if self.process_fds else 0
            }
        }
        
        return metrics
    
    def get_historical_metrics(self) -> Dict[str, List[float]]:
        """
        Get historical metrics for trend analysis.
        
        Returns:
            dict: Historical metrics
        """
        if not self.can_monitor:
            return {"error": "System monitoring not available"}
            
        # Get all historical data
        metrics = {
            "timestamps": list(range(len(self.cpu_percent))),  # Relative timestamps
            "cpu_percent": list(self.cpu_percent),
            "memory_percent": list(self.memory_percent),
            "memory_available_mb": list(self.memory_available),
            "swap_percent": list(self.swap_percent),
            "disk_percent": list(self.disk_usage),
            "network_sent_kbs": list(self.network_sent),
            "network_recv_kbs": list(self.network_recv),
            "process_cpu_percent": list(self.process_cpu),
            "process_memory_mb": list(self.process_memory)
        }
        
        return metrics
    
    def _get_system_info(self) -> Dict[str, Any]:
        """
        Get system information.
        
        Returns:
            dict: System information
        """
        if not HAS_PSUTIL:
            return {"error": "System info not available"}
            
        try:
            # Get CPU info
            cpu_info = {
                "count_physical": psutil.cpu_count(logical=False) or 0,
                "count_logical": psutil.cpu_count(logical=True) or 0
            }
            
            # Add CPU frequency if available
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                cpu_info["frequency_mhz"] = cpu_freq.current
            
            # Get memory info
            memory = psutil.virtual_memory()
            memory_info = {
                "total_gb": memory.total / (1024 ** 3),
                "available_gb": memory.available / (1024 ** 3)
            }
            
            # Get disk info
            disk = psutil.disk_usage('/')
            disk_info = {
                "total_gb": disk.total / (1024 ** 3),
                "free_gb": disk.free / (1024 ** 3)
            }
            
            # Get boot time
            boot_time = psutil.boot_time()
            import datetime
            boot_datetime = datetime.datetime.fromtimestamp(boot_time)
            
            # Combine system info
            return {
                "cpu": cpu_info,
                "memory": memory_info,
                "disk": disk_info,
                "boot_time": boot_datetime.isoformat(),
                "python_process": {
                    "pid": self.process.pid if self.process else None,
                    "create_time": datetime.datetime.fromtimestamp(
                        self.process.create_time()
                    ).isoformat() if self.process else None
                }
            }
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return {"error": f"Error getting system info: {e}"}
    
    def _calculate_average(self, values) -> float:
        """
        Calculate average of values.
        
        Args:
            values: Collection of values
            
        Returns:
            float: Average value
        """
        if not values:
            return 0.0
            
        return sum(values) / len(values)


# Create a singleton instance
_system_monitor: Optional[SystemMonitor] = None


def get_system_monitor(interval: float = 1.0) -> SystemMonitor:
    """
    Get system monitor singleton instance.
    
    Args:
        interval: Monitoring interval in seconds
        
    Returns:
        SystemMonitor: System monitor instance
    """
    global _system_monitor
    
    if _system_monitor is None:
        _system_monitor = SystemMonitor(interval=interval)
        
    return _system_monitor


# Main function for standalone testing
if __name__ == "__main__":
    # Configure basic logging
    logging.basicConfig(level=logging.INFO)
    
    # Create system monitor
    monitor = get_system_monitor(interval=1.0)
    
    # Start monitoring
    monitor.start_monitoring()
    
    try:
        # Display metrics for 10 seconds
        for _ in range(10):
            metrics = monitor.get_system_metrics()
            print("\n--- System Metrics ---")
            print(f"CPU: {metrics['cpu']['percent']:.1f}% (avg: {metrics['cpu']['avg_percent']:.1f}%)")
            print(f"Memory: {metrics['memory']['percent']:.1f}% ({metrics['memory']['available_mb']:.0f} MB free)")
            print(f"Disk: {metrics['disk']['percent']:.1f}%")
            print(f"Network: ↑ {metrics['network']['sent_kbs']:.1f} KB/s, ↓ {metrics['network']['recv_kbs']:.1f} KB/s")
            print(f"Process: {metrics['process']['cpu_percent']:.1f}% CPU, {metrics['process']['memory_mb']:.1f} MB memory")
            time.sleep(1)
    finally:
        # Stop monitoring
        monitor.stop_monitoring() 