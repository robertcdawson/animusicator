#!/usr/bin/env python3
"""
Crash reporting utility for monitoring and reporting application errors.

This module provides tools to capture and report unhandled exceptions
and application errors using Sentry or local logging as a fallback.
"""

import os
import sys
import platform
import logging
import traceback
from typing import Optional, Dict, Any, Callable

# Set up logging
logger = logging.getLogger(__name__)

# Try to import sentry_sdk, but don't fail if not available
try:
    import sentry_sdk
    from sentry_sdk.integrations.logging import LoggingIntegration
    HAS_SENTRY = True
except ImportError:
    logger.warning("sentry_sdk not available, using local crash reporting only")
    HAS_SENTRY = False


class CrashReporter:
    """
    Crash reporter for monitoring and reporting application errors.
    
    This class provides tools to capture and report unhandled exceptions
    and application errors to Sentry or local logs as a fallback.
    """
    
    def __init__(
        self, 
        dsn: Optional[str] = None,
        app_name: str = "Animusicator",
        app_version: str = "0.1.0",
        environment: str = "development",
        enable_reporting: bool = True,
        log_dir: Optional[str] = None
    ):
        """
        Initialize crash reporter.
        
        Args:
            dsn: Sentry DSN (Data Source Name) for remote reporting
            app_name: Application name
            app_version: Application version
            environment: Environment (development, production, etc.)
            enable_reporting: Whether to enable crash reporting
            log_dir: Directory for local crash logs
        """
        self.app_name = app_name
        self.app_version = app_version
        self.environment = environment
        self.enable_reporting = enable_reporting
        self.dsn = dsn
        self.log_dir = log_dir or os.path.join(os.path.expanduser("~"), ".musicviz", "logs")
        
        # System information
        self.system_info = self._get_system_info()
        
        # Crash file path (for local reporting)
        os.makedirs(self.log_dir, exist_ok=True)
        self.crash_log_path = os.path.join(self.log_dir, "crash_report.log")
        
        # Initialize crash reporting
        self.initialized = False
        if enable_reporting:
            self._initialize()
    
    def _initialize(self):
        """Initialize crash reporting backend."""
        # Set up Sentry SDK if available and configured
        if HAS_SENTRY and self.dsn:
            try:
                # Configure Sentry integrations
                sentry_logging = LoggingIntegration(
                    level=logging.INFO,        # Capture info and above as breadcrumbs
                    event_level=logging.ERROR  # Send errors as events
                )
                
                # Initialize Sentry SDK
                sentry_sdk.init(
                    dsn=self.dsn,
                    environment=self.environment,
                    release=f"{self.app_name}@{self.app_version}",
                    integrations=[sentry_logging],
                    
                    # Set trace sample rate for performance monitoring
                    traces_sample_rate=0.1,
                    
                    # Configure which data to include/exclude
                    send_default_pii=False,  # Don't send personal identifiable info
                    
                    # Set maximum breadcrumbs (for memory usage)
                    max_breadcrumbs=50
                )
                
                # Set user info (anonymized)
                # Generate a device ID that doesn't contain PII
                import hashlib
                system_id = hashlib.sha256(
                    f"{platform.node()}_{os.getlogin()}_{self.system_info['machine_id']}".encode()
                ).hexdigest()[:12]
                
                sentry_sdk.set_user({"id": system_id})
                
                # Set context for all events
                sentry_sdk.set_context("system", self.system_info)
                
                logger.info(f"Sentry crash reporting initialized ({self.environment})")
                self.initialized = True
                
            except Exception as e:
                logger.error(f"Failed to initialize Sentry: {e}")
                self._setup_local_reporting()
        else:
            logger.info("Using local crash reporting only")
            self._setup_local_reporting()
    
    def _setup_local_reporting(self):
        """Set up local crash reporting as fallback."""
        # Configure sys.excepthook to catch unhandled exceptions
        self.original_excepthook = sys.excepthook
        sys.excepthook = self._local_excepthook
        self.initialized = True
        logger.info(f"Local crash reporting initialized (logs at {self.crash_log_path})")
    
    def _local_excepthook(self, exc_type, exc_value, exc_traceback):
        """
        Local exception hook for handling unhandled exceptions.
        
        Args:
            exc_type: Exception type
            exc_value: Exception value
            exc_traceback: Exception traceback
        """
        # Log the exception
        logger.error("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))
        
        # Write crash report to file
        self._write_crash_report(exc_type, exc_value, exc_traceback)
        
        # Call the original excepthook
        self.original_excepthook(exc_type, exc_value, exc_traceback)
    
    def _write_crash_report(self, exc_type, exc_value, exc_traceback):
        """
        Write crash report to file.
        
        Args:
            exc_type: Exception type
            exc_value: Exception value
            exc_traceback: Exception traceback
        """
        try:
            with open(self.crash_log_path, "a") as f:
                import datetime
                
                # Write header
                f.write(f"\n{'='*80}\n")
                f.write(f"CRASH REPORT: {datetime.datetime.now().isoformat()}\n")
                f.write(f"Application: {self.app_name} {self.app_version}\n")
                f.write(f"Environment: {self.environment}\n")
                f.write(f"{'='*80}\n\n")
                
                # Write system info
                f.write("System Information:\n")
                for key, value in self.system_info.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
                
                # Write exception info
                f.write(f"Exception Type: {exc_type.__name__}\n")
                f.write(f"Exception Value: {exc_value}\n")
                f.write("\nTraceback:\n")
                
                # Format traceback
                tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
                for line in tb_lines:
                    f.write(line)
                
                f.write(f"\n{'='*80}\n")
        except Exception as e:
            logger.error(f"Failed to write crash report: {e}")
    
    def capture_exception(self, exc_info=None, **kwargs):
        """
        Capture and report an exception.
        
        Args:
            exc_info: Exception info tuple
            **kwargs: Additional context data
        """
        if not self.enable_reporting:
            return
            
        if HAS_SENTRY and self.dsn and self.initialized:
            # Use Sentry to capture the exception
            with sentry_sdk.push_scope() as scope:
                # Add additional tags and context data
                for key, value in kwargs.items():
                    scope.set_tag(key, value)
                    
                # Capture the exception
                sentry_sdk.capture_exception(exc_info)
                
        else:
            # Use local logging as fallback
            if exc_info:
                logger.error("Exception captured", exc_info=exc_info, extra=kwargs)
            else:
                logger.error("Exception captured", exc_info=True, extra=kwargs)
                
            # If we have exception info, write a crash report
            if exc_info:
                self._write_crash_report(*exc_info)
    
    def capture_message(self, message: str, level: str = "info", **kwargs):
        """
        Capture and report a message.
        
        Args:
            message: Message to capture
            level: Message level (info, warning, error)
            **kwargs: Additional context data
        """
        if not self.enable_reporting:
            return
            
        if HAS_SENTRY and self.dsn and self.initialized:
            # Convert level string to Sentry level
            sentry_level = {
                "debug": "debug",
                "info": "info",
                "warning": "warning",
                "error": "error",
                "critical": "fatal"
            }.get(level.lower(), "info")
            
            # Use Sentry to capture the message
            with sentry_sdk.push_scope() as scope:
                # Add additional tags and context data
                for key, value in kwargs.items():
                    scope.set_tag(key, value)
                    
                # Capture the message
                sentry_sdk.capture_message(message, level=sentry_level)
        else:
            # Use local logging as fallback
            log_method = getattr(logger, level.lower(), logger.info)
            log_method(message, extra=kwargs)
    
    def add_breadcrumb(self, category: str, message: str, level: str = "info", data: Optional[Dict[str, Any]] = None):
        """
        Add a breadcrumb to track application flow.
        
        Args:
            category: Breadcrumb category
            message: Breadcrumb message
            level: Breadcrumb level (info, warning, error)
            data: Additional data
        """
        if not self.enable_reporting:
            return
            
        if HAS_SENTRY and self.dsn and self.initialized:
            sentry_sdk.add_breadcrumb(
                category=category,
                message=message,
                level=level,
                data=data
            )
        else:
            # Use local logging as fallback
            log_method = getattr(logger, level.lower(), logger.info)
            log_method(f"{category}: {message}", extra={"data": data})
    
    def set_user(self, user_id: str):
        """
        Set user context for crash reports.
        
        Args:
            user_id: Anonymized user ID
        """
        if not self.enable_reporting:
            return
            
        if HAS_SENTRY and self.dsn and self.initialized:
            sentry_sdk.set_user({"id": user_id})
    
    def set_tag(self, key: str, value: str):
        """
        Set a tag for all subsequent events.
        
        Args:
            key: Tag key
            value: Tag value
        """
        if not self.enable_reporting:
            return
            
        if HAS_SENTRY and self.dsn and self.initialized:
            sentry_sdk.set_tag(key, value)
    
    def wrap_function(self, func: Callable):
        """
        Wrap a function to automatically capture exceptions.
        
        Args:
            func: Function to wrap
            
        Returns:
            Callable: Wrapped function
        """
        if not self.enable_reporting:
            return func
            
        def wrapped(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                self.capture_exception()
                raise
                
        return wrapped
            
    def _get_system_info(self) -> Dict[str, Any]:
        """
        Get system information for crash reports.
        
        Returns:
            dict: System information
        """
        info = {
            "os": platform.system(),
            "os_version": platform.release(),
            "os_build": platform.version(),
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "machine_id": self._get_machine_id()
        }
        
        return info
    
    def _get_machine_id(self) -> str:
        """
        Generate a unique machine ID.
        
        Returns:
            str: Machine ID
        """
        # Try to get a stable machine ID that doesn't contain PII
        try:
            if platform.system() == "Darwin":  # macOS
                import subprocess
                result = subprocess.run(
                    ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
                    capture_output=True,
                    text=True
                )
                for line in result.stdout.split("\n"):
                    if "IOPlatformUUID" in line:
                        return line.split("=")[1].strip().strip('"')
                        
            elif platform.system() == "Linux":
                try:
                    with open("/var/lib/dbus/machine-id", "r") as f:
                        return f.read().strip()
                except FileNotFoundError:
                    try:
                        with open("/etc/machine-id", "r") as f:
                            return f.read().strip()
                    except FileNotFoundError:
                        pass
                        
            elif platform.system() == "Windows":
                import winreg
                with winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    r"SOFTWARE\Microsoft\Cryptography"
                ) as key:
                    return winreg.QueryValueEx(key, "MachineGuid")[0]
        except Exception:
            pass
            
        # Fallback to a hash of hostname (less stable but better than nothing)
        import hashlib
        return hashlib.md5(platform.node().encode()).hexdigest()


# Create a singleton instance
_crash_reporter: Optional[CrashReporter] = None


def initialize_crash_reporting(
    dsn: Optional[str] = None,
    app_name: str = "Animusicator",
    app_version: str = "0.1.0",
    environment: str = "development",
    enable_reporting: bool = True,
    log_dir: Optional[str] = None
) -> CrashReporter:
    """
    Initialize the crash reporter singleton.
    
    Args:
        dsn: Sentry DSN
        app_name: Application name
        app_version: Application version
        environment: Environment name
        enable_reporting: Whether to enable crash reporting
        log_dir: Directory for local crash logs
        
    Returns:
        CrashReporter: Crash reporter instance
    """
    global _crash_reporter
    
    if _crash_reporter is None:
        _crash_reporter = CrashReporter(
            dsn=dsn,
            app_name=app_name,
            app_version=app_version,
            environment=environment,
            enable_reporting=enable_reporting,
            log_dir=log_dir
        )
        
    return _crash_reporter


def get_crash_reporter() -> CrashReporter:
    """
    Get the crash reporter singleton.
    
    Returns:
        CrashReporter: Crash reporter instance
    """
    global _crash_reporter
    
    if _crash_reporter is None:
        _crash_reporter = CrashReporter()
        
    return _crash_reporter


def capture_exception(*args, **kwargs):
    """Shorthand for get_crash_reporter().capture_exception()"""
    return get_crash_reporter().capture_exception(*args, **kwargs)


def capture_message(*args, **kwargs):
    """Shorthand for get_crash_reporter().capture_message()"""
    return get_crash_reporter().capture_message(*args, **kwargs)


def add_breadcrumb(*args, **kwargs):
    """Shorthand for get_crash_reporter().add_breadcrumb()"""
    return get_crash_reporter().add_breadcrumb(*args, **kwargs)


# Exception decorator
def report_exceptions(func):
    """
    Decorator to report exceptions in functions.
    
    Example:
        ```
        @report_exceptions
        def my_function():
            # This function's exceptions will be reported
            pass
        ```
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            capture_exception()
            raise
            
    return wrapper 