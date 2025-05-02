#!/usr/bin/env python3
"""
Main entry point for the Animusicator application.
"""

import os
import sys

from PyQt5.QtWidgets import QApplication

from .gui.main_window import MainWindow
from .utils import gpu_context
from .utils.config_loader import get_config
from .utils.logging_setup import setup_logging
from .utils.error_handler import get_error_handler, ErrorCategory, handle_exception


@handle_exception
def main():
    """Main entry point for the application."""
    # Set up logging
    setup_logging()  # This now uses our enhanced logging system
    
    # Get error handler
    error_handler = get_error_handler()
    
    # Register a callback for displaying errors in the UI
    def display_error_dialog(exception, message, context):
        # This will be called after the application is initialized
        from PyQt5.QtWidgets import QMessageBox
        if QApplication.instance():
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Critical)
            msg_box.setWindowTitle("Error")
            msg_box.setText(message)
            if context:
                msg_box.setDetailedText(str(context))
            msg_box.exec_()
    
    # Register for critical errors only
    error_handler.register_callback(display_error_dialog, [ErrorCategory.CRITICAL])

    # Create logger for main module
    from .utils.logging_setup import get_logger
    logger = get_logger(__name__)
    logger.info("Starting Animusicator")
    
    # Load configuration
    config = get_config()
    logger.info(f"Loaded configuration with {len(config.config)} settings")

    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("Animusicator")
    
    # Apply configuration settings
    fullscreen = config.get("visual.fullscreen", False)
    show_metrics = config.get("debug.show_metrics", False)

    # Create and show main window
    window = MainWindow()
    
    # Apply initial settings from config
    if fullscreen:
        window.fullscreen_checkbox.setChecked(True)
    if show_metrics:
        window.visual_widget.show_debug_overlay = show_metrics
    
    # Set audio device from config if specified
    default_device = config.get("audio.device")
    if default_device and default_device != "default":
        # Find the device by name
        for i in range(window.device_combo.count()):
            if window.device_combo.itemText(i) == default_device:
                window.device_combo.setCurrentIndex(i)
                break
    
    # Explicitly bring window to front
    print("About to show main window...")
    window.show()
    window.raise_()
    window.activateWindow()
    print(f"Window geometry: {window.geometry()}")
    print(f"Window is visible: {window.isVisible()}")

    # Log application startup
    logger.info("Application GUI initialized")

    # Start event loop
    try:
        print("Starting Qt event loop...")
        exit_code = app.exec_()
        print(f"Qt event loop exited with code: {exit_code}")
    except Exception as e:
        # Handle unhandled exceptions
        print(f"Exception in Qt event loop: {e}")
        error_handler.handle_error(
            e, 
            message="Unhandled exception in main event loop",
            category=ErrorCategory.CRITICAL
        )
        exit_code = 1

    # Log application exit
    logger.info(f"Application exiting with code {exit_code}")
    
    # Save any modified settings
    config.set("visual.fullscreen", window.fullscreen_checkbox.isChecked())
    config.set("debug.show_metrics", window.visual_widget.show_debug_overlay)
    
    # Save current device
    if window.device_combo.currentText() != "default":
        config.set("audio.device", window.device_combo.currentText())

    # Return exit code
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
