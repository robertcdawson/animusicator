#!/usr/bin/env python3
"""
Main application window for Animusicator.
"""

import os
import logging
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QComboBox, QPushButton, QCheckBox, QLabel, QStatusBar, QMessageBox
)
from PyQt5.QtCore import Qt, pyqtSlot, QTimer, QSize, QEvent

from .visual_widget import VisualWidget
from ..audio.engine import AudioEngine
from ..config.settings import load_settings

# Create logger
logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """
    Main application window for Animusicator.
    
    This window contains the visualization widget and controls for
    audio device selection and visualization settings.
    """
    
    def __init__(self, parent=None):
        """Initialize the main window."""
        super().__init__(parent)
        
        # Load settings
        self.settings = load_settings()
        
        # Set window properties
        self.setWindowTitle("Animusicator")
        self.setMinimumSize(QSize(800, 600))
        
        # Create audio engine
        self.audio_engine = AudioEngine(
            device=self.settings.get('audio_device'),
            sample_rate=self.settings.get('sample_rate', 44100),
            frame_size=self.settings.get('fft_size', 2048),
            hop_size=self.settings.get('hop_size', 512),
            buffer_size=self.settings.get('buffer_size', 10)
        )
        
        # Create UI components
        self.setup_ui()
        
        # Connect signals
        self.connect_signals()
        
        # Populate device list
        self.populate_devices()
        
        # Set up status bar
        self.statusBar().showMessage("Ready")
        
        # Start with test mode
        self._starting_audio = False
        self.start_stop_button.setText("Start")
        
        # Install event filter to debug events
        self.installEventFilter(self)
        
        logger.info("Main window initialized")
    
    def setup_ui(self):
        """Set up the user interface."""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Create control panel
        control_panel = QWidget()
        control_layout = QHBoxLayout(control_panel)
        control_layout.setContentsMargins(10, 10, 10, 10)
        
        # Create device selection
        device_label = QLabel("Audio Device:")
        device_label.setAccessibleName("Audio Device Label")
        device_label.setAccessibleDescription("Label for audio device selection dropdown")
        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(200)
        self.device_combo.setAccessibleName("Audio Device Selection")
        self.device_combo.setAccessibleDescription("Select the audio device for sound capture")
        
        # Create start/stop button
        self.start_stop_button = QPushButton("Start")
        self.start_stop_button.setAccessibleName("Start Stop Button")
        self.start_stop_button.setAccessibleDescription("Start or stop audio capture")
        self.start_stop_button.setShortcut("Ctrl+S")
        
        # Create fullscreen toggle
        self.fullscreen_checkbox = QCheckBox("Fullscreen")
        self.fullscreen_checkbox.setAccessibleName("Fullscreen Toggle")
        self.fullscreen_checkbox.setAccessibleDescription("Toggle fullscreen display mode")
        self.fullscreen_checkbox.setShortcut("F11")
        
        # Add test mode button
        self.test_mode_button = QPushButton("Test Mode")
        self.test_mode_button.setToolTip("Run with simulated audio (for testing/development)")
        self.test_mode_button.setAccessibleName("Test Mode Button")
        self.test_mode_button.setAccessibleDescription("Run with simulated audio data for testing")
        self.test_mode_button.setShortcut("Ctrl+T")
        
        # Add debug overlay toggle
        self.debug_overlay_button = QPushButton("Debug Info")
        self.debug_overlay_button.setToolTip("Toggle performance metrics display")
        self.debug_overlay_button.setAccessibleName("Debug Overlay Button")
        self.debug_overlay_button.setAccessibleDescription("Show or hide performance metrics")
        self.debug_overlay_button.setShortcut("Ctrl+D")
        
        # Add controls to layout
        control_layout.addWidget(device_label)
        control_layout.addWidget(self.device_combo)
        control_layout.addWidget(self.start_stop_button)
        control_layout.addWidget(self.fullscreen_checkbox)
        control_layout.addWidget(self.test_mode_button)
        control_layout.addWidget(self.debug_overlay_button)
        control_layout.addStretch()
        
        # Add control panel to main layout
        main_layout.addWidget(control_panel)
        
        # Create visualization widget
        self.visual_widget = VisualWidget(self)
        
        # Add visual widget to main layout (takes most of the space)
        main_layout.addWidget(self.visual_widget, 1)
        
        # Set dark background
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #121212;
                color: #FFFFFF;
            }
            QComboBox, QPushButton {
                background-color: #1E1E1E;
                border: 1px solid #333333;
                border-radius: 4px;
                padding: 6px;
                color: #FFFFFF;
            }
            QComboBox:hover, QPushButton:hover {
                background-color: #333333;
            }
            QPushButton:pressed {
                background-color: #1DB954;
            }
            QComboBox QAbstractItemView {
                background-color: #1E1E1E;
                border: 1px solid #333333;
                selection-background-color: #1DB954;
            }
            QCheckBox {
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QCheckBox::indicator:unchecked {
                border: 1px solid #333333;
                background-color: #1E1E1E;
            }
            QCheckBox::indicator:checked {
                background-color: #1DB954;
                border: 1px solid #1AA34A;
            }
            QStatusBar {
                background-color: #1E1E1E;
                color: #B3B3B3;
            }
        """)
    
    def connect_signals(self):
        """Connect UI signals to slots."""
        # Connect device combo box
        self.device_combo.currentIndexChanged.connect(self.on_device_changed)
        
        # Connect start/stop button
        self.start_stop_button.clicked.connect(self.on_start_stop_clicked)
        
        # Connect fullscreen toggle
        self.fullscreen_checkbox.stateChanged.connect(self.on_fullscreen_toggled)
        
        # Connect test mode button
        self.test_mode_button.clicked.connect(self.on_test_mode_clicked)
        
        # Connect debug overlay button
        self.debug_overlay_button.clicked.connect(self.on_debug_overlay_clicked)
        
        # Connect audio engine signals
        self.audio_engine.features_ready.connect(self.visual_widget.update_features)
        self.audio_engine.state_changed.connect(self.on_audio_state_changed)
        self.audio_engine.error_occurred.connect(self.on_audio_error)
        self.audio_engine.device_disconnected.connect(self.on_device_disconnected)
    
    def populate_devices(self):
        """Populate the device selection combo box."""
        try:
            # Clear the combo box
            self.device_combo.clear()
            
            # Get available devices
            devices = self.audio_engine.get_available_devices()
            
            # Add devices to combo box
            for device in devices:
                device_name = device['name']
                self.device_combo.addItem(device_name, device)
            
            # Select BlackHole device if available
            blackhole_index = self.device_combo.findText("BlackHole 2ch", Qt.MatchContains)
            if blackhole_index >= 0:
                self.device_combo.setCurrentIndex(blackhole_index)
            
            logger.info(f"Found {len(devices)} audio input devices")
            
        except Exception as e:
            logger.error(f"Error populating devices: {e}")
            self.statusBar().showMessage(f"Error: {e}")
    
    @pyqtSlot(int)
    def on_device_changed(self, index):
        """Handle device selection change."""
        if index < 0:
            return
            
        # Get selected device
        device_data = self.device_combo.itemData(index)
        device_name = self.device_combo.itemText(index)
        
        logger.info(f"Selected device: {device_name}")
        
        # Update visual widget device name
        self.visual_widget.set_device_name(device_name)
        
        # If audio engine is running, restart it with the new device
        was_running = self.audio_engine.running
        if was_running:
            self.audio_engine.stop()
        
        # Set new device for audio engine
        self.audio_engine.device = device_name
        
        # Restart if it was running
        if was_running:
            self.audio_engine.start()
        
        self.statusBar().showMessage(f"Selected device: {device_name}")
    
    @pyqtSlot()
    def on_start_stop_clicked(self):
        """Handle start/stop button click."""
        if not self.audio_engine.running and not self._starting_audio:
            # Start audio engine
            self._starting_audio = True
            self.start_stop_button.setEnabled(False)
            self.statusBar().showMessage("Starting audio...")
            
            # Set device name in visual widget
            self.visual_widget.set_device_name(self.device_combo.currentText())
            
            # Start the engine
            self.audio_engine.start()
            
        else:
            # Stop audio engine
            self.start_stop_button.setEnabled(False)
            self.statusBar().showMessage("Stopping audio...")
            self.audio_engine.requestInterruption()
            self.audio_engine.wait()
    
    @pyqtSlot(int)
    def on_fullscreen_toggled(self, state):
        """Handle fullscreen toggle."""
        if state == Qt.Checked:
            self.showFullScreen()
        else:
            self.showNormal()
    
    @pyqtSlot(str)
    def on_audio_state_changed(self, state):
        """Handle audio engine state changes."""
        self._starting_audio = False
        self.start_stop_button.setEnabled(True)
        
        if state == "started":
            self.start_stop_button.setText("Stop")
            self.statusBar().showMessage(f"Listening on: {self.audio_engine.device}")
        elif state == "stopped":
            self.start_stop_button.setText("Start")
            self.statusBar().showMessage("Audio stopped")
        elif state == "error":
            self.start_stop_button.setText("Start")
            # Error message will be set by the error handler
        elif state == "disconnected":
            self.start_stop_button.setText("Start")
            self.statusBar().showMessage("Device disconnected - in test mode")
    
    @pyqtSlot(str)
    def on_audio_error(self, error_message):
        """Handle audio engine errors."""
        self.statusBar().showMessage(f"Error: {error_message}")
        logger.error(f"Audio engine error: {error_message}")
        
        # Show error dialog for serious errors
        if "initialization" in error_message.lower() or "device" in error_message.lower():
            QMessageBox.critical(self, "Audio Error", 
                                error_message + "\n\nTry selecting a different device.")
    
    @pyqtSlot()
    def on_test_mode_clicked(self):
        """Toggle test mode with simulated audio."""
        if self.audio_engine:
            if self.audio_engine.is_running:
                # If already running, we need to stop first
                self.audio_engine.stop()
                # Give a moment to stop before starting test mode
                QTimer.singleShot(100, self.audio_engine.start_test_mode)
            else:
                self.audio_engine.start_test_mode()
    
    @pyqtSlot()
    def on_debug_overlay_clicked(self):
        """Toggle the performance metrics debug overlay."""
        self.visual_widget.toggle_debug_overlay()
    
    @pyqtSlot(str)
    def on_device_disconnected(self, reason):
        """Handle device disconnection events."""
        logger.warning(f"Device disconnected: {reason}")
        
        # Update UI
        self.statusBar().showMessage(f"Device disconnected: {reason}")
        
        # Show notification to user
        QMessageBox.warning(
            self, 
            "Device Disconnected",
            f"The audio device has been disconnected: {reason}\n\n"
            f"The application will continue in test mode until you select a new device."
        )
        
        # Refresh device list after a brief delay to allow system to detect changes
        self.populate_devices()
    
    def closeEvent(self, event):
        """Handle window close event."""
        # Print debugging info
        print("MainWindow is being closed")
        print(f"Close reason: {event.spontaneous()}")
        
        # Stop audio engine
        if self.audio_engine.running:
            self.audio_engine.requestInterruption()
            self.audio_engine.wait()
        
        # Clean up visual widget
        self.visual_widget.cleanup()
        
        # Accept the event
        event.accept()

    def keyPressEvent(self, event):
        """Handle keyboard navigation."""
        key = event.key()
        
        # ESC key to exit fullscreen
        if key == Qt.Key_Escape and self.isFullScreen():
            self.fullscreen_checkbox.setChecked(False)
            self.showNormal()
        
        # Space to toggle play/pause
        elif key == Qt.Key_Space:
            self.on_start_stop_clicked()
        
        # F to toggle fullscreen (in addition to F11)
        elif key == Qt.Key_F:
            self.fullscreen_checkbox.setChecked(not self.fullscreen_checkbox.isChecked())
            
        # D to toggle debug overlay
        elif key == Qt.Key_D:
            self.on_debug_overlay_clicked()
            
        # Pass unhandled events to parent
        else:
            super().keyPressEvent(event)

    def eventFilter(self, watched, event):
        """Debug event filter to trace events."""
        if event.type() not in [
            # Filter out frequent events to reduce noise
            QEvent.MouseMove, 
            QEvent.Paint, 
            QEvent.UpdateRequest,
            QEvent.Timer
        ]:
            print(f"Event: {event.type()} on {watched}")
        return super().eventFilter(watched, event) 