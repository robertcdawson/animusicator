#!/usr/bin/env python3
"""
Audio engine for capturing and processing audio in real-time.
"""

import numpy as np
import sounddevice as sd
from PyQt5.QtCore import QThread, pyqtSignal
from collections import deque
import logging
import time
import math
import random
from typing import Optional, List, Deque

from .feature_extractor import FeatureExtractor, AudioFeatures

# Create logger
logger = logging.getLogger(__name__)


class AudioEngine(QThread):
    """
    Audio engine that captures audio in real-time and extracts features.
    
    This class uses a separate thread to avoid blocking the GUI and
    provides real-time audio feature extraction for visualization.
    """
    
    # Signal emitted when new audio features are available
    features_ready = pyqtSignal(object)
    
    # Signal emitted when audio state changes
    state_changed = pyqtSignal(str)
    
    # Signal emitted when an error occurs
    error_occurred = pyqtSignal(str)
    
    # Signal emitted when device is disconnected
    device_disconnected = pyqtSignal(str)
    
    def __init__(self, device=None, sample_rate=44100, 
                 frame_size=2048, hop_size=512, buffer_size=10):
        """
        Initialize the audio engine.
        
        Args:
            device: Audio device name or ID (default: system default)
            sample_rate: Sample rate in Hz (default: 44100)
            frame_size: FFT frame size (default: 2048)
            hop_size: Hop size for overlapping windows (default: 512)
            buffer_size: Number of frames to buffer (default: 10)
        """
        super().__init__()
        
        # Audio parameters
        self.device = device
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size
        
        # Processing state
        self.running = False
        self.stream = None
        self.buffer: Deque[np.ndarray] = deque(maxlen=buffer_size)
        
        # Disconnection detection
        self.silent_frame_count = 0
        self.max_silent_frames = 50  # About 5 seconds at 10Hz processing
        self.disconnection_threshold = 1e-6  # Very low amplitude threshold
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor(
            sample_rate=sample_rate,
            frame_size=frame_size,
            hop_size=hop_size
        )
        
        # GPU context (will be initialized later if available)
        self.gpu_ctx = None
        
        # Current audio features
        self.current_features = AudioFeatures()
    
    def audio_callback(self, indata, frames, time, status):
        """
        Callback function for audio capture.
        
        Args:
            indata: Input audio data as numpy array
            frames: Number of frames
            time: Timestamp information
            status: Status information
        """
        if status:
            logger.warning(f"Audio callback status: {status}")
            
            # Check for input overflow (buffer overrun) or input underflow (data lost)
            if status.input_overflow:
                logger.warning("Input overflow detected - audio data was lost")
                
            # Check if device disconnection is indicated by status
            if "device unavailable" in str(status).lower() or "interrupted" in str(status).lower():
                logger.error(f"Device disconnection detected: {status}")
                self.handle_device_disconnection("Device disconnected or unavailable")
                return
        
        # Get mono audio (average channels if stereo)
        if indata.shape[1] > 1:
            data = np.mean(indata, axis=1)
        else:
            data = indata[:, 0]
        
        # Check for prolonged silence which might indicate disconnection
        rms = np.sqrt(np.mean(np.square(data)))
        if rms < self.disconnection_threshold:
            self.silent_frame_count += 1
            if self.silent_frame_count >= self.max_silent_frames:
                logger.warning(f"Prolonged silence detected ({self.silent_frame_count} frames), possible device disconnection")
                self.handle_device_disconnection("Prolonged silence detected, device might be disconnected")
                return
        else:
            # Reset counter if we get audio signal
            self.silent_frame_count = 0
        
        # Add to buffer
        self.buffer.append(data.copy())
        
        # Process audio if we have enough data
        if len(self.buffer) >= self.buffer.maxlen:
            self.process_audio()
    
    def handle_device_disconnection(self, reason):
        """Handle device disconnection event."""
        if not self.running:
            return
            
        logger.error(f"Audio device disconnection: {reason}")
        
        # Emit device disconnected signal
        self.device_disconnected.emit(reason)
        
        # Clean up stream
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                logger.error(f"Error closing stream after disconnection: {e}")
            finally:
                self.stream = None
        
        # Update state
        self.running = False
        self.state_changed.emit("disconnected")
        
        # Fall back to test mode to keep visuals running
        if not hasattr(self, '_test_mode') or not self._test_mode:
            logger.info("Falling back to test mode after device disconnection")
            self.run_test_mode()
    
    def process_audio(self):
        """Process buffered audio and extract features."""
        try:
            # Combine buffer into a single array
            # (taking the most recent frame_size samples)
            buffer_array = np.concatenate(list(self.buffer))
            if len(buffer_array) > self.frame_size:
                buffer_array = buffer_array[-self.frame_size:]
            
            # Extract features
            features = self.feature_extractor.process(buffer_array, self.gpu_ctx)
            
            # Store current features
            self.current_features = features
            
            # Emit signal with new features
            self.features_ready.emit(features)
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            self.error_occurred.emit(f"Audio processing error: {e}")
    
    def run(self):
        """Main thread execution loop."""
        # If test mode is requested, run that instead
        if hasattr(self, '_test_mode') and self._test_mode:
            self.run_test_mode()
            return
            
        try:
            # Reset disconnection detection
            self.silent_frame_count = 0
            
            # Query device capabilities – fall back gracefully if only mono is available
            dev_info = sd.query_devices(self.device, "input")
            max_ch = max(1, dev_info["max_input_channels"])
            ch = min(2, max_ch)  # prefer stereo, accept mono

            if dev_info["max_input_channels"] == 0:
                logger.error("Selected device has no input channels; reverting to test mode")
                raise ValueError("Selected device has no input channels")

            # Create an audio stream
            self.stream = sd.InputStream(
                device=self.device,
                channels=ch,
                samplerate=self.sample_rate,
                blocksize=self.hop_size,
                callback=self.audio_callback
            )
            
            # Start the stream
            self.stream.start()
            self.running = True
            self.state_changed.emit("started")
            
            logger.info(f"Audio engine started on device: {self.device or 'default'}")
            
            # Keep thread running until stopped
            while self.running and not self.isInterruptionRequested():
                self.msleep(100)  # Sleep to avoid high CPU usage
            
            # Clean up
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            
            self.running = False
            self.state_changed.emit("stopped")
            logger.info("Audio engine stopped")
            
        except Exception as e:
            logger.error(f"Error in audio engine: {e}")
            self.error_occurred.emit(f"Audio engine error: {e}")
            self.running = False
            self.state_changed.emit("error")
            
            # Fall back to test mode
            self.run_test_mode()
    
    def run_test_mode(self):
        """Generate simulated audio features for testing without actual audio input."""
        logger.info("Running in test mode with simulated audio features")
        
        # Set test mode flag and running state
        self._test_mode = True
        self.running = True
        self.state_changed.emit("test")
        
        # Create test features
        features = AudioFeatures()
        
        # Start time for time-based effects
        start_time = time.time()
        
        # Simulated update rate (fps)
        update_rate = 60  # Hz
        update_interval = 1.0 / update_rate
        
        # Run until interrupted
        while not self.isInterruptionRequested() and self.running:
            # Calculate elapsed time
            elapsed = time.time() - start_time
            
            # Simulate onset every 0.5 seconds
            onset_period = 0.5 
            onset_phase = (elapsed % onset_period) / onset_period
            onset = 1.0 - onset_phase if onset_phase < 0.1 else 0.0
            
            # Simulate beat phase (4/4 time signature) at 120 BPM
            beat_period = 60.0 / 120.0  # seconds per beat at 120 BPM
            beat_phase = (elapsed % beat_period) / beat_period
            
            # Simulate energy (volume)
            base_energy = 0.3 + 0.2 * math.sin(elapsed * 0.2)
            energy = base_energy + 0.5 * onset
            
            # Simulate spectral centroid
            centroid = 0.5 + 0.3 * math.sin(elapsed * 0.1)
            
            # Update features
            features.onset = onset
            features.beat_phase = beat_phase
            features.energy = min(1.0, energy)  # Clamp to 1.0
            features.centroid = centroid
            
            # Every ~8 seconds, add a random "drum hit" with higher energy
            if random.random() < 0.02:  # 2% chance per 100ms check
                features.onset = 1.0
                features.energy = 1.0
            
            # Emit updated features
            self.features_ready.emit(features)
            
            # Wait for next update
            self.msleep(100)  # 10 Hz update rate
        
        self.running = False
        self.state_changed.emit("stopped")
        logger.info("Test mode stopped")
    
    def stop(self):
        """Stop the audio engine."""
        self.running = False
        self.requestInterruption()
        self.wait()  # Wait for thread to finish
        # Reset test mode flag so subsequent starts use the selected device
        if hasattr(self, "_test_mode"):
            self._test_mode = False
        
    def start_test_mode(self):
        """Start the audio engine in test mode with simulated features."""
        # Make sure we're not already running
        if self.running:
            self.stop()

        # Start in test mode
        self._test_mode = True
        self.running = False  # Will be set to True in run_test_mode
        self.start()
        # The thread will call run() which will now call run_test_mode() directly
        
    @property
    def is_running(self):
        """Return True if the audio engine is running."""
        return self.running
        
    def get_available_devices(self):
        """Get list of available audio input devices."""
        try:
            devices = sd.query_devices()
            input_devices = [d for d in devices if d['max_input_channels'] > 0]
            return input_devices
        except Exception as e:
            logger.error(f"Error getting audio devices: {e}")
            self.error_occurred.emit(f"Failed to get audio devices: {e}")
            return []

    def select_device(self, device):
        """Select a new audio input device and restart the stream if needed."""
        was_running = self.running

        if was_running:
            try:
                self.stop()
            except Exception as e:
                logger.error(f"Failed to stop stream when switching device: {e}")
                self.error_occurred.emit(f"Failed to stop stream: {e}")
                was_running = False

        self.device = device

        if was_running:
            try:
                self.start()
            except Exception as e:
                logger.error(f"Failed to restart audio engine: {e}")
                self.error_occurred.emit(f"Failed to restart audio engine: {e}")
