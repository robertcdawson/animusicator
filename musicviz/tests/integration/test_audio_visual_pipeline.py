"""
Integration tests for the audio-visual pipeline.

These tests verify that audio processing, feature extraction, and visualization
components work together correctly.
"""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

# Try to import the necessary components
try:
    from musicviz.audio.engine import AudioEngine
    from musicviz.gui.visual_widget import VisualWidget
    COMPONENTS_AVAILABLE = True
except ImportError:
    COMPONENTS_AVAILABLE = False
    pytestmark = pytest.mark.skip("Required components not available")


@pytest.mark.integration
@pytest.mark.skipif(not COMPONENTS_AVAILABLE, reason="Required components not available")
class TestAudioVisualPipeline:
    """Test the integration between audio and visual components."""
    
    @pytest.fixture
    def mock_sounddevice(self):
        """Create a mock of the sounddevice module."""
        with patch("musicviz.audio.engine.sounddevice") as mock_sd:
            # Mock device list
            mock_sd.query_devices.return_value = [
                {"name": "Test Device 1", "max_input_channels": 2},
                {"name": "BlackHole 2ch", "max_input_channels": 2}
            ]
            
            # Mock stream
            mock_stream = MagicMock()
            mock_sd.InputStream.return_value = mock_stream
            
            yield mock_sd
    
    @pytest.fixture
    def mock_audio_callback(self, mock_audio_data):
        """Prepare a mock audio callback for testing."""
        _, audio_data = mock_audio_data
        
        def simulate_audio_callback(callback_fn):
            """Simulate audio data arriving through the callback."""
            # Convert audio data to the expected format (2D array with channels)
            frames = len(audio_data)
            indata = audio_data.reshape(-1, 1)
            
            # Call the callback with the synthetic audio data
            callback_fn(indata, frames, None, None)
        
        return simulate_audio_callback
    
    @pytest.fixture
    def mock_gl(self):
        """Mock PyOpenGL functions for testing."""
        with patch("musicviz.gui.visual_widget.GL") as mock_gl:
            with patch("musicviz.gui.visual_widget.GLU") as mock_glu:
                yield mock_gl
    
    @pytest.fixture
    def mock_shader_loader(self):
        """Mock shader loading for testing."""
        with patch("musicviz.visual.shaders.load_shader_program") as mock_loader:
            mock_loader.return_value = 123  # Mock shader program ID
            yield mock_loader
    
    def test_engine_to_visual_widget_connection(self, qapp, mock_sounddevice, 
                                              mock_audio_callback, mock_gl, 
                                              mock_shader_loader):
        """Test that audio engine signals connect properly to visual widget."""
        # Create the components
        audio_engine = AudioEngine()
        visual_widget = VisualWidget()
        
        # Initialize visual widget's OpenGL (normally done by Qt)
        visual_widget.initializeGL()
        
        # Connect the signal
        audio_engine.features_ready.connect(visual_widget.update_features)
        
        # Configure the audio engine
        audio_engine.select_device("BlackHole 2ch")
        
        # Start the audio engine
        audio_engine.start()
        
        # Get the callback function that was registered
        callback_fn = mock_sounddevice.InputStream.call_args[1]["callback"]
        
        # Create a spy on the update_features method
        original_update_features = visual_widget.update_features
        update_features_spy = MagicMock(wraps=original_update_features)
        visual_widget.update_features = update_features_spy
        
        # Simulate audio data arriving
        mock_audio_callback(callback_fn)
        
        # Verify that update_features was called with non-None features
        update_features_spy.assert_called_once()
        args, _ = update_features_spy.call_args
        assert args[0] is not None  # Features should be non-None
        
        # Stop the audio engine
        audio_engine.stop()
    
    def test_feature_values_affect_rendering(self, qapp, mock_sounddevice, 
                                           mock_audio_callback, mock_gl, 
                                           mock_shader_loader):
        """Test that different feature values result in different rendering."""
        # Create the components
        audio_engine = AudioEngine()
        visual_widget = VisualWidget()
        
        # Initialize visual widget's OpenGL (normally done by Qt)
        visual_widget.initializeGL()
        
        # Connect the signal
        audio_engine.features_ready.connect(visual_widget.update_features)
        
        # Configure the audio engine
        audio_engine.select_device("BlackHole 2ch")
        
        # Start the audio engine
        audio_engine.start()
        
        # Get the callback function that was registered
        callback_fn = mock_sounddevice.InputStream.call_args[1]["callback"]
        
        # Spy on the glUniform calls to check if features affect uniforms
        mock_gl.glUniform1f = MagicMock(wraps=mock_gl.glUniform1f)
        
        # Simulate audio data arriving
        mock_audio_callback(callback_fn)
        
        # Trigger a paint (normally done by Qt)
        visual_widget.paintGL()
        
        # Verify that uniform values were set based on features
        assert mock_gl.glUniform1f.call_count > 0
        
        # Store the uniform calls
        first_uniform_calls = list(mock_gl.glUniform1f.call_args_list)
        
        # Reset the mock to track new calls
        mock_gl.glUniform1f.reset_mock()
        
        # Create different audio features by modifying the feature extractor's output
        with patch('musicviz.audio.feature_extractor.FeatureExtractor.process') as mock_process:
            # Create significantly different features
            from dataclasses import dataclass
            
            @dataclass
            class DifferentFeatures:
                onset: float = 1.0  # Maximum onset
                beat_phase: float = 0.0
                energy: float = 1.0  # Maximum energy
                centroid: float = 1.0
                contrast: np.ndarray = np.array([1.0, 1.0, 1.0])
                chroma: np.ndarray = np.array([1.0] * 12)
                mfcc: np.ndarray = np.array([1.0, 1.0, 1.0, 1.0])
                embedding: np.ndarray = np.array([1.0] * 128)
            
            mock_process.return_value = DifferentFeatures()
            
            # Simulate audio data arriving again
            mock_audio_callback(callback_fn)
            
            # Trigger another paint
            visual_widget.paintGL()
            
            # Verify that uniform values were set again
            assert mock_gl.glUniform1f.call_count > 0
            
            # Store the new uniform calls
            second_uniform_calls = list(mock_gl.glUniform1f.call_args_list)
            
            # Verify that the uniform values are different
            # At least one call should be different
            assert first_uniform_calls != second_uniform_calls, "Uniform values should change with different features"
        
        # Stop the audio engine
        audio_engine.stop()
    
    def test_device_switching(self, qapp, mock_sounddevice, mock_gl, mock_shader_loader):
        """Test that switching audio devices works correctly."""
        # Create the components
        audio_engine = AudioEngine()
        visual_widget = VisualWidget()
        
        # Initialize visual widget's OpenGL (normally done by Qt)
        visual_widget.initializeGL()
        
        # Connect the signal
        audio_engine.features_ready.connect(visual_widget.update_features)
        
        # Configure and start with one device
        audio_engine.select_device("Test Device 1")
        audio_engine.start()
        
        # Verify stream was created with the right device
        first_device = mock_sounddevice.InputStream.call_args[1]["device"]
        assert "Test Device 1" in str(first_device)
        
        # Reset the mock to track new calls
        mock_sounddevice.reset_mock()
        
        # Switch to a different device
        audio_engine.stop()
        audio_engine.select_device("BlackHole 2ch")
        audio_engine.start()
        
        # Verify stream was recreated with the new device
        second_device = mock_sounddevice.InputStream.call_args[1]["device"]
        assert "BlackHole 2ch" in str(second_device)
        
        # Stop the audio engine
        audio_engine.stop()
    
    def test_end_to_end_with_gpu_fallback(self, qapp, mock_sounddevice, 
                                        mock_audio_callback, mock_gl, 
                                        mock_shader_loader):
        """Test the entire pipeline with GPU unavailable (fallback to CPU)."""
        # Mock GPU context to be unavailable
        with patch("musicviz.audio.feature_extractor.GPUContext") as mock_gpu:
            mock_instance = MagicMock()
            mock_instance.is_available = False
            mock_gpu.return_value = mock_instance
            
            # Create the components
            audio_engine = AudioEngine()
            visual_widget = VisualWidget()
            
            # Initialize visual widget's OpenGL (normally done by Qt)
            visual_widget.initializeGL()
            
            # Connect the signal
            audio_engine.features_ready.connect(visual_widget.update_features)
            
            # Configure the audio engine
            audio_engine.select_device("BlackHole 2ch")
            
            # Start the audio engine
            audio_engine.start()
            
            # Get the callback function that was registered
            callback_fn = mock_sounddevice.InputStream.call_args[1]["callback"]
            
            # Create a spy on the update_features method
            original_update_features = visual_widget.update_features
            update_features_spy = MagicMock(wraps=original_update_features)
            visual_widget.update_features = update_features_spy
            
            # Simulate audio data arriving
            mock_audio_callback(callback_fn)
            
            # Verify that update_features was still called despite GPU being unavailable
            update_features_spy.assert_called_once()
            
            # Verify that the GPU methods were not used
            assert mock_instance.stft.call_count == 0
            
            # Visual widget should still render
            visual_widget.paintGL()
            
            # Basic GL calls should still happen
            mock_gl.glClear.assert_called()
            
            # Stop the audio engine
            audio_engine.stop() 