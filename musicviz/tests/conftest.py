"""
Common pytest fixtures for Animusicator tests.
"""
import os
import sys
import pytest
from PyQt5.QtWidgets import QApplication

# Add src directory to path to make imports work in tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


@pytest.fixture(scope="session")
def qapp():
    """
    Create a QApplication instance for GUI tests.
    
    This fixture ensures that only one QApplication is created
    for the entire test session, which is required for Qt.
    """
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def mock_audio_data():
    """
    Generate mock audio data for testing audio processing.
    
    Returns:
        tuple: (sample_rate, audio_data) where audio_data is a numpy array
               containing a synthetic sine wave
    """
    import numpy as np
    
    # Create a simple sine wave for testing
    sample_rate = 44100  # Hz
    duration = 1.0       # second
    frequency = 440.0    # Hz (A4 note)
    
    # Generate time array
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Generate sine wave
    audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)
    
    return sample_rate, audio_data


@pytest.fixture
def mock_audio_features():
    """
    Create mock audio features for testing visualization components.
    
    Returns:
        dict: Dictionary containing mock audio features
    """
    import numpy as np
    from dataclasses import dataclass
    
    try:
        # Try to import the actual AudioFeatures class if available
        from musicviz.audio.features import AudioFeatures
        
        # Create mock features
        features = AudioFeatures(
            onset=0.75,
            beat_phase=0.5,
            energy=0.8,
            centroid=0.6,
            contrast=np.array([0.5, 0.6, 0.7]),
            chroma=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.8, 0.7, 0.6]),
            mfcc=np.array([0.1, 0.2, 0.3, 0.4]),
            embedding=np.array([0.1] * 128)
        )
        
    except (ImportError, ModuleNotFoundError):
        # If the actual class isn't available, create a basic dict
        @dataclass
        class MockAudioFeatures:
            onset: float
            beat_phase: float
            energy: float
            centroid: float
            contrast: np.ndarray
            chroma: np.ndarray
            mfcc: np.ndarray
            embedding: np.ndarray
        
        features = MockAudioFeatures(
            onset=0.75,
            beat_phase=0.5,
            energy=0.8,
            centroid=0.6,
            contrast=np.array([0.5, 0.6, 0.7]),
            chroma=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.8, 0.7, 0.6]),
            mfcc=np.array([0.1, 0.2, 0.3, 0.4]),
            embedding=np.array([0.1] * 128)
        )
    
    return features


@pytest.fixture
def temp_config_file(tmp_path):
    """
    Create a temporary configuration file for testing.
    
    Args:
        tmp_path: pytest built-in fixture providing a temporary directory
        
    Returns:
        Path: Path to the temporary configuration file
    """
    import yaml
    
    # Create a basic config for testing
    config = {
        "audio_device": "test_device",
        "fft_size": 1024,
        "hop_size": 256,
        "sample_rate": 44100,
        "visualization": {
            "fullscreen": False,
            "fps_limit": 60
        }
    }
    
    # Write config to temporary file
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    return config_path


@pytest.fixture
def mock_gpu_context():
    """
    Create a mock GPU context for testing GPU-dependent components.
    
    Returns:
        object: A simple object with GPU context method stubs
    """
    class MockGPUContext:
        def __init__(self):
            self.is_available = True
            self.device = "mock_gpu"
            
        def stft(self, audio_data):
            import numpy as np
            # Return mock STFT result
            return np.random.random((513, 10))
            
        def mel_filterbank(self, spectrogram):
            import numpy as np
            # Return mock mel spectrogram
            return np.random.random((128, 10))
            
        def embedding_model_encode(self, mel_spec):
            import numpy as np
            # Return mock embedding
            return np.random.random(128)
    
    return MockGPUContext() 