#!/usr/bin/env python3
"""
GPU context manager for audio processing.

This module handles GPU detection, initialization, and provides fallback
mechanisms for systems without GPU acceleration.
"""

import logging
import platform
import numpy as np
from typing import Optional, Dict, Tuple, Any

# Set up logging
logger = logging.getLogger(__name__)

# Define GPU types
GPU_TYPE_NONE = "none"
GPU_TYPE_CUDA = "cuda"
GPU_TYPE_MPS = "mps"  # Apple Metal Performance Shaders

# Try to import GPU libraries, but don't fail if not available
try:
    import torch
    import torchaudio
    HAS_TORCH = True
except ImportError:
    logger.warning("PyTorch not available, falling back to CPU processing")
    HAS_TORCH = False

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    logger.warning("CuPy not available, CUDA acceleration not available")
    HAS_CUPY = False


class GPUContext:
    """
    GPU context manager for audio processing.
    
    This class provides a unified interface for GPU-accelerated audio
    processing, with automatic fallback to CPU when GPU is not available.
    """
    
    def __init__(self, force_cpu: bool = False):
        """
        Initialize GPU context.
        
        Args:
            force_cpu: Force CPU processing even if GPU is available
        """
        self.force_cpu = force_cpu
        self.gpu_type = GPU_TYPE_NONE
        self.device = None
        self.initialized = False
        self.initialization_error = None
        
        # Performance metrics
        self.processing_times = []
        
        # Try to initialize GPU context
        if not force_cpu:
            self._initialize_gpu()
    
    def _initialize_gpu(self):
        """Initialize GPU context based on available hardware."""
        try:
            # Check if PyTorch is available
            if not HAS_TORCH:
                logger.warning("PyTorch not available, using CPU fallback")
                self.initialization_error = "PyTorch not available"
                return
                
            # Check for CUDA
            if torch.cuda.is_available():
                logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
                self.gpu_type = GPU_TYPE_CUDA
                self.device = torch.device("cuda:0")
                
                # Log CUDA device info
                cuda_props = {
                    "Name": torch.cuda.get_device_name(0),
                    "Capability": f"{torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}",
                    "Memory": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
                }
                logger.info(f"CUDA device info: {cuda_props}")
                
            # Check for MPS (Apple Silicon)
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                logger.info("MPS (Metal Performance Shaders) available")
                self.gpu_type = GPU_TYPE_MPS
                self.device = torch.device("mps")
                
                # Log basic MPS info
                mps_props = {
                    "Device": "Apple Silicon",
                    "Platform": platform.platform()
                }
                logger.info(f"MPS device info: {mps_props}")
                
            else:
                logger.warning("No GPU acceleration available, using CPU")
                self.device = torch.device("cpu")
            
            # Test GPU with a simple operation
            try:
                x = torch.randn(1000, 1000, device=self.device)
                y = x @ x
                del x, y  # Clean up
                logger.info(f"Successfully tested {self.gpu_type or 'cpu'} device")
            except Exception as e:
                logger.error(f"GPU test failed: {e}")
                self.gpu_type = GPU_TYPE_NONE
                self.device = torch.device("cpu")
                self.initialization_error = f"GPU test failed: {e}"
            
            self.initialized = True
            
        except Exception as e:
            logger.error(f"GPU initialization error: {e}")
            self.gpu_type = GPU_TYPE_NONE
            self.initialization_error = f"GPU initialization error: {e}"
    
    def stft(self, audio: np.ndarray, n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
        """
        Compute Short-Time Fourier Transform.
        
        Args:
            audio: Audio signal as numpy array
            n_fft: FFT size
            hop_length: Hop length
            
        Returns:
            np.ndarray: STFT power spectrum
        """
        try:
            # If PyTorch is available, use torchaudio for GPU acceleration
            if HAS_TORCH and self.gpu_type != GPU_TYPE_NONE:
                # Convert to PyTorch tensor
                audio_tensor = torch.tensor(audio, dtype=torch.float32, device=self.device)
                
                # Compute STFT
                stft = torchaudio.transforms.Spectrogram(
                    n_fft=n_fft,
                    hop_length=hop_length,
                    power=2.0,
                    center=True,
                    pad_mode="reflect"
                ).to(self.device)
                
                # Apply STFT transform
                spec = stft(audio_tensor)
                
                # Convert back to numpy
                return spec.cpu().numpy()
            else:
                # Fall back to CPU implementation
                return self._stft_cpu(audio, n_fft, hop_length)
                
        except Exception as e:
            logger.error(f"STFT error (falling back to CPU): {e}")
            return self._stft_cpu(audio, n_fft, hop_length)
    
    def _stft_cpu(self, audio: np.ndarray, n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
        """
        CPU fallback for STFT computation.
        
        Args:
            audio: Audio signal as numpy array
            n_fft: FFT size
            hop_length: Hop length
            
        Returns:
            np.ndarray: STFT power spectrum
        """
        # Import librosa only when needed (it's slow to import)
        try:
            import librosa
            return np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))**2
        except ImportError:
            # If librosa is not available, use scipy
            from scipy import signal
            return np.abs(signal.stft(audio, nperseg=n_fft, noverlap=n_fft-hop_length)[2])**2
    
    def mel_filterbank(self, spec: np.ndarray, sr: int = 44100, n_mels: int = 128) -> np.ndarray:
        """
        Apply mel filterbank to spectrogram.
        
        Args:
            spec: Spectrogram as numpy array
            sr: Sample rate
            n_mels: Number of mel bands
            
        Returns:
            np.ndarray: Mel spectrogram
        """
        try:
            # If PyTorch is available, use torchaudio for GPU acceleration
            if HAS_TORCH and self.gpu_type != GPU_TYPE_NONE:
                # Convert to PyTorch tensor
                spec_tensor = torch.tensor(spec, dtype=torch.float32, device=self.device)
                
                # Apply mel filterbank
                mel_transform = torchaudio.transforms.MelScale(
                    n_mels=n_mels,
                    sample_rate=sr,
                    f_min=0.0,
                    f_max=sr/2,
                    n_stft=spec.shape[0]
                ).to(self.device)
                
                mel_spec = mel_transform(spec_tensor)
                
                # Convert back to numpy
                return mel_spec.cpu().numpy()
            else:
                # Fall back to CPU implementation
                return self._mel_filterbank_cpu(spec, sr, n_mels)
                
        except Exception as e:
            logger.error(f"Mel filterbank error (falling back to CPU): {e}")
            return self._mel_filterbank_cpu(spec, sr, n_mels)
    
    def _mel_filterbank_cpu(self, spec: np.ndarray, sr: int = 44100, n_mels: int = 128) -> np.ndarray:
        """
        CPU fallback for mel filterbank.
        
        Args:
            spec: Spectrogram as numpy array
            sr: Sample rate
            n_mels: Number of mel bands
            
        Returns:
            np.ndarray: Mel spectrogram
        """
        try:
            import librosa
            # Create mel filterbank
            mel_basis = librosa.filters.mel(sr=sr, n_fft=(spec.shape[0]-1)*2, n_mels=n_mels)
            
            # Apply filterbank
            return mel_basis @ spec
        except ImportError:
            logger.warning("Librosa not available for mel filterbank, using linear approximation")
            # Simple approximation if librosa not available
            from scipy.signal import resample
            return resample(spec, n_mels, axis=0)
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """
        Get information about the GPU.
        
        Returns:
            dict: GPU information
        """
        info = {
            "type": self.gpu_type,
            "available": self.gpu_type != GPU_TYPE_NONE,
            "initialized": self.initialized,
            "error": self.initialization_error
        }
        
        # Add device-specific information
        if self.gpu_type == GPU_TYPE_CUDA and torch.cuda.is_available():
            info.update({
                "name": torch.cuda.get_device_name(0),
                "memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**2:.1f} MB",
                "memory_reserved": f"{torch.cuda.memory_reserved(0) / 1024**2:.1f} MB",
                "capability": f"{torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}"
            })
        elif self.gpu_type == GPU_TYPE_MPS:
            info.update({
                "name": "Apple Silicon",
                "platform": platform.platform()
            })
        
        return info
    
    def is_gpu_available(self) -> bool:
        """
        Check if GPU acceleration is available.
        
        Returns:
            bool: True if GPU acceleration is available
        """
        return self.gpu_type != GPU_TYPE_NONE
    
    def get_performance_stats(self) -> Dict[str, float]:
        """
        Get performance statistics.
        
        Returns:
            dict: Performance statistics
        """
        if not self.processing_times:
            return {"avg_time": 0.0, "min_time": 0.0, "max_time": 0.0}
            
        return {
            "avg_time": sum(self.processing_times) / len(self.processing_times),
            "min_time": min(self.processing_times),
            "max_time": max(self.processing_times)
        }
    
    def cleanup(self):
        """Clean up GPU resources."""
        if HAS_TORCH and self.gpu_type != GPU_TYPE_NONE:
            # Clear PyTorch cache
            if self.gpu_type == GPU_TYPE_CUDA:
                torch.cuda.empty_cache()


# Create a singleton instance
_gpu_context: Optional[GPUContext] = None


def get_gpu_context(force_new: bool = False, force_cpu: bool = False) -> GPUContext:
    """
    Get the GPU context singleton.
    
    Args:
        force_new: Force creation of a new context
        force_cpu: Force CPU processing
        
    Returns:
        GPUContext: GPU context instance
    """
    global _gpu_context
    
    if _gpu_context is None or force_new:
        _gpu_context = GPUContext(force_cpu=force_cpu)
        
    return _gpu_context


# Simple test code
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Create a test signal
    sample_rate = 44100
    duration = 1.0  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # Create a simple sine wave with some noise
    frequency = 440  # A4 note
    signal = 0.5 * np.sin(2 * np.pi * frequency * t) + 0.1 * np.random.randn(len(t))

    # Try to initialize GPU context
    ctx = get_gpu_context()

    # Compute STFT
    spec = ctx.stft(signal)

    # Print information about the result
    print(f"STFT shape: {spec.shape}")
    print(f"STFT dtype: {spec.dtype}")

    # Check if we're using GPU
    print(f"GPU available: {ctx.is_gpu_available()}")
    if ctx.is_gpu_available():
        print(f"GPU device: {ctx.device}")
        if ctx.gpu_type == GPU_TYPE_CUDA:
            print("Using CUDA for GPU acceleration")
        elif ctx.gpu_type == GPU_TYPE_MPS:
            print("Using MPS for GPU acceleration")
    else:
        print("Using CPU for processing")
