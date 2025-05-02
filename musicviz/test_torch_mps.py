#!/usr/bin/env python3
"""
Test script to verify PyTorch with MPS (Metal Performance Shaders) on Apple Silicon.
"""

import torch
import torchaudio
import numpy as np

def test_torch_backends():
    """Test available PyTorch backends."""
    print(f"PyTorch version: {torch.__version__}")
    print(f"TorchAudio version: {torchaudio.__version__}")
    
    # Check CUDA availability (not expected on Apple Silicon)
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Check MPS (Metal Performance Shaders) availability
    print(f"MPS available: {torch.backends.mps.is_available()}")
    
    # Check which device to use
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    
    # Create a small tensor and move it to the device
    try:
        x = torch.randn(3, 3)
        x = x.to(device)
        print(f"Created tensor on {device}:")
        print(x)
        
        # Simple matrix multiplication
        y = torch.matmul(x, x)
        print(f"Matrix multiplication result:")
        print(y)
        
        print("PyTorch is working correctly with the selected device!")
    except Exception as e:
        print(f"Error running PyTorch on {device}: {e}")
    
    # Test TorchAudio functionality
    try:
        # Create a simple sine wave
        sample_rate = 16000
        waveform = torch.sin(torch.arange(0, 1000).float() * 0.1)
        waveform = waveform.unsqueeze(0)  # Add batch dimension
        
        # Apply spectrogram transform
        spec = torchaudio.transforms.Spectrogram()(waveform)
        print(f"Successfully created spectrogram with shape: {spec.shape}")
        
        # Try to move it to GPU device if available
        try:
            spec = spec.to(device)
            print(f"Successfully moved spectrogram to {device}")
        except Exception as e:
            print(f"Could not move spectrogram to {device}: {e}")
        
        print("TorchAudio is working correctly!")
    except Exception as e:
        print(f"Error testing TorchAudio: {e}")

if __name__ == "__main__":
    test_torch_backends() 