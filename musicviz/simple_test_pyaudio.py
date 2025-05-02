#!/usr/bin/env python3
"""
Simple test script to verify PyAudio installation.
"""

try:
    import pyaudio
    print(f"PyAudio was imported successfully.")
    print(f"PyAudio version: {pyaudio.__version__}")
    
    # Try to initialize PyAudio
    p = pyaudio.PyAudio()
    print(f"PyAudio initialized successfully.")
    print(f"Number of devices: {p.get_device_count()}")
    p.terminate()
    
except ImportError as e:
    print(f"Error importing PyAudio: {e}")
except Exception as e:
    print(f"Error initializing PyAudio: {e}") 