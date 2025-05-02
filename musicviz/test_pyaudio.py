#!/usr/bin/env python3
"""
Simple test script to verify PyAudio installation.
Lists all available audio devices.
"""

import pyaudio

def list_audio_devices():
    """List all available audio input and output devices."""
    p = pyaudio.PyAudio()
    
    print("Available Audio Devices:")
    print("-----------------------")
    
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        device_name = device_info['name']
        input_channels = device_info['maxInputChannels']
        output_channels = device_info['maxOutputChannels']
        
        device_type = []
        if input_channels > 0:
            device_type.append("Input")
        if output_channels > 0:
            device_type.append("Output")
        
        device_type_str = " & ".join(device_type)
        
        print(f"[{i}] {device_name} ({device_type_str})")
        print(f"    Channels: {input_channels} in, {output_channels} out")
        print(f"    Default Sample Rate: {device_info['defaultSampleRate']} Hz")
        print()
    
    # Get default devices
    print("Default Input Device:")
    default_input = p.get_default_input_device_info()
    print(f"  [{default_input['index']}] {default_input['name']}")
    
    print("\nDefault Output Device:")
    default_output = p.get_default_output_device_info()
    print(f"  [{default_output['index']}] {default_output['name']}")
    
    # Clean up
    p.terminate()

if __name__ == "__main__":
    list_audio_devices() 