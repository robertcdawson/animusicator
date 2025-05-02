#!/usr/bin/env python3
"""
Simple test script to verify PyYAML functionality.
"""

import yaml
import os
import sys

def test_yaml():
    """Test YAML loading and dumping."""
    print(f"PyYAML version: {yaml.__version__}")
    
    # Create a sample configuration dictionary
    config = {
        'app': {
            'name': 'Animusicator',
            'version': '0.1.0',
        },
        'audio': {
            'device': 'BlackHole 2ch',
            'sample_rate': 44100,
            'frame_size': 2048,
            'hop_size': 512,
        },
        'visual': {
            'fullscreen': False,
            'fps_limit': 60,
            'shader': 'visualizer.frag',
        },
        'features': [
            'onset',
            'beat_phase',
            'energy',
            'spectral_centroid',
        ]
    }
    
    # Create a test config file
    config_path = 'test_config.yaml'
    
    # Dump the config to a YAML file
    print(f"Writing YAML configuration to {config_path}...")
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Read the config back
    print(f"Reading YAML configuration from {config_path}...")
    with open(config_path, 'r') as f:
        loaded_config = yaml.safe_load(f)
    
    # Verify the loaded config matches the original
    print(f"Verifying configuration...")
    assert loaded_config == config, "Loaded configuration doesn't match original"
    
    # Print the loaded config
    print(f"\nLoaded configuration:")
    print(yaml.dump(loaded_config, default_flow_style=False))
    
    # Clean up
    os.remove(config_path)
    print(f"Test completed successfully! PyYAML is working properly.")

if __name__ == "__main__":
    try:
        test_yaml()
    except Exception as e:
        print(f"Error testing PyYAML: {e}")
        sys.exit(1) 