#!/usr/bin/env python3
"""
Configuration loader utility for Animusicator.

This module provides utilities for loading, saving, and managing configuration
settings for the application.
"""

import os
import logging
import yaml
from typing import Dict, Any, Optional, Union

# Setup logger
logger = logging.getLogger(__name__)

# Default config locations
DEFAULT_CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'config')
DEFAULT_CONFIG_FILE = os.path.join(DEFAULT_CONFIG_DIR, 'default_config.yaml')
USER_CONFIG_FILE = os.path.join(DEFAULT_CONFIG_DIR, 'user_config.yaml')


class ConfigLoader:
    """
    Configuration loader and manager.
    
    Loads configuration from YAML files and provides methods to access and modify settings.
    """
    
    def __init__(self, config_file: Optional[str] = None, defaults: Optional[Dict[str, Any]] = None):
        """
        Initialize the configuration loader.
        
        Args:
            config_file: Path to the configuration file (YAML)
            defaults: Default configuration values
        """
        self.config_file = config_file
        self.config = {}
        
        # Apply defaults if provided
        if defaults:
            self.config = defaults.copy()
        
        # Load configuration from file if provided
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    file_config = yaml.safe_load(f)
                
                if file_config:
                    # Merge with defaults if both are provided
                    if defaults:
                        self._deep_merge(self.config, file_config)
                    else:
                        self.config = file_config
            except yaml.YAMLError:
                # Fall back to defaults on parsing error
                pass
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
        Args:
            key: Configuration key (can use dot notation for nested values)
            default: Default value if key is not found
            
        Returns:
            The configuration value or default
        """
        parts = key.split('.')
        value = self.config
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
                
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key (can use dot notation for nested values)
            value: Value to set
        """
        parts = key.split('.')
        config = self.config
        
        # Navigate to the right level, creating dictionaries as needed
        for i, part in enumerate(parts[:-1]):
            if part not in config or not isinstance(config[part], dict):
                config[part] = {}
            config = config[part]
            
        # Set the value
        config[parts[-1]] = value
    
    def save(self, file_path: Optional[str] = None) -> bool:
        """
        Save configuration to a file.
        
        Args:
            file_path: Path to save to (defaults to original config_file)
            
        Returns:
            bool: True if save was successful, False otherwise
        """
        save_path = file_path or self.config_file
        
        if not save_path:
            return False
            
        try:
            with open(save_path, 'w') as f:
                yaml.dump(self.config, f)
            return True
        except Exception:
            return False
    
    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        Deep merge two dictionaries.
        
        Values from source will overwrite values in target.
        
        Args:
            target: Target dictionary (modified in-place)
            source: Source dictionary
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                self._deep_merge(target[key], value)
            else:
                # Overwrite or set value
                target[key] = value


# Singleton instance for global access
_config_instance = None

def get_config() -> ConfigLoader:
    """
    Get singleton ConfigLoader instance.
    
    Returns:
        Shared ConfigLoader instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigLoader()
    return _config_instance


# Test code
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Get config instance
    config = get_config()
    
    # Print current config
    print("Current configuration:")
    print(yaml.dump(config.config, default_flow_style=False))
    
    # Test getting a value
    device = config.get('audio.device', 'default')
    print(f"Audio device: {device}")
    
    # Test setting a value
    config.set('audio.device', 'BlackHole 2ch')
    print(f"Updated audio device: {config.get('audio.device')}") 