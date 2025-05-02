"""
Settings management for the Animusicator application.
Provides functions to load, save, and manage application settings.
"""

import os
import yaml
import logging
from pathlib import Path

# Get logger
logger = logging.getLogger(__name__)

# Re-export config functionality
from ..utils.config_loader import get_config, ConfigLoader

def load_settings(defaults=None):
    """
    Load application settings from config files.
    
    Args:
        defaults: Optional dictionary of default settings
        
    Returns:
        Dict containing the application settings
    """
    config = get_config()
    
    # We're basically wrapping the ConfigLoader for backward compatibility
    if defaults:
        for key, value in defaults.items():
            if config.get(key) is None:
                config.set(key, value)
    
    return config

def save_settings(settings, filename=None):
    """
    Save application settings to a file.
    
    Args:
        settings: Settings dict or ConfigLoader instance
        filename: Optional filename to save to
        
    Returns:
        bool: True if save was successful
    """
    if isinstance(settings, ConfigLoader):
        return settings.save(filename)
    
    # If it's a dict, use the global config
    config = get_config()
    
    # Update with provided settings
    for key, value in settings.items():
        config.set(key, value)
    
    return config.save(filename) 