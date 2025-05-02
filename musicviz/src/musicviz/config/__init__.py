"""
Configuration module for the Animusicator application.
This module provides access to application configuration settings.
"""

import os
import sys

# Re-export config_loader functionality
from ..utils.config_loader import ConfigLoader, get_config

# Define config paths
config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))), 'config')
default_config_file = os.path.join(config_dir, 'default_config.yaml')
user_config_file = os.path.join(config_dir, 'user_config.yaml') 