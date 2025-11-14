"""
Utilities Package
Common utility functions and configurations.
"""

from src.utils.config import (
    Config,
    get_config,
    set_config,
    load_config,
    create_default_config
)
from src.utils.logger import setup_logger, get_logger

__all__ = [
    # Config
    'Config',
    'get_config',
    'set_config',
    'load_config',
    'create_default_config',
    
    # Logger
    'setup_logger',
    'get_logger',
]

__version__ = '1.0.0'