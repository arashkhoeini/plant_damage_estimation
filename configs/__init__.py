"""
Configuration management for plant damage estimation.

This module handles loading and parsing of configuration files,
including YAML configuration files and command-line argument processing.
"""

from .init_configs import init_config

__all__ = ["init_config"]
