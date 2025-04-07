#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration loader utility for SignSync
"""

import os
import logging
from typing import Dict, Any, Optional
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing configuration values
        
    Raises:
        FileNotFoundError: If configuration file doesn't exist
        yaml.YAMLError: If configuration file is not valid YAML
    """
    # Set default configuration
    default_config = {
        "app": {
            "name": "SignSync",
            "version": "0.1.0",
            "data_dir": os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"),
            "models_dir": os.path.join(os.path.dirname(os.path.dirname(__file__)), "models"),
        },
        "recognition": {
            "confidence_threshold": 0.85,
            "min_detection_confidence": 0.7,
            "min_tracking_confidence": 0.5,
            "sliding_window_size": 30,
            "model_path": "models/hand_gesture_classifier.tflite",
        },
        "translation": {
            "context_window_size": 5,
            "autocorrect_enabled": True,
            "tts_voice": "en-US-Neural2-F",
            "tts_speed": 1.0,
        },
        "video": {
            "width": 1280,
            "height": 720,
            "fps": 30,
            "camera_index": 0,
        },
        "gui": {
            "theme": "system",
            "high_contrast": False,
            "font_size": 12,
            "overlay_opacity": 0.8,
        },
        "performance": {
            "threads": max(2, os.cpu_count() - 1) if os.cpu_count() else 2,
            "force_cpu_only": False,
            "memory_limit_mb": 500,
        }
    }
    
    try:
        # Check if config file exists
        if not os.path.exists(config_path):
            # If default config path doesn't exist, create it with default values
            if config_path == "config/default.yaml":
                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                with open(config_path, "w") as f:
                    yaml.dump(default_config, f, default_flow_style=False)
                logging.info(f"Created default configuration file at {config_path}")
            else:
                raise FileNotFoundError(f"Configuration file {config_path} not found")
        
        # Load config from file
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            
        # Merge with default config (for any missing values)
        merged_config = default_config.copy()
        _deep_update(merged_config, config)
        
        return merged_config
    
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file: {e}")
        return default_config


def _deep_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
    """
    Recursively update a dictionary with another dictionary
    
    Args:
        base_dict: Base dictionary to update
        update_dict: Dictionary with values to update base_dict with
    """
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            _deep_update(base_dict[key], value)
        else:
            base_dict[key] = value 