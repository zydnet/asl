#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SignSync - Real-Time Sign Language Translation Suite
Main application entry point
"""

import sys
import logging
import argparse
import multiprocessing
from typing import Dict, Any, Optional

from PyQt6.QtWidgets import QApplication

# Import application modules
from app.gui.main_window import MainWindow
from utils.config_loader import load_config
from utils.logging_setup import setup_logging


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="SignSync - Real-Time Sign Language Translation Suite")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to configuration file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU-only mode (no GPU)")
    return parser.parse_args()


def initialize_app(config: Dict[str, Any]) -> None:
    """Initialize application components based on configuration."""
    # Set up GPU/CPU usage
    if config.get("force_cpu_only", False):
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        logging.info("Running in CPU-only mode")
    
    # Set process priority for real-time performance
    try:
        import psutil
        process = psutil.Process()
        process.nice(psutil.HIGH_PRIORITY_CLASS)
        logging.info("Set process priority to high")
    except (ImportError, PermissionError):
        logging.warning("Could not set process priority")


def main() -> int:
    """Main application entry point."""
    # Use multiprocessing to support proper concurrency
    multiprocessing.freeze_support()
    
    # Parse command line arguments and load configuration
    args = parse_arguments()
    config = load_config(args.config)
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level)
    
    # Override config with command line arguments
    if args.cpu_only:
        config["force_cpu_only"] = True
    
    # Initialize application components
    initialize_app(config)
    
    # Create and start Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("SignSync")
    
    # Create main window
    window = MainWindow(config)
    window.show()
    
    # Start the application event loop
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
