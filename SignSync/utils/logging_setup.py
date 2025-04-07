#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Logging setup utility for SignSync
"""

import os
import sys
import logging
import datetime
from typing import Optional


def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None) -> None:
    """
    Set up logging configuration
    
    Args:
        level: Logging level (default: INFO)
        log_file: Path to log file (default: logs/signsync_YYYY-MM-DD.log)
    """
    # Create logs directory if it doesn't exist
    if log_file is None:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(log_dir, f"signsync_{today}.log")
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    file_formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s - %(name)s:%(lineno)d - %(message)s"
    )
    console_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    
    # Create and configure console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)
    
    # Create and configure file handler
    try:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(min(level, logging.INFO))  # Always log at least INFO to file
        logger.addHandler(file_handler)
        logging.info(f"Logging to {log_file}")
    except (PermissionError, FileNotFoundError) as e:
        logging.warning(f"Could not set up file logging: {e}")
    
    # Log uncaught exceptions
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # Let KeyboardInterrupt pass through
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    
    sys.excepthook = handle_exception
    
    # Log start of application
    logging.info("=" * 80)
    logging.info("Starting SignSync application")
    logging.info("Log level: %s", logging.getLevelName(level))


class PerformanceLogger:
    """
    Performance logging utility for monitoring application performance
    """
    
    def __init__(self, name: str) -> None:
        """
        Initialize performance logger
        
        Args:
            name: Name of the component being monitored
        """
        self.name = name
        self.logger = logging.getLogger(f"perf.{name}")
        self.start_time = None
    
    def start(self) -> None:
        """Start timing"""
        self.start_time = datetime.datetime.now()
    
    def stop(self, message: str = "") -> float:
        """
        Stop timing and log duration
        
        Args:
            message: Optional message to add to log
            
        Returns:
            Duration in milliseconds
        """
        if self.start_time is None:
            return 0.0
        
        duration = (datetime.datetime.now() - self.start_time).total_seconds() * 1000
        if message:
            self.logger.debug(f"{message}: {duration:.2f}ms")
        else:
            self.logger.debug(f"Duration: {duration:.2f}ms")
            
        self.start_time = None
        return duration 