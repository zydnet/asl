#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Video preview widget for SignSync application
"""

import os
import time
import logging
import threading
import datetime
from typing import Dict, Any, Optional, Tuple, List, Callable

import cv2
import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QSize, QThread
from PyQt6.QtGui import QImage, QPixmap

from utils.logging_setup import PerformanceLogger


class VideoWorker(QThread):
    """Worker thread for video processing"""
    
    # Define signals
    frame_ready = pyqtSignal(np.ndarray)
    fps_updated = pyqtSignal(float)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize video worker
        
        Args:
            config: Application configuration
        """
        super().__init__()
        self.config = config
        self.running = False
        self.camera = None
        self.camera_index = 0
        self.fps = 0.0
        self.frame_count = 0
        self.last_fps_time = 0
        self.recording = False
        self.video_writer = None
        self.output_path = ""
        self.perf_logger = PerformanceLogger("video_worker")
    
    def run(self) -> None:
        """Run the thread"""
        try:
            # Try to open the camera
            self.camera = cv2.VideoCapture(self.camera_index)
            
            # Check if camera was opened successfully
            if not self.camera.isOpened():
                self.error_occurred.emit(f"Failed to open camera {self.camera_index}")
                return
            
            # Set camera properties
            width = self.config["video"]["width"]
            height = self.config["video"]["height"]
            fps = self.config["video"]["fps"]
            
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.camera.set(cv2.CAP_PROP_FPS, fps)
            
            # Get actual camera properties
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
            
            logging.info(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps} FPS")
            
            # Initialize FPS calculation
            self.frame_count = 0
            self.last_fps_time = time.time()
            
            # Main processing loop
            self.running = True
            while self.running:
                # Measure frame processing time
                self.perf_logger.start()
                
                # Read frame from camera
                ret, frame = self.camera.read()
                
                if not ret:
                    # If frame capture failed, try to recover
                    logging.warning("Failed to capture frame, attempting to recover")
                    time.sleep(0.1)
                    continue
                
                # Process frame
                processed_frame = self._process_frame(frame)
                
                # Emit processed frame
                self.frame_ready.emit(processed_frame)
                
                # Record frame if recording
                if self.recording and self.video_writer is not None:
                    self.video_writer.write(frame)
                
                # Calculate FPS
                self.frame_count += 1
                current_time = time.time()
                elapsed_time = current_time - self.last_fps_time
                
                if elapsed_time >= 1.0:
                    self.fps = self.frame_count / elapsed_time
                    self.fps_updated.emit(self.fps)
                    self.frame_count = 0
                    self.last_fps_time = current_time
                
                # Log performance
                self.perf_logger.stop("Frame processing")
                
                # Ensure we don't exceed the target FPS
                target_frame_time = 1.0 / fps
                elapsed = time.time() - current_time
                if elapsed < target_frame_time:
                    time.sleep(target_frame_time - elapsed)
            
            # Clean up
            if self.camera is not None:
                self.camera.release()
            
            if self.video_writer is not None:
                self.video_writer.release()
                
        except Exception as e:
            logging.error(f"Error in video worker: {e}", exc_info=True)
            self.error_occurred.emit(f"Error in video worker: {e}")
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process frame
        
        Args:
            frame: Input frame
            
        Returns:
            Processed frame
        """
        # TODO: Add hand landmark detection
        # For now, just convert to RGB for Qt
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    def start_recording(self, output_path: Optional[str] = None) -> bool:
        """
        Start recording
        
        Args:
            output_path: Path to save the recording (default: auto-generated)
            
        Returns:
            True if recording started successfully, False otherwise
        """
        if self.camera is None or not self.camera.isOpened():
            logging.error("Cannot start recording: Camera not initialized")
            return False
        
        try:
            # Create output directory if it doesn't exist
            if output_path is None:
                recordings_dir = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                    "data",
                    "recordings"
                )
                os.makedirs(recordings_dir, exist_ok=True)
                
                # Generate output path with timestamp
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(recordings_dir, f"recording_{timestamp}.mp4")
            
            # Get camera properties
            width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.camera.get(cv2.CAP_PROP_FPS)
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not self.video_writer.isOpened():
                logging.error(f"Failed to create video writer: {output_path}")
                return False
            
            self.output_path = output_path
            self.recording = True
            logging.info(f"Started recording to {output_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error starting recording: {e}", exc_info=True)
            return False
    
    def stop_recording(self) -> None:
        """Stop recording"""
        if self.recording and self.video_writer is not None:
            self.recording = False
            self.video_writer.release()
            self.video_writer = None
            logging.info(f"Stopped recording, saved to {self.output_path}")
    
    def stop(self) -> None:
        """Stop the thread"""
        self.running = False
        self.wait()


class VideoPreview(QWidget):
    """Video preview widget for displaying camera feed"""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize video preview
        
        Args:
            config: Application configuration
        """
        super().__init__()
        self.config = config
        self.is_running = False
        self.is_recording = False
        self.current_fps = 0.0
        
        # Create layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Create video display label
        self.video_label = QLabel("Camera not started")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setMinimumSize(640, 480)
        self.layout.addWidget(self.video_label)
        
        # Create video worker
        self.video_worker = None
    
    def start(self, camera_index: int = 0) -> bool:
        """
        Start video preview
        
        Args:
            camera_index: Camera index to use
            
        Returns:
            True if started successfully, False otherwise
        """
        if self.is_running:
            logging.warning("Video preview already running")
            return False
        
        try:
            # Create video worker
            self.video_worker = VideoWorker(self.config)
            self.video_worker.camera_index = camera_index
            
            # Connect signals
            self.video_worker.frame_ready.connect(self._update_frame)
            self.video_worker.fps_updated.connect(self._update_fps)
            self.video_worker.error_occurred.connect(self._on_error)
            
            # Start video worker
            self.video_worker.start()
            self.is_running = True
            logging.info(f"Started video preview with camera {camera_index}")
            return True
            
        except Exception as e:
            logging.error(f"Error starting video preview: {e}", exc_info=True)
            return False
    
    def stop(self) -> None:
        """Stop video preview"""
        if not self.is_running:
            return
        
        try:
            # Stop recording if active
            if self.is_recording:
                self.stop_recording()
            
            # Stop video worker
            if self.video_worker is not None:
                self.video_worker.stop()
                self.video_worker = None
            
            self.is_running = False
            self.video_label.setText("Camera not started")
            logging.info("Stopped video preview")
            
        except Exception as e:
            logging.error(f"Error stopping video preview: {e}", exc_info=True)
    
    def start_recording(self, output_path: Optional[str] = None) -> bool:
        """
        Start recording
        
        Args:
            output_path: Path to save the recording (default: auto-generated)
            
        Returns:
            True if recording started successfully, False otherwise
        """
        if not self.is_running or self.video_worker is None:
            logging.error("Cannot start recording: Video preview not running")
            return False
        
        if self.is_recording:
            logging.warning("Recording already in progress")
            return False
        
        success = self.video_worker.start_recording(output_path)
        if success:
            self.is_recording = True
        return success
    
    def stop_recording(self) -> None:
        """Stop recording"""
        if not self.is_recording or self.video_worker is None:
            return
        
        self.video_worker.stop_recording()
        self.is_recording = False
    
    def _update_frame(self, frame: np.ndarray) -> None:
        """
        Update video frame
        
        Args:
            frame: Video frame to display
        """
        height, width, channels = frame.shape
        bytes_per_line = channels * width
        
        # Convert OpenCV frame to QImage
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Convert QImage to QPixmap and display it
        pixmap = QPixmap.fromImage(q_image)
        
        # Scale pixmap to fit label while preserving aspect ratio
        pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        self.video_label.setPixmap(pixmap)
    
    def _update_fps(self, fps: float) -> None:
        """
        Update FPS counter
        
        Args:
            fps: Current FPS value
        """
        self.current_fps = fps
    
    def _on_error(self, error_message: str) -> None:
        """
        Handle error message from video worker
        
        Args:
            error_message: Error message
        """
        logging.error(f"Video worker error: {error_message}")
        self.video_label.setText(f"Error: {error_message}")
        self.stop() 