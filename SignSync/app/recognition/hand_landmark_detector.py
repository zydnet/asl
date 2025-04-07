#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hand landmark detector using MediaPipe
"""

import logging
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np
import mediapipe as mp

from utils.logging_setup import PerformanceLogger


class HandLandmarkDetector:
    """Hand landmark detector using MediaPipe"""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize hand landmark detector
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Get configuration values
        self.min_detection_confidence = config["recognition"]["min_detection_confidence"]
        self.min_tracking_confidence = config["recognition"]["min_tracking_confidence"]
        
        # Initialize MediaPipe Hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        
        self.perf_logger = PerformanceLogger("hand_landmark_detector")
        logging.info("Hand landmark detector initialized")
    
    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Detect hand landmarks in a frame
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (annotated frame, list of hand landmark data)
        """
        self.perf_logger.start()
        
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.hands.process(frame_rgb)
        
        # Create a copy of the frame for drawing
        annotated_frame = frame.copy()
        
        # Extract landmarks
        hand_landmarks_data = []
        
        if results.multi_hand_landmarks:
            for hand_idx, (hand_landmarks, handedness) in enumerate(
                zip(results.multi_hand_landmarks, results.multi_handedness)
            ):
                # Draw landmarks on frame
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Get handedness (left or right)
                handedness_label = handedness.classification[0].label
                
                # Extract normalized landmarks
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility if hasattr(landmark, 'visibility') else 1.0
                    })
                
                # Add hand data to result
                hand_landmarks_data.append({
                    'handedness': handedness_label,
                    'landmarks': landmarks,
                    'bounding_box': self._calculate_bounding_box(frame, hand_landmarks)
                })
        
        self.perf_logger.stop("Hand landmark detection")
        
        return annotated_frame, hand_landmarks_data
    
    def _calculate_bounding_box(
        self, 
        frame: np.ndarray, 
        hand_landmarks
    ) -> Dict[str, int]:
        """
        Calculate bounding box for a hand
        
        Args:
            frame: Input frame
            hand_landmarks: MediaPipe hand landmarks
            
        Returns:
            Dictionary with bounding box coordinates
        """
        height, width, _ = frame.shape
        
        # Initialize min and max values
        min_x, min_y = width, height
        max_x, max_y = 0, 0
        
        # Find min and max coordinates
        for landmark in hand_landmarks.landmark:
            x, y = int(landmark.x * width), int(landmark.y * height)
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        
        # Add padding
        padding = 20
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(width, max_x + padding)
        max_y = min(height, max_y + padding)
        
        return {
            'x_min': min_x,
            'y_min': min_y,
            'x_max': max_x,
            'y_max': max_y,
            'width': max_x - min_x,
            'height': max_y - min_y
        }
    
    def release(self) -> None:
        """Release resources"""
        self.hands.close() 