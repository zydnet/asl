#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gesture classifier for sign language recognition
"""

import os
import logging
import collections
from typing import Dict, Any, List, Tuple, Optional, Deque

import numpy as np
import tensorflow as tf

from utils.logging_setup import PerformanceLogger


class GestureClassifier:
    """Gesture classifier for sign language recognition"""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize gesture classifier
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.perf_logger = PerformanceLogger("gesture_classifier")
        
        # Load model
        self.model = self._load_model()
        
        # Initialize sliding window for continuous recognition
        self.window_size = config["recognition"]["sliding_window_size"]
        self.frame_buffer: Deque[List[Dict[str, Any]]] = collections.deque(maxlen=self.window_size)
        
        # Get confidence threshold
        self.confidence_threshold = config["recognition"]["confidence_threshold"]
        
        # Load label mapping (ID to sign name)
        self.labels = self._load_labels()
        
        logging.info(f"Gesture classifier initialized with {len(self.labels)} classes")
    
    def _load_model(self) -> Optional[tf.lite.Interpreter]:
        """
        Load TensorFlow Lite model
        
        Returns:
            TensorFlow Lite interpreter or None if loading failed
        """
        try:
            # Get model path
            model_path = self.config["recognition"]["model_path"]
            
            # Handle relative path
            if not os.path.isabs(model_path):
                model_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                    model_path
                )
            
            # Check if model exists
            if not os.path.exists(model_path):
                logging.warning(f"Model file not found: {model_path}")
                logging.warning("Using mock classifier instead")
                return None
            
            # Load model
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            
            # Get input and output details
            self.input_details = interpreter.get_input_details()
            self.output_details = interpreter.get_output_details()
            
            return interpreter
            
        except Exception as e:
            logging.error(f"Error loading model: {e}", exc_info=True)
            return None
    
    def _load_labels(self) -> List[str]:
        """
        Load label mapping
        
        Returns:
            List of label names
        """
        # TODO: Load labels from file
        # For now, use a mock label list
        return [
            "Hello",
            "Thank you",
            "Yes",
            "No",
            "Please",
            "Sorry",
            "Help",
            "Stop",
            "Good",
            "Bad"
        ]
    
    def _preprocess_landmarks(self, hand_landmarks_data: List[Dict[str, Any]]) -> np.ndarray:
        """
        Preprocess hand landmarks for model input
        
        Args:
            hand_landmarks_data: List of hand landmark data
            
        Returns:
            Preprocessed input features
        """
        # If no hands detected, return empty array
        if not hand_landmarks_data:
            return np.zeros((1, 63))  # 21 landmarks * 3 (x, y, z)
        
        # For simplicity, just use the first hand
        hand_data = hand_landmarks_data[0]
        landmarks = hand_data['landmarks']
        
        # Extract x, y, z coordinates and flatten into a single array
        features = []
        for lm in landmarks:
            features.extend([lm['x'], lm['y'], lm['z']])
        
        # Ensure we have 63 features (21 landmarks * 3 coordinates)
        if len(features) < 63:
            features.extend([0.0] * (63 - len(features)))
        
        # Reshape for model input
        return np.array(features, dtype=np.float32).reshape(1, 63)
    
    def process_landmarks(self, hand_landmarks_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process hand landmarks and recognize gesture
        
        Args:
            hand_landmarks_data: List of hand landmark data
            
        Returns:
            Dictionary with recognized gesture and confidence
        """
        self.perf_logger.start()
        
        # Add landmarks to frame buffer
        self.frame_buffer.append(hand_landmarks_data)
        
        # If no real model is loaded, use mock classification
        if self.model is None:
            mock_result = self._mock_classify(hand_landmarks_data)
            self.perf_logger.stop("Gesture classification (mock)")
            return mock_result
        
        try:
            # Preprocess landmarks
            input_features = self._preprocess_landmarks(hand_landmarks_data)
            
            # Set model input
            self.model.set_tensor(self.input_details[0]['index'], input_features)
            
            # Run inference
            self.model.invoke()
            
            # Get output
            output = self.model.get_tensor(self.output_details[0]['index'])
            
            # Find class with highest probability
            class_id = np.argmax(output[0])
            confidence = output[0][class_id]
            
            # Check confidence threshold
            if confidence < self.confidence_threshold:
                result = {
                    'gesture': 'unknown',
                    'confidence': confidence,
                    'recognized': False
                }
            else:
                gesture = self.labels[class_id] if class_id < len(self.labels) else f"unknown_{class_id}"
                result = {
                    'gesture': gesture,
                    'confidence': confidence,
                    'recognized': True
                }
            
            self.perf_logger.stop("Gesture classification")
            return result
            
        except Exception as e:
            logging.error(f"Error during gesture classification: {e}", exc_info=True)
            return {
                'gesture': 'error',
                'confidence': 0.0,
                'recognized': False,
                'error': str(e)
            }
    
    def _mock_classify(self, hand_landmarks_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Mock classification for development purposes
        
        Args:
            hand_landmarks_data: List of hand landmark data
            
        Returns:
            Dictionary with mock classification result
        """
        # If no hands detected, return unknown
        if not hand_landmarks_data:
            return {
                'gesture': 'unknown',
                'confidence': 0.0,
                'recognized': False
            }
        
        # Use a simple heuristic to determine a mock gesture
        # In a real implementation, this would be replaced by actual model inference
        import random
        
        # Select a random gesture with high confidence
        class_id = random.randint(0, len(self.labels) - 1)
        confidence = random.uniform(0.85, 0.99)
        
        return {
            'gesture': self.labels[class_id],
            'confidence': confidence,
            'recognized': True
        }
    
    def get_sentence_from_buffer(self) -> Tuple[str, float]:
        """
        Analyze the frame buffer to determine a complete sentence
        
        Returns:
            Tuple of (sentence, confidence)
        """
        # If buffer is nearly empty, return empty string
        if len(self.frame_buffer) < self.window_size // 2:
            return "", 0.0
        
        # Count occurrences of each gesture in the buffer
        gesture_counts = {}
        total_frames = len(self.frame_buffer)
        
        for frames in self.frame_buffer:
            if not frames:  # Skip empty frames
                continue
                
            result = self.process_landmarks(frames)
            if result['recognized']:
                gesture = result['gesture']
                if gesture in gesture_counts:
                    gesture_counts[gesture] += 1
                else:
                    gesture_counts[gesture] = 1
        
        # Find the most common gesture
        if not gesture_counts:
            return "", 0.0
            
        most_common = max(gesture_counts.items(), key=lambda x: x[1])
        gesture, count = most_common
        
        # Calculate confidence (proportion of frames with this gesture)
        confidence = count / total_frames
        
        return gesture, confidence 