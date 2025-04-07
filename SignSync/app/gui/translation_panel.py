#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Translation panel widget for SignSync application
"""

import os
import time
import logging
import threading
import queue
from typing import Dict, Any, Optional, List, Tuple

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton,
    QComboBox, QLabel, QSlider, QGroupBox, QCheckBox, QSpinBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QFont, QTextCursor, QIcon

from utils.logging_setup import PerformanceLogger


class TranslationWorker(QThread):
    """Worker thread for translation processing"""
    
    # Define signals
    translation_ready = pyqtSignal(str, float)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize translation worker
        
        Args:
            config: Application configuration
        """
        super().__init__()
        self.config = config
        self.running = False
        self.translation_queue = queue.Queue()
        self.perf_logger = PerformanceLogger("translation_worker")
    
    def run(self) -> None:
        """Run the thread"""
        try:
            self.running = True
            
            logging.info("Translation worker started")
            
            # Main processing loop
            while self.running:
                try:
                    # Wait for a new frame to process (timeout after 0.1 seconds)
                    try:
                        frame = self.translation_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    
                    # Process the frame
                    self.perf_logger.start()
                    
                    # TODO: Implement actual translation using hand gesture recognition
                    # For now, just use a mock translation
                    mock_translation = self._mock_translate(frame)
                    confidence = 0.95
                    
                    # Emit the translation result
                    self.translation_ready.emit(mock_translation, confidence)
                    
                    self.perf_logger.stop("Translation processing")
                    
                except Exception as e:
                    logging.error(f"Error processing translation: {e}", exc_info=True)
                    self.error_occurred.emit(f"Translation error: {e}")
                
                # Release the queue item
                self.translation_queue.task_done()
                
        except Exception as e:
            logging.error(f"Error in translation worker: {e}", exc_info=True)
            self.error_occurred.emit(f"Error in translation worker: {e}")
    
    def _mock_translate(self, frame) -> str:
        """
        Mock translation function (for development purposes)
        
        Args:
            frame: Input frame
            
        Returns:
            Mocked translation text
        """
        # Simulate processing time
        time.sleep(0.1)
        
        # Return a mock translation
        mock_phrases = [
            "Hello, how are you?",
            "My name is John",
            "Nice to meet you",
            "Thank you",
            "Yes",
            "No",
            "Please",
            "Excuse me",
            "I don't understand",
            "Can you help me?",
        ]
        
        import random
        return random.choice(mock_phrases)
    
    def add_frame(self, frame) -> None:
        """
        Add a frame to the translation queue
        
        Args:
            frame: Frame to translate
        """
        if self.running:
            self.translation_queue.put(frame)
    
    def stop(self) -> None:
        """Stop the thread"""
        self.running = False
        self.wait()


class TranslationPanel(QWidget):
    """Translation panel widget for displaying translations"""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize translation panel
        
        Args:
            config: Application configuration
        """
        super().__init__()
        self.config = config
        self.is_active = False
        self.last_translation = ""
        
        # Create layout
        self.layout = QVBoxLayout(self)
        
        # Create translation text area
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setFont(QFont("Arial", 12))
        self.text_edit.setMinimumHeight(200)
        self.layout.addWidget(self.text_edit)
        
        # Create settings group
        self.settings_group = QGroupBox("Translation Settings")
        self.settings_layout = QVBoxLayout(self.settings_group)
        self.layout.addWidget(self.settings_group)
        
        # Create TTS settings
        self.tts_settings = QGroupBox("Text-to-Speech")
        self.tts_layout = QVBoxLayout(self.tts_settings)
        
        # TTS enable checkbox
        self.tts_enabled = QCheckBox("Enable Text-to-Speech")
        self.tts_enabled.setChecked(True)
        self.tts_layout.addWidget(self.tts_enabled)
        
        # TTS voice selection
        voice_layout = QHBoxLayout()
        voice_layout.addWidget(QLabel("Voice:"))
        self.voice_combo = QComboBox()
        self.voice_combo.addItem("Default")
        self.voice_combo.addItem("Male")
        self.voice_combo.addItem("Female")
        voice_layout.addWidget(self.voice_combo)
        self.tts_layout.addLayout(voice_layout)
        
        # TTS speed slider
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(50)
        self.speed_slider.setMaximum(200)
        self.speed_slider.setValue(100)
        self.speed_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.speed_slider.setTickInterval(25)
        speed_layout.addWidget(self.speed_slider)
        speed_layout.addWidget(QLabel("1.0x"))
        self.tts_layout.addLayout(speed_layout)
        
        self.settings_layout.addWidget(self.tts_settings)
        
        # Create translation settings
        self.trans_settings = QGroupBox("Translation Options")
        self.trans_layout = QVBoxLayout(self.trans_settings)
        
        # Context window size
        context_layout = QHBoxLayout()
        context_layout.addWidget(QLabel("Context Window:"))
        self.context_spin = QSpinBox()
        self.context_spin.setMinimum(1)
        self.context_spin.setMaximum(10)
        self.context_spin.setValue(self.config["translation"]["context_window_size"])
        context_layout.addWidget(self.context_spin)
        self.trans_layout.addLayout(context_layout)
        
        # Auto-correct checkbox
        self.autocorrect = QCheckBox("Enable Auto-Correction")
        self.autocorrect.setChecked(self.config["translation"]["autocorrect_enabled"])
        self.trans_layout.addWidget(self.autocorrect)
        
        self.settings_layout.addWidget(self.trans_settings)
        
        # Create action buttons
        button_layout = QHBoxLayout()
        
        self.clear_btn = QPushButton("Clear")
        button_layout.addWidget(self.clear_btn)
        
        self.copy_btn = QPushButton("Copy")
        button_layout.addWidget(self.copy_btn)
        
        self.save_btn = QPushButton("Save")
        button_layout.addWidget(self.save_btn)
        
        self.layout.addLayout(button_layout)
        
        # Connect signals and slots
        self.clear_btn.clicked.connect(self._clear_translation)
        self.copy_btn.clicked.connect(self._copy_translation)
        self.save_btn.clicked.connect(self._save_translation)
        
        # Create translation worker
        self.translation_worker = None
    
    def start(self) -> bool:
        """
        Start translation processing
        
        Returns:
            True if started successfully, False otherwise
        """
        if self.is_active:
            logging.warning("Translation panel already active")
            return False
        
        try:
            # Create translation worker
            self.translation_worker = TranslationWorker(self.config)
            
            # Connect signals
            self.translation_worker.translation_ready.connect(self._update_translation)
            self.translation_worker.error_occurred.connect(self._on_error)
            
            # Start translation worker
            self.translation_worker.start()
            self.is_active = True
            logging.info("Started translation processing")
            
            # Add initial message
            self._append_message("Translation started. Ready to interpret sign language...")
            
            return True
            
        except Exception as e:
            logging.error(f"Error starting translation: {e}", exc_info=True)
            return False
    
    def stop(self) -> None:
        """Stop translation processing"""
        if not self.is_active:
            return
        
        try:
            # Stop translation worker
            if self.translation_worker is not None:
                self.translation_worker.stop()
                self.translation_worker = None
            
            self.is_active = False
            self._append_message("Translation stopped.")
            logging.info("Stopped translation processing")
            
        except Exception as e:
            logging.error(f"Error stopping translation: {e}", exc_info=True)
    
    def process_frame(self, frame) -> None:
        """
        Process a video frame for translation
        
        Args:
            frame: Video frame to process
        """
        if self.is_active and self.translation_worker is not None:
            self.translation_worker.add_frame(frame)
    
    def _update_translation(self, translation: str, confidence: float) -> None:
        """
        Update translation text
        
        Args:
            translation: Translation text
            confidence: Confidence score (0.0-1.0)
        """
        # Skip if it's the same as the last translation
        if translation == self.last_translation:
            return
        
        self.last_translation = translation
        
        # Format confidence as percentage
        confidence_str = f"{confidence * 100:.1f}%"
        
        # Add timestamp
        timestamp = time.strftime("%H:%M:%S")
        
        # Format message
        message = f"[{timestamp}, {confidence_str}] {translation}"
        
        # Append to text edit
        self._append_message(message)
        
        # TODO: Implement text-to-speech if enabled
        if self.tts_enabled.isChecked():
            # For now, just log that we would speak the text
            logging.debug(f"TTS would speak: {translation}")
    
    def _append_message(self, message: str) -> None:
        """
        Append message to text edit
        
        Args:
            message: Message text
        """
        self.text_edit.append(message)
        
        # Auto-scroll to bottom
        cursor = self.text_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.text_edit.setTextCursor(cursor)
    
    def _clear_translation(self) -> None:
        """Clear translation text"""
        self.text_edit.clear()
        self.last_translation = ""
    
    def _copy_translation(self) -> None:
        """Copy translation text to clipboard"""
        self.text_edit.selectAll()
        self.text_edit.copy()
        self.text_edit.moveCursor(QTextCursor.MoveOperation.End)
    
    def _save_translation(self) -> None:
        """Save translation text to file"""
        try:
            # Create output directory if it doesn't exist
            translations_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "data",
                "translations"
            )
            os.makedirs(translations_dir, exist_ok=True)
            
            # Generate output path with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(translations_dir, f"translation_{timestamp}.txt")
            
            # Save translation text
            with open(output_path, "w") as f:
                f.write(self.text_edit.toPlainText())
            
            # Add success message
            self._append_message(f"Translation saved to {output_path}")
            logging.info(f"Translation saved to {output_path}")
            
        except Exception as e:
            logging.error(f"Error saving translation: {e}", exc_info=True)
            self._append_message(f"Error saving translation: {e}")
    
    def _on_error(self, error_message: str) -> None:
        """
        Handle error message from translation worker
        
        Args:
            error_message: Error message
        """
        logging.error(f"Translation worker error: {error_message}")
        self._append_message(f"Error: {error_message}") 