#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main window for SignSync application
"""

import os
import logging
from typing import Dict, Any, Optional

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QTabWidget, QMenuBar, QStatusBar, QDockWidget,
    QToolBar, QPushButton, QLabel, QComboBox, QSlider
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QIcon, QAction

from app.gui.video_preview import VideoPreview
from app.gui.translation_panel import TranslationPanel


class MainWindow(QMainWindow):
    """Main window for SignSync application"""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize main window
        
        Args:
            config: Application configuration
        """
        super().__init__()
        self.config = config
        self.setWindowTitle(f"SignSync v{config['app']['version']}")
        
        # Set window size and position
        self.resize(1280, 720)
        self.setMinimumSize(800, 600)
        
        # Create central widget and main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Initialize components
        self._init_menu_bar()
        self._init_toolbar()
        self._init_components()
        self._init_status_bar()
        
        # Connect signals and slots
        self._connect_signals()
        
        logging.info("Main window initialized")
    
    def _init_menu_bar(self) -> None:
        """Initialize menu bar"""
        menu_bar = self.menuBar()
        
        # File menu
        file_menu = menu_bar.addMenu("&File")
        
        # Create actions
        new_session_action = QAction("&New Session", self)
        new_session_action.setShortcut("Ctrl+N")
        file_menu.addAction(new_session_action)
        
        open_action = QAction("&Open Recording", self)
        open_action.setShortcut("Ctrl+O")
        file_menu.addAction(open_action)
        
        save_action = QAction("&Save Recording", self)
        save_action.setShortcut("Ctrl+S")
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        settings_action = QAction("S&ettings", self)
        file_menu.addAction(settings_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Alt+F4")
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menu_bar.addMenu("&View")
        
        fullscreen_action = QAction("&Fullscreen", self)
        fullscreen_action.setShortcut("F11")
        view_menu.addAction(fullscreen_action)
        
        # Help menu
        help_menu = menu_bar.addMenu("&Help")
        
        help_action = QAction("&Help", self)
        help_action.setShortcut("F1")
        help_menu.addAction(help_action)
        
        about_action = QAction("&About", self)
        help_menu.addAction(about_action)
    
    def _init_toolbar(self) -> None:
        """Initialize toolbar"""
        self.toolbar = QToolBar("Main Toolbar")
        self.toolbar.setMovable(False)
        self.toolbar.setIconSize(QSize(32, 32))
        self.addToolBar(self.toolbar)
        
        # Camera control
        self.camera_btn = QPushButton("Start Camera")
        self.toolbar.addWidget(self.camera_btn)
        
        self.toolbar.addSeparator()
        
        # Recording control
        self.record_btn = QPushButton("Record")
        self.toolbar.addWidget(self.record_btn)
        
        self.toolbar.addSeparator()
        
        # Translation control
        self.translate_btn = QPushButton("Translate")
        self.toolbar.addWidget(self.translate_btn)
        
        self.toolbar.addSeparator()
        
        # Camera selection
        self.toolbar.addWidget(QLabel("Camera:"))
        self.camera_combo = QComboBox()
        self.camera_combo.addItem("Default Camera")
        self.toolbar.addWidget(self.camera_combo)
    
    def _init_components(self) -> None:
        """Initialize main components"""
        # Create horizontal layout for main content
        content_layout = QHBoxLayout()
        self.main_layout.addLayout(content_layout)
        
        # Create video preview
        self.video_preview = VideoPreview(self.config)
        content_layout.addWidget(self.video_preview, 2)  # 2/3 of space
        
        # Create translation panel
        self.translation_panel = TranslationPanel(self.config)
        content_layout.addWidget(self.translation_panel, 1)  # 1/3 of space
        
        # Create tabs for different modes
        self.mode_tabs = QTabWidget()
        self.main_layout.addWidget(self.mode_tabs)
        
        # Create Translation tab
        self.translation_tab = QWidget()
        self.mode_tabs.addTab(self.translation_tab, "Translation")
        
        # Create Video Call tab
        self.video_call_tab = QWidget()
        self.mode_tabs.addTab(self.video_call_tab, "Video Call")
        
        # Create Learning tab
        self.learning_tab = QWidget()
        self.mode_tabs.addTab(self.learning_tab, "Learning")
    
    def _init_status_bar(self) -> None:
        """Initialize status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Add status message
        self.status_message = QLabel("Ready")
        self.status_bar.addWidget(self.status_message)
        
        # Add spacer
        self.status_bar.addPermanentWidget(QWidget(), 1)
        
        # Add FPS counter
        self.fps_label = QLabel("FPS: --")
        self.status_bar.addPermanentWidget(self.fps_label)
        
        # Add confidence indicator
        self.confidence_label = QLabel("Confidence: --")
        self.status_bar.addPermanentWidget(self.confidence_label)
    
    def _connect_signals(self) -> None:
        """Connect signals and slots"""
        # Camera button
        self.camera_btn.clicked.connect(self._toggle_camera)
        
        # Record button
        self.record_btn.clicked.connect(self._toggle_recording)
        
        # Translate button
        self.translate_btn.clicked.connect(self._toggle_translation)
    
    def _toggle_camera(self) -> None:
        """Toggle camera on/off"""
        if self.video_preview.is_running:
            self.video_preview.stop()
            self.camera_btn.setText("Start Camera")
            self.status_message.setText("Camera stopped")
        else:
            camera_index = self.config["video"]["camera_index"]
            self.video_preview.start(camera_index)
            self.camera_btn.setText("Stop Camera")
            self.status_message.setText("Camera started")
    
    def _toggle_recording(self) -> None:
        """Toggle recording on/off"""
        if self.video_preview.is_recording:
            self.video_preview.stop_recording()
            self.record_btn.setText("Record")
            self.status_message.setText("Recording stopped")
        else:
            if not self.video_preview.is_running:
                self._toggle_camera()
            
            self.video_preview.start_recording()
            self.record_btn.setText("Stop Recording")
            self.status_message.setText("Recording started")
    
    def _toggle_translation(self) -> None:
        """Toggle translation on/off"""
        if self.translation_panel.is_active:
            self.translation_panel.stop()
            self.translate_btn.setText("Translate")
            self.status_message.setText("Translation stopped")
        else:
            if not self.video_preview.is_running:
                self._toggle_camera()
            
            self.translation_panel.start()
            self.translate_btn.setText("Stop Translation")
            self.status_message.setText("Translation started")
    
    def closeEvent(self, event) -> None:
        """Handle window close event"""
        # Stop all processes
        if self.video_preview.is_running:
            self.video_preview.stop()
        
        if self.translation_panel.is_active:
            self.translation_panel.stop()
        
        # Accept close event
        event.accept() 