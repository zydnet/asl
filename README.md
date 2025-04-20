# SignSync - Real-Time Sign Language Translation Suite

A cross-platform desktop application with integrated AI capabilities for bidirectional sign language translation, video call support, and adaptive learning features.

## Features

- **Real-Time Sign Recognition**: 21-point hand landmark detection with MediaPipe Hands and gesture classification
- **Multimodal Translation**: Sign-to-Text and Text-to-Speech with context-aware translation
- **Video Call Integration**: WebRTC-based video communication with real-time overlay system
- **Adaptive Learning Module**: Gamified learning experience with personalized lessons

## Requirements

- Python 3.9+
- GPU recommended (but not required) for optimal performance
- Webcam
- Microphone
- Speakers

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/SignSync.git
   cd SignSync
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Run the application:
   ```
   python main.py
   ```

## Project Structure

```
SignSync/
├── app/                # Application components
│   ├── gui/            # PyQt6 UI components
│   ├── recognition/    # Sign language recognition module
│   ├── translation/    # Translation pipeline
│   ├── video_call/     # Video call integration
│   └── learning/       # Adaptive learning module
├── models/             # Pre-trained models and model definitions
├── utils/              # Utility functions and helper classes
├── scripts/            # Setup and maintenance scripts
├── tests/              # Unit and integration tests
├── data/               # Training and test data
├── config/             # Configuration files
├── requirements.txt    # Python dependencies
├── main.py             # Application entry point
└── setup.py            # Package installation script
```
