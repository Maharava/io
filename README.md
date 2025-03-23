# Wake Word Detection Engine

A fully offline wake word detection system for Windows operating systems using a lightweight CNN architecture. The system operates entirely on the local machine without requiring internet connectivity.

## Features

- **Completely Offline**: Works without internet connection
- **Lightweight**: Low CPU and memory usage
- **Customisable**: Train your own wake words
- **Privacy-Focused**: Audio never leaves your device
- **Configurable Actions**: Run commands, press keyboard shortcuts, or show notifications

## Installation

### Prerequisites

- Python 3.9 or higher
- Windows, macOS, or Linux operating system
- A microphone

### Standard Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/wakeword.git
cd wakeword

# Install the package
pip install -e .
```

### Run the Application

```bash
wakeword
```

## Quick Start Guide

1. **First Launch**: On first launch, you'll be prompted to select an audio input device and create a wake word model.

2. **Training a Wake Word**:
   - Click on "Training" in the system tray menu
   - Record at least 50 samples of your wake word (e.g., "Hey Computer")
   - Record at least 10 background noise samples (typical environment sounds)
   - Enter a name for your model and click "Start Training"
   - Training takes ~10-15 minutes depending on your hardware

3. **Configuration**:
   - Click on "Settings" in the system tray menu
   - Select your audio device and wake word model
   - Adjust detection threshold (higher = fewer false positives, but may miss actual wake words)
   - Configure what happens when the wake word is detected

4. **Activation**:
   - Click "Enable" in the system tray menu to start listening
   - The application will run in the background and listen for your wake word
   - When detected, it will perform your configured action

## Directory Structure

- `audio/`: Audio processing modules
  - `capture.py`: Microphone input handling
  - `features.py`: MFCC feature extraction
  - `vad.py`: Voice activity detection

- `model/`: Neural network components
  - `architecture.py`: CNN model definition
  - `inference.py`: Real-time inference engine
  - `training.py`: Model training pipeline

- `ui/`: User interface
  - `tray.py`: System tray application
  - `config.py`: Configuration window
  - `training_ui.py`: Training interface

- `utils/`: Utility functions
  - `config.py`: Configuration handling
  - `actions.py`: Trigger action execution

## Advanced Configuration

The application stores configuration and models in the `~/.wakeword` directory:

- `~/.wakeword/config/config.json`: Main configuration file
- `~/.wakeword/models/`: Trained wake word models
- `~/.wakeword/training_data/`: Audio samples for training

## Troubleshooting

- **No Sound Detected**: Check your microphone settings and make sure it's the default input device
- **False Positives**: Increase the detection threshold in the settings
- **Missed Wake Words**: Lower the detection threshold or train with more diverse samples
- **High CPU Usage**: Reduce the sample rate or frame size in the config file

## License

This project is licensed under the MIT License - see the LICENSE file for details.
