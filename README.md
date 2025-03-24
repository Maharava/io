# Io Wake Word Detection Engine

A fully offline wake word detection system using a lightweight CNN architecture. The system operates entirely on the local machine without requiring internet connectivity, making it suitable for privacy-conscious applications.

## Features

- **Completely Offline**: Works without internet connection
- **Lightweight**: Low CPU and memory usage
- **Customisable**: Train your own wake words
- **Privacy-Focused**: Audio never leaves your device
- **Modern UI**: Sleek, sci-fi inspired interface using CustomTkinter
- **Configurable Actions**: Run commands, press keyboard shortcuts, or show notifications

## Installation

### Prerequisites

- Python 3.9 or higher
- Windows, macOS, or Linux operating system
- A microphone

### Standard Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/io-wake-word.git
cd io-wake-word

# Install the package
pip install -e .
```

### Run the Application

```bash
io
```

## Quick Start Guide

1. **First Launch**: On first launch, you'll be prompted to select an audio input device and create a wake word model.

2. **Training a Wake Word**:
   - Go to the "Training" tab
   - Record at least 50 samples of your wake word (e.g., "Hey Computer")
   - Record at least 10 background noise samples (typical environment sounds)
   - Enter a name for your model and click "Start Training"
   - Training takes ~10-15 minutes depending on your hardware

3. **Configuration**:
   - Go to the "Configuration" tab
   - Select your audio device and wake word model
   - Adjust detection threshold (higher = fewer false positives, but may miss actual wake words)
   - Configure what happens when the wake word is detected

4. **Activation**:
   - On the "Dashboard" tab, click "Start Detection" to begin listening
   - The application will listen for your wake word
   - When detected, it will perform your configured action

## Project Structure

- `audio/`: Audio processing modules
  - `capture.py`: Microphone input handling
  - `features.py`: MFCC feature extraction
  - `vad.py`: Voice activity detection

- `model/`: Neural network components
  - `architecture.py`: CNN model definition
  - `inference.py`: Real-time inference engine
  - `training.py`: Model training pipeline

- `ui/`: User interface
  - `app.py`: Main application window
  - `config_panel.py`: Configuration panel
  - `training_panel.py`: Training interface

- `utils/`: Utility functions
  - `config.py`: Configuration handling
  - `actions.py`: Trigger action execution

## Configuration

The application stores configuration and models in the `~/.io` directory:

- `~/.io/config/config.json`: Main configuration file
- `~/.io/models/`: Trained wake word models
- `~/.io/training_data/`: Audio samples for training

## Troubleshooting

- **No Sound Detected**: Check your microphone settings and make sure it's the default input device
- **False Positives**: Increase the detection threshold in the settings
- **Missed Wake Words**: Lower the detection threshold or train with more diverse samples
- **High CPU Usage**: Reduce the sample rate or frame size in the config file

## License

This project is licensed under the MIT License - see the LICENSE file for details.