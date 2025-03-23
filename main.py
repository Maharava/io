"""
Wake Word Detection Engine - Main Application
"""
import os
import sys
import json
import logging
import threading
from pathlib import Path

# Local imports
from audio.capture import AudioCapture
from audio.features import FeatureExtractor
from model.inference import WakeWordDetector
from utils.config import load_config, save_config
from utils.actions import TriggerHandler
from ui.tray import SystemTrayApp

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path.home() / ".wakeword" / "wakeword.log")
    ]
)
logger = logging.getLogger("WakeWord")

def ensure_app_directories():
    """Create necessary application directories if they don't exist"""
    app_dir = Path.home() / ".wakeword"
    models_dir = app_dir / "models"
    config_dir = app_dir / "config"
    
    for directory in [app_dir, models_dir, config_dir]:
        directory.mkdir(exist_ok=True)
    
    # Create default config if it doesn't exist
    config_file = config_dir / "config.json"
    if not config_file.exists():
        default_config = {
            "audio_device": None,  # Will be selected on first run
            "sample_rate": 16000,
            "frame_size": 512,
            "model_path": None,    # Will be selected on first run
            "threshold": 0.85,
            "debounce_time": 3.0,  # seconds
            "action": {
                "type": "notification",
                "params": {"message": "Wake word detected!"}
            }
        }
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=4)

def main():
    """Main application entry point"""
    # Ensure app directories and configs exist
    ensure_app_directories()
    
    # Load configuration
    config = load_config()
    
    # Initialize components
    feature_extractor = FeatureExtractor(
        sample_rate=config["sample_rate"],
        frame_size=config["frame_size"]
    )
    
    detector = WakeWordDetector(
        model_path=config["model_path"],
        threshold=config["threshold"]
    )
    
    action_handler = TriggerHandler(
        action_config=config["action"],
        debounce_time=config["debounce_time"]
    )
    
    # Set up the processing pipeline
    def process_audio(audio_frame):
        """Process incoming audio and trigger actions on detection"""
        features = feature_extractor.extract(audio_frame)
        if features is not None:
            detection, confidence = detector.detect(features)
            if detection:
                logger.info(f"Wake word detected with confidence: {confidence:.4f}")
                action_handler.trigger()
    
    # Start audio capture with the processing callback
    audio_capture = AudioCapture(
        device_index=config["audio_device"],
        sample_rate=config["sample_rate"],
        frame_size=config["frame_size"],
        callback=process_audio
    )
    
    # Start the audio capture in a separate thread
    capture_thread = threading.Thread(target=audio_capture.start, daemon=True)
    capture_thread.start()
    
    # Start the system tray application (this will block until exit)
    app = SystemTrayApp(
        audio_capture=audio_capture,
        detector=detector,
        config=config
    )
    app.run()
    
    # Cleanup on exit
    audio_capture.stop()
    logger.info("Application terminated")

if __name__ == "__main__":
    main()
