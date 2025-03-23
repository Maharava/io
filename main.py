"""
Wake Word Detection Engine - Main Application
"""
import os
import sys
import json
import logging
import threading
import queue
import time
from pathlib import Path

# Local imports
from audio.capture import AudioCapture
from audio.features import FeatureExtractor
from model.inference import WakeWordDetector
from utils.config import load_config, save_config, create_default_config
from utils.actions import TriggerHandler
from ui.tray import SystemTrayApp

# Set up logging
def setup_logging():
    """Set up logging with proper directory creation"""
    log_dir = Path.home() / ".wakeword"
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "wakeword.log")
        ]
    )
    return logging.getLogger("WakeWord")

logger = setup_logging()

def ensure_app_directories():
    """Create necessary application directories if they don't exist"""
    app_dir = Path.home() / ".wakeword"
    models_dir = app_dir / "models"
    config_dir = app_dir / "config"
    training_dir = app_dir / "training_data"
    
    for directory in [app_dir, models_dir, config_dir, training_dir]:
        directory.mkdir(exist_ok=True)
    
    # Create default config if it doesn't exist
    config_file = config_dir / "config.json"
    if not config_file.exists():
        default_config = create_default_config()
        # No need to save here as create_default_config does it

class AudioProcessor:
    """Thread-safe audio processing pipeline"""
    def __init__(self, config):
        """Initialize the audio processing pipeline"""
        self.config = config
        self.running = False
        self.processing_thread = None
        
        # Processing queue for passing audio between threads
        self.audio_queue = queue.Queue(maxsize=100)  # Limit queue size to prevent memory issues
        
        # Initialize components
        self.feature_extractor = FeatureExtractor(
            sample_rate=config["sample_rate"],
            frame_size=config["frame_size"]
        )
        
        self.detector = WakeWordDetector(
            model_path=config["model_path"],
            threshold=config["threshold"]
        )
        
        self.action_handler = TriggerHandler(
            action_config=config["action"],
            debounce_time=config["debounce_time"]
        )
        
        # Audio capture with callback to enqueue audio
        self.audio_capture = AudioCapture(
            device_index=config["audio_device"],
            sample_rate=config["sample_rate"],
            frame_size=config["frame_size"],
            callback=self.enqueue_audio
        )
    
    def enqueue_audio(self, audio_frame):
        """Add audio frame to processing queue"""
        if self.running:
            try:
                # Use non-blocking put with timeout to avoid deadlocks
                self.audio_queue.put(audio_frame, block=True, timeout=0.1)
            except queue.Full:
                logger.warning("Audio processing queue full, dropping frame")
    
    def processing_loop(self):
        """Main audio processing loop"""
        logger.info("Audio processing thread started")
        
        while self.running:
            try:
                # Get audio frame from queue with timeout
                audio_frame = self.audio_queue.get(block=True, timeout=0.1)
                
                # Extract features
                features = self.feature_extractor.extract(audio_frame)
                
                # Perform detection if features were extracted
                if features is not None:
                    detection, confidence = self.detector.detect(features)
                    
                    # Trigger action if wake word detected
                    if detection:
                        logger.info(f"Wake word detected with confidence: {confidence:.4f}")
                        self.action_handler.trigger()
                
                # Mark task as done
                self.audio_queue.task_done()
                
            except queue.Empty:
                # Timeout waiting for audio, just continue
                pass
            except Exception as e:
                logger.error(f"Error in audio processing: {e}")
                # Short sleep to avoid CPU spinning on repeated errors
                time.sleep(0.1)
        
        logger.info("Audio processing thread stopped")
    
    def start(self):
        """Start the audio processor"""
        if self.running:
            return
            
        self.running = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self.processing_loop,
            daemon=True
        )
        self.processing_thread.start()
        
        # Start audio capture
        self.audio_capture.start()
        
        logger.info("Audio processor started")
    
    def stop(self):
        """Stop the audio processor"""
        if not self.running:
            return
            
        self.running = False
        
        # Stop audio capture
        self.audio_capture.stop()
        
        # Wait for processing thread to finish
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
            self.processing_thread = None
        
        # Clear the queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
            except queue.Empty:
                break
                
        logger.info("Audio processor stopped")
    
    def update_config(self, config):
        """Update configuration"""
        self.config = config
        
        # Update components with new config
        self.detector.set_threshold(config["threshold"])
        if config["model_path"] != self.config["model_path"]:
            self.detector.load_model(config["model_path"])
            
        self.action_handler.update_config(config["action"])
        self.action_handler.debounce_time = config["debounce_time"]
        
        # If audio device changed, restart audio capture
        if config["audio_device"] != self.config["audio_device"]:
            was_running = self.audio_capture.is_running
            if was_running:
                self.audio_capture.stop()
                
            self.audio_capture = AudioCapture(
                device_index=config["audio_device"],
                sample_rate=config["sample_rate"],
                frame_size=config["frame_size"],
                callback=self.enqueue_audio
            )
            
            if was_running:
                self.audio_capture.start()

def main():
    """Main application entry point"""
    # Ensure app directories and configs exist
    ensure_app_directories()
    
    # Load configuration
    config = load_config()
    
    # Create audio processor
    processor = AudioProcessor(config)
    
    # Start the system tray application (this will block until exit)
    app = SystemTrayApp(
        audio_processor=processor,
        config=config
    )
    app.run()
    
    # Cleanup on exit
    processor.stop()
    logger.info("Application terminated")

if __name__ == "__main__":
    main()