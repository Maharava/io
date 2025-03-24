"""
Io Wake Word Detection Engine - Main Entry Point
"""
import os
import sys
import logging
import threading
import queue
import time
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Local imports (absolute for direct script execution)
from audio.capture import AudioCapture
from audio.features import FeatureExtractor
from model.inference import WakeWordDetector
from utils.config import Config
from utils.actions import ActionHandler
from ui.app import IoApp

# Set up logging
def setup_logging():
    """Set up logging with proper directory creation"""
    log_dir = Path.home() / ".io"
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "io.log")
        ]
    )
    return logging.getLogger("Io")

logger = setup_logging()

def ensure_app_directories():
    """Create necessary application directories"""
    app_dir = Path.home() / ".io"
    models_dir = app_dir / "models"
    config_dir = app_dir / "config"
    training_dir = app_dir / "training_data"
    training_dir_wake = training_dir / "wake_word"
    training_dir_neg = training_dir / "negative"
    
    for directory in [app_dir, models_dir, config_dir, training_dir, 
                      training_dir_wake, training_dir_neg]:
        directory.mkdir(exist_ok=True)
    
    # Ensure config exists
    if not (config_dir / "config.json").exists():
        Config.create_default()


class AudioProcessor:
    """Audio processing pipeline for wake word detection"""
    
    def __init__(self, config):
        """Initialize the audio processing pipeline"""
        self.config = config
        self.running = False
        self.processing_thread = None
        
        # Processing queue for passing audio between threads
        self.audio_queue = queue.Queue(maxsize=100)
        
        # UI callback
        self.ui_callback = None
        
        # Initialize components
        self.feature_extractor = FeatureExtractor(
            sample_rate=config["sample_rate"],
            frame_size=config["frame_size"]
        )
        
        self.detector = WakeWordDetector(
            model_path=config["model_path"],
            threshold=config["threshold"]
        )
        
        self.action_handler = ActionHandler(
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
    
    def set_ui_callback(self, callback):
        """Set callback for UI updates"""
        self.ui_callback = callback
        # Register this callback with the detector
        self.detector.register_detection_callback(self.ui_callback)
    
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
                        # Log message moved to detector for better debouncing
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
        restart_needed = False
        
        # Check if audio device changed
        if config["audio_device"] != self.config["audio_device"]:
            restart_needed = True
        
        # Check if model path changed BEFORE updating config
        model_path_changed = config.get("model_path") != self.config.get("model_path")
        
        # Update configuration
        self.config = config
        
        # Update detector threshold
        self.detector.set_threshold(config["threshold"])
        
        # Update model if path changed
        if model_path_changed:
            logger.info(f"Loading model from path: {config['model_path']}")
            self.detector.load_model(config["model_path"])
                
        # Update action handler
        self.action_handler.update_config(config["action"], config["debounce_time"])
        
        # Restart audio capture if needed
        if restart_needed and self.running:
            self.stop()
            
            # Update audio capture
            self.audio_capture = AudioCapture(
                device_index=config["audio_device"],
                sample_rate=config["sample_rate"],
                frame_size=config["frame_size"],
                callback=self.enqueue_audio
            )
            
            self.start()


def main():
    """Main application entry point"""
    # Ensure app directories and configs exist
    ensure_app_directories()
    
    # Load configuration
    config = Config.load()
    
    # Create audio processor
    processor = AudioProcessor(config)
    
    # Create and run the application
    app = IoApp(processor)
    app.run()
    
    # Cleanup on exit
    processor.stop()
    logger.info("Application terminated")


if __name__ == "__main__":
    main()