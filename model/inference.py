"""
Inference engine for wake word detection
"""
import torch
import numpy as np
import logging
import collections
import threading
from .architecture import load_model

logger = logging.getLogger("Io.Model.Inference")

class WakeWordDetector:
    """Real-time wake word detector with confidence smoothing"""
    
    def __init__(self, model_path=None, threshold=0.85, window_size=3):
        """Initialize detector with given model and parameters"""
        self.model_path = model_path
        self.threshold = threshold
        self.window_size = window_size
        
        # Store recent predictions for smoothing
        self.recent_predictions = collections.deque(maxlen=window_size)
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Detection callback for testing
        self.test_callback = None
        
        # Load the model
        self.model = None
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load a wake word model"""
        try:
            new_model = load_model(model_path)
            if new_model:
                # Switch to evaluation mode
                new_model.eval()
                
                with self.lock:
                    self.model = new_model
                    # Clear predictions history
                    self.recent_predictions.clear()
                
                logger.info(f"Model loaded successfully from {model_path}")
                return True
            else:
                logger.error(f"Failed to load model from {model_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def set_threshold(self, threshold):
        """Update detection threshold"""
        with self.lock:
            self.threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Detection threshold set to {self.threshold}")
    
    def register_test_callback(self, callback):
        """Register callback for testing detection"""
        self.test_callback = callback
    
    def unregister_test_callback(self):
        """Unregister test callback"""
        self.test_callback = None
    
    def detect(self, features):
        """Detect wake word in audio features"""
        # First check if model is loaded
        with self.lock:
            if self.model is None:
                logger.warning("No model loaded, cannot perform detection")
                return False, 0.0
            
            current_threshold = self.threshold
        
        # Convert numpy array to torch tensor
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float()
        
        # Ensure correct input shape
        if len(features.shape) != 3:
            logger.error(f"Invalid feature shape: {features.shape}")
            return False, 0.0
        
        try:
            # Make prediction
            with torch.no_grad():
                prediction = self.model(features)
                confidence = prediction.item()
            
            # Add to recent predictions - with lock for thread safety
            with self.lock:
                self.recent_predictions.append(confidence)
                
                # Make a copy inside the lock to avoid race conditions
                recent_preds = list(self.recent_predictions)
            
            # Process predictions
            avg_confidence = sum(recent_preds) / len(recent_preds)
            
            # Determine if wake word is detected
            # Require minimum 3 frames above threshold in the window to trigger
            high_confidence_count = sum(1 for p in recent_preds if p > current_threshold)
            is_detected = (avg_confidence > current_threshold and 
                          high_confidence_count >= min(3, len(recent_preds)))
            
            # Call test callback if registered
            if is_detected and self.test_callback:
                self.test_callback(avg_confidence)
            
            return is_detected, avg_confidence
            
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            return False, 0.0