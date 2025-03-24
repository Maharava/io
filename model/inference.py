"""
Inference engine for wake word detection
"""
import torch
import numpy as np
import logging
import collections
import threading
import time
from .architecture import load_model as load_model_function

logger = logging.getLogger("Io.Model.Inference")

class WakeWordDetector:
    """Real-time wake word detector with advanced filtering"""
    
    def __init__(self, model_path=None, threshold=0.85, window_size=5):
        """Initialize detector with given model and parameters"""
        self.model_path = model_path
        self.threshold = threshold
        # Increase window size for better smoothing
        self.window_size = window_size
        
        # Store recent predictions for smoothing with timestamps
        self.recent_predictions = collections.deque(maxlen=window_size)
        
        # Detection debouncing to prevent rapid-fire detections
        self.last_detection_time = 0
        self.detection_cooldown = 3.0  # seconds, increased for more stability
        
        # Consecutive frames counter
        self.high_confidence_streak = 0
        self.required_streak = 3  # Need consecutive high confidence frames
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Detection callback for testing and UI updates
        self.test_callback = None
        self.detection_callback = None
        
        # Load the model
        self.model = None
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load a wake word model"""
        if not model_path:
            logger.error("Model path is None, cannot load model")
            return False
            
        try:
            # Check if path exists
            import os
            if not os.path.exists(model_path):
                logger.error(f"Model file does not exist: {model_path}")
                return False
                
            # Load model from the architecture module
            new_model = load_model_function(model_path)
            
            if new_model:
                # Switch to evaluation mode
                new_model.eval()
                
                with self.lock:
                    self.model = new_model
                    # Clear predictions history
                    self.recent_predictions.clear()
                    self.high_confidence_streak = 0
                
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
        
    def register_detection_callback(self, callback):
        """Register callback for dashboard updates"""
        self.detection_callback = callback
        
    def unregister_detection_callback(self):
        """Unregister dashboard callback"""
        self.detection_callback = None
    
    def detect(self, features):
        """Detect wake word in audio features with robust filtering"""
        # Check if features is None (indicates silence)
        if features is None:
            # Reset streak counter for silence
            self.high_confidence_streak = 0
            return False, 0.0
            
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
            
            # Debug for suspicious values
            if confidence > 0.99 or confidence < 0.01:
                logger.debug(f"Extreme confidence value: {confidence:.4f}")
            
            # Add to recent predictions with timestamp - with lock for thread safety
            current_time = time.time()
            with self.lock:
                self.recent_predictions.append((confidence, current_time))
                
                # Make a copy inside the lock to avoid race conditions
                recent_preds = list(self.recent_predictions)
            
            # Process predictions with time-weighting (more recent = more important)
            if len(recent_preds) >= 3:
                # Calculate time-weighted average
                total_weight = 0
                weighted_sum = 0
                
                # Get most recent timestamp as reference
                latest_time = recent_preds[-1][1]
                
                for conf, timestamp in recent_preds:
                    # Exponential time decay weight (more recent = higher weight)
                    # Weight decreases with age of prediction
                    time_diff = latest_time - timestamp
                    weight = np.exp(-time_diff * 2)  # Decay factor
                    
                    weighted_sum += conf * weight
                    total_weight += weight
                
                avg_confidence = weighted_sum / total_weight if total_weight > 0 else 0
                
                # Count high confidence predictions in recent window
                high_conf_count = sum(1 for conf, _ in recent_preds if conf > current_threshold)
                
                # Check if current sample exceeds threshold
                if confidence > current_threshold:
                    self.high_confidence_streak += 1
                else:
                    self.high_confidence_streak = 0
                
                # Apply debouncing - only allow detection every few seconds
                time_since_last = current_time - self.last_detection_time
                can_trigger = time_since_last > self.detection_cooldown
                
                # Final detection decision with multiple criteria
                is_detected = (
                    avg_confidence > current_threshold and  # Average must exceed threshold
                    high_conf_count >= min(3, len(recent_preds)) and  # Multiple samples must be high
                    self.high_confidence_streak >= self.required_streak and  # Consecutive high samples required
                    can_trigger  # Respect cooldown period
                )
                
                # Update last detection time if triggered
                if is_detected:
                    self.last_detection_time = current_time
                    # Reset streak counter after successful detection
                    self.high_confidence_streak = 0
                    
                    # Call test callback if registered
                    if self.test_callback:
                        self.test_callback(avg_confidence)
                        
                    # Call detection callback for dashboard updates if registered
                    if self.detection_callback:
                        self.detection_callback(avg_confidence)
                    
                    # Log only when actual detection occurs
                    logger.info(f"Wake word detected with confidence: {avg_confidence:.4f}")
                
                return is_detected, avg_confidence
            else:
                # Not enough samples yet
                return False, confidence
                
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            return False, 0.0