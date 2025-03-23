"""
Inference engine for wake word detection
"""
import torch
import numpy as np
import logging
import collections
from .architecture import load_model

logger = logging.getLogger("WakeWord.Inference")

class WakeWordDetector:
    def __init__(self, model_path, threshold=0.85, window_size=3):
        """
        Real-time wake word detector
        
        Args:
            model_path: Path to trained model
            threshold: Detection threshold (0.0 to 1.0)
            window_size: Number of frames to consider for smoothing
        """
        self.model_path = model_path
        self.threshold = threshold
        self.window_size = window_size
        
        # Store recent predictions for smoothing
        self.recent_predictions = collections.deque(maxlen=window_size)
        
        # Load the model
        self.model = None
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load a wake word model"""
        self.model = load_model(model_path)
        if self.model:
            # Switch to evaluation mode
            self.model.eval()
            # Clear predictions history
            self.recent_predictions.clear()
            logger.info(f"Model loaded successfully from {model_path}")
            return True
        else:
            logger.error(f"Failed to load model from {model_path}")
            return False
    
    def set_threshold(self, threshold):
        """Update detection threshold"""
        self.threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Detection threshold set to {self.threshold}")
    
    def detect(self, features):
        """
        Detect wake word in audio features
        
        Args:
            features: MFCC features of shape [1, n_mfcc, num_frames]
            
        Returns:
            tuple: (detection_result, confidence)
        """
        if self.model is None:
            logger.warning("No model loaded, cannot perform detection")
            return False, 0.0
        
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
            
            # Add to recent predictions
            self.recent_predictions.append(confidence)
            
            # Apply smoothing by averaging recent predictions
            avg_confidence = sum(self.recent_predictions) / len(self.recent_predictions)
            
            # Determine if wake word is detected
            # Require minimum 3 frames above threshold in the window to trigger
            high_confidence_count = sum(1 for p in self.recent_predictions if p > self.threshold)
            is_detected = (avg_confidence > self.threshold and 
                          high_confidence_count >= min(3, len(self.recent_predictions)))
            
            return is_detected, avg_confidence
            
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            return False, 0.0
