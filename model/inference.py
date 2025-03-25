"""
Diagnostic version of the inference engine with enhanced logging
"""
import torch
import numpy as np
import logging
import collections
import threading
import time
import os
from .architecture import load_model as load_model_function

# Create a special diagnostic logger
diagnostic_logger = logging.getLogger("Io.Diagnostic")
diagnostic_logger.setLevel(logging.INFO)

# Add a file handler to log diagnostics to a separate file
diagnostic_file = os.path.join(os.path.expanduser("~"), ".io", "diagnostic.log")
file_handler = logging.FileHandler(diagnostic_file, mode='w')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
diagnostic_logger.addHandler(file_handler)

diagnostic_logger.info("----- STARTING DIAGNOSTIC LOGGING -----")

class WakeWordDetector:
    """Real-time wake word detector with diagnostic capabilities"""
    
    def __init__(self, model_path=None, threshold=0.85, window_size=5, debug_mode=True):
        """Initialize detector with given model and parameters"""
        self.model_path = model_path
        self.threshold = threshold
        self.window_size = window_size
        self.debug_mode = debug_mode
        
        # Store recent predictions for smoothing with timestamps
        self.recent_predictions = collections.deque(maxlen=window_size)
        
        # Detection debouncing to prevent rapid-fire detections
        self.last_detection_time = 0
        self.detection_cooldown = 2.0  # seconds
        
        # Consecutive frames counter
        self.high_confidence_streak = 0
        self.required_streak = 2
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Detection callback for testing and UI updates
        self.test_callback = None
        self.detection_callback = None
        
        # Logging counters
        self.prediction_count = 0
        self.last_log_time = time.time()
        self.log_interval = 5.0  # Log summary every 5 seconds
        
        # Last few confidence scores for debugging
        self.last_confidences = collections.deque(maxlen=20)
        
        # Load the model
        self.model = None
        if model_path:
            self.load_model(model_path)
        
        # Log initialization
        diagnostic_logger.info(f"Initialized detector with threshold={threshold}, window_size={window_size}")
    
    def load_model(self, model_path):
        """Load a wake word model"""
        if not model_path:
            diagnostic_logger.error("Model path is None, cannot load model")
            return False
            
        try:
            # Check if path exists
            if not os.path.exists(model_path):
                diagnostic_logger.error(f"Model file does not exist: {model_path}")
                return False
                
            # Log model loading attempt
            diagnostic_logger.info(f"Attempting to load model from: {model_path}")
            
            # Load model, specifying n_mfcc=13
            new_model = load_model_function(model_path, n_mfcc=13, num_frames=101)
            
            if new_model:
                # Switch to evaluation mode
                new_model.eval()
                
                # Get model parameter count for debugging
                param_count = sum(p.numel() for p in new_model.parameters())
                
                with self.lock:
                    self.model = new_model
                    self.recent_predictions.clear()
                    self.high_confidence_streak = 0
                
                diagnostic_logger.info(f"Model loaded successfully! Parameters: {param_count}")
                
                # Print model structure for debugging
                diagnostic_logger.info(f"Model structure:\n{new_model}")
                return True
            else:
                diagnostic_logger.error("Failed to load model (returned None)")
                return False
                
        except Exception as e:
            diagnostic_logger.error(f"Error loading model: {str(e)}", exc_info=True)
            return False
    
    def set_threshold(self, threshold):
        """Update detection threshold"""
        with self.lock:
            self.threshold = max(0.0, min(1.0, threshold))
        diagnostic_logger.info(f"Detection threshold set to {self.threshold}")
    
    def register_test_callback(self, callback):
        """Register callback for testing detection"""
        self.test_callback = callback
    
    def unregister_test_callback(self):
        """Unregister test callback"""
        self.test_callback = None
        
    def register_detection_callback(self, callback):
        """Register callback for dashboard updates"""
        self.detection_callback = callback
        diagnostic_logger.info("Detection callback registered")
        
    def unregister_detection_callback(self):
        """Unregister dashboard callback"""
        self.detection_callback = None
        diagnostic_logger.info("Detection callback unregistered")
    
    def detect(self, features):
        """Detect wake word in audio features with robust filtering"""
        # Debugging time tracker
        start_time = time.time()
        
        # Check if features is None (indicates silence)
        if features is None:
            self.high_confidence_streak = 0
            if self.debug_mode and self.prediction_count % 50 == 0:
                diagnostic_logger.debug("Silent frame detected (None features)")
            return False, 0.0
            
        # First check if model is loaded
        with self.lock:
            if self.model is None:
                if self.prediction_count % 100 == 0:
                    diagnostic_logger.warning("No model loaded, cannot perform detection")
                return False, 0.0
            
            current_threshold = self.threshold
        
        # Convert numpy array to torch tensor
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float()
        
        # Validate feature dimensions
        if len(features.shape) != 3:
            if self.prediction_count % 100 == 0:
                diagnostic_logger.error(f"Invalid feature shape: {features.shape}")
            return False, 0.0
            
        # Log input shape occasionally
        if self.prediction_count % 500 == 0:
            diagnostic_logger.info(f"Input feature shape: {features.shape}")
            
            # Analyze the audio features for debugging
            if isinstance(features, torch.Tensor):
                mean = float(features.mean().item())
                std = float(features.std().item())
                min_val = float(features.min().item())
                max_val = float(features.max().item())
                diagnostic_logger.info(f"Feature stats: mean={mean:.3f}, std={std:.3f}, min={min_val:.3f}, max={max_val:.3f}")
        
        try:
            # Make prediction
            with torch.no_grad():
                outputs = self.model(features)
                confidence = outputs.item()
            
            # Store confidence for diagnostics
            self.last_confidences.append(confidence)
            
            # Log confidence periodically
            self.prediction_count += 1
            current_time = time.time()
            
            # Log a summary periodically
            if current_time - self.last_log_time > self.log_interval:
                if len(self.last_confidences) > 0:
                    avg_conf = sum(self.last_confidences) / len(self.last_confidences)
                    max_conf = max(self.last_confidences) if self.last_confidences else 0
                    diagnostic_logger.info(f"Recent confidence stats: avg={avg_conf:.4f}, max={max_conf:.4f}, threshold={current_threshold:.4f}")
                    diagnostic_logger.info(f"Last few confidences: {[f'{c:.3f}' for c in self.last_confidences]}")
                    diagnostic_logger.info(f"High confidence streak: {self.high_confidence_streak}")
                self.last_log_time = current_time
                self.last_confidences.clear()
            
            # Process high confidence scores
            if confidence > current_threshold:
                self.high_confidence_streak += 1
                diagnostic_logger.info(f"High confidence detected: {confidence:.4f}, streak={self.high_confidence_streak}")
            else:
                self.high_confidence_streak = 0
            
            # Add to recent predictions
            current_time = time.time()
            with self.lock:
                self.recent_predictions.append((confidence, current_time))
                recent_preds = list(self.recent_predictions)
            
            # Process predictions with time-weighting
            if len(recent_preds) >= 3:
                # Calculate time-weighted average
                total_weight = 0
                weighted_sum = 0
                
                latest_time = recent_preds[-1][1]
                
                for conf, timestamp in recent_preds:
                    time_diff = latest_time - timestamp
                    weight = np.exp(-time_diff * 2)
                    
                    weighted_sum += conf * weight
                    total_weight += weight
                
                avg_confidence = weighted_sum / total_weight if total_weight > 0 else 0
                
                # Count high confidence predictions
                high_conf_count = sum(1 for conf, _ in recent_preds if conf > current_threshold)
                
                # Check debounce time
                time_since_last = current_time - self.last_detection_time
                can_trigger = time_since_last > self.detection_cooldown
                
                # Final detection decision
                is_detected = (
                    avg_confidence > current_threshold and
                    high_conf_count >= min(3, len(recent_preds)) and
                    self.high_confidence_streak >= self.required_streak and
                    can_trigger
                )
                
                # Log detection criteria when close to threshold
                if avg_confidence > current_threshold * 0.8:
                    diagnostic_logger.info(
                        f"Detection check: avg_conf={avg_confidence:.4f}, "
                        f"high_count={high_conf_count}, streak={self.high_confidence_streak}, "
                        f"can_trigger={can_trigger}, DETECTED={is_detected}"
                    )
                
                # Update last detection time if triggered
                if is_detected:
                    self.last_detection_time = current_time
                    self.high_confidence_streak = 0
                    
                    # Call callbacks
                    if self.test_callback:
                        self.test_callback(avg_confidence)
                    if self.detection_callback:
                        self.detection_callback(avg_confidence)
                    
                    diagnostic_logger.info(f"WAKE WORD DETECTED! Confidence: {avg_confidence:.4f}")
                
                # Log processing time occasionally
                if self.prediction_count % 100 == 0:
                    process_time = time.time() - start_time
                    diagnostic_logger.debug(f"Detection processing time: {process_time*1000:.2f}ms")
                
                return is_detected, avg_confidence
            else:
                return False, confidence
                
        except Exception as e:
            diagnostic_logger.error(f"Error during inference: {str(e)}", exc_info=True)
            return False, 0.0