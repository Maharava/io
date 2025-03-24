"""
CNN model architecture for wake word detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
import math
from pathlib import Path

logger = logging.getLogger("Io.Model.Architecture")

def calculate_conv_output_length(input_length, kernel_size, stride, padding=0):
    """Calculate output length after a conv/pool layer with precise PyTorch formula"""
    return math.floor((input_length + 2 * padding - kernel_size) / stride + 1)

class WakeWordModel(nn.Module):
    """1D CNN model for wake word detection"""
    
    def __init__(self, n_mfcc=13, num_frames=101):
        """Initialize the model with given parameters"""
        super(WakeWordModel, self).__init__()
        
        # Calculate exact output dimensions for each layer
        # First MaxPool: kernel=3, stride=2, padding=0
        after_pool1 = calculate_conv_output_length(num_frames, 3, 2, 0)
        # Second MaxPool: kernel=3, stride=2, padding=0
        after_pool2 = calculate_conv_output_length(after_pool1, 3, 2, 0)
        # Final flattened size
        self.fc_input_size = 64 * after_pool2
        
        logger.info(f"Model dimensions calculation: input={num_frames}, after_pool1={after_pool1}, "
                   f"after_pool2={after_pool2}, fc_input_size={self.fc_input_size}")
        
        # Simplified architecture with clear dimensions tracking
        self.conv_layers = nn.Sequential(
            # First conv block - ensure n_mfcc is used here as input channels
            nn.Conv1d(n_mfcc, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=0),
            
            # Second conv block
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=0)
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.fc_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """Forward pass through the model"""
        # Apply conv layers
        x = self.conv_layers(x)
        
        # Log shape for debugging
        batch_size, channels, width = x.shape
        logger.debug(f"Shape before flattening: {x.shape}")
        
        # Verify dimensions match what we calculated
        expected_size = self.fc_input_size // channels
        if width != expected_size:
            logger.warning(f"Dimension mismatch! Expected width: {expected_size}, got: {width}")
        
        # Flatten for FC layers
        x = x.view(x.size(0), -1)
        
        # Apply FC layers
        x = self.fc_layers(x)
        
        return x


def create_model(n_mfcc=13, num_frames=101):
    """Create a new wake word model"""
    return WakeWordModel(n_mfcc=n_mfcc, num_frames=num_frames)


def save_model(model, path):
    """Save model to disk with proper resource management"""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the model
        torch.save(model.state_dict(), path)
        logger.info(f"Model saved to {path}")
        return True
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return False


def load_model(path, n_mfcc=13, num_frames=101):
    """Load model from disk with robust error handling"""
    if not path or not os.path.exists(path):
        logger.error(f"Model file not found: {path}")
        return None
    
    try:
        # Create a new model with exact same architecture
        model = create_model(n_mfcc=n_mfcc, num_frames=num_frames)
        
        # Load state dictionary
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        
        # Check for compatibility
        expected_keys = set(model.state_dict().keys())
        loaded_keys = set(state_dict.keys())
        
        if expected_keys != loaded_keys:
            logger.error(f"Model architecture mismatch. Expected keys: {expected_keys}, got: {loaded_keys}")
            logger.error("This probably means the loaded model was trained with a different architecture.")
            return None
            
        # Check for shape compatibility in first layer
        first_layer_key = "conv_layers.0.weight"
        if first_layer_key in state_dict:
            loaded_shape = state_dict[first_layer_key].shape
            expected_shape = model.state_dict()[first_layer_key].shape
            
            if loaded_shape != expected_shape:
                logger.error(f"First layer shape mismatch. Expected: {expected_shape}, got: {loaded_shape}")
                logger.error("This means the model was trained with different MFCC parameters.")
                return None
        
        # All checks passed, load the weights
        model.load_state_dict(state_dict)
        model.eval()  # Set to evaluation mode
        
        logger.info(f"Model loaded successfully from {path}")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None