"""
CNN model architecture for wake word detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
from pathlib import Path

logger = logging.getLogger("Io.Model.Architecture")

class WakeWordModel(nn.Module):
    """1D CNN model for wake word detection"""
    
    def __init__(self, n_mfcc=13, num_frames=101):
        """Initialize the model with given parameters"""
        super(WakeWordModel, self).__init__()
        
        # Simplified architecture with clear dimensions tracking
        self.conv_layers = nn.Sequential(
            # First conv block - ensure n_mfcc is used here as input channels
            nn.Conv1d(n_mfcc, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),
            
            # Second conv block
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )
        
        # Calculate output size of conv layers for FC layer
        # Input: [batch, n_mfcc, num_frames]
        # After conv1 + pool1: [batch, 64, num_frames//2]
        # After conv2 + pool2: [batch, 64, num_frames//4]
        fc_input_size = 64 * (num_frames // 4)
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(fc_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """Forward pass through the model"""
        # Apply conv layers
        x = self.conv_layers(x)
        
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
        model = create_model(n_mfcc=n_mfcc, num_frames=num_frames)
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        model.eval()  # Set to evaluation mode
        logger.info(f"Model loaded from {path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None