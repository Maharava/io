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
    """1D CNN model for wake word detection with improved regularization"""
    
    def __init__(self, n_mfcc=26, num_frames=101):  # Doubled n_mfcc to account for delta features
        """Initialize the model with given parameters"""
        super(WakeWordModel, self).__init__()
        
        # Architecture with regularization
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv1d(n_mfcc, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),  # Add dropout for regularization
            nn.MaxPool1d(kernel_size=3, stride=2),
            
            # Second conv block
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),  # Add dropout for regularization
            nn.MaxPool1d(kernel_size=3, stride=2),
            
            # Add a third conv block for more capacity
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),  # Higher dropout in deeper layers
            nn.MaxPool1d(kernel_size=3, stride=2)
        )
        
        # Calculate output size of conv layers for FC layer
        # With kernel=3, stride=2, the formula is:
        # output_size = ((input_size - kernel_size) // stride) + 1
        
        # After first pooling: ((101 - 3) // 2) + 1 = 50
        # After second pooling: ((50 - 3) // 2) + 1 = 24
        # After third pooling: ((24 - 3) // 2) + 1 = 11
        fc_input_size = 128 * 11
        
        # Fully connected layers with dropout
        self.fc_layers = nn.Sequential(
            nn.Linear(fc_input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),  # Substantial dropout before final classification
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
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


def create_model(n_mfcc=26, num_frames=101):  # Updated default for delta features
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


def load_model(path, n_mfcc=26, num_frames=101):  # Updated default for delta features
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