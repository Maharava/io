"""
CNN model architecture for wake word detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger("WakeWord.Model")

class WakeWordModel(nn.Module):
    def __init__(self, n_mfcc=13, num_frames=101):
        """
        1D CNN model for wake word detection
        
        Args:
            n_mfcc: Number of MFCC coefficients
            num_frames: Number of time frames
        """
        super(WakeWordModel, self).__init__()
        
        # Input shape: [batch, 1, n_mfcc, num_frames]
        self.conv1 = nn.Conv1d(n_mfcc, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2)
        
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2)
        
        # Calculate size after pooling layers
        pooled_size = num_frames
        pooled_size = (pooled_size + 2*1 - 3) // 1 + 1  # conv1 with padding
        pooled_size = (pooled_size - 3) // 2 + 1        # pool1
        pooled_size = (pooled_size + 2*1 - 3) // 1 + 1  # conv2 with padding
        pooled_size = (pooled_size - 3) // 2 + 1        # pool2
        
        self.fc1 = nn.Linear(64 * pooled_size, 128)
        self.fc2 = nn.Linear(128, 1)
    
    def forward(self, x):
        """
        Forward pass through the model
        
        Args:
            x: Input tensor of shape [batch, n_mfcc, num_frames]
            
        Returns:
            Tensor: Model output (wake word confidence)
        """
        # Apply first conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Apply second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Flatten and apply fully connected layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        
        return x

def create_model(n_mfcc=13, num_frames=101):
    """Create a new wake word model"""
    return WakeWordModel(n_mfcc=n_mfcc, num_frames=num_frames)

def save_model(model, path):
    """Save model to disk"""
    try:
        torch.save(model.state_dict(), path)
        logger.info(f"Model saved to {path}")
        return True
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return False

def load_model(path, n_mfcc=13, num_frames=101):
    """Load model from disk"""
    try:
        model = create_model(n_mfcc=n_mfcc, num_frames=num_frames)
        model.load_state_dict(torch.load(path))
        model.eval()
        logger.info(f"Model loaded from {path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None
