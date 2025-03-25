#!/usr/bin/env python3
"""
Wake Word Model Recovery Tool

This script helps recover trained models by:
1. Finding checkpoint files from training
2. Converting the checkpoint to a usable model
3. Creating a simple model architecture that matches the checkpoint

Usage:
    python model_recovery.py
"""
import os
import sys
import torch
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModelRecovery")

class SimpleWakeWordModel(torch.nn.Module):
    """Simplified CNN model for wake word detection"""
    
    def __init__(self, n_mfcc=13, num_frames=101):
        super(SimpleWakeWordModel, self).__init__()
        
        # A simpler architecture with fewer layers
        self.conv_layer = torch.nn.Sequential(
            torch.nn.Conv1d(n_mfcc, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=0)
        )
        
        # Calculate output size
        output_width = (num_frames - 3) // 2 + 1  # Simple calculation for MaxPool1d
        self.fc_input_size = 32 * output_width
        
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Linear(self.fc_input_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

class StandardWakeWordModel(torch.nn.Module):
    """Standard CNN model for wake word detection"""
    
    def __init__(self, n_mfcc=13, num_frames=101):
        super(StandardWakeWordModel, self).__init__()
        
        # Calculate output dimensions
        after_pool1 = (num_frames - 3) // 2 + 1
        after_pool2 = (after_pool1 - 3) // 2 + 1
        self.fc_input_size = 64 * after_pool2
        
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv1d(n_mfcc, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=0),
            
            torch.nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=0)
        )
        
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Linear(self.fc_input_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
            torch.nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

def find_latest_checkpoint():
    """Find the latest checkpoint from training"""
    # Possible checkpoint locations
    checkpoint_dirs = [
        Path.home() / ".io" / "training_diagnostics",
        Path.home() / ".io" / "models",
        Path("training_diagnostics"),
        Path("models"),
    ]
    
    # Checkpoint files to look for, in priority order
    checkpoint_files = [
        "model_best.pth",
        "model_epoch_80.pth",
        "model_epoch_60.pth",
        "model_epoch_40.pth",
        "model_epoch_20.pth",
        "model_epoch_10.pth",
    ]
    
    # Find the first existing checkpoint
    for directory in checkpoint_dirs:
        if directory.exists():
            for filename in checkpoint_files:
                checkpoint_path = directory / filename
                if checkpoint_path.exists():
                    return checkpoint_path
    
    return None

def detect_model_type(state_dict):
    """Detect model type from state dict keys"""
    keys = list(state_dict.keys())
    
    if not keys:
        return None
    
    # Check for SimpleWakeWordModel (has conv_layer)
    if any("conv_layer" in key for key in keys):
        return "simple"
    
    # Check for standard WakeWordModel (has conv_layers)
    if any("conv_layers" in key for key in keys):
        return "standard"
    
    return None

def create_matching_model(state_dict, n_mfcc=13, num_frames=101):
    """Create a model that matches the state dict architecture"""
    model_type = detect_model_type(state_dict)
    
    if model_type == "simple":
        print("Detected SimpleWakeWordModel architecture")
        return SimpleWakeWordModel(n_mfcc=n_mfcc, num_frames=num_frames)
    elif model_type == "standard":
        print("Detected StandardWakeWordModel architecture")
        return StandardWakeWordModel(n_mfcc=n_mfcc, num_frames=num_frames)
    else:
        print("Could not detect model architecture")
        return None

def recover_model():
    """Attempt to recover the latest checkpoint as a usable model"""
    # Find checkpoint
    checkpoint_path = find_latest_checkpoint()
    if not checkpoint_path:
        print("❌ No checkpoint files found")
        return False
    
    print(f"Found checkpoint: {checkpoint_path}")
    
    # Create output directory
    models_dir = Path.home() / ".io" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # New model path
    recovered_path = models_dir / "recovered_model.pth"
    
    try:
        # Load state dict
        print(f"Loading checkpoint from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        
        # Create a matching model
        model = create_matching_model(state_dict)
        if model is None:
            # Just copy the checkpoint directly
            print("Could not create matching model, copying checkpoint directly")
            torch.save(state_dict, recovered_path)
        else:
            # Load state dict into model
            model.load_state_dict(state_dict)
            
            # Save the model
            torch.save(model.state_dict(), recovered_path)
        
        print(f"✅ Model recovered to: {recovered_path}")
        
        # Create info file
        info_path = models_dir / "recovered_model_info.txt"
        with open(info_path, 'w') as f:
            f.write(f"Original checkpoint: {checkpoint_path}\n")
            f.write(f"Recovery date: {__import__('datetime').datetime.now()}\n")
            f.write(f"Model type: {detect_model_type(state_dict) or 'unknown'}\n")
        
        return True
    except Exception as e:
        print(f"❌ Error recovering model: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print(" WAKE WORD MODEL RECOVERY TOOL ")
    print("=" * 60)
    print("This tool recovers trained models from checkpoints")
    
    recover_model()
