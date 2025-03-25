#!/usr/bin/env python3
"""
Wake Word Model Recovery Tool

This script helps you recover trained models that failed to save properly.
It searches for checkpoint files and creates a usable model.

Usage:
    python recover_model.py
"""
import os
import sys
import torch
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModelRecovery")

def recover_model():
    """Attempt to recover the latest trained model"""
    # Find the diagnostic directory
    diagnostics_dir = Path.home() / ".io" / "training_diagnostics"
    if not diagnostics_dir.exists():
        print(f"❌ Diagnostic directory not found: {diagnostics_dir}")
        return False
    
    # Look for model checkpoints in priority order
    checkpoint_files = [
        "model_best.pth",
        "model_epoch_80.pth",
        "model_epoch_60.pth", 
        "model_epoch_40.pth",
        "model_epoch_20.pth",
        "model_epoch_10.pth",
    ]
    
    # Find the first existing checkpoint
    checkpoint_path = None
    for filename in checkpoint_files:
        path = diagnostics_dir / filename
        if path.exists():
            checkpoint_path = path
            print(f"Found checkpoint: {path}")
            break
    
    if checkpoint_path is None:
        print("❌ No checkpoint files found in diagnostic directory")
        return False
    
    # Create models directory
    models_dir = Path.home() / ".io" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Create the recovery destination
    recovered_path = models_dir / "recovered_model.pth"
    
    # Try to load and save the checkpoint
    try:
        print(f"Loading checkpoint from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        
        print(f"Saving recovered model to {recovered_path}...")
        torch.save(state_dict, recovered_path)
        
        print(f"✅ SUCCESS: Model recovered to {recovered_path}")
        print("You can now use this model in your wake word detector")
        
        # Create a simple note about the recovery
        note_path = models_dir / "recovered_model_info.txt"
        with open(note_path, 'w') as f:
            f.write(f"Model recovered from checkpoint: {checkpoint_path.name}\n")
            f.write(f"Recovery date: {__import__('datetime').datetime.now()}\n")
        
        return True
    except Exception as e:
        print(f"❌ Error recovering model: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print(" WAKE WORD MODEL RECOVERY TOOL ")
    print("=" * 60)
    print("This tool will try to recover trained models from checkpoints")
    
    recover_model()
