#!/usr/bin/env python3
"""
Wake Word Model Rescue Utility

This script attempts to rescue trained wake word models from diagnostic checkpoints.
Use this if your model training completed but the model failed to save.
"""
import os
import sys
import torch
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModelRescue")

def rescue_model():
    """Attempt to rescue the latest trained model"""
    try:
        # Find the diagnostic directory
        diagnostics_dir = Path.home() / ".io" / "training_diagnostics"
        if not diagnostics_dir.exists():
            logger.error(f"Diagnostics directory not found: {diagnostics_dir}")
            return False
        
        # Look for model checkpoints in priority order
        checkpoint_files = [
            "model_best.pth",
            "model_epoch_80.pth",
            "model_epoch_60.pth", 
            "model_epoch_40.pth",
            "model_epoch_20.pth"
        ]
        
        checkpoint_path = None
        for filename in checkpoint_files:
            path = diagnostics_dir / filename
            if path.exists():
                checkpoint_path = path
                logger.info(f"Found checkpoint: {path}")
                break
        
        if checkpoint_path is None:
            logger.error("No checkpoint files found")
            return False
        
        # Create models directory
        models_dir = Path.home() / ".io" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Rescued model path
        rescued_path = models_dir / "rescued_model.pth"
        
        # Load checkpoint and save to rescued path
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        torch.save(state_dict, rescued_path)
        
        logger.info(f"Model rescued to {rescued_path}")
        print(f"\nSUCCESS: Model rescued to {rescued_path}")
        print("You can now use this model in your wake word detector")
        
        return True
    except Exception as e:
        logger.error(f"Error rescuing model: {e}")
        print(f"\nERROR: Could not rescue model: {e}")
        return False

if __name__ == "__main__":
    print("Wake Word Model Rescue Utility")
    print("Attempting to rescue your trained model from checkpoints...")
    rescue_model()
