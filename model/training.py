def train(self, train_loader, val_loader, num_epochs=100, patience=20, progress_callback=None):
    # ... existing code ...
    
    # Final model preparation
    if best_model is not None:
        model.load_state_dict(best_model)
        training_logger.info("Loaded best model from validation")
    
    # If we were using BCEWithLogitsLoss, create inference model with sigmoid
    if isinstance(criterion, nn.BCEWithLogitsLoss):
        training_logger.info("Creating inference model with sigmoid")
        try:
            # Create a new model with sigmoid for inference
            if self.use_simple_model:
                inference_model = SimpleWakeWordModel(n_mfcc=self.n_mfcc, num_frames=self.num_frames)
            else:
                inference_model = WakeWordModel(n_mfcc=self.n_mfcc, num_frames=self.num_frames)
            
            # Transfer the trained weights
            inference_model.load_state_dict(model.state_dict())
            model = inference_model
            training_logger.info("Successfully created inference model with sigmoid")
        except Exception as e:
            training_logger.error(f"Error creating inference model: {e}")
            training_logger.warning("Using training model for inference (without sigmoid)")
    
    # Final inference test
    training_logger.info("Testing inference on validation data")
    self.test_inference(model, val_loader)
    
    return model

def save_trained_model(self, model, path):
    """Save trained model to disk with robust error handling"""
    if model is None:
        training_logger.error("Cannot save model: model is None")
        return False
    
    try:
        # Ensure the directory exists
        path = Path(path) if not isinstance(path, Path) else path
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the model
        torch.save(model.state_dict(), path)
        training_logger.info(f"Model saved to {path}")
        
        # Save metadata
        try:
            metadata_path = path.parent / f"{path.stem}_info.txt"
            with open(metadata_path, 'w') as f:
                f.write(f"Model trained with wake word engine\n")
                f.write(f"n_mfcc: {self.n_mfcc}\n")
                f.write(f"n_fft: {self.n_fft}\n")
                f.write(f"hop_length: {self.hop_length}\n")
                f.write(f"num_frames: {self.num_frames}\n")
                f.write(f"model_type: {'simple' if self.use_simple_model else 'standard'}\n")
                f.write(f"Date: {__import__('datetime').datetime.now()}\n")
        except Exception as e:
            training_logger.warning(f"Error saving metadata: {e}")
        
        return True
    except Exception as e:
        training_logger.error(f"Error saving model: {e}")
        return False