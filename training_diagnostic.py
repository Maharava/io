"""
Wake Word Model Training Diagnostic

This script diagnoses issues with the wake word model training process
and provides a modified trainer that addresses early stopping problems.
"""
import os
import sys
from pathlib import Path
import torch
import numpy as np
import logging
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("WakeWordDiagnostic")

# Add parent directory to path to import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try importing from the wake word package
try:
    from model.training import WakeWordTrainer
    from model.architecture import SimpleWakeWordModel, WakeWordModel
    logger.info("Successfully imported wake word modules")
except ImportError as e:
    logger.error(f"Error importing wake word modules: {e}")
    logger.error("Make sure you're running this script from the project root directory")
    sys.exit(1)

class EnhancedWakeWordTrainer(WakeWordTrainer):
    """Enhanced trainer with fixes for early stopping and learning issues"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info("Initializing enhanced trainer with improved parameters")
        
    def train(self, train_loader, val_loader, num_epochs=100, patience=20, progress_callback=None):
        """
        Train the wake word model with enhanced diagnostics and fixed early stopping.
        Increased patience and improved optimization parameters.
        """
        # Get feature shape from a sample batch
        for features, _ in train_loader:
            input_shape = features.shape
            logger.info(f"Input feature shape: {input_shape}")
            break
        
        # Create model - either standard or simple
        if self.use_simple_model:
            logger.info("Using simple model architecture")
            model = SimpleWakeWordModel(n_mfcc=self.n_mfcc, num_frames=self.num_frames)
        else:
            logger.info("Using standard model architecture")
            model = WakeWordModel(n_mfcc=self.n_mfcc, num_frames=self.num_frames)
        
        model.to(self.device)
        
        # Log model details
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model created with {num_params} trainable parameters")
        logger.info(f"Model structure:\n{model}")
        
        # Define loss function and optimizer with better parameters
        criterion = torch.nn.BCELoss()
        
        # Lower learning rate for more stable training
        learning_rate = 0.001  # Decreased from 0.005
        weight_decay = 1e-5    # L2 regularization to prevent overfitting
        
        # Use AdamW instead of Adam for better weight decay handling
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # Add learning rate scheduler with gentler reduction
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6
        )
        
        # Track metrics
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        # Early stopping with increased patience
        best_val_loss = float('inf')
        best_model = None
        best_epoch = 0
        patience_counter = 0
        patience_threshold = patience  # Increased from default 10
        
        # Training loop
        import time
        start_time = time.time()
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Training phase
            model.train()
            train_loss = 0.0
            correct_train = 0
            total_train = 0
            
            # Create progress logger
            batch_count = len(train_loader)
            
            # Track gradients for diagnostics in first few epochs
            if epoch < 3:
                grad_norms = []
            
            for batch_idx, (features, labels) in enumerate(train_loader):
                features, labels = features.to(self.device), labels.to(self.device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                
                # Track gradient norms for diagnostics
                if epoch < 3:
                    total_norm = 0
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5
                    grad_norms.append(total_norm)
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                predicted = (outputs > 0.5).float()
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
                
                # Print batch progress in early epochs
                if epoch < 3 and batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch+1}, Batch {batch_idx}/{batch_count}, Loss: {loss.item():.4f}")
            
            # Print gradient diagnostics for early epochs
            if epoch < 3:
                avg_grad = sum(grad_norms) / len(grad_norms) if grad_norms else 0
                max_grad = max(grad_norms) if grad_norms else 0
                logger.info(f"Epoch {epoch+1} gradient stats - Avg: {avg_grad:.6f}, Max: {max_grad:.6f}")
            
            train_loss = train_loss / len(train_loader)
            train_acc = correct_train / total_train
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            
            # Track predictions for analysis
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for features, labels in val_loader:
                    features, labels = features.to(self.device), labels.to(self.device)
                    
                    # Forward pass
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                    
                    # Track predictions and labels
                    all_preds.extend(outputs.squeeze().cpu().numpy())
                    all_labels.extend(labels.squeeze().cpu().numpy())
                    
                    # Statistics
                    val_loss += loss.item()
                    predicted = (outputs > 0.5).float()
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()
            
            val_loss = val_loss / len(val_loader)
            val_acc = correct_val / total_val
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            # Update learning rate
            prev_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Calculate time taken
            epoch_time = time.time() - epoch_start
            
            # Log detailed progress
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.2f}s) - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                f"LR: {current_lr:.6f}"
            )
            
            # Log prediction distribution for monitoring
            if epoch % 5 == 0:
                # Convert to numpy arrays for analysis
                preds_array = np.array(all_preds)
                labels_array = np.array(all_labels)
                
                # Calculate mean prediction by class
                pos_preds = preds_array[labels_array == 1]
                neg_preds = preds_array[labels_array == 0]
                
                logger.info(f"Prediction stats - Positive: mean={np.mean(pos_preds):.4f}, "
                           f"std={np.std(pos_preds):.4f}, Negative: mean={np.mean(neg_preds):.4f}, "
                           f"std={np.std(neg_preds):.4f}")
            
            # Update progress callback
            if progress_callback:
                progress_pct = int(30 + (epoch + 1) / num_epochs * 60)
                progress_callback(
                    f"Training: Epoch {epoch+1}/{num_epochs}, Accuracy: {val_acc:.2f}", 
                    progress_pct
                )
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(
                    self.diagnostic_dir, 
                    f"model_epoch_{epoch+1}.pth"
                )
                torch.save(model.state_dict(), checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            # Check for improvement with numerical stability threshold
            improvement_threshold = 1e-5  # Small threshold to account for floating point issues
            if best_val_loss - val_loss > improvement_threshold:
                logger.info(f"Validation loss improved from {best_val_loss:.6f} to {val_loss:.6f}")
                best_val_loss = val_loss
                best_model = model.state_dict().copy()
                best_epoch = epoch + 1
                patience_counter = 0
                
                # Save best model
                torch.save(
                    best_model, 
                    os.path.join(self.diagnostic_dir, "model_best.pth")
                )
            else:
                patience_counter += 1
                logger.info(f"No improvement for {patience_counter} epochs. "
                          f"(Best: {best_val_loss:.6f}, Current: {val_loss:.6f})")
                
                if patience_counter >= patience_threshold:
                    logger.info(f"Early stopping at epoch {epoch+1} after {patience_counter} epochs without improvement")
                    break
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")
        logger.info(f"Best model from epoch {best_epoch} with val_loss={best_val_loss:.6f}")
        
        # Plot training curves
        epochs = range(1, len(train_losses) + 1)
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accs, 'b-', label='Training Accuracy')
        plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.diagnostic_dir, "training_curves.png"))
        plt.close()
        
        # Load best model for final evaluation
        model.load_state_dict(best_model)
        
        return model

def fix_training_issues():
    """
    Run diagnostics and fix training issues
    """
    logger.info("Starting wake word training diagnostics")
    
    # Create output directory
    output_dir = Path.home() / ".io" / "diagnostic_fix"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Path to training and validation data
    training_dir = Path.home() / ".io" / "training_data"
    wake_word_dir = training_dir / "wake_word"
    negative_dir = training_dir / "negative"
    
    # Verify directories exist
    for dir_path in [training_dir, wake_word_dir, negative_dir]:
        if not dir_path.exists():
            logger.error(f"Directory not found: {dir_path}")
            return
    
    # Count files
    wake_files = list(wake_word_dir.glob("*.wav"))
    neg_files = list(negative_dir.glob("*.wav"))
    logger.info(f"Found {len(wake_files)} wake word files and {len(neg_files)} negative files")
    
    # Create the enhanced trainer
    trainer = EnhancedWakeWordTrainer()
    logger.info("Preparing data...")
    
    # Prepare data
    train_loader, val_loader = trainer.prepare_data(wake_files, neg_files)
    
    # Train the model
    logger.info("Starting enhanced training...")
    model = trainer.train(train_loader, val_loader)
    
    # Save the model
    model_path = output_dir / "enhanced_model.pth"
    trainer.save_trained_model(model, str(model_path))
    logger.info(f"Enhanced model saved to {model_path}")
    
    return model_path

if __name__ == "__main__":
    try:
        fixed_model_path = fix_training_issues()
        if fixed_model_path:
            print(f"\nEnhanced wake word model saved to: {fixed_model_path}")
            print("You can now use this model in the wake word detector.")
    except Exception as e:
        logger.exception("Error during diagnostic fix")
        print(f"\nError: {str(e)}")
        print("Check the logs for more details.")
