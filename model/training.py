"""
Training module for Io wake word detection
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import logging
import time
from pathlib import Path
import random
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

from .architecture import SimpleWakeWordModel, WakeWordModel, save_model

# Set up logging
training_logger = logging.getLogger("Io.Model.Training")

class WakeWordDataset(Dataset):
    """Dataset for wake word training"""
    
    def __init__(self, wake_word_files, negative_files, n_mfcc=13, n_fft=2048, hop_length=160):
        """Initialize dataset with file paths"""
        self.wake_word_files = wake_word_files
        self.negative_files = negative_files
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = 16000
        
        # All files
        self.files = wake_word_files + negative_files
        
        # Labels (1 for wake word, 0 for negative)
        self.labels = [1] * len(wake_word_files) + [0] * len(negative_files)
    
    def __len__(self):
        """Get dataset size"""
        return len(self.files)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        file_path = self.files[idx]
        label = self.labels[idx]
        
        try:
            # Load audio file
            y, sr = librosa.load(file_path, sr=self.sample_rate)
            
            # Pad or truncate to 1 second
            if len(y) < self.sample_rate:
                y = np.pad(y, (0, self.sample_rate - len(y)))
            elif len(y) > self.sample_rate:
                y = y[:self.sample_rate]
            
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(
                y=y, 
                sr=sr, 
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft, 
                hop_length=self.hop_length
            )
            
            # Ensure consistent dimensions
            if mfccs.shape[1] > 101:
                mfccs = mfccs[:, :101]
            elif mfccs.shape[1] < 101:
                pad_width = 101 - mfccs.shape[1]
                mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)))
            
            # Normalize features
            mfccs = (mfccs - np.mean(mfccs)) / (np.std(mfccs) + 1e-8)
            
            return torch.from_numpy(mfccs).float(), torch.tensor([label]).float()
            
        except Exception as e:
            training_logger.error(f"Error loading {file_path}: {e}")
            # Return zeros for failed files
            mfccs = np.zeros((self.n_mfcc, 101))
            return torch.from_numpy(mfccs).float(), torch.tensor([0]).float()


class WakeWordTrainer:
    """Train wake word detection models"""
    
    def __init__(self, n_mfcc=13, n_fft=2048, hop_length=160, num_frames=101, use_simple_model=False):
        """Initialize trainer with feature parameters"""
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_frames = num_frames
        self.use_simple_model = use_simple_model
        
        # Create diagnostics directory
        self.diagnostics_dir = Path.home() / ".io" / "training_diagnostics"
        self.diagnostics_dir.mkdir(parents=True, exist_ok=True)
        
        training_logger.info(
            f"Initialized trainer with n_mfcc={n_mfcc}, n_fft={n_fft}, "
            f"hop_length={hop_length}, num_frames={num_frames}, "
            f"use_simple_model={use_simple_model}"
        )
    
    def prepare_data(self, wake_word_files, negative_files, val_split=0.2, batch_size=32):
        """Prepare training and validation data loaders"""
        if not wake_word_files or not negative_files:
            training_logger.error("No training files provided")
            return None, None
        
        # Convert Path objects to strings if needed
        wake_word_files = [str(f) for f in wake_word_files]
        negative_files = [str(f) for f in negative_files]
        
        # Shuffle files
        random.seed(42)  # For reproducibility
        random.shuffle(wake_word_files)
        random.shuffle(negative_files)
        
        # Split into training and validation
        n_wake_val = max(1, int(len(wake_word_files) * val_split))
        n_neg_val = max(1, int(len(negative_files) * val_split))
        
        train_wake = wake_word_files[:-n_wake_val]
        val_wake = wake_word_files[-n_wake_val:]
        
        train_neg = negative_files[:-n_neg_val]
        val_neg = negative_files[-n_neg_val:]
        
        # Create datasets
        train_dataset = WakeWordDataset(
            train_wake, train_neg, 
            n_mfcc=self.n_mfcc, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length
        )
        
        val_dataset = WakeWordDataset(
            val_wake, val_neg, 
            n_mfcc=self.n_mfcc, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0
        )
        
        training_logger.info(
            f"Data prepared: {len(train_wake)} wake word and {len(train_neg)} "
            f"negative samples for training, {len(val_wake)} wake word and "
            f"{len(val_neg)} negative samples for validation"
        )
        
        return train_loader, val_loader
    
    def train(self, train_loader, val_loader, num_epochs=100, patience=20, progress_callback=None):
        """Train the wake word model"""
        training_logger.info("Starting model training")
        
        # Create model
        if self.use_simple_model:
            model = SimpleWakeWordModel(n_mfcc=self.n_mfcc, num_frames=self.num_frames)
            training_logger.info("Using SimpleWakeWordModel")
        else:
            model = WakeWordModel(n_mfcc=self.n_mfcc, num_frames=self.num_frames)
            training_logger.info("Using WakeWordModel")
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training variables
        best_val_loss = float('inf')
        best_model = None
        patience_counter = 0
        
        # Training loop
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            # Update progress
            if progress_callback:
                progress_value = 30 + int(50 * epoch / num_epochs)
                progress_callback(f"Training epoch {epoch+1}/{num_epochs}...", progress_value)
            
            # Training loop
            for inputs, labels in train_loader:
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            train_loss = running_loss / len(train_loader.dataset)
            train_acc = correct / total
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * inputs.size(0)
                    predicted = (outputs > 0.5).float()
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            val_loss = val_loss / len(val_loader.dataset)
            val_acc = val_correct / val_total
            
            # Calculate precision, recall, f1
            precision, recall, f1, _ = precision_recall_fscore_support(
                np.array(all_labels).flatten(),
                np.array(all_preds).flatten(),
                average='binary',
                zero_division=0
            )
            
            # Log metrics
            training_logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, "
                f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
            )
            
            # Save checkpoint
            if epoch % 20 == 0:
                checkpoint_path = self.diagnostics_dir / f"model_epoch_{epoch}.pth"
                torch.save(model.state_dict(), checkpoint_path)
                training_logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model.state_dict().copy()
                patience_counter = 0
                
                # Save best model
                best_model_path = self.diagnostics_dir / "model_best.pth"
                torch.save(best_model, best_model_path)
                training_logger.info(f"New best model saved! Val Loss: {val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    training_logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
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
    
    def test_inference(self, model, val_loader):
        """Test inference on validation data"""
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                predicted = (outputs > 0.5).float()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            np.array(all_labels).flatten(),
            np.array(all_preds).flatten(),
            average='binary',
            zero_division=0
        )
        
        training_logger.info(
            f"Final model performance: "
            f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
        )
        
        return precision, recall, f1
    
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