"""
Training pipeline for wake word detection model
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import librosa
import logging
import math
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

logger = logging.getLogger("Io.Model.Training")

# Define the model architecture directly in this file to ensure consistency
class WakeWordModel(nn.Module):
    """1D CNN model for wake word detection"""
    
    def __init__(self, n_mfcc=13, num_frames=101):
        """Initialize the model with given parameters"""
        super(WakeWordModel, self).__init__()
        
        # Simplified architecture with clear dimensions tracking
        self.conv_layers = nn.Sequential(
            # First conv block - using n_mfcc as input channels
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
        # After first MaxPool: frame_count is approximately ceil(num_frames/2)
        # After second MaxPool: frame_count is approximately ceil(ceil(num_frames/2)/2)
        frame_count_after_first = math.ceil(num_frames / 2)
        frame_count_after_second = math.ceil(frame_count_after_first / 2)
        fc_input_size = 64 * frame_count_after_second
        
        logger.info(f"FC input size calculated as: {fc_input_size}")
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(fc_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """Forward pass through the model"""
        # Apply conv layers and capture intermediate shape
        x = self.conv_layers(x)
        
        # Log the actual shape before flattening (helps with debugging)
        batch_size, channels, width = x.shape
        logger.info(f"Shape before flattening: {x.shape}, flattened size: {channels * width}")
        
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

class WakeWordDataset(Dataset):
    """Dataset for wake word training"""
    
    def __init__(self, features, labels):
        """Initialize dataset with features and labels"""
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = torch.from_numpy(self.features[idx]).float()
        label = torch.tensor([self.labels[idx]]).float()
        return feature, label


class WakeWordTrainer:
    """Trainer for wake word models"""
    
    def __init__(self, sample_rate=16000, n_mfcc=13, n_fft=2048, 
                 hop_length=160, num_frames=101):
        """Initialize trainer with given parameters"""
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_frames = num_frames
        
        # Use CPU for better compatibility
        self.device = torch.device("cpu")
        logger.info(f"Using device: {self.device}")
    
    def extract_features(self, audio_files, augment=False):
        """Extract MFCC features from audio files with optional augmentation"""
        features = []
        
        for file_path in audio_files:
            try:
                # Load audio file
                y, sr = librosa.load(file_path, sr=self.sample_rate)
                
                # Basic data augmentations if requested
                versions = [y]
                if augment:
                    # Pitch shift (up and down)
                    y_pitch_up = librosa.effects.pitch_shift(y, sr=sr, n_steps=1)
                    y_pitch_down = librosa.effects.pitch_shift(y, sr=sr, n_steps=-1)
                    versions.extend([y_pitch_up, y_pitch_down])
                    
                    # Time stretch (faster and slower)
                    y_faster = librosa.effects.time_stretch(y, rate=1.1)
                    y_slower = librosa.effects.time_stretch(y, rate=0.9)
                    versions.extend([y_faster, y_slower])
                    
                    # Volume variation
                    y_louder = y * 1.2
                    y_softer = y * 0.8
                    versions.extend([y_louder, y_softer])
                
                # Process each version
                for audio in versions:
                    # Ensure minimum length
                    if len(audio) < self.sample_rate:
                        audio = np.pad(audio, (0, self.sample_rate - len(audio)))
                    
                    # For long recordings, extract multiple segments
                    if len(audio) > self.sample_rate * 2:
                        # Extract segments of 1 second with 0.5 second overlap
                        segment_length = self.sample_rate
                        hop = self.sample_rate // 2
                        
                        for start in range(0, len(audio) - segment_length, hop):
                            segment = audio[start:start + segment_length]
                            mfcc = self._extract_mfcc(segment)
                            if mfcc is not None:
                                features.append(mfcc)
                    else:
                        # Process the whole file
                        mfcc = self._extract_mfcc(audio)
                        if mfcc is not None:
                            features.append(mfcc)
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
        
        return features
    
    def _extract_mfcc(self, audio):
        """Extract MFCC features from audio segment"""
        try:
            # Extract MFCCs - explicitly using self.n_mfcc
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,  # This should be 13
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # Ensure consistent length
            if mfcc.shape[1] < self.num_frames:
                # Pad if too short
                pad_width = self.num_frames - mfcc.shape[1]
                mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)))
            elif mfcc.shape[1] > self.num_frames:
                # Truncate if too long
                mfcc = mfcc[:, :self.num_frames]
            
            # Log the MFCC shape to help with debugging
            logger.debug(f"Extracted MFCC shape: {mfcc.shape}")
            
            return mfcc
            
        except Exception as e:
            logger.error(f"Error extracting MFCCs: {e}")
            return None
    
    def prepare_data(self, wake_word_files, negative_files, test_size=0.2):
        """Prepare training and validation datasets"""
        logger.info("Extracting features from wake word samples")
        wake_word_features = self.extract_features(wake_word_files, augment=True)
        
        logger.info("Extracting features from negative samples")
        negative_features = self.extract_features(negative_files, augment=False)
        
        # Create labels (1 for wake word, 0 for negative)
        wake_word_labels = np.ones(len(wake_word_features))
        negative_labels = np.zeros(len(negative_features))
        
        # Combine features and labels
        features = wake_word_features + negative_features
        labels = np.concatenate([wake_word_labels, negative_labels])
        
        # Split into train/validation sets
        features_train, features_val, labels_train, labels_val = train_test_split(
            features, labels, test_size=test_size, stratify=labels, random_state=42
        )
        
        # Create datasets
        train_dataset = WakeWordDataset(features_train, labels_train)
        val_dataset = WakeWordDataset(features_val, labels_val)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=32, shuffle=True, drop_last=False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=32, shuffle=False, drop_last=False
        )
        
        # Get feature dimensions for debugging
        if len(features) > 0:
            sample_feature = features[0]
            logger.info(f"Sample feature shape: {sample_feature.shape}")
        
        logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
        return train_loader, val_loader
    
    def train(self, train_loader, val_loader, num_epochs=50, patience=5, progress_callback=None):
        """Train the wake word model"""
        # Sample the first batch to get feature dimensions
        for features, _ in train_loader:
            logger.info(f"Actual input shape: {features.shape}")
            break
            
        # Create model and move to device - using our local create_model
        model = create_model(n_mfcc=self.n_mfcc, num_frames=self.num_frames)
        model.to(self.device)
        
        # Define loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Early stopping
        best_val_loss = float('inf')
        best_model = None
        patience_counter = 0
        
        # Training loop
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            correct_train = 0
            total_train = 0
            
            for features, labels in train_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                predicted = (outputs > 0.5).float()
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
            
            train_loss = train_loss / len(train_loader)
            train_acc = correct_train / total_train
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            
            with torch.no_grad():
                for features, labels in val_loader:
                    features, labels = features.to(self.device), labels.to(self.device)
                    
                    # Forward pass
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                    
                    # Statistics
                    val_loss += loss.item()
                    predicted = (outputs > 0.5).float()
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()
            
            val_loss = val_loss / len(val_loader)
            val_acc = correct_val / total_val
            
            # Progress update
            logger.info(f"Epoch {epoch+1}/{num_epochs}, "
                       f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            if progress_callback:
                progress_percent = int(30 + (epoch + 1) / num_epochs * 60)
                progress_callback(f"Training: Epoch {epoch+1}/{num_epochs}, Accuracy: {val_acc:.2f}", 
                                 progress_percent)
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        model.load_state_dict(best_model)
        return model
    
    def save_trained_model(self, model, path):
        """Save trained model to disk"""
        # Ensure the directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save the model
        return save_model(model, path)