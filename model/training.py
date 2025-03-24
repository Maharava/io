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
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from .architecture import create_model, save_model

logger = logging.getLogger("Io.Model.Training")

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
            # Extract MFCCs
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
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
            
            return mfcc
            
        except Exception as e:
            logger.error(f"Error extracting MFCCs: {e}")
            return None
    
    def prepare_data(self, wake_word_files, negative_files, test_size=0.2):
        """Prepare training and validation datasets with more balanced approach"""
        logger.info("Extracting features from wake word samples")
        wake_word_features = self.extract_features(wake_word_files, augment=True)
        
        logger.info("Extracting features from negative samples")
        # Apply augmentation to negative samples too for better balance
        negative_features = self.extract_features(negative_files, augment=True)
        
        # Create labels (1 for wake word, 0 for negative)
        wake_word_labels = np.ones(len(wake_word_features))
        negative_labels = np.zeros(len(negative_features))
        
        # Check if we need to balance the dataset
        wake_count = len(wake_word_features)
        neg_count = len(negative_features)
        logger.info(f"Raw dataset: {wake_count} wake word samples, {neg_count} negative samples")
        
        # Balance dataset if needed - undersample the majority class
        if wake_count > neg_count * 3:  # If wake word samples are more than 3x negative samples
            # Randomly select wake word samples to match ratio
            indices = np.random.choice(wake_count, size=neg_count * 3, replace=False)
            wake_word_features = [wake_word_features[i] for i in indices]
            wake_word_labels = np.ones(len(wake_word_features))
            logger.info(f"Balanced dataset by reducing wake word samples to {len(wake_word_features)}")
        elif neg_count > wake_count * 3:  # If negative samples are more than 3x wake word samples
            # Randomly select negative samples to match ratio
            indices = np.random.choice(neg_count, size=wake_count * 3, replace=False)
            negative_features = [negative_features[i] for i in indices]
            negative_labels = np.zeros(len(negative_features))
            logger.info(f"Balanced dataset by reducing negative samples to {len(negative_features)}")
        
        # Combine features and labels
        features = wake_word_features + negative_features
        labels = np.concatenate([wake_word_labels, negative_labels])
        
        # Split into train/validation sets with stratification
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
        
        logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
        return train_loader, val_loader

    def train(self, train_loader, val_loader, num_epochs=50, patience=5, progress_callback=None):
        """Train the wake word model with improved regularization"""
        # Create model and move to device
        model = create_model(n_mfcc=self.n_mfcc * 2, num_frames=self.num_frames)  # Account for delta features
        model.to(self.device)
        
        # Define loss function with weighting for better balance
        # This helps particularly if classes are still somewhat imbalanced
        pos_weight = torch.tensor([1.0])  # Can be adjusted if needed
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Add weight decay for L2 regularization
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        
        # Learning rate scheduler for better convergence
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Early stopping
        best_val_loss = float('inf')
        best_model = None
        patience_counter = 0
        
        # F1 score tracking
        best_f1 = 0.0
        
        # Training loop
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            true_positives = 0
            false_positives = 0
            false_negatives = 0
            
            for features, labels in train_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                
                # Apply gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                
                # Calculate metrics
                true_positives += ((predicted == 1) & (labels == 1)).sum().item()
                false_positives += ((predicted == 1) & (labels == 0)).sum().item()
                false_negatives += ((predicted == 0) & (labels == 1)).sum().item()
            
            train_loss = train_loss / len(train_loader)
            
            # Calculate F1 score
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_true_positives = 0
            val_false_positives = 0
            val_false_negatives = 0
            
            with torch.no_grad():
                for features, labels in val_loader:
                    features, labels = features.to(self.device), labels.to(self.device)
                    
                    # Forward pass
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                    
                    # Statistics
                    val_loss += loss.item()
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                    
                    # Calculate metrics
                    val_true_positives += ((predicted == 1) & (labels == 1)).sum().item()
                    val_false_positives += ((predicted == 1) & (labels == 0)).sum().item()
                    val_false_negatives += ((predicted == 0) & (labels == 1)).sum().item()
            
            val_loss = val_loss / len(val_loader)
            
            # Calculate validation F1 score
            val_precision = val_true_positives / (val_true_positives + val_false_positives) if (val_true_positives + val_false_positives) > 0 else 0
            val_recall = val_true_positives / (val_true_positives + val_false_negatives) if (val_true_positives + val_false_negatives) > 0 else 0
            val_f1 = 2 * val_precision * val_recall / (val_precision + val_recall) if (val_precision + val_recall) > 0 else 0
            
            # Update learning rate based on validation loss
            scheduler.step(val_loss)
            
            # Progress update
            logger.info(f"Epoch {epoch+1}/{num_epochs}, "
                       f"Train Loss: {train_loss:.4f}, Train F1: {f1:.4f}, "
                       f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
            
            if progress_callback:
                progress_percent = int(30 + (epoch + 1) / num_epochs * 60)
                progress_callback(f"Training: Epoch {epoch+1}/{num_epochs}, Val F1: {val_f1:.2f}", 
                                 progress_percent)
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model.state_dict().copy()
                best_f1 = val_f1
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Log final model performance
        logger.info(f"Training complete. Best validation F1 score: {best_f1:.4f}")
        
        # Load best model
        model.load_state_dict(best_model)
        return model
    
    def save_trained_model(self, model, path):
        """Save trained model to disk"""
        # Ensure the directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save the model
        return save_model(model, path)