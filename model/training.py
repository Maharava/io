"""
Enhanced training pipeline with robust debugging and training improvements
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import librosa
import logging
import math
import time
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Create a dedicated training logger
training_logger = logging.getLogger("Io.Training.Debug")
training_logger.setLevel(logging.INFO)

# Add a file handler for training logs
training_log_file = os.path.join(os.path.expanduser("~"), ".io", "training_debug.log")
if not any(isinstance(h, logging.FileHandler) and h.baseFilename == training_log_file for h in training_logger.handlers):
    file_handler = logging.FileHandler(training_log_file, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    training_logger.addHandler(file_handler)

training_logger.info("----- STARTING ENHANCED TRAINING -----")

def calculate_conv_output_length(input_length, kernel_size, stride, padding=0):
    """Calculate output length after a conv/pool layer with precise PyTorch formula"""
    return math.floor((input_length + 2 * padding - kernel_size) / stride + 1)

# A simpler model that might be easier to train
class SimpleWakeWordModel(nn.Module):
    """Simplified CNN model for wake word detection"""
    
    def __init__(self, n_mfcc=13, num_frames=101):
        super(SimpleWakeWordModel, self).__init__()
        
        # A simpler architecture with fewer layers
        self.conv_layer = nn.Sequential(
            nn.Conv1d(n_mfcc, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=0)
        )
        
        # Calculate output size
        output_width = calculate_conv_output_length(num_frames, 3, 2, 0)
        self.fc_input_size = 32 * output_width
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.fc_input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# Standard model architecture
class WakeWordModel(nn.Module):
    """1D CNN model for wake word detection"""
    
    def __init__(self, n_mfcc=13, num_frames=101):
        super(WakeWordModel, self).__init__()
        
        # Calculate output dimensions
        after_pool1 = calculate_conv_output_length(num_frames, 3, 2, 0)
        after_pool2 = calculate_conv_output_length(after_pool1, 3, 2, 0)
        self.fc_input_size = 64 * after_pool2
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(n_mfcc, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=0),
            
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=0)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.fc_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

class WakeWordDataset(Dataset):
    """Dataset for wake word training"""
    
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = torch.from_numpy(self.features[idx]).float()
        label = torch.tensor([self.labels[idx]]).float()
        return feature, label

class WakeWordTrainer:
    """Enhanced trainer for wake word models with debugging"""
    
    def __init__(self, sample_rate=16000, n_mfcc=13, n_fft=2048, 
                 hop_length=160, num_frames=101, use_simple_model=True):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_frames = num_frames
        self.use_simple_model = use_simple_model
        
        # Use CPU for better compatibility
        self.device = torch.device("cpu")
        
        # Save directory for checkpoints and diagnostics
        self.diagnostic_dir = os.path.join(os.path.expanduser("~"), ".io", "training_diagnostics")
        os.makedirs(self.diagnostic_dir, exist_ok=True)
        
        training_logger.info(f"Trainer initialized with: n_mfcc={n_mfcc}, simple_model={use_simple_model}")
    
    def visualize_sample(self, audio_file, save_path=None):
        """Visualize MFCC features from an audio sample"""
        try:
            # Load audio
            y, sr = librosa.load(audio_file, sr=self.sample_rate)
            
            # Extract MFCC
            mfcc = librosa.feature.mfcc(
                y=y, sr=sr, n_mfcc=self.n_mfcc, 
                n_fft=self.n_fft, hop_length=self.hop_length
            )
            
            # Create plot
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(
                mfcc, x_axis='time', y_axis='mel', 
                sr=sr, hop_length=self.hop_length
            )
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'MFCC features from {os.path.basename(audio_file)}')
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path)
                training_logger.info(f"Saved MFCC visualization to {save_path}")
            
            plt.close()
            return True
        except Exception as e:
            training_logger.error(f"Error visualizing sample: {e}")
            return False
    
    def extract_features(self, audio_files, augment=False):
        """Extract MFCC features from audio files with optional augmentation"""
        features = []
        processed_files = 0
        total_files = len(audio_files)
        
        for file_path in audio_files:
            try:
                # Progress tracking
                processed_files += 1
                if processed_files % 20 == 0 or processed_files == total_files:
                    training_logger.info(f"Feature extraction progress: {processed_files}/{total_files}")
                
                # Load audio file
                y, sr = librosa.load(file_path, sr=self.sample_rate)
                
                # Basic statistics for debugging
                audio_duration = len(y) / sr
                audio_energy = np.mean(y**2)
                audio_max = np.max(np.abs(y))
                
                if processed_files % 20 == 0:
                    training_logger.info(
                        f"Audio stats for {os.path.basename(file_path)}: "
                        f"duration={audio_duration:.2f}s, energy={audio_energy:.6f}, max={audio_max:.6f}"
                    )
                
                # Skip if audio is too silent
                if audio_energy < 0.0001:
                    training_logger.warning(f"Very quiet audio in {file_path}, energy={audio_energy:.6f}")
                    continue
                
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
                training_logger.error(f"Error processing {file_path}: {e}")
        
        # Log feature statistics
        if features:
            features_array = np.array(features)
            mean_val = np.mean(features_array)
            std_val = np.std(features_array)
            min_val = np.min(features_array)
            max_val = np.max(features_array)
            
            training_logger.info(
                f"Feature statistics: count={len(features)}, shape={features[0].shape}, "
                f"mean={mean_val:.4f}, std={std_val:.4f}, min={min_val:.4f}, max={max_val:.4f}"
            )
        else:
            training_logger.error("No features extracted! Check audio files and extraction process.")
            
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
            
            # Check for NaN or Inf values
            if np.isnan(mfcc).any() or np.isinf(mfcc).any():
                training_logger.warning("Found NaN or Inf in extracted features!")
                return None
                
            return mfcc
            
        except Exception as e:
            training_logger.error(f"Error extracting MFCCs: {e}")
            return None
    
    def prepare_data(self, wake_word_files, negative_files, test_size=0.2):
        """Prepare training and validation datasets"""
        # Visualize a few random samples for verification
        if wake_word_files:
            sample_file = random.choice(wake_word_files)
            self.visualize_sample(
                sample_file, 
                os.path.join(self.diagnostic_dir, "wake_word_sample.png")
            )
        
        if negative_files:
            sample_file = random.choice(negative_files)
            self.visualize_sample(
                sample_file, 
                os.path.join(self.diagnostic_dir, "negative_sample.png")
            )
        
        training_logger.info(f"Extracting features from {len(wake_word_files)} wake word samples")
        wake_word_features = self.extract_features(wake_word_files, augment=True)
        
        training_logger.info(f"Extracting features from {len(negative_files)} negative samples")
        negative_features = self.extract_features(negative_files, augment=False)
        
        # Validate that we have enough data
        if len(wake_word_features) < 10:
            training_logger.error("Not enough wake word features extracted (< 10)!")
        if len(negative_features) < 10:
            training_logger.error("Not enough negative features extracted (< 10)!")
        
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
        
        # Log class distribution
        train_pos = np.sum(labels_train)
        train_neg = len(labels_train) - train_pos
        val_pos = np.sum(labels_val)
        val_neg = len(labels_val) - val_pos
        
        training_logger.info(f"Training set: {len(train_dataset)} samples ({train_pos} positive, {train_neg} negative)")
        training_logger.info(f"Validation set: {len(val_dataset)} samples ({val_pos} positive, {val_neg} negative)")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=32, shuffle=True, drop_last=False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=32, shuffle=False, drop_last=False
        )
        
        return train_loader, val_loader
    
    def test_inference(self, model, val_loader):
        """Test inference on validation data and log results"""
        model.eval()
        confidences = []
        correct = 0
        total = 0
        pos_conf = []
        neg_conf = []
        
        with torch.no_grad():
            for features, labels in val_loader:
                outputs = model(features)
                confidences.extend(outputs.numpy().flatten().tolist())
                
                # Track confidences by class
                for i, (output, label) in enumerate(zip(outputs, labels)):
                    if label.item() == 1:
                        pos_conf.append(output.item())
                    else:
                        neg_conf.append(output.item())
                
                # Calculate accuracy
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Log detailed statistics
        training_logger.info(f"Inference test - Accuracy: {correct/total:.4f}")
        
        if confidences:
            conf_mean = np.mean(confidences)
            conf_std = np.std(confidences)
            conf_min = np.min(confidences)
            conf_max = np.max(confidences)
            
            training_logger.info(
                f"Confidence stats: mean={conf_mean:.4f}, std={conf_std:.4f}, "
                f"min={conf_min:.4f}, max={conf_max:.4f}"
            )
        
        # Log class-specific confidence
        if pos_conf:
            pos_mean = np.mean(pos_conf)
            pos_std = np.std(pos_conf)
            training_logger.info(f"Positive class: mean={pos_mean:.4f}, std={pos_std:.4f}")
        
        if neg_conf:
            neg_mean = np.mean(neg_conf)
            neg_std = np.std(neg_conf)
            training_logger.info(f"Negative class: mean={neg_mean:.4f}, std={neg_std:.4f}")
        
        # Plot histogram of confidences
        plt.figure(figsize=(8, 6))
        if pos_conf and neg_conf:
            plt.hist([pos_conf, neg_conf], bins=20, alpha=0.5, label=['Positive', 'Negative'])
            plt.legend()
        else:
            plt.hist(confidences, bins=20)
        
        plt.title("Distribution of Confidence Scores")
        plt.xlabel("Confidence")
        plt.ylabel("Count")
        plt.savefig(os.path.join(self.diagnostic_dir, "confidence_distribution.png"))
        plt.close()
    
    def train(self, train_loader, val_loader, num_epochs=100, patience=10, progress_callback=None):
        """Train the wake word model with enhanced diagnostics"""
        # Get feature shape from a sample batch
        for features, _ in train_loader:
            input_shape = features.shape
            training_logger.info(f"Input feature shape: {input_shape}")
            break
        
        # Create model - either standard or simple
        if self.use_simple_model:
            training_logger.info("Using simple model architecture")
            model = SimpleWakeWordModel(n_mfcc=self.n_mfcc, num_frames=self.num_frames)
        else:
            training_logger.info("Using standard model architecture")
            model = WakeWordModel(n_mfcc=self.n_mfcc, num_frames=self.num_frames)
        
        model.to(self.device)
        
        # Log model details
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        training_logger.info(f"Model created with {num_params} trainable parameters")
        training_logger.info(f"Model structure:\n{model}")
        
        # Define loss function and optimizer
        criterion = nn.BCELoss()
        
        # Use a higher learning rate
        learning_rate = 0.005  # Increased from 0.001
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Add learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Track metrics
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        # Early stopping
        best_val_loss = float('inf')
        best_model = None
        best_epoch = 0
        patience_counter = 0
        
        # Save initial model
        torch.save(
            model.state_dict(), 
            os.path.join(self.diagnostic_dir, "model_initial.pth")
        )
        
        # Test inference before training
        training_logger.info("Testing inference before training:")
        self.test_inference(model, val_loader)
        
        # Training loop
        start_time = time.time()
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
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
                
                # Gradient clipping to prevent exploding gradients
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                predicted = (outputs > 0.5).float()
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
            
            train_loss = train_loss / len(train_loader)
            train_acc = correct_train / total_train
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
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
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Calculate time taken
            epoch_time = time.time() - epoch_start
            
            # Log detailed progress
            training_logger.info(
                f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.2f}s) - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )
            
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
                
                # Test inference
                if (epoch + 1) % 20 == 0:
                    training_logger.info(f"Testing inference at epoch {epoch+1}:")
                    self.test_inference(model, val_loader)
            
            # Check for improvement
            if val_loss < best_val_loss:
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
                if patience_counter >= patience:
                    training_logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        total_time = time.time() - start_time
        training_logger.info(f"Training completed in {total_time:.2f} seconds")
        training_logger.info(f"Best model from epoch {best_epoch} with val_loss={best_val_loss:.4f}")
        
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
        
        # Final inference test
        training_logger.info("Final inference test:")
        self.test_inference(model, val_loader)
        
        return model
    
    def save_trained_model(self, model, path):
        """Save trained model to disk"""
        # Ensure the directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        try:
            torch.save(model.state_dict(), path)
            training_logger.info(f"Model saved to {path}")
            
            # Create a README with training info
            readme_path = os.path.join(os.path.dirname(path), "training_info.txt")
            with open(readme_path, 'w') as f:
                f.write(f"Model trained with enhanced training pipeline\n")
                f.write(f"n_mfcc: {self.n_mfcc}\n")
                f.write(f"n_fft: {self.n_fft}\n")
                f.write(f"hop_length: {self.hop_length}\n")
                f.write(f"num_frames: {self.num_frames}\n")
                f.write(f"model_type: {'simple' if self.use_simple_model else 'standard'}\n")
                f.write(f"Training diagnostic files in: {self.diagnostic_dir}\n")
            
            return True
        except Exception as e:
            training_logger.error(f"Error saving model: {e}")
            return False