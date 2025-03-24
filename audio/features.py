"""
Audio Feature Extraction - Converts raw audio to MFCC features
"""
import numpy as np
import librosa
import logging

logger = logging.getLogger("Io.Audio.Features")

class FeatureExtractor:
    """Extract MFCC features from audio frames for wake word detection"""
    
    def __init__(self, sample_rate=16000, frame_size=512, 
                 n_mfcc=13, n_fft=2048, hop_length=160):
        """Initialize feature extractor with given parameters"""
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Feature normalization parameters (updated during use)
        self.mean = np.zeros(n_mfcc)
        self.std = np.ones(n_mfcc)
        
        # Feature cache for frequently used audio
        self.feature_cache = {}
        self.max_cache_size = 100
        
        # Number of frames to generate 1 second of context (101 frames)
        self.num_frames = 101
        
        # Running buffer for audio context
        self.audio_buffer = np.zeros(0)
    
    def extract(self, audio_frame):
        """Extract MFCC features from audio frame"""
        # Add current frame to buffer
        self.audio_buffer = np.append(self.audio_buffer, audio_frame)
        
        # We need at least 1 second of audio
        min_samples = self.sample_rate + self.frame_size
        
        if len(self.audio_buffer) < min_samples:
            return None
        
        # Keep only the most recent audio (1 second plus margin)
        if len(self.audio_buffer) > min_samples * 1.2:
            self.audio_buffer = self.audio_buffer[-min_samples:]
        
        # Calculate buffer hash for cache lookup
        buffer_hash = hash(self.audio_buffer.tobytes())
        
        # Check cache first
        if buffer_hash in self.feature_cache:
            return self.feature_cache[buffer_hash]
        
        # Extract MFCCs
        try:
            # Calculate MFCCs
            mfccs = librosa.feature.mfcc(
                y=self.audio_buffer, 
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # Ensure we have enough frames
            if mfccs.shape[1] < self.num_frames:
                # Pad with zeros if necessary
                pad_width = self.num_frames - mfccs.shape[1]
                mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)))
            elif mfccs.shape[1] > self.num_frames:
                # Take the most recent frames
                mfccs = mfccs[:, -self.num_frames:]
            
            # Apply normalization
            mfccs = (mfccs - self.mean[:, np.newaxis]) / (self.std[:, np.newaxis] + 1e-6)
            
            # Update normalization parameters with running average
            alpha = 0.01  # Learning rate for updating mean and std
            current_mean = np.mean(mfccs, axis=1)
            current_std = np.std(mfccs, axis=1)
            self.mean = (1 - alpha) * self.mean + alpha * current_mean
            self.std = (1 - alpha) * self.std + alpha * current_std
            
            # Reshape for the model [batch, channels, mfcc, time]
            mfccs = np.expand_dims(mfccs, axis=0)
            
            # Cache the result
            self._update_cache(buffer_hash, mfccs)
            
            return mfccs
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None
    
    def _update_cache(self, key, value):
        """Update feature cache with size limit"""
        # Add to cache
        self.feature_cache[key] = value
        
        # Remove oldest items if cache is too large
        if len(self.feature_cache) > self.max_cache_size:
            # Get oldest items (first items in the cache)
            oldest_key = next(iter(self.feature_cache))
            del self.feature_cache[oldest_key]
    
    def clear_buffer(self):
        """Clear the audio buffer"""
        self.audio_buffer = np.zeros(0)
    
    def clear_cache(self):
        """Clear the feature cache"""
        self.feature_cache.clear()