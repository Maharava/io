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
        
        # Energy threshold for silence detection
        self.energy_threshold = 0.005  # Minimum energy to consider non-silence
        
        # Number of silence frames in a row
        self.silence_counter = 0
        self.max_silence_frames = 5  # How many frames of silence to allow before discarding
    
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
        
        # Check if audio is silent
        energy = np.mean(self.audio_buffer**2)
        if energy < self.energy_threshold:
            self.silence_counter += 1
            if self.silence_counter > self.max_silence_frames:
                logger.debug(f"Silent frame detected, energy: {energy:.6f}")
                return None  # Don't process silent audio
        else:
            self.silence_counter = 0
        
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
            
            # Also extract delta features for better discrimination
            delta_mfccs = librosa.feature.delta(mfccs)
            
            # Combine original and delta features
            combined_features = np.vstack([mfccs, delta_mfccs])
            
            # Ensure we have enough frames
            if combined_features.shape[1] < self.num_frames:
                # Pad with zeros if necessary
                pad_width = self.num_frames - combined_features.shape[1]
                combined_features = np.pad(combined_features, ((0, 0), (0, pad_width)))
            elif combined_features.shape[1] > self.num_frames:
                # Take the most recent frames
                combined_features = combined_features[:, -self.num_frames:]
            
            # Apply normalization (feature-wise for better robustness)
            # Use fixed standardization for more consistent results
            # This is critical for inference stability
            for i in range(combined_features.shape[0]):
                feature_mean = np.mean(combined_features[i])
                feature_std = np.std(combined_features[i])
                if feature_std > 1e-6:  # Prevent division by zero
                    combined_features[i] = (combined_features[i] - feature_mean) / feature_std
            
            # Reshape for the model [batch, channels, features, time]
            features = np.expand_dims(combined_features, axis=0)
            
            # Cache the result
            self._update_cache(buffer_hash, features)
            
            return features
            
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
        self.silence_counter = 0
    
    def clear_cache(self):
        """Clear the feature cache"""
        self.feature_cache.clear()