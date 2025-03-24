"""
Enhanced feature extraction with diagnostics
"""
import numpy as np
import librosa
import logging
import os
import time

logger = logging.getLogger("Io.Audio.Features")

# Create a special diagnostic logger for feature extraction
diagnostic_logger = logging.getLogger("Io.Diagnostic.Features")
diagnostic_logger.setLevel(logging.INFO)

# Add a file handler if it doesn't already have one
if not diagnostic_logger.handlers:
    diagnostic_file = os.path.join(os.path.expanduser("~"), ".io", "feature_diagnostic.log")
    file_handler = logging.FileHandler(diagnostic_file, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    diagnostic_logger.addHandler(file_handler)

diagnostic_logger.info("----- STARTING FEATURE DIAGNOSTIC LOGGING -----")

class FeatureExtractor:
    """Extract MFCC features from audio frames with diagnostic logging"""
    
    def __init__(self, sample_rate=16000, frame_size=512, 
                 n_mfcc=13, n_fft=2048, hop_length=160, debug_mode=True):
        """Initialize feature extractor with given parameters"""
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.debug_mode = debug_mode
        
        # Feature cache for frequently used audio
        self.feature_cache = {}
        self.max_cache_size = 100
        
        # Number of frames to generate 1 second of context
        self.num_frames = 101
        
        # Running buffer for audio context
        self.audio_buffer = np.zeros(0)
        
        # Energy threshold for silence detection
        self.energy_threshold = 0.005  # Minimum energy to consider non-silence
        
        # Number of silence frames in a row
        self.silence_counter = 0
        self.max_silence_frames = 5
        
        # Counters for diagnostics
        self.extraction_count = 0
        self.last_log_time = time.time()
        self.log_interval = 5.0  # Log summary every 5 seconds
        
        # Audio level tracking
        self.max_audio_level = 0.0
        self.avg_audio_level = 0.0
        self.audio_level_samples = 0
        
        # Log initialization
        diagnostic_logger.info(
            f"FeatureExtractor initialized with: sample_rate={sample_rate}, frame_size={frame_size}, "
            f"n_mfcc={n_mfcc}, n_fft={n_fft}, hop_length={hop_length}"
        )
    
    def extract(self, audio_frame):
        """Extract MFCC features from audio frame with diagnostic logging"""
        start_time = time.time()
        
        # Add current frame to buffer
        self.audio_buffer = np.append(self.audio_buffer, audio_frame)
        
        # Update audio level statistics
        frame_energy = np.mean(audio_frame**2) if len(audio_frame) > 0 else 0
        self.max_audio_level = max(self.max_audio_level, frame_energy)
        self.avg_audio_level = (self.avg_audio_level * self.audio_level_samples + frame_energy) / (self.audio_level_samples + 1)
        self.audio_level_samples += 1
        
        # Periodically log audio levels
        current_time = time.time()
        if current_time - self.last_log_time > self.log_interval:
            diagnostic_logger.info(f"Audio levels - avg: {self.avg_audio_level:.6f}, max: {self.max_audio_level:.6f}")
            self.last_log_time = current_time
            self.max_audio_level = 0.0
            self.avg_audio_level = 0.0
            self.audio_level_samples = 0
        
        # We need at least 1 second of audio
        min_samples = self.sample_rate + self.frame_size
        
        if len(self.audio_buffer) < min_samples:
            return None
        
        # Keep only the most recent audio
        if len(self.audio_buffer) > min_samples * 1.2:
            self.audio_buffer = self.audio_buffer[-min_samples:]
        
        # Check if audio is silent
        energy = np.mean(self.audio_buffer**2)
        if energy < self.energy_threshold:
            self.silence_counter += 1
            if self.silence_counter > self.max_silence_frames:
                if self.extraction_count % 50 == 0:  # Only log occasionally to avoid spam
                    diagnostic_logger.debug(f"Silent frame detected, energy: {energy:.6f}")
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
            
            # Ensure consistent length
            if mfccs.shape[1] < self.num_frames:
                # Pad if too short
                pad_width = self.num_frames - mfccs.shape[1]
                mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)))
            elif mfccs.shape[1] > self.num_frames:
                # Truncate if too long
                mfccs = mfccs[:, -self.num_frames:]
            
            # Apply normalization (feature-wise for better robustness)
            for i in range(mfccs.shape[0]):
                feature_mean = np.mean(mfccs[i])
                feature_std = np.std(mfccs[i])
                if feature_std > 1e-6:  # Prevent division by zero
                    mfccs[i] = (mfccs[i] - feature_mean) / feature_std
            
            # Reshape for the model [batch, channels, features, time]
            features = np.expand_dims(mfccs, axis=0)
            
            # Log extraction details periodically
            self.extraction_count += 1
            if self.extraction_count % 100 == 0:
                extraction_time = time.time() - start_time
                diagnostic_logger.info(f"Feature extraction #{self.extraction_count}")
                diagnostic_logger.info(f"  Audio energy: {energy:.6f}")
                diagnostic_logger.info(f"  MFCC shape: {mfccs.shape}")
                diagnostic_logger.info(f"  Feature shape: {features.shape}")
                diagnostic_logger.info(f"  Extraction time: {extraction_time*1000:.2f}ms")
                
                # Basic feature statistics
                feature_mean = np.mean(features)
                feature_std = np.std(features)
                feature_min = np.min(features)
                feature_max = np.max(features)
                diagnostic_logger.info(f"  Feature stats: mean={feature_mean:.4f}, std={feature_std:.4f}, min={feature_min:.4f}, max={feature_max:.4f}")
            
            # Cache the result
            self._update_cache(buffer_hash, features)
            
            return features
            
        except Exception as e:
            diagnostic_logger.error(f"Error extracting features: {str(e)}", exc_info=True)
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
        diagnostic_logger.info("Audio buffer cleared")
    
    def clear_cache(self):
        """Clear the feature cache"""
        cache_size = len(self.feature_cache)
        self.feature_cache.clear()
        diagnostic_logger.info(f"Feature cache cleared ({cache_size} items)")
        
    def get_diagnostic_info(self):
        """Return diagnostic information about the feature extractor"""
        return {
            "buffer_length": len(self.audio_buffer),
            "cache_size": len(self.feature_cache),
            "extraction_count": self.extraction_count,
            "silence_counter": self.silence_counter,
            "max_audio_level": self.max_audio_level,
            "avg_audio_level": self.avg_audio_level
        }