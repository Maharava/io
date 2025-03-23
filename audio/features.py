"""
Audio Feature Extraction - Converts raw audio to MFCC features
"""
import numpy as np
import librosa
import logging

logger = logging.getLogger("WakeWord.Features")

class FeatureExtractor:
    def __init__(self, sample_rate=16000, frame_size=512, 
                 n_mfcc=13, n_fft=2048, hop_length=160):
        """
        Extract MFCC features from audio frames
        
        Args:
            sample_rate: Audio sample rate in Hz
            frame_size: Number of samples per audio frame
            n_mfcc: Number of MFCC coefficients to extract
            n_fft: FFT window size
            hop_length: Hop length for feature extraction (10ms at 16kHz)
        """
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Feature normalization parameters (will be updated during use)
        self.mean = np.zeros(n_mfcc)
        self.std = np.ones(n_mfcc)
        self.feature_cache = {}
        
        # Number of frames to generate 1 second of context (101 frames)
        self.num_frames = 101
        
        # Running buffer for context
        self.audio_buffer = np.zeros(0)
    
    def extract(self, audio_frame):
        """
        Extract MFCC features from audio frame
        
        Args:
            audio_frame: Numpy array of audio samples
            
        Returns:
            numpy.ndarray: MFCC features of shape [1, n_mfcc, num_frames]
                          or None if not enough audio context yet
        """
        # Add current frame to buffer
        self.audio_buffer = np.append(self.audio_buffer, audio_frame)
        
        # We need at least 1 second of audio (plus a bit more for processing)
        min_samples = self.sample_rate + self.frame_size
        
        if len(self.audio_buffer) < min_samples:
            return None
        
        # Keep only the most recent audio (1 second plus some margin)
        if len(self.audio_buffer) > min_samples * 1.2:
            self.audio_buffer = self.audio_buffer[-min_samples:]
        
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
            # For our CNN: [1, 1, n_mfcc, num_frames]
            mfccs = np.expand_dims(mfccs, axis=0)
            
            return mfccs
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None
