"""
Voice Activity Detection - Filters out silent audio frames
"""
import numpy as np

class VoiceActivityDetector:
    def __init__(self, sample_rate=16000, frame_duration_ms=30, 
                 threshold_energy=0.0001, threshold_zero_crossings=10):
        """
        Simple energy-based voice activity detector
        
        Args:
            sample_rate: Audio sample rate in Hz
            frame_duration_ms: Frame size in milliseconds
            threshold_energy: Energy threshold for speech detection
            threshold_zero_crossings: Minimum zero crossings for speech
        """
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.threshold_energy = threshold_energy
        self.threshold_zero_crossings = threshold_zero_crossings
    
    def is_speech(self, audio_frame):
        """
        Determine if audio frame contains speech
        
        Args:
            audio_frame: Numpy array of audio samples (-1.0 to 1.0 range)
            
        Returns:
            bool: True if speech detected, False otherwise
        """
        # Calculate frame energy
        energy = np.mean(audio_frame**2)
        
        # Calculate zero crossing rate
        zero_crossings = np.sum(np.abs(np.diff(np.signbit(audio_frame).astype(int))))
        
        # Consider it speech if both energy and zero crossings are above thresholds
        is_speech = (energy > self.threshold_energy and 
                     zero_crossings > self.threshold_zero_crossings)
        
        return is_speech
