"""
Voice Activity Detection - Filters out silent audio frames
"""
import numpy as np
import logging

logger = logging.getLogger("WakeWord.VAD")

class VoiceActivityDetector:
    def __init__(self, sample_rate=16000, frame_duration_ms=30, 
                 threshold_energy=0.0001, threshold_zero_crossings=10):
        """Simple energy-based voice activity detector"""
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.threshold_energy = threshold_energy
        self.threshold_zero_crossings = threshold_zero_crossings
        
        # Adaptive threshold parameters
        self.use_adaptive_threshold = True
        self.adaptive_energy_threshold = threshold_energy
        self.noise_level = threshold_energy
        self.energy_history = []
        self.history_size = 50  # Keep last 50 frames for adaptation
        self.adaptation_rate = 0.05  # Slow adaptation rate
    
    def is_speech(self, audio_frame):
        """Determine if audio frame contains speech"""
        try:
            # Calculate frame energy
            energy = np.mean(audio_frame**2)
            
            # Calculate zero crossing rate
            zero_crossings = np.sum(np.abs(np.diff(np.signbit(audio_frame).astype(int))))
            
            # Update adaptive threshold if enabled
            if self.use_adaptive_threshold:
                self._update_adaptive_threshold(energy)
                threshold_to_use = self.adaptive_energy_threshold
            else:
                threshold_to_use = self.threshold_energy
            
            # Consider it speech if both energy and zero crossings are above thresholds
            is_speech = (energy > threshold_to_use and 
                        zero_crossings > self.threshold_zero_crossings)
            
            return is_speech
        except Exception as e:
            logger.error(f"Error in VAD processing: {e}")
            # Default to returning True to avoid missing potential speech
            return True
    
    def _update_adaptive_threshold(self, current_energy):
        """Update adaptive energy threshold based on recent audio"""
        # Add current energy to history
        self.energy_history.append(current_energy)
        
        # Keep history size limited
        if len(self.energy_history) > self.history_size:
            self.energy_history.pop(0)
        
        # Calculate noise level from the lowest 20% of energies
        if len(self.energy_history) >= 5:
            sorted_energies = sorted(self.energy_history)
            noise_count = max(1, int(len(sorted_energies) * 0.2))
            recent_noise_level = sum(sorted_energies[:noise_count]) / noise_count
            
            # Smoothly update noise level
            self.noise_level = ((1 - self.adaptation_rate) * self.noise_level + 
                               self.adaptation_rate * recent_noise_level)
            
            # Set adaptive threshold as a multiple of noise level with a minimum
            self.adaptive_energy_threshold = max(self.threshold_energy, 
                                               self.noise_level * 3.0)

class ImprovedVoiceActivityDetector(VoiceActivityDetector):
    """Advanced VAD with better noise handling"""
    def __init__(self, sample_rate=16000, frame_duration_ms=30):
        super().__init__(sample_rate, frame_duration_ms)
        
        # Enhanced parameters
        self.use_adaptive_threshold = True
        self.energy_history_size = 100
        self.energy_history = []
        self.speech_mode = False
        self.speech_frame_count = 0
        self.silence_frame_count = 0
        
        # Hangover for smoother transitions
        self.speech_hangover = 8  # Continue detecting speech for this many frames after energy drops
        self.silence_hangover = 3  # Wait this many frames of silence before switching modes
    
    def is_speech(self, audio_frame):
        """Improved speech detection with hangover"""
        try:
            # Calculate frame energy
            energy = np.mean(audio_frame**2)
            
            # Calculate zero crossing rate
            zero_crossings = np.sum(np.abs(np.diff(np.signbit(audio_frame).astype(int))))
            
            # Update adaptive threshold
            self._update_adaptive_threshold(energy)
            
            # Raw speech detection
            raw_speech_detected = (energy > self.adaptive_energy_threshold and 
                                  zero_crossings > self.threshold_zero_crossings)
            
            # State machine with hangover for smoother detection
            if self.speech_mode:
                if raw_speech_detected:
                    # Continue in speech mode, reset silence counter
                    self.silence_frame_count = 0
                    self.speech_frame_count += 1
                    return True
                else:
                    # Possible end of speech, increment silence counter
                    self.silence_frame_count += 1
                    
                    # Stay in speech mode during hangover
                    if self.silence_frame_count < self.speech_hangover:
                        return True
                    else:
                        # Switch to silence mode
                        self.speech_mode = False
                        self.speech_frame_count = 0
                        return False
            else:
                if raw_speech_detected:
                    # Possible start of speech, increment speech counter
                    self.speech_frame_count += 1
                    
                    # Switch to speech mode if enough consecutive speech frames
                    if self.speech_frame_count > self.silence_hangover:
                        self.speech_mode = True
                        self.silence_frame_count = 0
                        return True
                    else:
                        return False
                else:
                    # Definite silence
                    self.speech_frame_count = 0
                    return False
                    
        except Exception as e:
            logger.error(f"Error in improved VAD processing: {e}")
            return True  # Default to speech on error
    
    def _update_adaptive_threshold(self, current_energy):
        """Enhanced adaptive threshold with better noise estimation"""
        # Add current energy to history
        self.energy_history.append(current_energy)
        
        # Keep history size limited
        if len(self.energy_history) > self.energy_history_size:
            self.energy_history.pop(0)
        
        # Need enough history for reliable estimation
        if len(self.energy_history) < 10:
            return
        
        # Calculate noise level from the lowest 20% of energies
        sorted_energies = sorted(self.energy_history)
        noise_count = max(1, int(len(sorted_energies) * 0.2))
        recent_noise_level = sum(sorted_energies[:noise_count]) / noise_count
        
        # Calculate speech level from the highest 10% of energies
        speech_count = max(1, int(len(sorted_energies) * 0.1))
        speech_level = sum(sorted_energies[-speech_count:]) / speech_count
        
        # Smooth update of noise level
        self.noise_level = 0.95 * self.noise_level + 0.05 * recent_noise_level
        
        # Set adaptive threshold based on noise and speech levels
        if speech_level > self.noise_level * 2:
            # Clear speech/noise separation, set threshold in between
            self.adaptive_energy_threshold = self.noise_level * 1.5
        else:
            # Poor separation, use more conservative threshold
            self.adaptive_energy_threshold = self.noise_level * 2.5
        
        # Ensure minimum threshold
        self.adaptive_energy_threshold = max(self.threshold_energy, self.adaptive_energy_threshold)