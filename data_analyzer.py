"""
Wake Word Data Analysis Script

This script analyzes the training data to ensure it's being correctly processed
and properly balanced between positive and negative samples.
"""
import os
import sys
from pathlib import Path
import numpy as np
import librosa
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("WakeWordDataAnalyzer")

# Add parent directory to path to import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try importing from the wake word package
try:
    from audio.features import FeatureExtractor
    logger.info("Successfully imported wake word modules")
except ImportError as e:
    logger.error(f"Error importing wake word modules: {e}")
    logger.error("Make sure you're running this script from the project root directory")
    sys.exit(1)

def analyze_audio_files(audio_dir, output_dir, limit=None):
    """
    Analyze audio files in the given directory
    
    Args:
        audio_dir: Directory containing audio files
        output_dir: Directory to save analysis results
        limit: Maximum number of files to analyze
    """
    logger.info(f"Analyzing audio files in {audio_dir}")
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get all WAV files
    wav_files = list(Path(audio_dir).glob("*.wav"))
    if limit:
        wav_files = wav_files[:limit]
    
    logger.info(f"Found {len(wav_files)} WAV files")
    
    # Track audio statistics
    durations = []
    energies = []
    max_values = []
    
    # Create feature extractor
    feature_extractor = FeatureExtractor()
    
    # Process each file
    for i, file_path in enumerate(wav_files):
        try:
            # Load audio
            y, sr = librosa.load(str(file_path), sr=16000)
            
            # Calculate statistics
            duration = len(y) / sr
            energy = np.mean(y**2)
            max_val = np.max(np.abs(y))
            
            durations.append(duration)
            energies.append(energy)
            max_values.append(max_val)
            
            # Extract features for a few examples
            if i < 5:
                # Extract MFCC
                mfcc = librosa.feature.mfcc(
                    y=y, sr=sr, n_mfcc=13, 
                    n_fft=2048, hop_length=160
                )
                
                # Save visualization
                plt.figure(figsize=(10, 4))
                librosa.display.specshow(
                    mfcc, x_axis='time', y_axis='mel', 
                    sr=sr, hop_length=160
                )
                plt.colorbar(format='%+2.0f dB')
                plt.title(f'MFCC features from {file_path.name}')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{file_path.stem}_mfcc.png"))
                plt.close()
                
                # Show waveform
                plt.figure(figsize=(10, 4))
                plt.plot(np.arange(len(y))/sr, y)
                plt.title(f'Waveform of {file_path.name}')
                plt.xlabel('Time (s)')
                plt.ylabel('Amplitude')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{file_path.stem}_waveform.png"))
                plt.close()
            
            # Print progress
            if (i + 1) % 10 == 0 or i + 1 == len(wav_files):
                logger.info(f"Processed {i + 1}/{len(wav_files)} files")
                
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
    
    # Plot statistics
    logger.info("Generating statistics plots")
    
    # Duration histogram
    plt.figure(figsize=(10, 6))
    plt.hist(durations, bins=20)
    plt.title('Audio Duration Distribution')
    plt.xlabel('Duration (s)')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, "duration_histogram.png"))
    plt.close()
    
    # Energy histogram
    plt.figure(figsize=(10, 6))
    plt.hist(energies, bins=20)
    plt.title('Audio Energy Distribution')
    plt.xlabel('Energy')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, "energy_histogram.png"))
    plt.close()
    
    # Max value histogram
    plt.figure(figsize=(10, 6))
    plt.hist(max_values, bins=20)
    plt.title('Audio Max Value Distribution')
    plt.xlabel('Max Value')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, "max_value_histogram.png"))
    plt.close()
    
    # Print summary statistics
    logger.info(f"Duration - Mean: {np.mean(durations):.2f}s, Min: {np.min(durations):.2f}s, Max: {np.max(durations):.2f}s")
    logger.info(f"Energy - Mean: {np.mean(energies):.6f}, Min: {np.min(energies):.6f}, Max: {np.max(energies):.6f}")
    logger.info(f"Max Value - Mean: {np.mean(max_values):.4f}, Min: {np.min(max_values):.4f}, Max: {np.max(max_values):.4f}")
    
    return {
        "files_analyzed": len(wav_files),
        "mean_duration": np.mean(durations),
        "mean_energy": np.mean(energies),
        "mean_max_value": np.mean(max_values)
    }

def analyze_feature_extraction(audio_dir, output_dir, limit=5):
    """
    Analyze how features are extracted from audio files
    
    Args:
        audio_dir: Directory containing audio files
        output_dir: Directory to save analysis results
        limit: Maximum number of files to analyze
    """
    logger.info(f"Analyzing feature extraction from {audio_dir}")
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get WAV files
    wav_files = list(Path(audio_dir).glob("*.wav"))[:limit]
    
    logger.info(f"Analyzing feature extraction for {len(wav_files)} files")
    
    # Create feature extractor
    feature_extractor = FeatureExtractor(
        sample_rate=16000,
        frame_size=512,
        n_mfcc=13,
        n_fft=2048,
        hop_length=160
    )
    
    # Process each file
    for file_path in wav_files:
        try:
            # Load audio
            y, sr = librosa.load(str(file_path), sr=16000)
            
            # Log the file and its basic properties
            logger.info(f"File: {file_path.name}, Length: {len(y)} samples, Duration: {len(y)/sr:.2f}s")
            
            # Use our feature extractor - need to reset the buffer first
            feature_extractor.clear_buffer()
            
            # Add audio to buffer
            feature_extractor.audio_buffer = y
            
            # Extract features
            features = feature_extractor.extract(np.array([]))
            
            if features is not None:
                # Log feature shape
                logger.info(f"Extracted features shape: {features.shape}")
                
                # Plot features
                plt.figure(figsize=(10, 6))
                plt.imshow(features[0], aspect='auto', origin='lower')
                plt.colorbar(format='%+2.0f')
                plt.title(f'Features from {file_path.name}')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{file_path.stem}_features.png"))
                plt.close()
                
                # Feature statistics
                mean_val = np.mean(features)
                std_val = np.std(features)
                min_val = np.min(features)
                max_val = np.max(features)
                
                logger.info(f"Feature stats: mean={mean_val:.4f}, std={std_val:.4f}, min={min_val:.4f}, max={max_val:.4f}")
            else:
                logger.warning(f"Failed to extract features for {file_path.name}")
                
        except Exception as e:
            logger.error(f"Error analyzing features for {file_path}: {e}")

def run_analysis():
    """
    Run analysis on wake word data
    """
    logger.info("Starting wake word data analysis")
    
    # Create output directory
    output_dir = Path.home() / ".io" / "data_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Path to training data
    training_dir = Path.home() / ".io" / "training_data"
    wake_word_dir = training_dir / "wake_word"
    negative_dir = training_dir / "negative"
    
    # Verify directories exist
    for dir_path in [training_dir, wake_word_dir, negative_dir]:
        if not dir_path.exists():
            logger.error(f"Directory not found: {dir_path}")
            return
    
    # Analyze wake word files
    wake_output_dir = output_dir / "wake_word"
    wake_stats = analyze_audio_files(wake_word_dir, wake_output_dir, limit=20)
    
    # Analyze negative files
    negative_output_dir = output_dir / "negative"
    negative_stats = analyze_audio_files(negative_dir, negative_output_dir, limit=20)
    
    # Compare statistics
    logger.info("\nComparison of wake word vs negative samples:")
    logger.info(f"Wake word - Files: {wake_stats['files_analyzed']}, Duration: {wake_stats['mean_duration']:.2f}s, Energy: {wake_stats['mean_energy']:.6f}")
    logger.info(f"Negative - Files: {negative_stats['files_analyzed']}, Duration: {negative_stats['mean_duration']:.2f}s, Energy: {negative_stats['mean_energy']:.6f}")
    
    # Analyze feature extraction
    logger.info("\nAnalyzing feature extraction:")
    feature_output_dir = output_dir / "features"
    analyze_feature_extraction(wake_word_dir, feature_output_dir / "wake_word")
    analyze_feature_extraction(negative_dir, feature_output_dir / "negative")
    
    logger.info(f"\nAnalysis complete. Results saved to {output_dir}")
    return output_dir

if __name__ == "__main__":
    try:
        analysis_dir = run_analysis()
        if analysis_dir:
            print(f"\nData analysis complete. Results saved to: {analysis_dir}")
    except Exception as e:
        logger.exception("Error during data analysis")
        print(f"\nError: {str(e)}")
        print("Check the logs for more details.")
