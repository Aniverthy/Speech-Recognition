#!/usr/bin/env python3
"""
Audio Preprocessing Service

Handles audio file loading, conversion, and preparation for processing.
"""

import os
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from .service_state import ServiceState


class PreprocessService:
    """Audio preprocessing and preparation service."""
    
    def __init__(self, state: ServiceState):
        self.state = state
        self.audio_config = state.get_audio_config()
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and convert to target format.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        # Load audio with librosa
        audio, sr = librosa.load(file_path, sr=None, mono=True)
        
        # Resample if necessary
        if sr != self.audio_config["target_sample_rate"]:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.audio_config["target_sample_rate"])
            sr = self.audio_config["target_sample_rate"]
        
        # Ensure float32 format
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        return audio, sr
    
    def validate_audio(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Validate audio data for processing.
        
        Args:
            audio: Audio waveform
            sr: Sample rate
            
        Returns:
            Validation results
        """
        duration = len(audio) / sr
        
        # Check duration limits
        if duration < self.audio_config["min_segment_duration"]:
            raise ValueError(f"Audio too short: {duration:.2f}s (min: {self.audio_config['min_segment_duration']}s)")
        
        if duration > self.state.MAX_AUDIO_DURATION:
            raise ValueError(f"Audio too long: {duration:.2f}s (max: {self.state.MAX_AUDIO_DURATION}s)")
        
        # Check for invalid values
        if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
            raise ValueError("Audio contains invalid values (NaN or Inf)")
        
        return {
            "duration": duration,
            "sample_rate": sr,
            "samples": len(audio),
            "is_valid": True
        }
    
    def segment_audio(self, audio: np.ndarray, sr: int, chunk_duration: Optional[float] = None) -> list:
        """
        Segment audio into chunks for processing.
        
        Args:
            audio: Audio waveform
            sr: Sample rate
            chunk_duration: Duration of each chunk (uses default if None)
            
        Returns:
            List of audio chunks
        """
        if chunk_duration is None:
            chunk_duration = self.audio_config["chunk_duration"]
        
        chunk_samples = int(chunk_duration * sr)
        chunks = []
        
        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i:i + chunk_samples]
            if len(chunk) >= int(self.audio_config["min_segment_duration"] * sr):
                chunks.append(chunk)
        
        return chunks
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to prevent clipping.
        
        Args:
            audio: Audio waveform
            
        Returns:
            Normalized audio
        """
        # RMS normalization
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0:
            target_rms = 0.1  # Target RMS level
            audio = audio * (target_rms / rms)
        
        # Peak normalization
        max_val = np.max(np.abs(audio))
        if max_val > 0.95:  # If close to clipping
            audio = audio * (0.95 / max_val)
        
        return audio
    
    def apply_high_pass_filter(self, audio: np.ndarray, sr: int, cutoff: float = 80.0) -> np.ndarray:
        """
        Apply high-pass filter to remove low-frequency noise.
        
        Args:
            audio: Audio waveform
            sr: Sample rate
            cutoff: Cutoff frequency in Hz
            
        Returns:
            Filtered audio
        """
        # Design high-pass filter
        nyquist = sr / 2
        normalized_cutoff = cutoff / nyquist
        
        # Use Butterworth filter
        b, a = signal.butter(4, normalized_cutoff, btype='high', analog=False)
        
        # Apply filter
        filtered_audio = signal.filtfilt(b, a, audio)
        
        return filtered_audio
    
    def apply_preprocessing(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply full preprocessing pipeline.
        
        Args:
            audio: Audio waveform
            sr: Sample rate
            
        Returns:
            Preprocessed audio
        """
        # Apply high-pass filter to remove low-frequency noise
        audio = self.apply_high_pass_filter(audio, sr, cutoff=80.0)
        
        # Normalize audio
        audio = self.normalize_audio(audio)
        
        return audio
    
    def save_audio(self, audio: np.ndarray, sr: int, output_path: str, format: str = "wav") -> str:
        """
        Save processed audio to file.
        
        Args:
            audio: Audio waveform
            sr: Sample rate
            output_path: Output file path
            format: Audio format
            
        Returns:
            Path to saved file
        """
        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save with soundfile
        sf.write(output_path, audio, sr, format=format.upper())
        
        return output_path
    
    def get_audio_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get audio file information.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Audio file information
        """
        try:
            # Load audio
            audio, sr = self.load_audio(file_path)
            
            # Get basic info
            duration = len(audio) / sr
            channels = 1 if len(audio.shape) == 1 else audio.shape[1]
            
            # Get file size
            file_size = os.path.getsize(file_path)
            file_size_mb = file_size / (1024 * 1024)
            
            return {
                "file_path": file_path,
                "duration": duration,
                "sample_rate": sr,
                "channels": channels,
                "samples": len(audio),
                "file_size_mb": file_size_mb,
                "format": Path(file_path).suffix.lower(),
                "is_valid": True
            }
            
        except Exception as e:
            return {
                "file_path": file_path,
                "is_valid": False,
                "error": str(e)
            }
