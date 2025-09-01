#!/usr/bin/env python3
"""
Service State Management

Manages global state, configuration, and shared resources across all services.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import torch


class ServiceState:
    """Global service state and configuration manager."""
    
    def __init__(self):
        # Audio processing settings
        self.TARGET_SAMPLE_RATE = 16000  # Hz - standard for voice recognition
        self.SUPPORTED_FORMATS = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.m4b', '.aac']
        self.CHUNK_DURATION = 3.0  # seconds - audio chunk size
        
        # ASR settings
        self.ASR_BACKEND = "faster-whisper"  # Only faster-whisper backend
        self.DEFAULT_MODEL_SIZE = "base"  # Model size: tiny, base, small, medium, large-v3
        self.WORD_TIMESTAMPS = True  # Enable word-level timing
        self.BEAM_SIZE = 5  # Beam search size
        
        # Speaker identification settings
        self.MFCC_FEATURES = 13  # Number of MFCC coefficients
        self.EMBEDDING_THRESHOLD = 0.65  # Neural embedding similarity threshold
        self.FEATURES_THRESHOLD = 0.40  # Spectral features similarity threshold
        self.CLUSTERING_THRESHOLD = 0.7  # Speaker clustering threshold
        
        # File paths
        self.OUTPUT_DIR = Path("out")
        self.TEMP_DIR = Path("temp")
        self.MODEL_CACHE = Path("temp_speechbrain")
        self.ENROLLMENT_DIR = Path("Reference")
        
        # Device settings
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.COMPUTE_TYPE = "float16" if self.DEVICE == "cuda" else "int8"
        
        # Processing limits
        self.MAX_AUDIO_DURATION = 3600  # seconds - maximum audio length
        self.MIN_SEGMENT_DURATION = 0.5  # seconds - minimum segment length
        self.MAX_SEGMENT_DURATION = 30.0  # seconds - maximum segment length
        
        # Initialize directories
        self._setup_directories()
    
    def _setup_directories(self):
        """Create necessary directories if they don't exist."""
        for directory in [self.OUTPUT_DIR, self.TEMP_DIR, self.MODEL_CACHE]:
            directory.mkdir(exist_ok=True)
    
    def get_audio_config(self) -> Dict[str, Any]:
        """Get audio processing configuration."""
        return {
            "target_sample_rate": self.TARGET_SAMPLE_RATE,
            "supported_formats": self.SUPPORTED_FORMATS,
            "chunk_duration": self.CHUNK_DURATION,
            "min_segment_duration": self.MIN_SEGMENT_DURATION,
            "max_segment_duration": self.MAX_SEGMENT_DURATION
        }
    
    def get_asr_config(self) -> Dict[str, Any]:
        """Get ASR configuration."""
        return {
            "backend": self.ASR_BACKEND,
            "model_size": self.DEFAULT_MODEL_SIZE,
            "word_timestamps": self.WORD_TIMESTAMPS,
            "beam_size": self.BEAM_SIZE,
            "device": self.DEVICE,
            "compute_type": self.COMPUTE_TYPE
        }
    
    def get_speaker_config(self) -> Dict[str, Any]:
        """Get speaker identification configuration."""
        return {
            "mfcc_features": self.MFCC_FEATURES,
            "embedding_threshold": self.EMBEDDING_THRESHOLD,
            "features_threshold": self.FEATURES_THRESHOLD,
            "clustering_threshold": self.CLUSTERING_THRESHOLD
        }
    
    def get_paths(self) -> Dict[str, Path]:
        """Get file path configuration."""
        return {
            "output": self.OUTPUT_DIR,
            "temp": self.TEMP_DIR,
            "model_cache": self.MODEL_CACHE,
            "enrollment": self.ENROLLMENT_DIR
        }
    
    def is_gpu_available(self) -> bool:
        """Check if GPU is available."""
        return self.DEVICE == "cuda"
    
    def get_model_cache_path(self) -> Path:
        """Get model cache directory path."""
        return self.MODEL_CACHE
    
    def get_output_path(self) -> Path:
        """Get output directory path."""
        return self.OUTPUT_DIR
    
    def get_temp_directory(self) -> Path:
        """Get temporary directory path."""
        return self.TEMP_DIR
