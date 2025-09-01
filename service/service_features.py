#!/usr/bin/env python3
"""
Feature Extraction Service

Handles voice embedding extraction and spectral feature computation using Resemblyzer.
"""

import numpy as np
import librosa
from typing import Tuple, Optional, Dict, Any
from .service_state import ServiceState


class FeatureService:
    """Feature extraction service for voice analysis using Resemblyzer."""
    
    def __init__(self, state: ServiceState):
        self.state = state
        self.speaker_config = state.get_speaker_config()
        self.resemblyzer_encoder = None
        self._initialize_encoders()
    
    def _initialize_encoders(self):
        """Initialize Resemblyzer encoder."""
        try:
            from resemblyzer import VoiceEncoder
            self.resemblyzer_encoder = VoiceEncoder()
        except Exception:
            self.resemblyzer_encoder = None
    
    def extract_embedding_and_features(self, waveform: np.ndarray, sample_rate: int) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        Extract voice embedding and spectral features.
        
        Args:
            waveform: Audio waveform
            sample_rate: Sample rate
            
        Returns:
            Tuple of (embedding, features)
        """
        embedding = None
        
        # Use Resemblyzer encoder
        if self.resemblyzer_encoder is not None:
            try:
                # Resample to 16kHz if needed
                if sample_rate != 16000:
                    wav16 = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
                else:
                    wav16 = waveform
                
                # Extract embedding
                embedding = self.resemblyzer_encoder.embed_utterance(wav16.astype(np.float32))
                embedding = embedding.astype(np.float32)
                
            except Exception:
                pass
        
        # Always extract spectral features as fallback
        features = self._extract_spectral_features(waveform, sample_rate)
        
        return embedding, features
    
    def _extract_spectral_features(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Extract MFCC and spectral features.
        
        Args:
            waveform: Audio waveform
            sample_rate: Sample rate
            
        Returns:
            Feature vector
        """
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=self.speaker_config["mfcc_features"])
        
        # Add delta features
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Extract spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=waveform, sr=sample_rate)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sample_rate)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=waveform, sr=sample_rate)
        
        # Concatenate all features
        features = np.concatenate([
            mfcc.flatten(),
            mfcc_delta.flatten(),
            mfcc_delta2.flatten(),
            spectral_centroid.flatten(),
            spectral_rolloff.flatten(),
            spectral_bandwidth.flatten()
        ])
        
        # Normalize
        features = (features - np.mean(features)) / (np.std(features) + 1e-8)
        # print(features.astype(np.float32))
        return features.astype(np.float32)
    
    def normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        L2 normalize embedding vector.
        
        Args:
            embedding: Embedding vector
            
        Returns:
            Normalized embedding
        """
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding
    
    def compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Similarity score
        """
        # Ensure vectors are 1D
        vec1 = vec1.flatten()
        vec2 = vec2.flatten()
        
        # Compute cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 > 0 and norm2 > 0:
            return dot_product / (norm1 * norm2)
        return 0.0
    
    def is_encoder_available(self) -> bool:
        """Check if Resemblyzer encoder is available."""
        return self.resemblyzer_encoder is not None
    
    def get_feature_info(self) -> Dict[str, Any]:
        """Get feature extraction information."""
        return {
            "mfcc_features": self.speaker_config["mfcc_features"],
            "resemblyzer_available": self.resemblyzer_encoder is not None,
            "embedding_threshold": self.speaker_config["embedding_threshold"],
            "features_threshold": self.speaker_config["features_threshold"]
        }
