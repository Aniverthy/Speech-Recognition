#!/usr/bin/env python3
"""
Diarization Service

Handles speaker diarization and clustering.
"""

import numpy as np
import librosa
from typing import List, Dict, Any
from sklearn.cluster import AgglomerativeClustering
from .service_state import ServiceState
from .service_features import FeatureService


class DiarizationService:
    """Speaker diarization service."""
    
    def __init__(self, state: ServiceState, feature_service: FeatureService):
        self.state = state
        self.feature_service = feature_service
        self.speaker_config = state.get_speaker_config()
    
    def perform_diarization(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        Perform speaker diarization on audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of segments with speaker labels
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        # Voice activity detection
        voice_segments = self._voice_activity_detection(audio, sr)
        
        if len(voice_segments) < 2:
            return voice_segments
        
        # Extract features for each segment
        for segment in voice_segments:
            start_sample = int(segment["start_time"] * sr)
            end_sample = int(segment["end_time"] * sr)
            segment_audio = audio[start_sample:end_sample]
            
            # Extract features
            embedding, features = self.feature_service.extract_embedding_and_features(segment_audio, sr)
            
            if embedding is not None:
                segment["embedding"] = embedding
            segment["features"] = features
        
        # Perform clustering
        clustered_segments = self._cluster_speakers(voice_segments)
        
        return clustered_segments
    
    def _voice_activity_detection(self, audio: np.ndarray, sr: int) -> List[Dict[str, Any]]:
        """
        Perform voice activity detection.
        
        Args:
            audio: Audio waveform
            sr: Sample rate
            
        Returns:
            List of voice segments
        """
        # Compute energy
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.010 * sr)    # 10ms hop
        
        energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)
        energy = energy.flatten()
        
        # Threshold-based detection
        threshold = np.percentile(energy, 30)
        voice_frames = energy > threshold
        
        # Convert to segments
        segments = []
        start_frame = None
        
        for i, is_voice in enumerate(voice_frames):
            if is_voice and start_frame is None:
                start_frame = i
            elif not is_voice and start_frame is not None:
                # End of voice segment
                start_time = start_frame * hop_length / sr
                end_time = i * hop_length / sr
                duration = end_time - start_time
                
                if duration >= self.state.MIN_SEGMENT_DURATION:
                    segments.append({
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration": duration
                    })
                
                start_frame = None
        
        # Handle last segment
        if start_frame is not None:
            start_time = start_frame * hop_length / sr
            end_time = len(audio) / sr
            duration = end_time - start_time
            
            if duration >= self.state.MIN_SEGMENT_DURATION:
                segments.append({
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": duration
                })
        
        return segments
    
    def _cluster_speakers(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Cluster segments by speaker.
        
        Args:
            segments: List of segments with features
            
        Returns:
            Segments with speaker labels
        """
        if len(segments) < 2:
            return segments
        
        # Prepare feature matrix
        features_list = []
        for segment in segments:
            if "embedding" in segment:
                features_list.append(segment["embedding"])
            else:
                features_list.append(segment["features"])
        
        features_matrix = np.array(features_list)
        
        # Determine number of clusters
        n_clusters = min(len(segments), 10)  # Max 10 speakers
        
        # Try different cluster counts
        best_n_clusters = 1
        best_silhouette = -1
        
        for n in range(1, min(n_clusters + 1, len(segments))):
            try:
                clustering = AgglomerativeClustering(
                    n_clusters=n,
                    metric='cosine',
                    linkage='average'
                )
                labels = clustering.fit_predict(features_matrix)
                
                # Compute silhouette score
                from sklearn.metrics import silhouette_score
                silhouette = silhouette_score(features_matrix, labels, metric='cosine')
                
                if silhouette > best_silhouette:
                    best_silhouette = silhouette
                    best_n_clusters = n
                    
            except Exception:
                continue
        
        # Perform final clustering
        try:
            final_clustering = AgglomerativeClustering(
                n_clusters=best_n_clusters,
                metric='cosine',
                linkage='average'
            )
            final_labels = final_clustering.fit_predict(features_matrix)
            
            # Assign labels to segments
            for i, segment in enumerate(segments):
                segment["clustered_speaker"] = f"User{final_labels[i] + 1}"
                
        except Exception:
            # Fallback: assign all to User1
            for segment in segments:
                segment["clustered_speaker"] = "User1"
        
        return segments
    
    def get_diarization_info(self) -> Dict[str, Any]:
        """Get diarization service information."""
        return {
            "clustering_threshold": self.speaker_config["clustering_threshold"],
            "min_segment_duration": self.state.MIN_SEGMENT_DURATION,
            "max_segment_duration": self.state.MAX_SEGMENT_DURATION
        }
