#!/usr/bin/env python3
"""
Enrollment Service

Handles speaker enrollment and profile management.
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from .service_state import ServiceState
from .service_features import FeatureService


class EnrollmentService:
    """Speaker enrollment service for voice identification."""
    
    def __init__(self, state: ServiceState, feature_service: FeatureService):
        self.state = state
        self.feature_service = feature_service
        self.enrollment_profiles = {}
        self.speaker_name_map = {}
        self.enrollment_dir = state.ENROLLMENT_DIR
        
        # Load existing profiles if available
        if self.enrollment_dir.exists():
            self.load_profiles()
    
    def load_profiles(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Load enrollment profiles from directory.
        
        Returns:
            Dictionary of enrollment profiles
        """
        if not self.enrollment_dir.exists():
            return {}
        
        # Find audio files
        audio_files = []
        for ext in self.state.SUPPORTED_FORMATS:
            audio_files.extend(self.enrollment_dir.rglob(f"*{ext}"))
        
        if not audio_files:
            return {}
        
        # Group by speaker
        speaker_files = {}
        for audio_file in audio_files:
            if audio_file.parent != self.enrollment_dir:
                # Hierarchical layout
                speaker_name = audio_file.parent.name
            else:
                # Flat layout
                speaker_name = audio_file.stem.split('_')[0]
            
            if speaker_name not in speaker_files:
                speaker_files[speaker_name] = []
            speaker_files[speaker_name].append(audio_file)
        
        # Process each speaker
        for speaker_name, files in speaker_files.items():
            embeddings = []
            features = []
            
            for audio_file in files:
                try:
                    # Load audio
                    import librosa
                    audio, sr = librosa.load(str(audio_file), sr=16000, mono=True)
                    
                    # Extract features
                    embedding, feature = self.feature_service.extract_embedding_and_features(audio, sr)
                    
                    if embedding is not None:
                        embeddings.append(embedding)
                    features.append(feature)
                    
                except Exception:
                    continue
            
            # Create profile
            profile = {}
            if embeddings:
                avg_embedding = np.mean(embeddings, axis=0)
                avg_embedding = self.feature_service.normalize_embedding(avg_embedding)
                profile["embedding"] = avg_embedding.astype(np.float32)
            
            if features:
                avg_features = np.mean(features, axis=0)
                avg_features = avg_features / (np.linalg.norm(avg_features) + 1e-8)
                profile["features"] = avg_features.astype(np.float32)
            
            if profile:
                self.enrollment_profiles[speaker_name] = profile
        
        return self.enrollment_profiles
    
    def map_speakers(self, segments: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        """
        Map detected speakers to enrolled names.
        
        Args:
            segments: List of segments with speaker labels
            
        Returns:
            Tuple of (updated segments, speaker mapping)
        """
        if not self.enrollment_profiles:
            return segments, {}
        
        # Group segments by speaker
        speaker_groups = {}
        for segment in segments:
            speaker = segment.get("speaker") or segment.get("clustered_speaker") or "User1"
            if speaker not in speaker_groups:
                speaker_groups[speaker] = []
            speaker_groups[speaker].append(segment)
        
        # Build cluster representatives
        cluster_representatives = {}
        for speaker, group_segments in speaker_groups.items():
            embeddings = []
            features = []
            
            for segment in group_segments:
                if "embedding" in segment:
                    embeddings.append(segment["embedding"])
                if "features" in segment:
                    features.append(segment["features"])
            
            if embeddings:
                cluster_representatives[speaker] = {
                    "embedding": np.mean(embeddings, axis=0),
                    "type": "embedding"
                }
            elif features:
                cluster_representatives[speaker] = {
                    "features": np.mean(features, axis=0),
                    "type": "features"
                }
        
        # Prepare enrollment vectors
        enrollment_vectors = {}
        for name, profile in self.enrollment_profiles.items():
            if "embedding" in profile:
                enrollment_vectors[name] = {
                    "embedding": profile["embedding"],
                    "type": "embedding"
                }
            elif "features" in profile:
                enrollment_vectors[name] = {
                    "features": profile["features"],
                    "type": "features"
                }
        
        # Perform mapping
        speaker_name_map = {}
        used_names = set()
        
        # Sort clusters by size
        sorted_clusters = sorted(
            cluster_representatives.items(),
            key=lambda x: len(speaker_groups[x[0]]),
            reverse=True
        )
        
        for cluster_speaker, cluster_data in sorted_clusters:
            best_match = None
            best_score = -1
            
            for name, enroll_data in enrollment_vectors.items():
                if name in used_names:
                    continue
                
                if cluster_data["type"] != enroll_data["type"]:
                    continue
                
                # Compute similarity
                if cluster_data["type"] == "embedding":
                    similarity = self.feature_service.compute_similarity(
                        cluster_data["embedding"], enroll_data["embedding"]
                    )
                    threshold = self.state.EMBEDDING_THRESHOLD
                else:
                    similarity = self.feature_service.compute_similarity(
                        cluster_data["features"], enroll_data["features"]
                    )
                    threshold = self.state.FEATURES_THRESHOLD
                
                if similarity > best_score and similarity >= threshold:
                    best_score = similarity
                    best_match = name
            
            if best_match:
                speaker_name_map[cluster_speaker] = best_match
                used_names.add(best_match)
        
        # Apply mapping
        updated_segments = []
        for segment in segments:
            updated_segment = segment.copy()
            speaker = segment.get("speaker") or segment.get("clustered_speaker") or "User1"
            
            if speaker in speaker_name_map:
                updated_segment["speaker"] = speaker_name_map[speaker]
                if "clustered_speaker" in updated_segment:
                    updated_segment["clustered_speaker"] = speaker_name_map[speaker]
            
            updated_segments.append(updated_segment)
        
        return updated_segments, speaker_name_map
    
    def get_enrollment_info(self) -> Dict[str, Any]:
        """Get enrollment status information."""
        return {
            "enrollment_directory": str(self.enrollment_dir),
            "directory_exists": self.enrollment_dir.exists(),
            "profiles_loaded": len(self.enrollment_profiles),
            "speaker_names": list(self.enrollment_profiles.keys()),
            "speaker_name_map": self.speaker_name_map.copy()
        }
    
    def is_enrollment_available(self) -> bool:
        """Check if enrollment profiles are available."""
        return len(self.enrollment_profiles) > 0
    
    def reload_profiles(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Reload enrollment profiles from disk."""
        self.enrollment_profiles.clear()
        self.speaker_name_map.clear()
        return self.load_profiles()
