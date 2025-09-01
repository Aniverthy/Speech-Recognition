#!/usr/bin/env python3
"""
ASR Service

Handles Automatic Speech Recognition using faster-whisper.
"""

import os
from typing import List, Dict, Any
from .service_state import ServiceState


class ASRService:
    """ASR service using faster-whisper backend."""
    
    def __init__(self, state: ServiceState):
        self.state = state
        self.asr_config = state.get_asr_config()
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize faster-whisper model."""
        try:
            from faster_whisper import WhisperModel
            
            self.model = WhisperModel(
                self.asr_config["model_size"],
                device=self.asr_config["device"],
                compute_type=self.asr_config["compute_type"]
            )
        except Exception:
            self.model = None
    
    def transcribe_audio(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        Transcribe audio file to text with timestamps.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of transcription segments
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if self.model is None:
            raise RuntimeError("ASR model not available")
        
        try:
            # Transcribe with faster-whisper
            segments_result, info = self.model.transcribe(
                audio_path,
                word_timestamps=self.asr_config["word_timestamps"],
                beam_size=self.asr_config["beam_size"]
            )
            
            # Process segments
            segments = []
            for segment in segments_result:
                segments.append({
                    "start_time": float(segment.start),
                    "end_time": float(segment.end),
                    "duration": float(segment.end - segment.start),
                    "text": segment.text.strip(),
                    "confidence": getattr(segment, 'avg_logprob', 0.0)
                })
            
            return segments
            
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}")
    
    def transcribe_chunks(self, audio_chunks: List, sample_rate: int) -> List[Dict[str, Any]]:
        """
        Transcribe audio chunks.
        
        Args:
            audio_chunks: List of audio chunks
            sample_rate: Sample rate
            
        Returns:
            List of transcription segments
        """
        if self.model is None:
            raise RuntimeError("ASR model not available")
        
        all_segments = []
        current_time = 0.0
        
        for i, chunk in enumerate(audio_chunks):
            try:
                # Save chunk to temporary file
                temp_path = self.state.TEMP_DIR / f"chunk_{i}.wav"
                import soundfile as sf
                sf.write(str(temp_path), chunk, sample_rate)
                
                # Transcribe chunk
                chunk_segments = self.transcribe_audio(str(temp_path))
                
                # Adjust timestamps
                for segment in chunk_segments:
                    segment["start_time"] += current_time
                    segment["end_time"] += current_time
                    all_segments.append(segment)
                
                # Update current time
                if chunk_segments:
                    current_time = chunk_segments[-1]["end_time"]
                else:
                    current_time += len(chunk) / sample_rate
                
                # Clean up temp file
                temp_path.unlink(missing_ok=True)
                
            except Exception:
                # Continue with next chunk if one fails
                current_time += len(chunk) / sample_rate
                continue
        
        return all_segments
    
    def is_available(self) -> bool:
        """Check if ASR service is available."""
        return self.model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "backend": self.asr_config["backend"],
            "model_size": self.asr_config["model_size"],
            "device": self.asr_config["device"],
            "compute_type": self.asr_config["compute_type"],
            "model_loaded": self.model is not None
        }
