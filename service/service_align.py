#!/usr/bin/env python3
"""
Alignment Service

Handles audio-text alignment and synchronization.
"""

from typing import List, Dict, Any, Tuple
from .service_state import ServiceState


class AlignmentService:
    """Audio-text alignment service."""
    
    def __init__(self, state: ServiceState):
        self.state = state
    
    def align_segments(self, asr_segments: List[Dict[str, Any]], diarized_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Align ASR segments with diarized segments.
        
        Args:
            asr_segments: ASR transcription segments
            diarized_segments: Speaker diarization segments
            
        Returns:
            Aligned segments with both text and speaker info
        """
        if not asr_segments or not diarized_segments:
            return []
        
        # Sort segments by start time
        asr_segments = sorted(asr_segments, key=lambda x: x["start_time"])
        diarized_segments = sorted(diarized_segments, key=lambda x: x["start_time"])
        
        aligned_segments = []
        
        for asr_seg in asr_segments:
            # Find overlapping diarized segments
            overlapping = []
            for diarized_seg in diarized_segments:
                if self._segments_overlap(asr_seg, diarized_seg):
                    overlap = self._compute_overlap(asr_seg, diarized_seg)
                    overlapping.append((diarized_seg, overlap))
            
            if overlapping:
                # Sort by overlap amount
                overlapping.sort(key=lambda x: x[1], reverse=True)
                best_match = overlapping[0][0]
                
                # Create aligned segment
                aligned_seg = asr_seg.copy()
                aligned_seg["speaker"] = best_match.get("clustered_speaker", "Unknown")
                aligned_seg["embedding"] = best_match.get("embedding")
                aligned_seg["features"] = best_match.get("features")
                
                aligned_segments.append(aligned_seg)
            else:
                # No overlap found, create segment with unknown speaker
                aligned_seg = asr_seg.copy()
                aligned_seg["speaker"] = "Unknown"
                aligned_segments.append(aligned_seg)
        
        return aligned_segments
    
    def _segments_overlap(self, seg1: Dict[str, Any], seg2: Dict[str, Any]) -> bool:
        """
        Check if two segments overlap in time.
        
        Args:
            seg1: First segment
            seg2: Second segment
            
        Returns:
            True if segments overlap
        """
        start1, end1 = seg1["start_time"], seg1["end_time"]
        start2, end2 = seg2["start_time"], seg2["end_time"]
        
        return start1 < end2 and start2 < end1
    
    def _compute_overlap(self, seg1: Dict[str, Any], seg2: Dict[str, Any]) -> float:
        """
        Compute overlap amount between segments.
        
        Args:
            seg1: First segment
            seg2: Second segment
            
        Returns:
            Overlap duration
        """
        start1, end1 = seg1["start_time"], seg1["end_time"]
        start2, end2 = seg2["start_time"], seg2["end_time"]
        
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        
        return max(0, overlap_end - overlap_start)
    
    def merge_short_segments(self, segments: List[Dict[str, Any]], min_duration: float = 0.5) -> List[Dict[str, Any]]:
        """
        Merge short segments with adjacent ones.
        
        Args:
            segments: List of segments
            min_duration: Minimum duration threshold
            
        Returns:
            Merged segments
        """
        if not segments:
            return []
        
        # Sort by start time
        segments = sorted(segments, key=lambda x: x["start_time"])
        
        merged = []
        current = segments[0].copy()
        
        for next_seg in segments[1:]:
            # Check if we can merge
            if (next_seg["speaker"] == current["speaker"] and 
                next_seg["start_time"] - current["end_time"] < 0.1 and
                current["duration"] < min_duration):
                
                # Merge segments
                current["end_time"] = next_seg["end_time"]
                current["duration"] = current["end_time"] - current["start_time"]
                current["text"] = current["text"] + " " + next_seg["text"]
                
            else:
                # Add current segment and start new one
                merged.append(current)
                current = next_seg.copy()
        
        # Add last segment
        merged.append(current)
        
        return merged
    
    def validate_alignment(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate alignment quality.
        
        Args:
            segments: Aligned segments
            
        Returns:
            Validation results
        """
        if not segments:
            return {"is_valid": False, "errors": ["No segments to validate"]}
        
        errors = []
        warnings = []
        
        # Check for missing speaker labels
        missing_speakers = [seg for seg in segments if not seg.get("speaker")]
        if missing_speakers:
            errors.append(f"{len(missing_speakers)} segments missing speaker labels")
        
        # Check for timing issues
        for i, seg in enumerate(segments):
            if seg["start_time"] >= seg["end_time"]:
                errors.append(f"Segment {i}: invalid timing (start >= end)")
            
            if seg["duration"] < 0:
                errors.append(f"Segment {i}: negative duration")
        
        # Check for gaps
        gaps = []
        for i in range(1, len(segments)):
            prev_end = segments[i-1]["end_time"]
            curr_start = segments[i]["start_time"]
            gap = curr_start - prev_end
            
            if gap > 1.0:  # Gap larger than 1 second
                gaps.append(gap)
        
        if gaps:
            warnings.append(f"Found {len(gaps)} gaps larger than 1 second")
        
        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "total_segments": len(segments),
            "segments_with_speakers": len([s for s in segments if s.get("speaker")]),
            "avg_gap": sum(gaps) / len(gaps) if gaps else 0
        }
