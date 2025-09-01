#!/usr/bin/env python3
"""
Output Service

Handles multiple output formats and file generation.
"""

import json
import csv
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from .service_state import ServiceState


class OutputService:
    """Output generation service for multiple formats."""
    
    def __init__(self, state: ServiceState):
        self.state = state
        self.output_dir = state.get_output_path()
    
    def _convert_numpy_types(self, obj):
        """
        Convert numpy types to Python types for JSON serialization.
        
        Args:
            obj: Object to convert
            
        Returns:
            Converted object
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    def generate_json_output(self, segments: List[Dict[str, Any]], filename: str = None) -> str:
        """
        Generate JSON output with complete analysis.
        
        Args:
            segments: List of processed segments
            filename: Output filename (without extension)
            
        Returns:
            Path to generated file
        """
        if filename is None:
            filename = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        output_path = self.output_dir / f"{filename}.json"
        
        # Prepare output data
        output_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_segments": len(segments),
                "total_duration": sum(seg.get("duration", 0) for seg in segments),
                "unique_speakers": len(set(seg.get("speaker", "Unknown") for seg in segments))
            },
            "segments": segments
        }
        
        # Convert numpy types to Python types for JSON serialization
        output_data = self._convert_numpy_types(output_data)
        
        # Write JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        return str(output_path)
    
    def generate_text_output(self, segments: List[Dict[str, Any]], filename: str = None) -> str:
        """
        Generate human-readable text output.
        
        Args:
            segments: List of processed segments
            filename: Output filename (without extension)
            
        Returns:
            Path to generated file
        """
        if filename is None:
            filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        output_path = self.output_dir / f"{filename}.txt"
        
        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda x: x["start_time"])
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("Voice Recognition Analysis\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Segments: {len(segments)}\n")
            f.write(f"Total Duration: {sum(seg.get('duration', 0) for seg in segments):.2f}s\n\n")
            
            f.write("Conversation:\n")
            f.write("-" * 30 + "\n\n")
            
            for i, segment in enumerate(sorted_segments):
                speaker = segment.get("speaker", "Unknown")
                start_time = segment.get("start_time", 0)
                end_time = segment.get("end_time", 0)
                text = segment.get("text", "")
                confidence = segment.get("confidence", 0)
                
                f.write(f"[{i+1:03d}] {start_time:06.2f}s - {end_time:06.2f}s | {speaker}\n")
                f.write(f"       {text}\n")
                if confidence != 0:
                    f.write(f"       Confidence: {confidence:.3f}\n")
                f.write("\n")
        
        return str(output_path)
    
    def generate_csv_output(self, segments: List[Dict[str, Any]], filename: str = None) -> str:
        """
        Generate CSV output for analysis.
        
        Args:
            segments: List of processed segments
            filename: Output filename (without extension)
            
        Returns:
            Path to generated file
        """
        if filename is None:
            filename = f"segments_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        output_path = self.output_dir / f"{filename}.csv"
        
        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda x: x["start_time"])
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                "Segment", "Start_Time", "End_Time", "Duration", "Speaker", 
                "Text", "Confidence", "Embedding_Available", "Features_Available"
            ])
            
            # Write data
            for i, segment in enumerate(sorted_segments):
                writer.writerow([
                    i + 1,
                    f"{segment.get('start_time', 0):.3f}",
                    f"{segment.get('end_time', 0):.3f}",
                    f"{segment.get('duration', 0):.3f}",
                    segment.get("speaker", "Unknown"),
                    segment.get("text", ""),
                    f"{segment.get('confidence', 0):.3f}",
                    "embedding" in segment,
                    "features" in segment
                ])
        
        return str(output_path)
    
    def generate_summary_report(self, segments: List[Dict[str, Any]], filename: str = None) -> str:
        """
        Generate summary report.
        
        Args:
            segments: List of processed segments
            filename: Output filename (without extension)
            
        Returns:
            Path to generated file
        """
        if filename is None:
            filename = f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        output_path = self.output_dir / f"{filename}.txt"
        
        # Calculate statistics
        total_duration = sum(seg.get("duration", 0) for seg in segments)
        speakers = {}
        total_words = 0
        
        for segment in segments:
            speaker = segment.get("speaker", "Unknown")
            duration = segment.get("duration", 0)
            text = segment.get("text", "")
            words = len(text.split())
            
            if speaker not in speakers:
                speakers[speaker] = {"duration": 0, "words": 0, "segments": 0}
            
            speakers[speaker]["duration"] += duration
            speakers[speaker]["words"] += words
            speakers[speaker]["segments"] += 1
            total_words += words
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("Voice Recognition Summary Report\n")
            f.write("=" * 40 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Overall Statistics:\n")
            f.write(f"  Total Segments: {len(segments)}\n")
            f.write(f"  Total Duration: {total_duration:.2f} seconds\n")
            f.write(f"  Total Words: {total_words}\n")
            f.write(f"  Unique Speakers: {len(speakers)}\n\n")
            
            f.write("Speaker Analysis:\n")
            f.write("-" * 20 + "\n")
            
            # Sort speakers by duration
            sorted_speakers = sorted(speakers.items(), key=lambda x: x[1]["duration"], reverse=True)
            
            for speaker, stats in sorted_speakers:
                percentage = (stats["duration"] / total_duration) * 100 if total_duration > 0 else 0
                f.write(f"\n{speaker}:\n")
                f.write(f"  Duration: {stats['duration']:.2f}s ({percentage:.1f}%)\n")
                f.write(f"  Segments: {stats['segments']}\n")
                f.write(f"  Words: {stats['words']}\n")
                f.write(f"  Avg Words/Segment: {stats['words']/stats['segments']:.1f}\n")
        
        return str(output_path)
    
    def generate_all_outputs(self, segments: List[Dict[str, Any]], base_filename: str = None) -> Dict[str, str]:
        """
        Generate all output formats.
        
        Args:
            segments: List of processed segments
            base_filename: Base filename for outputs
            
        Returns:
            Dictionary of output file paths
        """
        outputs = {}
        
        outputs["json"] = self.generate_json_output(segments, base_filename)
        outputs["text"] = self.generate_text_output(segments, base_filename)
        outputs["csv"] = self.generate_csv_output(segments, base_filename)
        outputs["summary"] = self.generate_summary_report(segments, base_filename)
        
        return outputs
    
    def get_output_info(self) -> Dict[str, Any]:
        """Get output service information."""
        return {
            "output_directory": str(self.output_dir),
            "directory_exists": self.output_dir.exists(),
            "available_formats": ["json", "text", "csv", "summary"]
        }
