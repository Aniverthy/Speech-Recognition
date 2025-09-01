#!/usr/bin/env python3
"""
Pipeline Service

Orchestrates the entire voice recognition pipeline.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from .service_state import ServiceState
from .service_preprocess import PreprocessService
from .service_asr import ASRService
from .service_features import FeatureService
from .service_enroll import EnrollmentService
from .service_diarize import DiarizationService
from .service_align import AlignmentService
from .service_output import OutputService
from .service_base64 import Base64Service


class PipelineService:
    """Main pipeline service that coordinates all processing steps."""
    
    def __init__(self, state: ServiceState):
        self.state = state
        
        # Initialize all services
        self.preprocess = PreprocessService(state)
        self.asr = ASRService(state)
        self.features = FeatureService(state)
        self.enrollment = EnrollmentService(state, self.features)
        self.diarization = DiarizationService(state, self.features)
        self.alignment = AlignmentService(state)
        self.output = OutputService(state)
        self.base64 = Base64Service(state)
    
    def process_audio(self, audio_path: str, output_formats: List[str] = None) -> Dict[str, Any]:
        """
        Process audio file through the complete pipeline.
        
        Args:
            audio_path: Path to audio file
            output_formats: List of output formats (json, txt, csv, summary)
            
        Returns:
            Pipeline results and output file paths
        """
        if output_formats is None:
            output_formats = ["json", "txt", "csv", "summary"]
        
        results = {
            "input_file": audio_path,
            "success": False,
            "outputs": {},
            "errors": [],
            "processing_info": {}
        }
        
        try:
            # Step 1: Preprocess audio
            audio, sr = self.preprocess.load_audio(audio_path)
            audio_info = self.preprocess.validate_audio(audio, sr)
            audio = self.preprocess.apply_preprocessing(audio, sr)
            
            results["processing_info"]["audio"] = audio_info
            
            # Step 2: Perform ASR
            asr_segments = self.asr.transcribe_audio(audio_path)
            results["processing_info"]["asr"] = {
                "segments": len(asr_segments),
                "total_duration": sum(seg.get("duration", 0) for seg in asr_segments)
            }
            
            # Step 3: Perform diarization
            diarized_segments = self.diarization.perform_diarization(audio_path)
            results["processing_info"]["diarization"] = {
                "segments": len(diarized_segments),
                "speakers_detected": len(set(seg.get("clustered_speaker", "Unknown") for seg in diarized_segments))
            }
            
            # Step 4: Align ASR and diarization
            aligned_segments = self.alignment.align_segments(asr_segments, diarized_segments)
            
            # Step 5: Apply speaker enrollment mapping
            if self.enrollment.is_enrollment_available():
                mapped_segments, speaker_mapping = self.enrollment.map_speakers(aligned_segments)
                results["processing_info"]["enrollment"] = {
                    "profiles_loaded": len(self.enrollment.enrollment_profiles),
                    "speaker_mapping": speaker_mapping
                }
            else:
                mapped_segments = aligned_segments
                results["processing_info"]["enrollment"] = {"profiles_loaded": 0}
            
            # Step 6: Merge short segments
            final_segments = self.alignment.merge_short_segments(mapped_segments)
            
            # Step 7: Generate outputs
            base_filename = Path(audio_path).stem
            outputs = self.output.generate_all_outputs(final_segments, base_filename)
            
            # Filter requested formats
            for format_name in output_formats:
                if format_name in outputs:
                    results["outputs"][format_name] = outputs[format_name]
            
            results["success"] = True
            results["processing_info"]["final"] = {
                "segments": len(final_segments),
                "unique_speakers": len(set(seg.get("speaker", "Unknown") for seg in final_segments)),
                "total_duration": sum(seg.get("duration", 0) for seg in final_segments)
            }
            
        except Exception as e:
            results["errors"].append(str(e))
            results["success"] = False
        
        return results
    
    def process_base64(self, base64_string: str, audio_format: str = "wav", output_formats: List[str] = None) -> Dict[str, Any]:
        """
        Process Base64 encoded audio through the pipeline.
        
        Args:
            base64_string: Base64 encoded audio data
            audio_format: Audio format
            output_formats: List of output formats
            
        Returns:
            Pipeline results
        """
        try:
            # Decode Base64 to audio file
            audio_file = self.base64.decode_base64_audio(base64_string, audio_format)
            
            # Process the audio file
            results = self.process_audio(audio_file, output_formats)
            
            # Add Base64 info
            results["base64_info"] = self.base64.get_base64_info(base64_string)
            results["temp_audio_file"] = audio_file
            
            return results
            
        except Exception as e:
            return {
                "success": False,
                "errors": [str(e)],
                "input_type": "base64"
            }
    
    def process_base64_file(self, file_path: str, audio_format: str = "wav", output_formats: List[str] = None) -> Dict[str, Any]:
        """
        Process Base64 audio from text file.
        
        Args:
            file_path: Path to text file containing Base64 data
            audio_format: Audio format
            output_formats: List of output formats
            
        Returns:
            Pipeline results
        """
        try:
            # Decode Base64 file to audio file
            audio_file = self.base64.decode_base64_file(file_path, audio_format)
            
            # Process the audio file
            results = self.process_audio(audio_file, output_formats)
            
            # Add file info
            results["base64_file"] = file_path
            results["temp_audio_file"] = audio_file
            
            return results
            
        except Exception as e:
            return {
                "success": False,
                "errors": [str(e)],
                "input_type": "base64_file",
                "file_path": file_path
            }
    
    def process_base64_chunks(self, base64_chunks, audio_format: str = "wav", output_formats: List[str] = None) -> Dict[str, Any]:
        """
        Process an iterable/generator of Base64 chunks.
        
        Args:
            base64_chunks: Iterable or generator yielding Base64 string chunks
            audio_format: Audio format for decoding
            output_formats: List of output formats (json, txt, csv, summary)
            
        Returns:
            Pipeline results
        """
        try:
            # Decode chunks to a temp audio file
            audio_file = self.base64.decode_base64_stream(base64_chunks, audio_format)
            
            # Process the audio file
            results = self.process_audio(audio_file, output_formats)
            results["temp_audio_file"] = audio_file
            results["input_type"] = "base64_chunks"
            return results
        except Exception as e:
            return {
                "success": False,
                "errors": [str(e)],
                "input_type": "base64_chunks"
            }

    def process_directory(self, directory_path: str, output_formats: List[str] = None) -> Dict[str, Any]:
        """
        Process all audio files in a directory.
        
        Args:
            directory_path: Path to directory
            output_formats: List of output formats
            
        Returns:
            Directory processing results
        """
        if not os.path.isdir(directory_path):
            return {"success": False, "errors": ["Directory not found"]}
        
        results = {
            "directory": directory_path,
            "success": True,
            "files_processed": 0,
            "files_failed": 0,
            "results": [],
            "errors": []
        }
        
        # Find audio files
        audio_files = []
        for ext in self.state.SUPPORTED_FORMATS:
            audio_files.extend(Path(directory_path).glob(f"*{ext}"))
        
        if not audio_files:
            results["errors"].append("No audio files found in directory")
            results["success"] = False
            return results
        
        # Process each file
        for audio_file in audio_files:
            try:
                file_result = self.process_audio(str(audio_file), output_formats)
                results["results"].append(file_result)
                
                if file_result["success"]:
                    results["files_processed"] += 1
                else:
                    results["files_failed"] += 1
                    
            except Exception as e:
                results["files_failed"] += 1
                results["errors"].append(f"Failed to process {audio_file}: {e}")
        
        return results
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get pipeline service information."""
        return {
            "services": {
                "preprocess": self.preprocess.get_audio_info("test") if hasattr(self.preprocess, 'get_audio_info') else "Available",
                "asr": self.asr.get_model_info(),
                "features": self.features.get_feature_info(),
                "enrollment": self.enrollment.get_enrollment_info(),
                "diarization": self.diarization.get_diarization_info(),
                "alignment": "Available",
                "output": self.output.get_output_info(),
                "base64": self.base64.get_service_info()
            },
            "state": {
                "gpu_available": self.state.is_gpu_available(),
                "device": self.state.DEVICE,
                "compute_type": self.state.COMPUTE_TYPE
            }
        }
    
    def cleanup(self):
        """Clean up temporary files and resources."""
        try:
            # Clean up Base64 temp files
            self.base64.cleanup_temp_files()
            
            # Clean up other temp files
            for temp_file in self.state.TEMP_DIR.glob("*"):
                if temp_file.is_file():
                    temp_file.unlink(missing_ok=True)
        except Exception:
            pass
