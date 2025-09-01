#!/usr/bin/env python3
"""
Base64 Service

Handles Base64 encoded audio input and conversion.
"""

import base64
import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Iterable
from .service_state import ServiceState


class Base64Service:
    """Base64 audio handling service."""
    
    def __init__(self, state: ServiceState):
        self.state = state
        self.temp_dir = state.get_temp_directory()
    
    def decode_base64_audio(self, base64_string: str, audio_format: str = "wav", filename: str = None) -> str:
        """
        Decode Base64 string to audio file.
        
        Args:
            base64_string: Base64 encoded audio data
            audio_format: Audio format (wav, mp3, etc.)
            filename: Optional filename (without extension)
            
        Returns:
            Path to decoded audio file
        """
        try:
            # Decode Base64 string
            audio_data = base64.b64decode(base64_string)
            
            # Generate filename
            if filename is None:
                filename = f"base64_audio_{int(os.urandom(4).hex(), 16)}"
            
            # Create output path
            output_path = self.temp_dir / f"{filename}.{audio_format}"
            
            # Write audio file
            with open(output_path, 'wb') as f:
                f.write(audio_data)
            
            return str(output_path)
            
        except Exception as e:
            raise ValueError(f"Failed to decode Base64 audio: {e}")
    
    def decode_base64_file(self, file_path: str, audio_format: str = "wav", filename: str = None) -> str:
        """
        Decode Base64 audio from text file.
        
        Args:
            file_path: Path to text file containing Base64 data
            audio_format: Audio format (wav, mp3, etc.)
            filename: Optional filename (without extension)
            
        Returns:
            Path to decoded audio file
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Base64 file not found: {file_path}")
        
        try:
            # Read Base64 string from file
            with open(file_path, 'r', encoding='utf-8') as f:
                base64_string = f.read().strip()
            
            # Decode the string
            return self.decode_base64_audio(base64_string, audio_format, filename)
            
        except Exception as e:
            raise ValueError(f"Failed to read Base64 file: {e}")
    
    def decode_base64_stream(self, base64_chunks: Iterable[str], audio_format: str = "wav", filename: str = None) -> str:
        """
        Incrementally decode a stream/iterable of Base64 chunks into an audio file.
        
        This safely handles arbitrary chunk boundaries by buffering incomplete
        Base64 quanta (multiples of 4 characters) and decoding only the
        complete portion on each iteration.
        
        Args:
            base64_chunks: Iterable of Base64 string chunks
            audio_format: Output audio format (e.g., wav, mp3)
            filename: Optional filename (without extension)
            
        Returns:
            Path to the decoded audio file
        """
        try:
            if filename is None:
                filename = f"base64_audio_{int(os.urandom(4).hex(), 16)}"
            output_path = self.temp_dir / f"{filename}.{audio_format}"

            # Buffer keeps leftover base64 chars that do not make a full quantum of 4
            buffer = ""
            is_urlsafe: Optional[bool] = None

            with open(output_path, 'wb') as out_f:
                for chunk in base64_chunks:
                    if not chunk:
                        continue

                    # Strip common data URI prefix if provided in the first chunk(s)
                    if "base64," in chunk:
                        chunk = chunk.split("base64,", 1)[1]

                    # Remove whitespace/newlines and normalize
                    chunk = "".join(str(chunk).split())
                    if not chunk:
                        continue

                    # Detect urlsafe alphabet once
                    if is_urlsafe is None:
                        is_urlsafe = ('-' in chunk) or ('_' in chunk)

                    buffer += chunk

                    # Decode only complete 4-char quanta to avoid padding errors
                    complete_len = len(buffer) - (len(buffer) % 4)
                    if complete_len >= 4:
                        to_decode = buffer[:complete_len]
                        if is_urlsafe:
                            decoded = base64.urlsafe_b64decode(to_decode)
                        else:
                            decoded = base64.b64decode(to_decode)
                        out_f.write(decoded)
                        buffer = buffer[complete_len:]

                # Finalize: decode any remaining buffer with padding
                if buffer:
                    pad_len = (-len(buffer)) % 4
                    to_decode = buffer + ("=" * pad_len)
                    if is_urlsafe:
                        decoded = base64.urlsafe_b64decode(to_decode)
                    else:
                        decoded = base64.b64decode(to_decode)
                    out_f.write(decoded)

            return str(output_path)
        except Exception as e:
            raise ValueError(f"Failed to decode Base64 stream: {e}")
    
    def validate_base64_string(self, base64_string: str) -> Dict[str, Any]:
        """
        Validate Base64 string.
        
        Args:
            base64_string: Base64 string to validate
            
        Returns:
            Validation results
        """
        result = {
            "is_valid": False,
            "length": len(base64_string),
            "estimated_size_mb": 0,
            "errors": []
        }
        
        try:
            # Check if string is valid Base64
            decoded = base64.b64decode(base64_string)
            result["is_valid"] = True
            result["estimated_size_mb"] = len(decoded) / (1024 * 1024)
            
            # Check size limits
            if result["estimated_size_mb"] > 100:  # 100MB limit
                result["errors"].append("Audio file too large (>100MB)")
                result["is_valid"] = False
                
        except Exception as e:
            result["errors"].append(f"Invalid Base64 string: {e}")
        
        return result
    
    def get_base64_info(self, base64_string: str) -> Dict[str, Any]:
        """
        Get information about Base64 audio data.
        
        Args:
            base64_string: Base64 string
            
        Returns:
            Information about the audio data
        """
        validation = self.validate_base64_string(base64_string)
        
        info = {
            "is_valid": validation["is_valid"],
            "string_length": validation["length"],
            "estimated_size_mb": validation["estimated_size_mb"],
            "errors": validation["errors"]
        }
        
        if validation["is_valid"]:
            # Try to determine audio format from header
            try:
                decoded = base64.b64decode(base64_string)
                header = decoded[:16]
                
                # Check common audio format headers
                if header.startswith(b'RIFF') and header.endswith(b'WAVE'):
                    info["detected_format"] = "wav"
                elif header.startswith(b'ID3') or header.startswith(b'\xff\xfb'):
                    info["detected_format"] = "mp3"
                elif header.startswith(b'fLaC'):
                    info["detected_format"] = "flac"
                elif header.startswith(b'OggS'):
                    info["detected_format"] = "ogg"
                else:
                    info["detected_format"] = "unknown"
                    
            except Exception:
                info["detected_format"] = "unknown"
        
        return info
    
    def cleanup_temp_files(self, pattern: str = "base64_audio_*") -> int:
        """
        Clean up temporary Base64 audio files.
        
        Args:
            pattern: File pattern to match
            
        Returns:
            Number of files removed
        """
        removed_count = 0
        
        try:
            for file_path in self.temp_dir.glob(pattern):
                if file_path.is_file():
                    file_path.unlink()
                    removed_count += 1
        except Exception:
            pass
        
        return removed_count
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get Base64 service information."""
        return {
            "temp_directory": str(self.temp_dir),
            "directory_exists": self.temp_dir.exists(),
            "supported_formats": ["wav", "mp3", "flac", "ogg", "m4a"],
            "max_file_size_mb": 100
        }
