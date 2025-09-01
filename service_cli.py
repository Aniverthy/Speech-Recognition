#!/usr/bin/env python3
"""
Voice Recognition CLI Service

Command-line interface for the voice recognition system.
"""

import argparse
import sys
import os
from pathlib import Path
from service import PipelineService, ServiceState


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Voice Recognition System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single audio file
  python service_cli.py -f audio.mp3
  
  # Process with specific output formats
  python service_cli.py -f audio.mp3 --formats json txt
  
  # Process all files in directory
  python service_cli.py -d audio_folder/
  
  # Custom output directory
  python service_cli.py -f audio.mp3 -o results/
  
  # Process Base64 encoded audio
  python service_cli.py --base64 "UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBSuBzvLZiTYIG2m98OScTgwOUarm7blmGgU7k9n1unEiBC13yO/eizEIHWq+8+OWT..."
  
  # Process Base64 from text file
  python service_cli.py --base64-file audio_base64.txt --audio-format wav
  
  # Base64 with custom filename and format
  python service_cli.py --base64 "..." --audio-format mp3 --base64-filename my_audio
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-f", "--file",
        help="Process a single audio file"
    )
    input_group.add_argument(
        "-d", "--directory",
        help="Process all audio files in a directory"
    )
    input_group.add_argument(
        "--base64",
        help="Process Base64 encoded audio data"
    )
    input_group.add_argument(
        "--base64-file",
        help="Process audio from a text file containing Base64 encoded data"
    )
    
    # Output options
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["json", "txt", "csv", "summary"],
        default=["json", "txt", "csv", "summary"],
        help="Output formats to generate (default: all)"
    )
    
    parser.add_argument(
        "-o", "--output-dir",
        help="Custom output directory (default: out/)"
    )
    
    # Base64 options
    parser.add_argument(
        "--audio-format",
        default="wav",
        choices=["wav", "mp3", "flac", "ogg", "m4a"],
        help="Audio format for Base64 decoding (default: wav)"
    )
    
    parser.add_argument(
        "--base64-filename",
        help="Custom filename for Base64 audio (without extension)"
    )
    
    # Processing options
    parser.add_argument(
        "--model-size",
        default="base",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="ASR model size (default: base)"
    )
    
    parser.add_argument(
        "--no-enrollment",
        action="store_true",
        help="Skip speaker enrollment mapping"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize services
        state = ServiceState()
        
        # Override output directory if specified
        if args.output_dir:
            state.OUTPUT_DIR = Path(args.output_dir)
            state.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Override model size if specified
        if args.model_size:
            state.DEFAULT_MODEL_SIZE = args.model_size
        
        # Initialize pipeline
        pipeline = PipelineService(state)
        
        # Process based on input type
        if args.file:
            print(f"Processing audio file: {args.file}")
            results = pipeline.process_audio(args.file, args.formats)
            
        elif args.directory:
            print(f"Processing directory: {args.directory}")
            results = pipeline.process_directory(args.directory, args.formats)
            
        elif args.base64:
            print("Processing Base64 encoded audio")
            results = pipeline.process_base64(
                args.base64, 
                args.audio_format, 
                args.formats
            )
            
        elif args.base64_file:
            print(f"Processing Base64 file: {args.base64_file}")
            results = pipeline.process_base64_file(
                args.base64_file, 
                args.audio_format, 
                args.formats
            )
        
        # Display results
        if results.get("success", False):
            print("\n✅ Processing completed successfully!")
            
            if "outputs" in results:
                print("\nGenerated output files:")
                for format_name, file_path in results["outputs"].items():
                    print(f"  {format_name.upper()}: {file_path}")
            
            if "processing_info" in results:
                info = results["processing_info"]
                if "final" in info:
                    final = info["final"]
                    print(f"\nFinal results:")
                    print(f"  Segments: {final.get('segments', 0)}")
                    print(f"  Speakers: {final.get('unique_speakers', 0)}")
                    print(f"  Duration: {final.get('total_duration', 0):.2f}s")
                
                if "enrollment" in info and info["enrollment"].get("profiles_loaded", 0) > 0:
                    enrollment = info["enrollment"]
                    print(f"\nSpeaker enrollment:")
                    print(f"  Profiles loaded: {enrollment.get('profiles_loaded', 0)}")
                    if "speaker_mapping" in enrollment:
                        mapping = enrollment["speaker_mapping"]
                        for old, new in mapping.items():
                            print(f"  {old} → {new}")
            
        else:
            print("\n❌ Processing failed!")
            if "errors" in results:
                print("\nErrors:")
                for error in results["errors"]:
                    print(f"  {error}")
            
            sys.exit(1)
        
        # Cleanup
        pipeline.cleanup()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
