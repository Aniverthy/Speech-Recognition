# Voice Recognition System

A modular, production-ready voice recognition system using **faster-whisper** for ASR and **Resemblyzer** for voice embeddings.

## Features

✅ **ASR**: faster-whisper (fast, accurate speech recognition)  
✅ **Voice Embeddings**: Resemblyzer (speaker identification)  
✅ **Speaker Diarization**: Automatic speaker detection and clustering  
✅ **Speaker Enrollment**: Map detected speakers to known names  
✅ **Multiple Outputs**: JSON, TXT, CSV, Summary formats  
✅ **Base64 Support**: Process encoded audio from web APIs (string, file, chunks)  
✅ **Production Ready**: Hardcoded config, no loggers, clean code  

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Basic Usage
```bash
# Process audio file
python service_cli.py -f audio.mp3

# Process directory
python service_cli.py -d audio_folder/

# Process Base64 audio from file (recommended on Windows)
python service_cli.py --base64-file src/base64.txt --audio-format wav
```

## Service Architecture

```
service/
├── __init__.py              # Service package exports
├── service_state.py         # Global configuration & state
├── service_preprocess.py    # Audio loading & preparation
├── service_asr.py          # Speech recognition (faster-whisper)
├── service_features.py     # Voice embeddings (Resemblyzer)
├── service_enroll.py       # Speaker enrollment
├── service_diarize.py      # Speaker diarization
├── service_align.py        # Audio-text alignment
├── service_output.py       # Multiple output formats
├── service_base64.py       # Base64 audio handling
└── service_pipeline.py     # Main pipeline orchestration
```

## What Each Service Does

### **Service State** (`service_state.py`)
- Manages global configuration and hardcoded production settings
- Handles device detection (GPU/CPU), file paths, thresholds
- Centralized configuration for all services

### **Preprocess Service** (`service_preprocess.py`)
- Loads audio files and converts to target format (16kHz mono)
- Validates audio quality and duration limits
- Segments audio into chunks for processing

### **ASR Service** (`service_asr.py`)
- **Uses only faster-whisper** for speech recognition
- Handles transcription with word-level timestamps
- Supports multiple model sizes (tiny, base, small, medium, large-v3)

### **Feature Service** (`service_features.py`)
- **Uses only Resemblyzer** for voice embeddings
- Extracts 256-dimensional voice embeddings
- Falls back to spectral features if embeddings fail

### **Enrollment Service** (`service_enroll.py`)
- Manages speaker enrollment profiles
- Loads reference audio from Reference/ directory
- Maps detected speakers to enrolled names using cosine similarity

### **Diarization Service** (`service_diarize.py`)
- Performs voice activity detection
- Clusters audio segments by speaker
- Uses cosine similarity for speaker identification

### **Alignment Service** (`service_align.py`)
- Aligns ASR segments with diarized segments
- Merges short segments and validates timing
- Ensures proper audio-text synchronization

### **Output Service** (`service_output.py`)
- Generates multiple output formats:
  - **JSON**: Complete analysis with metadata
  - **TXT**: Human-readable conversation format
  - **CSV**: Structured data for analysis
  - **Summary**: Statistical report

### **Base64 Service** (`service_base64.py`)
- Handles Base64 encoded audio input
- Decodes and converts to audio files
- Useful for web APIs and data transmission

### **Pipeline Service** (`service_pipeline.py`)
- Orchestrates the entire processing pipeline
- Coordinates all services in sequence
- Handles error handling and progress tracking

## Configuration

All settings are hardcoded in `service_state.py`:

```python
# Audio processing
TARGET_SAMPLE_RATE = 16000        # Hz - standard for voice recognition
SUPPORTED_FORMATS = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.m4b', '.aac']
CHUNK_DURATION = 3.0              # seconds - audio chunk size

# ASR settings
ASR_BACKEND = "faster-whisper"    # Only faster-whisper backend
DEFAULT_MODEL_SIZE = "base"       # Model size: tiny, base, small, medium, large-v3
WORD_TIMESTAMPS = True            # Enable word-level timing
BEAM_SIZE = 5                     # Beam search size

# Speaker identification
MFCC_FEATURES = 13                # Number of MFCC coefficients
EMBEDDING_THRESHOLD = 0.65        # Neural embedding similarity threshold
FEATURES_THRESHOLD = 0.40         # Spectral features similarity threshold
CLUSTERING_THRESHOLD = 0.7        # Speaker clustering threshold
```

## Speaker Enrollment

### Setup
1. Create a `Reference/` folder in your project
2. Add subfolders for each person: `Reference/John/`, `Reference/Sarah/`
3. Place 3-5 audio samples per person (10-30 seconds each)
4. Supported formats: WAV, MP3, FLAC, M4A, OGG, M4B, AAC

### How It Works
1. System loads reference audio and extracts voice embeddings using Resemblyzer
2. When processing target audio, it detects speaker clusters
3. Uses cosine similarity to match clusters with enrolled profiles
4. Applies the mapping to replace generic labels (User1, User2) with real names (John, Sarah)

### Similarity Thresholds
- **Neural embeddings**: 0.65 (high confidence)
- **Spectral features**: 0.40 (fallback method)

## Usage Examples

### Python API
```python
from service import PipelineService, ServiceState

# Initialize services
state = ServiceState()
pipeline = PipelineService(state)

# Process audio file
results = pipeline.process_audio("audio.mp3")

# Process Base64 audio
results = pipeline.process_base64("BASE64_STRING")

# Process Base64 in chunks (for long inputs / streaming)
def chunker(path, size=8192):
    with open(path, 'r', encoding='utf-8') as f:
        while True:
            chunk = f.read(size)
            if not chunk:
                break
            yield chunk

results = pipeline.process_base64_chunks(chunker("src/base64.txt"), audio_format="wav")

# Process directory
results = pipeline.process_directory("audio_folder/")
```

### Command Line
```bash
# Process single file
python service_cli.py -f audio.mp3

# Process with specific outputs
python service_cli.py -f audio.mp3 --formats json txt

# Process directory
python service_cli.py -d audio_folder/

# Custom output directory
python service_cli.py -f audio.mp3 -o results/

# Process Base64 from file
python service_cli.py --base64-file src/base64.txt --audio-format wav

# Use different model size
python service_cli.py -f audio.mp3 --model-size medium

# Verbose output
python service_cli.py -f audio.mp3 --verbose
```

## Output Formats

### JSON Output
Complete analysis with metadata, segments, and processing information.

### Text Output
Human-readable conversation format with timestamps and speaker labels.

### CSV Output
Structured data for analysis with segment details.

### Summary Report
Statistical analysis including speaker duration, word counts, and quality metrics.

## Performance

- **ASR**: faster-whisper provides 2-4x speed improvement over OpenAI Whisper
- **Voice Embeddings**: Resemblyzer provides fast, accurate speaker identification
- **GPU Support**: Automatic GPU detection and utilization when available
- **Memory Efficient**: Streaming processing for large audio files

## Requirements

- Python 3.8+
- 4GB+ RAM (8GB+ recommended)
- GPU optional but recommended for large files
- Audio files: WAV, MP3, FLAC, M4A, OGG, M4B, AAC

## Installation Issues

### Common Problems
1. **Resemblyzer installation**: May require Visual Studio Build Tools on Windows
2. **faster-whisper**: Requires C++ compiler on some systems
3. **Audio codecs**: Install ffmpeg for additional audio format support

### Solutions
```bash
# Windows: Install Visual Studio Build Tools
# Linux: Install build-essential
sudo apt-get install build-essential

# macOS: Install Xcode Command Line Tools
xcode-select --install

# Alternative: Use conda
conda install -c conda-forge resemblyzer faster-whisper
```

## Production Deployment

The system is designed for production with:
- ✅ Hardcoded configuration values
- ✅ No external logging dependencies
- ✅ Consistent error handling
- ✅ Performance-optimized settings
- ✅ Comprehensive validation
- ✅ Quality assessment tools

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:
1. Check the configuration in `service_state.py`
2. Verify all dependencies are installed correctly
3. Check audio file format and quality
4. Use `--verbose` flag for detailed error information
