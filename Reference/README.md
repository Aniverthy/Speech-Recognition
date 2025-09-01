# Reference Audio Setup for Speaker Mapping

This folder contains reference audio samples for mapping detected speakers to known identities using cosine similarity.

## How It Works

The system uses **cosine similarity** to match audio segments with reference audio samples:
- **Embedding-based matching**: Uses neural network embeddings (threshold: 0.65)
- **Feature-based matching**: Uses MFCC features as fallback (threshold: 0.40)

## Folder Structure

```
Reference/
├── John/           # Speaker: John
│   ├── sample1.wav # Reference audio samples
│   ├── sample2.mp3 # Multiple samples improve accuracy
│   └── ...
├── Sarah/          # Speaker: Sarah
│   ├── voice1.wav
│   └── ...
└── Mike/           # Speaker: Mike
    ├── mike_voice.mp3
    └── ...
```

## Audio Requirements

- **Format**: WAV, MP3, FLAC, M4A, OGG, M4B, AAC
- **Quality**: Clear speech, minimal background noise
- **Duration**: 10-30 seconds per sample (longer = better)
- **Content**: Natural speech, not music or effects
- **Sample Rate**: Will be converted to 16kHz automatically

## Usage

1. **Place reference audio** in the appropriate speaker folder
2. **Run the pipeline** with your target audio:
   ```bash
   python cli_streamlined.py -f your_audio.mp3
   ```
3. **The system will automatically**:
   - Load enrollment profiles from Reference/
   - Process your audio with speaker diarization
   - Map detected speakers to enrolled names using cosine similarity
   - Generate output with real names instead of "User1", "User2", etc.

## Tips for Best Results

- **Multiple samples per person**: Include 3-5 different audio samples per speaker
- **Varied content**: Different phrases, emotions, speaking styles
- **Clean audio**: Minimal background noise, clear speech
- **Consistent voice**: Same person, same recording conditions
- **Avoid duplicates**: Don't use the same audio clip multiple times

## Example Output

Instead of:
```
User1: "Hello, how are you?"
User2: "I'm doing well, thank you."
```

You'll get:
```
John: "Hello, how are you?"
Sarah: "I'm doing well, thank you."
```

## Troubleshooting

- **Low similarity scores**: Check audio quality and try more reference samples
- **No matches found**: Ensure reference audio is clear and representative
- **Wrong matches**: Add more diverse reference samples for each speaker
