# Yoga CD Processor

A complete pipeline for transforming fragmented yoga audio recordings into professionally organized track collections. Built in honor of [Rajyogi Caruso](#honoring-rajyogi-caruso), whose wisdom continues to guide countless practitioners.

## What This Tool Accomplishes

Transform any collection of fragmented audio tracks into a cohesive, searchable session with intelligent context based segmentation.

### Complete 3-Phase Processing

- ✅ **Phase 1**: Combine fragmented tracks into single master recordings
- ✅ **Phase 2**: AI-powered transcription with OpenAI Whisper
- ✅ **Phase 3**: Intelligent content-based splitting with M3U playlist generation

### Key Features

- Content aware track naming based on yoga practice flow
- Professional metadata tagging (artist, album, track numbers)
- High quality audio preservation (defaults to 192k)
- M3U playlists for seamless navigation
- Batch processing for multiple CD collections
- Comprehensive transcripts with timestamp accuracy

## Installation

### Pre-requisites

- Python 3.8+
- ffmpeg (required for audio processing)

```bash
# Install ffmpeg on macOS
brew install ffmpeg

# Install ffmpeg on Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg
```

### Setup

1. Clone or navigate to the project directory:

```bash
cd /path/to/yoga-cd-processor
```

2. Create and activate virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Complete Processing Pipeline

Process any CD collection with full transcription and intelligent splitting:

```bash
python yoga_cd_processor.py "/path/to/yoga/session" --output ./processed_cds --use-openai-whisper --chunk-audio
```

### Expected Output

When processing completes successfully, you'll see comprehensive progress tracking:

```
Processing single CD directory

Processing CD: Red Practice #2
Found 52 audio files
Combining 52 audio files...
Combining files: 100%|████████████████████| 52/52 [00:20<00:00,  2.49it/s]
Exporting combined audio to: /path/to/processed/combined/Red Practice #2/Red Practice #2_master.m4a
Combined audio: 50.52 minutes
Converting Red Practice #2_master.m4a to mp3
Converted to: /path/to/processed/converted/Red Practice #2/Red Practice #2_master.mp3
Transcribing audio: Red Practice #2_master.mp3
Using chunked transcription with 3-minute chunks...
Splitting audio into 3-minute chunks with optimized encoding...
Created chunk 1: chunk_000.mp3
...
Transcribing chunk 17/17: chunk_016.mp3
Combined transcript saved: /path/to/processed/transcripts/Red Practice #2_transcript.json

============================================================
PHASE 3: AI-ASSISTED INTELLIGENT SPLITTING
============================================================
Actual audio duration: 50.5 minutes
Creating 8 segments of ~6 minutes each
Created 8 intelligent segments:
   1. Opening Meditation & Intention      (6.3 min)
   2. Warm-Up & Breath Work               (6.3 min)
   3. Standing Poses & Movement           (6.3 min)
   4. Table Pose & Push-Up Practice       (6.3 min)
   5. Floor Poses & Spinal Work           (6.3 min)
   6. Bridge Pose & Spinal Lifting        (6.3 min)
   7. Seated Poses & Forward Bends        (6.3 min)
   8. Spinal Twists & Releases            (6.3 min)

✅ Successfully created 8 tracks
📁 Each track includes:
   • Proper metadata (title, artist, album, track number)
   • High-quality 192k MP3 encoding
   • Intelligent content-based segmentation
📱 M3U playlist created for easy playback

All phases completed!
```

### Advanced Options

#### Resume Processing (Skip Completed Phases)

```bash
# Skip to Phase 3 if transcription already exists
python yoga_cd_processor.py /path/to/cds --output ./processed_cds --skip-combine --skip-convert --skip-transcribe --mp3-file /path/to/master.mp3

# Custom audio quality
python yoga_cd_processor.py /path/to/cds --output ./processed_cds --crossfade 1000 --bitrate 320k
```

#### OpenAI Configuration

```bash
# Use OpenAI API key from environment
export OPENAI_API_KEY="your-api-key-here"
python yoga_cd_processor.py /path/to/cds --output ./processed_cds --use-openai-whisper

# Custom chunk duration for processing
python yoga_cd_processor.py /path/to/cds --output ./processed_cds --chunk-audio --chunk-duration 240
```

## Generated Files

The processor creates a comprehensive output structure for each yoga session:

```txt
processed_cds/
├── combined/
│   └── Session_Name/
│       └── Session_Name_master.m4a      # Master combined file (original quality)
├── converted/
│   └── Session_Name/
│       └── Session_Name_master.mp3      # High-quality MP3 for processing
├── transcripts/
│   └── Session_Name_transcript.json     # Complete transcript with timestamps
│   └── Session_Name_transcript.txt      # Human-readable transcript
└── splits/
    └── Session_Name/
        ├── 01_Opening_Meditation_Intention.mp3
        ├── 02_Warm-Up_Dragons_Breath.mp3
        ├── 03_Standing_Poses_Movement.mp3
        ├── 04_Table_Pose_Push-Up_Practice.mp3
        ├── 05_Floor_Poses_Spinal_Work.mp3
        ├── 06_Bridge_Pose_Spinal_Lifting.mp3
        ├── 07_Head_to_Knee_Seated_Forward_Bends.mp3
        ├── 08_Spinal_Twists_Releases.mp3
        ├── 09_Breathing_Practice_Pranayama.mp3
        ├── 10_Rest_Integration.mp3
        └── Session_Name.m3u                 # Playlist file for easy playback
```

## Examples

### Process Single Yoga Session

```bash
python yoga_cd_processor.py "/path/to/yoga/session" --output ./processed_cds --use-openai-whisper --chunk-audio
```

### Batch Process Multiple Sessions

```bash
python yoga_cd_processor.py "/path/to/yoga/collection" --output ./processed_cds --use-openai-whisper --chunk-audio
```

## Audio Quality Preservation

Each phase maintains the highest possible audio quality:

- **Master files**: Preserved in original M4A format
- **Processing files**: High-quality MP3 (192k default, customizable up to 320k)
- **Crossfade transitions**: Smooth blending between combined tracks (500ms default)
- **Smart processing**: No unnecessary re-encoding from source material

## Troubleshooting

### Common Setup Issues

**FFmpeg not found**: Install using your system package manager

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg
```

**OpenAI API errors**: Ensure your API key is properly set

```bash
export OPENAI_API_KEY="your-api-key-here"
```

**Permission errors**: Check write permissions to your output directory

### Processing Issues

**Large file timeouts**: Use chunked processing for sessions longer than 60 minutes

```bash
python yoga_cd_processor.py /path/to/session --chunk-audio --chunk-duration 180
```

**Memory issues**: Process sessions individually rather than batch processing

## Attribution & License

This work is released under Creative Commons Zero v1.0 Universal ([CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)) - Public Domain 

### Honoring Rajyogi Caruso

This tool was created in honor of Rajyogi Caruso, whose yoga teachings have transformed countless lives. I hope his approach to yoga as a complete spiritual practice continues to guide practitioners toward deeper understanding and inner peace.

This project serves as a technical tool to help preserve and organize audio recordings for educational purposes.

### License Terms

You can copy, modify, distribute and perform the work, even for commercial purposes, all without asking permission.
