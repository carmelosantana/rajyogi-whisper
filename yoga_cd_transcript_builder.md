# 📄 Build Document — Yoga CD Transcript and Smart Splitting

## 🌟 Goal

Process a collection of yoga audio CDs (often split arbitrarily every minute) to:

1. **Combine fragmented tracks** into coherent single audio files.
2. **Transcribe** each audio file using Whisper with **speaker diarization**.
3. **Intelligently split** the audio into meaningful tracks based on subject and natural pauses using AI.
4. **Generate new metadata** and titles aligned with transcript themes.
5. **Save** and structure all files for future debugging, training chatbots, and content reuse.

---

## 📆 Workflow Overview

### ✅ 1. Input Handling

- Audio source: Folder per CD (`./CD_Title/track01.mp3 ... trackNN.mp3`)
- Combine tracks if duration < threshold (e.g. 2 min) or uniform duration
- Output: `./combined/CD_Title/combined.mp3`

### ✅ 2. Transcription + Speaker Diarization

- Use `Whisper` (local or OpenAI API, configurable)
- Primary speaker assumed: `Rajyogi Caruso`
- Add `speaker` field to each segment
- Save raw transcript for each file: `./transcripts/CD_Title.json`

### ✅ 3. Intelligent Audio Splitting (AI-Guided)

- Identify candidate breakpoints:
  - Natural pauses (silence detection)
  - Low energy audio
  - Transcript punctuation/semantic breaks
- Use OpenAI (or local LLM) to:
  - Suggest ideal split points
  - Generate new **titles** per split segment based on transcript themes
- Export:
  - Split audio tracks: `./output/CD_Title/track01.mp3`, `track02.mp3`, ...
  - New titles + cue data: `splits.json`

### ✅ 4. Metadata Assignment

- Pull CD title from folder name
- Auto-generate per-track titles from transcript summary
- Optional fields:
  - Speaker: Always "Rajyogi Caruso"
  - Track length
  - Subject tags

### ✅ 5. Output & Debugging Artifacts

- `combined.mp3`
- `transcript.json`
- `splits.json`
- Split `mp3` files
- Summary JSON of all metadata

---

## 🔧 Config Options

```python
USE_OPENAI_WHISPER = False     # Set to True to use OpenAI API
TARGET_SAMPLE_RATE = 16000     # Standard for Whisper processing
SAVE_DEBUG_FILES = True        # Save intermediate files (e.g. temp transcript, breakpoints)
```

---

## 🔍 Example Output

```json
{
  "cd_title": "Path of Inner Silence",
  "speaker": "Rajyogi Caruso",
  "original_duration": "00:58:12",
  "transcript": [
    { "timestamp": "00:00:03", "speaker": "Rajyogi Caruso", "text": "Welcome to the path of inner stillness..." },
    { "timestamp": "00:12:49", "speaker": "Rajyogi Caruso", "text": "As we breathe in..." }
  ],
  "splits": [
    {
      "start_time": "00:00:00",
      "end_time": "00:11:45",
      "title": "Introduction to Stillness",
      "file": "track01.mp3"
    },
    {
      "start_time": "00:11:45",
      "end_time": "00:27:30",
      "title": "Breath Awareness",
      "file": "track02.mp3"
    }
  ]
}
```

---

## 🛠 Tools & Libraries

| Task                    | Tool                                 |
| ----------------------- | ------------------------------------ |
| Combine audio tracks    | `pydub`, `ffmpeg`                    |
| Transcription           | `Whisper` (CLI or API)               |
| Silence detection       | `pydub.silence`, `librosa`, `ffmpeg` |
| Subject-based splitting | `OpenAI GPT-4` or LLM-based parser   |
| Audio splitting/export  | `pydub`, `ffmpeg`                    |
| JSON output             | `json`, `os`, `pathlib`              |

---

## 🛌 Next Steps

We will:

1. Build CLI script that accepts a folder path of audio files.
2. Auto-combine and transcribe.
3. Use transcript + AI to create subject-based audio splits.
4. Save all outputs for chatbot and reuse.

