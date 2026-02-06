# YouTube Automation Pipeline

A complete pipeline for automated YouTube video creation from script or audio input to final video rendering. It handles transcription, voiceover generation, visual generation (placeholders), video editing, and background music integration.

## Features

- **Script Processing**: segments text into scenes with duration estimation.
- **Audio Processing**: TTS generation and speech-to-text transcription.
- **Visual Generator**: Creates automated visual placeholders with gradients and text.
- **Video Editing**: Composes scenes with transitions, Ken Burns effect, and subtitles.
- **Background Music**: Adds music with volume ducking.
- **Thumbnail Generation**: Automatically generates a video thumbnail.

## Usage

```bash
# From script text
python yt_automation.py --script script.txt --style tech

# From audio file
python yt_automation.py --audio voiceover.mp3 --music background.mp3
```

## Dependencies

- `moviepy`
- `pydub`
- `librosa`
- `soundfile`
- `opencv-python`
- `pillow`
- `pyttsx3`
- `SpeechRecognition`
- `numpy`
