# Audio Redaction Service

AudioSafeguard is an intelligent audio redaction tool powered by Gemini AI. It allows users to securely redact sensitive information from audio files using advanced AI detection, regex patterns, or keyword lists.

## Features

- **Intelligent Redaction**: Uses Gemini AI to understand context and redact PII/sensitive entities.
- **Multiple Modes**:
  - **Gemini AI**: Context-aware redaction.
  - **BERT / NER**: Named Entity Recognition (if configured).
  - **Regex**: Pattern-based redaction for standard formats (emails, phones, etc.).
- **Redaction Styles**: Replace sensitive audio with a "Beep" or "Silence".
- **Noise Reduction**: Integrated audio pre-processing to improve transcription accuracy.
- **Visual Interface**: Interactive waveform visualization using WaveSurfer.js.

## Getting Started

### Prerequisites

- Python 3.8+
- [FFmpeg](https://ffmpeg.org/download.html) (required for audio processing)

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/yourusername/audio-redaction-service.git
    cd audio-redaction-service
    ```

2.  Install dependencies:
    ```bash
    pip install fastapi "uvicorn[standard]" python-multipart
    # Add other dependencies as per your local setup (e.g., libraries for detection)
    ```

### Running the Server

Start the FastAPI backend:

```bash
python server.py
```

The application will be available at `http://localhost:8000`.

## Deployment

### GitHub Pages (Frontend Demo)

The `docs/` folder in this repository contains a static version of the frontend for GitHub Pages.
Note: The static site on GitHub Pages is a UI demo and requires the running Python backend to function fully.

## License

[MIT](LICENSE)
