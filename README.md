# SPECTRA AI Detector

Real Flask backend + Hugging Face image classifier. Video is analyzed by sampling frames. Audio is intentionally disabled until a dedicated audio deepfake model is added.

## Windows
1. Install Python 3.11+ and FFmpeg (ffmpeg must be on PATH).
2. Open this folder in Command Prompt.
3. Run `run.bat`.
4. Open http://127.0.0.1:5000

The default model is `umm-maybe/AI-image-detector`. Change it with `SPECTRA_MODEL` if desired.

Results are probabilistic, not proof. No detector catches every generator.
