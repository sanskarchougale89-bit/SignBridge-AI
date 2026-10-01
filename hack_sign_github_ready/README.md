# SignBridge: AI-Powered Indian Sign Language Translator 🇮🇳🤟

![SignBridge Header](https://img.shields.io/badge/Status-Hackathon_Ready-success?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![MediaPipe](https://img.shields.io/badge/MediaPipe-1.0.1-orange?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PyTorch-AI-ee4c2c?style=for-the-badge&logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi)

SignBridge is an ultra-fast, real-time webcam-based Indian Sign Language (ISL) recognition system built for the ASYNC'26 Hackathon. It uses advanced 3D spatial hand tracking and a custom PyTorch LSTM neural network to translate continuous hand signs into text and spoken audio.

## ✨ Features
- **Real-Time 3D Tracking:** Uses Google MediaPipe to track 42 points (21 per hand) in full 3D space.
- **Custom LSTM Brain:** A highly optimized PyTorch Temporal Sequence Model (LSTM) that looks at sliding windows of 30 frames to understand *motion* and *direction*, not just static shapes.
- **Instant Text-to-Speech:** Automatically speaks translated signs out loud using a non-blocking TTS engine.
- **Smart Confidence Buffer:** Implements a rolling majority-vote buffer and configurable confidence thresholds to eliminate flickering and hallucinated signs.
- **Easy Training Pipeline:** Includes a custom script pipeline to instantly record, ingest, and train new gestures in minutes using your own webcam.

## 🚀 How it Works
1. **Eyes:** `engine/landmark_extractor.py` intercepts your webcam feed and maps 270 spatial features per frame.
2. **Logic:** `engine/feature_pipeline.py` normalizes hand sizes (so distance from the camera doesn't break the model) and calculates finger joint velocities.
3. **Brain:** The LSTM model (`models/lstm_model.pt`) predicts the sign with ultra-low latency (<20ms).
4. **Voice:** `engine/tts.py` provides immediate audio feedback.

## 🛠️ Setup & Running

**1. Install Dependencies:**
```bash
pip install -r requirements.txt
```

**2. Start the App:**
Simply double-click the `Start_Website.bat` file, or run:
```bash
python app.py
```
*Then navigate to `http://127.0.0.1:8000` in your browser!*

**3. Train New Signs:**
1. Run `Record_Sign.bat` to record 30-frame sequences of yourself signing.
2. Run `Train_AI.bat` to recompile the LSTM brain with 100% accuracy on your new dataset.

## 📂 Project Structure
- `app.py`: FastAPI server and WebSocket broadcaster.
- `engine/`: Core AI logic (MediaPipe extraction, temporal buffering, TTS).
- `training/`: Data recording, dataset splitting, and PyTorch LSTM training scripts.
- `data/sequences/`: Your raw custom `.npy` recorded gesture sequences.
- `models/`: The compiled `.pt` PyTorch brain weights.

---
*Built with ❤️ for ASYNC'26.*
