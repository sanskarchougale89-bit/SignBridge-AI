import cv2
import json
import logging
import asyncio
import threading
import time
from pathlib import Path
from typing import Optional, List
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse

from engine.recognizer import SignRecognitionEngine
from engine.tts import TTSEngine

logging.basicConfig(level=logging.INFO)

PROJECT_ROOT = Path(__file__).parent
MODEL_PATH = PROJECT_ROOT / "models" / "lstm_model.pt"
HAND_LANDMARKER_PATH = PROJECT_ROOT / "hand_landmarker.task"
CAMERA_INDEX = 0

class AppState:
    def __init__(self):
        self.camera: Optional[cv2.VideoCapture] = None
        self.active_clients = 0
        self.client_lock = threading.Lock()
        self.engine: Optional[object] = None
        self.tts: Optional[object] = None
        self.is_running = False
        self.processing_thread: Optional[threading.Thread] = None
        
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_prediction: dict = {}
        
        self.ws_clients: List[WebSocket] = []
        self.lock = threading.Lock()
        self.translation_history = []

    def start_camera(self):
        if self.camera is not None and self.camera.isOpened():
            return True
        self.camera = cv2.VideoCapture(CAMERA_INDEX)
        if not self.camera.isOpened():
            self.camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        return self.camera.isOpened()

    def stop_camera(self):
        if self.camera is not None:
            self.camera.release()
            self.camera = None

    def client_connected(self):
        with self.client_lock:
            self.active_clients += 1
            if self.active_clients == 1:
                logging.info("Client connected: Starting camera")
                self.start_camera()

    def client_disconnected(self):
        with self.client_lock:
            self.active_clients -= 1
            if self.active_clients <= 0:
                self.active_clients = 0
                logging.info("All clients disconnected: Stopping camera")
                self.stop_camera()

state = AppState()

def processing_loop():
    while state.is_running:
        if state.camera is None or not state.camera.isOpened():
            time.sleep(0.1)
            continue
            
        ret, frame = state.camera.read()
        if not ret:
            time.sleep(0.01)
            continue
            
        frame = cv2.flip(frame, 1)
        
        with state.lock:
            if state.engine:
                result = state.engine.process_frame(frame, time.monotonic_ns() // 1_000_000)
                state.latest_frame = result.get("annotated_frame", frame)
                
                # Update main prediction state
                pred_data = {
                    "sign": result.get("current_sign") or result.get("live_sign", ""),
                    "confidence": result.get("confidence", 0.0),
                    "status": result.get("status", "idle"),
                    "sequence": result.get("sequence", []),
                    "sentence": result.get("sentence", [])
                }
                state.latest_prediction = pred_data
                
                # Removed ISL PATH
            else:
                state.latest_frame = frame

        time.sleep(0.01)

def generate_mjpeg():
    while True:
        with state.lock:
            frame = state.latest_frame
            
        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Starting camera...", (180, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2, cv2.LINE_AA)
                        
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.03)

async def broadcast_predictions():
    while True:
        if state.ws_clients and state.latest_prediction:
            dead_clients = []
            for ws in state.ws_clients:
                try:
                    await ws.send_json(state.latest_prediction)
                except Exception:
                    dead_clients.append(ws)
            for ws in dead_clients:
                if ws in state.ws_clients:
                    state.ws_clients.remove(ws)
        await asyncio.sleep(0.1)

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    try:
        state.tts = TTSEngine(cooldown=3.0)
        logging.info("TTS engine initialized")
    except Exception as e:
        logging.warning(f"TTS init failed: {e}")

    try:
        state.engine = SignRecognitionEngine(
            model_path=str(MODEL_PATH) if MODEL_PATH.exists() else None,
            hand_landmarker_path=str(HAND_LANDMARKER_PATH),
            tts_engine=state.tts,
            auto_speak=True,
        )
        logging.info(f"Recognition engine initialized")
    except Exception as e:
        logging.warning(f"Engine init failed: {e}")
        state.engine = None

    state.is_running = True
    state.processing_thread = threading.Thread(target=processing_loop, daemon=True)
    state.processing_thread.start()
    logging.info("Processing loop started")

    asyncio.create_task(broadcast_predictions())

@app.on_event("shutdown")
async def shutdown_event():
    state.is_running = False
    state.stop_camera()
    if state.tts:
        state.tts.shutdown()
    if state.processing_thread:
        state.processing_thread.join(timeout=1.0)

def get_frontend_html() -> str:
    with open('frontend_backup.html', 'r', encoding='utf-8') as f:
        return f.read()

@app.get("/", response_class=HTMLResponse)
async def index():
    return get_frontend_html()

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    state.ws_clients.append(websocket)
    state.client_connected()
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            if msg.get("action") == "speak":
                text = msg.get("text", "")
                if state.tts and text:
                    state.tts.speak(text)
            elif msg.get("action") == "clear":
                with state.lock:
                    state.translation_history.clear()
                    if state.engine:
                        state.engine.reset()
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in state.ws_clients:
            state.ws_clients.remove(websocket)
        state.client_disconnected()

@app.post("/api/speak")
async def speak_text(payload: dict):
    text = payload.get("text", "")
    if state.tts and text:
        state.tts.speak(text)
        return {"status": "success"}
    return {"status": "error", "message": "TTS not available or empty text"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
