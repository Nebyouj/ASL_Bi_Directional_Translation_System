import tempfile
from starlette.background import BackgroundTask
import cv2
import numpy as np
import logging
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from model import load_model
from recognizer import ASLRecognizer
from gtts import gTTS
import os
from translator import Translator

app = FastAPI()
translator = Translator()

# ----------------------------
# Logger setup
# ----------------------------
logger = logging.getLogger("reverse_landmarks")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter(
    "[%(levelname)s] %(asctime)s - %(message)s"
)
handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model, labels = load_model("asl_bilstm.pth")
recognizer = ASLRecognizer(model, labels)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))

from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import base64

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ WebSocket connected")
    
    # Each WebSocket connection gets its OWN recognizer — solves the singleton problem
    ws_recognizer = ASLRecognizer(model, labels)
    
    try:
        while True:
            # Receive frame as base64 string
            data = await websocket.receive_text()
            
            # Decode base64 → numpy frame
            img_bytes = base64.b64decode(data)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue
            
            # Resize to 320x240 — MediaPipe doesn't need full resolution
            frame = cv2.resize(frame, (320, 240))
            
            word = ws_recognizer.process_frame(frame)
            
            response = {
                "word": word,
                "sentence": " ".join(ws_recognizer.sentence),
            }
            
            await websocket.send_json(response)
            
    except WebSocketDisconnect:
        print("❌ WebSocket disconnected")

@app.post("/clear")
def clear_sentence():
    recognizer.sentence.clear()
    recognizer.last_word = None
    return {"status": "cleared"}

@app.post("/translate")
def translate_text(body: dict):
    text = body.get("text", "")
    target_lang = body.get("lang", "en")
    source_lang = body.get("source_lang", None)
    
    if not text:
        return {"translated": text}
    
    # Reverse direction: non-English → English
    if source_lang and source_lang != "en" and target_lang == "en":
        reverse_map = {
            "fr": "Helsinki-NLP/opus-mt-fr-en",
            "ar": "Helsinki-NLP/opus-mt-ar-en",
            "de": "Helsinki-NLP/opus-mt-de-en",
        }
        result = translator.translate_with_model(text, reverse_map.get(source_lang))
        return {"translated": result or text}
    
    if target_lang == "en":
        return {"translated": text}
    
    result = translator.translate(text, target_lang)
    return {"translated": result or text}

# Serve static folder
# app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def serve_home():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()
    
@app.post("/predict")
async def predict(frame: UploadFile = File(...), lang: str = "en"):
    image_bytes = await frame.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    word = recognizer.process_frame(frame)

    translated_sentence = " ".join(recognizer.sentence)

    if lang != "en":
        translated_sentence = translator.translate(
            translated_sentence,
            lang
        )

    return {
        "word": word,
        "sentence": " ".join(recognizer.sentence),
        "translated": translated_sentence
}

@app.get("/reverse_landmarks")
def reverse_landmarks(sentence: str):
    words = sentence.upper().split()
    sequences = []
    for word in words:
        print("Pose")
        path = os.path.join(BASE_DIR, "reverse_dataset", f"{word}.npy")
        if os.path.exists(path):
            seq = np.load(path).tolist()
            sequences.append({"word": word, "frames": seq, "type": "pose+hands"})
        else:
            # fallback to old hand-only dataset
            folder = os.path.join(BASE_DIR, "asl_dataset", word)
            if os.path.isdir(folder):
                files = sorted(os.listdir(folder))
                if files:
                    seq = np.load(os.path.join(folder, files[0])).tolist()
                    sequences.append({"word": word, "frames": seq, "type": "hands-only"})
    return {"sequences": sequences}

@app.get("/tts")
def tts(sentence: str):
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        path = f.name
    gTTS(sentence).save(path)
    return FileResponse(path, media_type="audio/mpeg",
                        background=BackgroundTask(os.unlink, path))
