# import tempfile
# from starlette.background import BackgroundTask
# import cv2
# import numpy as np
# import logging
# from fastapi import FastAPI, UploadFile, File
# from fastapi.responses import FileResponse
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import HTMLResponse
# from fastapi.staticfiles import StaticFiles
# from model import load_model
# from recognizer import ASLRecognizer
# from gtts import gTTS
# import os
# from translator import Translator

# app = FastAPI()
# translator = Translator()

# # ----------------------------
# # Logger setup
# # ----------------------------
# logger = logging.getLogger("reverse_landmarks")
# logger.setLevel(logging.INFO)

# handler = logging.StreamHandler()
# formatter = logging.Formatter(
#     "[%(levelname)s] %(asctime)s - %(message)s"
# )
# handler.setFormatter(formatter)

# if not logger.handlers:
#     logger.addHandler(handler)

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # allow all for development
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# model, labels = load_model("asl_bilstm.pth")
# recognizer = ASLRecognizer(model, labels)
# BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))

# from fastapi import WebSocket, WebSocketDisconnect
# import asyncio
# import base64

# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     print("✅ WebSocket connected")
    
#     # Each WebSocket connection gets its OWN recognizer — solves the singleton problem
#     ws_recognizer = ASLRecognizer(model, labels)
    
#     try:
#         while True:
#             # Receive frame as base64 string
#             data = await websocket.receive_text()
            
#             # Decode base64 → numpy frame
#             img_bytes = base64.b64decode(data)
#             np_arr = np.frombuffer(img_bytes, np.uint8)
#             frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
#             if frame is None:
#                 continue
            
#             # Resize to 320x240 — MediaPipe doesn't need full resolution
#             frame = cv2.resize(frame, (320, 240))
            
#             word = ws_recognizer.process_frame(frame)
            
#             response = {
#                 "word": word,
#                 "sentence": " ".join(ws_recognizer.sentence),
#             }
            
#             await websocket.send_json(response)
            
#     except WebSocketDisconnect:
#         print("❌ WebSocket disconnected")

# @app.post("/clear")
# def clear_sentence():
#     recognizer.sentence.clear()
#     recognizer.last_word = None
#     return {"status": "cleared"}

# @app.post("/translate")
# def translate_text(body: dict):
#     text = body.get("text", "")
#     target_lang = body.get("lang", "en")
#     source_lang = body.get("source_lang", None)
    
#     if not text:
#         return {"translated": text}
    
#     # Reverse direction: non-English → English
#     if source_lang and source_lang != "en" and target_lang == "en":
#         reverse_map = {
#             "fr": "Helsinki-NLP/opus-mt-fr-en",
#             "ar": "Helsinki-NLP/opus-mt-ar-en",
#             "de": "Helsinki-NLP/opus-mt-de-en",
#         }
#         result = translator.translate_with_model(text, reverse_map.get(source_lang))
#         return {"translated": result or text}
    
#     if target_lang == "en":
#         return {"translated": text}
    
#     result = translator.translate(text, target_lang)
#     return {"translated": result or text}

# # Serve static folder
# # app.mount("/static", StaticFiles(directory="static"), name="static")

# @app.get("/", response_class=HTMLResponse)
# def serve_home():
#     with open("static/index.html", "r", encoding="utf-8") as f:
#         return f.read()
    
# @app.post("/predict")
# async def predict(frame: UploadFile = File(...), lang: str = "en"):
#     image_bytes = await frame.read()
#     np_arr = np.frombuffer(image_bytes, np.uint8)
#     frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

#     word = recognizer.process_frame(frame)

#     translated_sentence = " ".join(recognizer.sentence)

#     if lang != "en":
#         translated_sentence = translator.translate(
#             translated_sentence,
#             lang
#         )

#     return {
#         "word": word,
#         "sentence": " ".join(recognizer.sentence),
#         "translated": translated_sentence
# }

# @app.get("/reverse_landmarks")
# def reverse_landmarks(sentence: str):
#     words = sentence.upper().split()
#     sequences = []
#     for word in words:
#         print("Pose")
#         path = os.path.join(BASE_DIR, "reverse_dataset", f"{word}.npy")
#         if os.path.exists(path):
#             seq = np.load(path).tolist()
#             sequences.append({"word": word, "frames": seq, "type": "pose+hands"})
#         else:
#             # fallback to old hand-only dataset
#             folder = os.path.join(BASE_DIR, "asl_dataset", word)
#             if os.path.isdir(folder):
#                 files = sorted(os.listdir(folder))
#                 if files:
#                     seq = np.load(os.path.join(folder, files[0])).tolist()
#                     sequences.append({"word": word, "frames": seq, "type": "hands-only"})
#     return {"sequences": sequences}

# @app.get("/tts")
# def tts(sentence: str):
#     with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
#         path = f.name
#     gTTS(sentence).save(path)
#     return FileResponse(path, media_type="audio/mpeg",
#                         background=BackgroundTask(os.unlink, path))


import tempfile
import asyncio
from concurrent.futures import ThreadPoolExecutor
from starlette.background import BackgroundTask
import cv2
import numpy as np
import logging
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from model import load_model
from recognizer import ASLRecognizer
from gtts import gTTS
import os
import psutil
from translator import Translator

app = FastAPI()
translator = Translator()

# ----------------------------
# Thread pool for blocking inference
# Keeps the async event loop unblocked during torch/mediapipe calls
# ----------------------------
executor = ThreadPoolExecutor(max_workers=2)

# ----------------------------
# Logger setup
# ----------------------------
logger = logging.getLogger("asl_app")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s")
handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(handler)

# ----------------------------
# CORS
# ----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Load model ONCE at startup
# recognizer.py already builds the shared MediaPipe detector at import time
# ----------------------------
model, labels = load_model("asl_bilstm.pth")

# Put model in eval mode — no gradient tracking needed
model.eval()

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))

import base64


# ----------------------------
# Memory guard helper
# ----------------------------
def memory_ok(min_free_mb: int = 80) -> bool:
    """Returns True if at least min_free_mb of RAM is available."""
    mem = psutil.virtual_memory()
    free_mb = mem.available / (1024 * 1024)
    logger.info(f"Available memory: {free_mb:.1f} MB")
    return free_mb >= min_free_mb


# ----------------------------
# WebSocket endpoint
# ----------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # Reject new connections if memory is critically low
    if not memory_ok(min_free_mb=80):
        logger.warning("Low memory — rejecting WebSocket connection")
        await websocket.close(code=1013)  # 1013 = Try Again Later
        return

    await websocket.accept()
    logger.info("✅ WebSocket connected")

    # Each connection gets its own state (sequence, sentence, etc.)
    # but shares the singleton MediaPipe detector from recognizer.py
    ws_recognizer = ASLRecognizer(model, labels)

    loop = asyncio.get_event_loop()

    try:
        while True:
            # --- Receive binary frame directly (no base64 overhead) ---
            # Frontend should send raw bytes: ws.send(blob) not base64
            # Falls back to base64 text if needed
            try:
                raw = await websocket.receive_bytes()
                img_bytes = raw
            except Exception:
                # Fallback: receive as base64 text
                data = await websocket.receive_text()
                img_bytes = base64.b64decode(data)

            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            # Resize to 224x168 — smaller than before, MediaPipe handles it fine
            frame = cv2.resize(frame, (224, 168))

            # Run blocking inference in thread pool — keeps event loop free
            word = await loop.run_in_executor(
                executor,
                ws_recognizer.process_frame,
                frame
            )

            response = {
                "word": word,
                "sentence": " ".join(ws_recognizer.sentence),
            }

            await websocket.send_json(response)

    except WebSocketDisconnect:
        logger.info("❌ WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Explicitly clear buffers to free memory immediately
        ws_recognizer.sequence.clear()
        ws_recognizer.sentence.clear()
        ws_recognizer.prev_features = None


# ----------------------------
# HTTP endpoints
# ----------------------------

@app.post("/clear")
def clear_sentence():
    # No global recognizer singleton anymore
    # Clients should track their own state via WebSocket
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


@app.get("/", response_class=HTMLResponse)
def serve_home():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/predict")
async def predict(frame: UploadFile = File(...), lang: str = "en"):
    """
    HTTP fallback endpoint for single-frame prediction.
    Uses a fresh per-request recognizer — stateless.
    """
    image_bytes = await frame.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return {"word": None, "sentence": "", "translated": ""}

    img = cv2.resize(img, (224, 168))

    # Use a short-lived recognizer for HTTP requests (no persistent state)
    http_recognizer = ASLRecognizer(model, labels)

    loop = asyncio.get_event_loop()
    word = await loop.run_in_executor(executor, http_recognizer.process_frame, img)

    sentence = " ".join(http_recognizer.sentence)
    translated_sentence = sentence

    if lang != "en" and sentence:
        translated_sentence = translator.translate(sentence, lang)

    return {
        "word": word,
        "sentence": sentence,
        "translated": translated_sentence,
    }


@app.get("/reverse_landmarks")
def reverse_landmarks(sentence: str):
    words = sentence.upper().split()
    sequences = []
    for word in words:
        path = os.path.join(BASE_DIR, "reverse_dataset", f"{word}.npy")
        if os.path.exists(path):
            seq = np.load(path).tolist()
            sequences.append({"word": word, "frames": seq, "type": "pose+hands"})
        else:
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
    return FileResponse(
        path,
        media_type="audio/mpeg",
        background=BackgroundTask(os.unlink, path)
    )


@app.get("/health")
def health():
    """Health check with memory stats — useful for Render monitoring."""
    mem = psutil.virtual_memory()
    return {
        "status": "ok",
        "memory_used_mb": round((mem.total - mem.available) / (1024 * 1024), 1),
        "memory_available_mb": round(mem.available / (1024 * 1024), 1),
        "memory_percent": mem.percent,
    }
