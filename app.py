# app.py
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import asyncio
from model_utils import download_from_drive, load_yolo_model, WEIGHT_FILE

app = FastAPI(title="YOLO API")

# Put your Drive FILE_ID here or set as ENV variable on Render
DRIVE_FILE_ID = os.getenv("DRIVE_FILE_ID", "1y_4SxvfBQiCvG5KbewIu-fN-ctoYODng")

# Globals
MODEL = None

@app.on_event("startup")
async def startup_event():
    global MODEL
    # 1) download model if needed
    if not WEIGHT_FILE.exists():
        if DRIVE_FILE_ID == "https://drive.google.com/file/d/1y_4SxvfBQiCvG5KbewIu-fN-ctoYODng/view?usp=drive_link" or not DRIVE_FILE_ID:
            raise RuntimeError("Set DRIVE_FILE_ID env var or put ID in code (change it).")
        download_from_drive(DRIVE_FILE_ID)
    # 2) load model
    MODEL = load_yolo_model(str(WEIGHT_FILE))
    print("Model loaded. Ready to serve.")

class TextIn(BaseModel):
    text: str

@app.get("/")
def root():
    return {"status": "ok"}

# Example image detection endpoint (POST an image file)
@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        import cv2
        import numpy as np
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # BGR

        # Depending on MODEL type API:
        # ultralytics YOLO: results = MODEL(img)  (it accepts numpy images)
        results = MODEL(img)
        # format results to JSON
        out = []
        for r in results:
            # for ultralytics r.boxes.xyxy, r.boxes.conf, r.boxes.cls
            boxes = r.boxes.xyxy.tolist() if hasattr(r, "boxes") else []
            out.append({"boxes": boxes})
        return {"predictions": out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
