from fastapi import FastAPI
import uvicorn
import gdown
import os
from ultralytics import YOLO

MODEL_PATH = "model.pt"
MODEL_URL = "https://drive.google.com/file/d/1y_4SxvfBQiCvG5KbewIu-fN-ctoYODng/view?usp=drive_link"

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load model
model = YOLO(MODEL_PATH)

app = FastAPI()

@app.get("/")
def home():
    return {"message": "API is working"}

@app.post("/predict")
def predict(data: dict):
    img_url = data.get("img")

    if not img_url:
        return {"error": "Image URL missing"}

    results = model(img_url)
    return {"result": results[0].tojson()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
