# models_utils.py
from ultralytics import YOLO
import torch
import requests
import os

MODEL_URL = "https://huggingface.co/Paraskchauhan/paras-yolo-model/resolve/main/Open%20CV.zip"
MODEL_PATH = "yolo.pt"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
        print("Model downloaded.")

def load_model():
    download_model()
    model = YOLO(MODEL_PATH)
    return model

def run_inference(model, image_path):
    results = model(image_path)
    return results[0]
