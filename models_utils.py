# model_utils.py
import os
from pathlib import Path

MODEL_LOCAL_PATH = Path("model_weights")
MODEL_LOCAL_PATH.mkdir(exist_ok=True)
WEIGHT_FILE = MODEL_LOCAL_PATH / "yolo.pt"

def download_from_drive(drive_file_id):
    # downloads only if not present
    if WEIGHT_FILE.exists():
        print("Model already exists locally:", WEIGHT_FILE)
        return str(WEIGHT_FILE)
    print("Downloading model from Google Drive...")
    import gdown
    url = f"https://drive.google.com/uc?id={drive_file_id}"
    gdown.download(url, str(WEIGHT_FILE), quiet=False)
    return str(WEIGHT_FILE)

def load_yolo_model(weight_path):
    # using ultralytics loader or torch.hub for YOLOv5
    try:
        # Try ultralytics (YOLOv8) if installed
        from ultralytics import YOLO
        model = YOLO(weight_path)
        print("Loaded YOLO model with ultralytics.")
        return model
    except Exception:
        import torch
        # fallback to YOLOv5 hub (if weight compatible)
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=weight_path, force_reload=False)
        print("Loaded YOLO model via torch.hub")
        return model
