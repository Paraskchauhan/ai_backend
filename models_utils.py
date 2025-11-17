# model_utils.py
import os
from pathlib import Path

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "yolo.pt"

def download_from_gdrive(file_id, out_path):
    """Download using gdown (works for large files)."""
    try:
        import gdown
    except ImportError:
        raise Exception("gdown not installed. Add it to requirements.txt")
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading model from {url} to {out_path} ...")
    gdown.download(url, str(out_path), quiet=False)

def load_model_from_file(model_path):
    """Load YOLO model - adjust loader if you used a specific library."""
    try:
        import torch
        # Try ultralytics/YOLO approach first (modern)
        try:
            from ultralytics import YOLO
            model = YOLO(str(model_path))
            print("Loaded model with ultralytics.YOLO")
            return model, "ultralytics"
        except Exception:
            # fallback to torch.hub (yolov5 style)
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(model_path), force_reload=False)
            print("Loaded model with torch.hub (yolov5 custom)")
            return model, "yolov5"
    except Exception as e:
        print("Model load error:", e)
        raise

def ensure_model(file_id_env_var="MODEL_GDRIVE_ID"):
    """
    Ensure model file exists locally; if not, download from Google Drive.
    Returns (model_object, loader_type)
    """
    import os
    file_id = os.getenv(file_id_env_var)
    if not file_id:
        raise Exception(f"Environment variable {file_id_env_var} not set (Google Drive file id).")
    if not MODEL_PATH.exists():
        download_from_gdrive(file_id, MODEL_PATH)
    model, loader = load_model_from_file(MODEL_PATH)
    return model, loader
