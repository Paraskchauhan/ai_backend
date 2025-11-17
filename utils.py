# model_utils.py
import os
import sys
from pathlib import Path

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

def download_from_drive(drive_id, dest_path):
    """
    Uses gdown library to download large files from drive.
    drive_id: the file id (string)
    dest_path: local path to save file
    """
    try:
        import gdown
    except Exception as e:
        raise RuntimeError("gdown not installed: " + str(e))

    url = f"https://drive.google.com/uc?id={drive_id}"
    # gdown will handle large files
    gdown.download(url, str(dest_path), quiet=False, fuzzy=True)
    return dest_path

def load_yolo_model(local_path):
    """
    Load YOLO model using ultralytics.YOLO
    local_path: path to .pt file
    """
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError("ultralytics not installed: " + str(e))

    model = YOLO(str(local_path))
    return model

def ensure_model_loaded(drive_id_env="MODEL_DRIVE_ID", model_filename="yolo.pt"):
    """
    Ensure model exists locally; if not, download from Drive using env var MODEL_DRIVE_ID.
    Returns loaded model object.
    """
    import os
    drive_id = os.getenv(drive_id_env)
    if not drive_id:
        raise RuntimeError(f"Environment variable {drive_id_env} not set. Set it to Google Drive FILE_ID.")

    local_path = MODEL_DIR / model_filename
    if not local_path.exists():
        print("Model not found locally — downloading from Google Drive...")
        download_from_drive(drive_id, local_path)
        print("Download finished.")
    else:
        print("Model already exists at", local_path)

    # load model
    model = load_yolo_model(local_path)
    print("YOLO model loaded.")
    return model
