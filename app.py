from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
from PIL import Image
import torch
import os

app = Flask(__name__)
CORS(app)

# Load YOLOv5 Model
model = torch.hub.load("ultralytics/yolov5", "custom", path="best.pt", trust_repo=true)
model.eval()

@app.route("/")
def home():
    return {"message": "Backend is running!"}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get Base64 Image
        image_base64 = request.json["image"]
        image_base64 = image_base64.split(",")[1]

        img_bytes = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Inference
        results = model(img, size=640)

        df = results.pandas().xyxy[0]

        if len(df) > 0:
            label = df.iloc[0]['name']
        else:
            label = "no_object"

        return jsonify({"result": label})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # IMPORTANT for Render
    app.run(host="0.0.0.0", port=port)
