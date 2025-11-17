from flask import Flask, request, jsonify
import torch
import requests
import os

app = Flask(__name__)

MODEL_URL = "https://huggingface.co/Paraskchauhan/paras-yolo-model/resolve/main/Open%20CV.zip"
MODEL_PATH = "yolo.pt"

# Step A: Model download function
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
        print("Model downloaded!")

# Step B: Load model
download_model()
model = torch.hub.load('ultralytics/yolov5', 'custom', MODEL_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    img_file = request.files['image']
    img_path = "input.jpg"
    img_file.save(img_path)

    results = model(img_path)
    data = results.pandas().xyxy[0].to_dict(orient="records")

    return jsonify({"detections": data})

@app.route('/')
def home():
    return "AI Model Running Successfully!"

if __name__ == '__main__':
    app.run()
